"""
Script Étape 2 - Test des pipelines dual sur TOUS les 11 modèles d'embeddings
Fichier: step2_compare_all_models_dual.py
"""

import os
import json
import time
import pandas as pd
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

# --------------------------------------------------
# CONFIGURATION ÉTAPE 2
# --------------------------------------------------
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

MAX_TOKENS = 512
TOP_K = 20
TOP_N_STAGE1 = 10  # Après stage 1 (miniReranker)
TOP_N_FINAL = 5    # Après stage 2 (Cross-Encoder ou ARA-Reranker)

# TOUS les 11 modèles d'embeddings
MODELS_TO_TEST = {
    "MiniLM-L12": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "AraModernBert-STS": "NAMAA-Space/AraModernBert-Base-STS",
    "MarBERTv2": "UBC-NLP/MARBERTv2",
    "Multilingual-E5": "intfloat/multilingual-e5-base",
    "Arabic-Triplet-Matryoshka-V2": "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    "Arabic-SBERT-100K": "akhooli/Arabic-SBERT-100K",
    "AraBERTv2": "aubmindlab/bert-base-arabertv2",
    "CAMeLBERT": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "DistilBERT_Arabic": "asafaya/bert-base-arabic",
    "AraBERT_Large": "aubmindlab/bert-large-arabertv02",
    "IslamQA-BGE-M3": "IslamQA/bge-m3-finetuned"
}

# 2 configurations de dual reranking
DUAL_RERANKER_CONFIGURATIONS = {
    "dual_mini_cross": {
        "use_dual": True,
        "stage1_model": "prithivida/miniReranker_arabic_v1",
        "stage2_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "miniReranker → Cross-Encoder"
    },
    "dual_mini_ara": {
        "use_dual": True,
        "stage1_model": "prithivida/miniReranker_arabic_v1",
        "stage2_model": "Omartificial-Intelligence-Space/ARA-Reranker-V1",
        "description": "miniReranker → ARA-Reranker"
    }
}

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
FIQH_TEMPLATE = """أنت خبير في الفقه المالكي. أجب عن السؤال التالي بالاعتماد على السياق المعطى.
 قدم اجابة مختصرة مع تحديد الخيار الصحيح (أ، ب، ج).

السياق:
{context}

السؤال: {question}

الخيارات:
{options}

"""

# --------------------------------------------------
# DUAL RERANKING RETRIEVER
# --------------------------------------------------
class DualRerankerRetriever(BaseRetriever):
    """Retriever combinant 2 Cross-Encoders en pipeline"""
    
    base_retriever: Any
    stage1_model: str
    stage2_model: str
    top_n_stage1: int
    top_n_final: int
    stage1_reranker: Any = None
    stage2_reranker: Any = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, base_retriever: Any, stage1_model: str, stage2_model: str, 
                 top_n_stage1: int, top_n_final: int):
        super().__init__(
            base_retriever=base_retriever,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            top_n_stage1=top_n_stage1,
            top_n_final=top_n_final
        )
        
        # Stage 1: Premier reranker
        print(f"  Stage 1: Loading {stage1_model}")
        self.stage1_reranker = CrossEncoder(stage1_model)
        print(f"  ✅ Stage 1 loaded")
        
        # Stage 2: Deuxième reranker
        print(f"  Stage 2: Loading {stage2_model}")
        self.stage2_reranker = CrossEncoder(stage2_model)
        print(f"  ✅ Stage 2 loaded")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Récupérer les documents initiaux
        docs = self.base_retriever.get_relevant_documents(query)
        
        if not docs:
            return []
        
        # Stage 1: Premier filtrage
        pairs_stage1 = [[query, doc.page_content] for doc in docs]
        scores_stage1 = self.stage1_reranker.predict(pairs_stage1)
        
        # Garder top_n_stage1 documents
        sorted_indices_stage1 = scores_stage1.argsort()[::-1][:self.top_n_stage1]
        docs_stage1 = [docs[i] for i in sorted_indices_stage1]
        
        # Stage 2: Re-ranking final
        if len(docs_stage1) <= self.top_n_final:
            return docs_stage1
        
        pairs_stage2 = [[query, doc.page_content] for doc in docs_stage1]
        scores_stage2 = self.stage2_reranker.predict(pairs_stage2)
        
        # Garder top_n_final documents
        sorted_indices_stage2 = scores_stage2.argsort()[::-1][:self.top_n_final]
        final_docs = [docs_stage1[i] for i in sorted_indices_stage2]
        
        return final_docs

# --------------------------------------------------
# UTILITAIRES
# --------------------------------------------------
def create_dual_retriever(base_retriever, config: Dict):
    """Créer le dual retriever"""
    try:
        return DualRerankerRetriever(
            base_retriever=base_retriever,
            stage1_model=config["stage1_model"],
            stage2_model=config["stage2_model"],
            top_n_stage1=TOP_N_STAGE1,
            top_n_final=TOP_N_FINAL
        )
    except Exception as e:
        print(f"⚠️ Erreur lors de la création du dual retriever: {e}")
        raise e

def validate_answer(model_answer: str, correct_answer: str) -> bool:
    """Méthode d'extraction de réponse améliorée"""
    if not model_answer or not correct_answer:
        return False
    
    arabic_to_latin = {'أ': 'A', 'ب': 'B', 'ج': 'C'}
    
    correct_normalized = correct_answer.strip().upper()
    if correct_normalized in arabic_to_latin:
        correct_normalized = arabic_to_latin[correct_normalized]
    
    primary_pattern = r'[ABCأبج](?=[.\s:،؛]|$)'
    matches = re.findall(primary_pattern, model_answer)
    
    if not matches:
        fallback_pattern = r'\b[ABCأبج]\b'
        matches = re.findall(fallback_pattern, model_answer)
    
    if not matches:
        emergency_pattern = r'[ABCأبج]'
        matches = re.findall(emergency_pattern, model_answer)
    
    if not matches:
        return False
    
    last_answer = matches[-1].upper()
    
    if last_answer in arabic_to_latin:
        last_answer = arabic_to_latin[last_answer]
    
    return last_answer == correct_normalized

# --------------------------------------------------
# CLASSE PRINCIPALE
# --------------------------------------------------
class Step2RAGComparator:
    def __init__(self, qcm_data: Dict, text_file_path: str, models_dict: Dict):
        self.qcm_data = qcm_data
        self.text_file_path = text_file_path
        self.models_dict = models_dict

    def _save_summary_by_method(self, df: pd.DataFrame):
        """Sauvegarder un fichier résumé par méthode de re-ranking"""
        print("\n" + "="*70)
        print("💾 Création des fichiers résumés par méthode (ÉTAPE 2)...")
        print("="*70)
        
        df_valid = df[df['error_message'].isna()].copy()
        
        if len(df_valid) == 0:
            print("⚠️ Aucune donnée valide pour créer les résumés")
            return
        
        method_files = {
            "dual_mini_cross": "step2_summary_dual_mini_cross.csv",
            "dual_mini_ara": "step2_summary_dual_mini_ara.csv"
        }
        
        for config_name, filename in method_files.items():
            df_config = df_valid[df_valid['configuration'] == config_name].copy()
            
            if len(df_config) == 0:
                print(f"⚠️ Aucune donnée pour {config_name}")
                continue
            
            summary = df_config.groupby('model').agg({
                'is_correct': 'mean',
                'response_time': 'mean'
            }).reset_index()
            
            summary.columns = ['Model', 'Accuracy (%)', 'Mean Response Time (s)']
            summary['Accuracy (%)'] = (summary['Accuracy (%)'] * 100).round(2)
            summary['Mean Response Time (s)'] = summary['Mean Response Time (s)'].round(2)
            
            summary = summary.sort_values('Accuracy (%)', ascending=False)
            
            summary.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ {filename} créé ({len(summary)} modèles)")
            
            print(f"\n📊 Aperçu {config_name}:")
            print(summary.to_string(index=False))
            print()

    def initialize_rag_system(self, model_name: str, model_path: str, config: Dict):
        print(f"\n{'='*70}")
        print(f"Initialisation: {model_name}")
        print(f"Pipeline: {config['description']}")
        print(f"{'='*70}")

        with open(self.text_file_path, 'r', encoding='utf-8') as f:
            raw_chunks = f.read().split('***')
        texts = [Document(page_content=chunk.strip()) for chunk in raw_chunks if chunk.strip()]

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )

        vectorstore = FAISS.from_documents(texts, embeddings)
        base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

        retriever = base_retriever
        if config["use_dual"]:
            retriever = create_dual_retriever(base_retriever, config)
            print(f"✅ Dual pipeline configured: {config['description']}")

        llm = OllamaLLM(
            model=OLLAMA_MODEL, 
            base_url=OLLAMA_BASE_URL,
            temperature=0.05, 
            top_k=40, 
            top_p=0.95, 
            num_predict=MAX_TOKENS
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=FIQH_TEMPLATE, 
                    input_variables=["context", "question", "options"]
                )
            },
            return_source_documents=True,
            verbose=False
        )
        return qa_chain

    def run_comparison(self, question_ids: List[str]):
        results = []
        total_start_time = time.time()

        total_combinations = len(self.models_dict) * len(DUAL_RERANKER_CONFIGURATIONS)
        print(f"\n🚀 ÉTAPE 2: Test des pipelines dual sur TOUS les modèles")
        print(f"📊 Total combinations: {total_combinations}")
        print(f"📦 Modèles d'embeddings: {len(self.models_dict)}")
        print(f"🔄 Dual pipelines: {len(DUAL_RERANKER_CONFIGURATIONS)}")

        for model_name, model_path in self.models_dict.items():
            
            for config_name, config_params in DUAL_RERANKER_CONFIGURATIONS.items():
                print(f"\n{'*'*70}")
                print(f"🔎 Testing: {model_name} + {config_name}")
                print(f"{'*'*70}")
                
                try:
                    qa_chain = self.initialize_rag_system(model_name, model_path, config_params)
                except Exception as e:
                    print(f"❌ Init error {model_name}+{config_name}: {e}")
                    results.append({
                        "model": model_name,
                        "configuration": config_name,
                        "error_message": str(e)
                    })
                    continue

                config_results = []
                for qid in question_ids:
                    q = self.qcm_data[qid]
                    question = q["question"]
                    options = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
                    correct_answer = q["answer_letter"]

                    try:
                        start_time = time.time()
                        response = qa_chain({"question": question, "options": options, "chat_history": []})
                        response_time = time.time() - start_time

                        model_answer = response.get("answer", "")
                        is_correct = validate_answer(model_answer, correct_answer)

                        entry = {
                            "model": model_name,
                            "configuration": config_name,
                            "question_id": qid,
                            "question": question,
                            "correct_answer": correct_answer,
                            "model_answer": model_answer,
                            "is_correct": is_correct,
                            "response_time": response_time,
                            "error_message": None
                        }
                        results.append(entry)
                        config_results.append(entry)

                    except Exception as e:
                        print(f"⚠️ Error {model_name}+{config_name} on Q{qid}: {e}")
                        results.append({
                            "model": model_name,
                            "configuration": config_name,
                            "question_id": qid,
                            "error_message": str(e)
                        })
                        continue

                if config_results:
                    df_config = pd.DataFrame(config_results)
                    acc = df_config["is_correct"].mean()
                    mean_time = df_config["response_time"].mean()
                    
                    print(f"\n📊 Résumé {model_name} + {config_name}:")
                    print(f"   Accuracy       : {acc:.2%}")
                    print(f"   Mean Resp. Time: {mean_time:.2f} sec")

        df = pd.DataFrame(results)
        df.to_csv("step2_results_full.csv", index=False, encoding="utf-8-sig")
        print("\n✅ Fichier complet sauvegardé: step2_results_full.csv")

        self._save_summary_by_method(df)

        print("\n" + "="*70)
        print("=== STEP 2 FINAL SUMMARY ===")
        print("="*70)
        
        if not df.empty:
            df_valid = df[df['error_message'].isna()].copy()
            
            if len(df_valid) > 0:
                summary = df_valid.groupby(["model", "configuration"]).agg({
                    "is_correct": "mean",
                    "response_time": "mean"
                }).round(4)
                print(summary)
                
                print("\n🏆 Ranking by Accuracy (Étape 2):")
                ranking = summary.sort_values("is_correct", ascending=False)
                for i, (idx, row) in enumerate(ranking.iterrows(), 1):
                    model, config = idx
                    print(f"{i}. {model} ({config}): {row['is_correct']:.2%} | {row['response_time']:.2f}s avg")
                
                # Meilleure combinaison globale
                print("\n" + "="*70)
                print("🥇 MEILLEURE COMBINAISON GLOBALE")
                print("="*70)
                best_combo = ranking.iloc[0]
                best_model, best_config = ranking.index[0]
                print(f"Modèle: {best_model}")
                print(f"Pipeline: {best_config}")
                print(f"Accuracy: {best_combo['is_correct']:.2%}")
                print(f"Mean Time: {best_combo['response_time']:.2f}s")

        print(f"\n⏱️ Total execution time: {(time.time()-total_start_time)/60:.2f} minutes")
        
        print("\n" + "="*70)
        print("📁 FICHIERS GÉNÉRÉS (ÉTAPE 2):")
        print("="*70)
        print("1. step2_results_full.csv (données complètes)")
        print("2. step2_summary_dual_mini_cross.csv")
        print("3. step2_summary_dual_mini_ara.csv")
        print("="*70)
        
        return df

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    qcm_file = "qcm_test_140QA.json"
    text_file = "dataset_700QA.txt"

    # Charger les données QCM
    if not os.path.exists(qcm_file):
        print(f"❌ File not found: {qcm_file}")
        return

    with open(qcm_file, 'r', encoding='utf-8') as f:
        qcm_data = json.load(f)

    print("\n" + "="*70)
    print("🚀 ÉTAPE 2: Test des Pipelines Dual sur TOUS les 11 modèles")
    print("="*70)
    print(f"📦 {len(MODELS_TO_TEST)} modèles d'embeddings")
    print(f"❓ Processing {len(qcm_data)} questions")
    print(f"🔄 Dual pipelines: {len(DUAL_RERANKER_CONFIGURATIONS)}")
    
    print("\n📋 Modèles d'embeddings à tester:")
    for i, (model_name, model_path) in enumerate(MODELS_TO_TEST.items(), 1):
        print(f"   {i}. {model_name}")
    
    print("\n📋 Configurations dual reranking:")
    for config_name, config_info in DUAL_RERANKER_CONFIGURATIONS.items():
        print(f"   • {config_name}: {config_info['description']}")
    
    print("="*70)
    
    try:
        comparator = Step2RAGComparator(qcm_data, text_file, MODELS_TO_TEST)
        question_ids = list(qcm_data.keys())
        results = comparator.run_comparison(question_ids)
    except Exception as e:
        print(f"\n❌ SCRIPT TERMINATED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
