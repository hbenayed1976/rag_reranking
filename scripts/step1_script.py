import os
import json
import time
import pandas as pd
import re
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import CrossEncoder

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Metrics
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --------------------------------------------------
# CONFIGURATION ÉTAPE 1
# --------------------------------------------------
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

MAX_TOKENS = 512
TOP_K = 20
TOP_N_FINAL = 5

# 10 modèles d'embeddings à tester
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
    "AraBERT_Large": "aubmindlab/bert-large-arabertv02"
}

# 3 configurations de reranking (chacun seul)
RERANKER_CONFIGURATIONS = {
    "cross_encoder_only": {
        "use_reranking": True,
        "method": "cross_encoder",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Cross-Encoder only"
    },
    "mini_reranker_only": {
        "use_reranking": True,
        "method": "mini_reranker",
        "model": "prithivida/miniReranker_arabic_v1",
        "description": "miniReranker only"
    },
    "ara_reranker_only": {
        "use_reranking": True,
        "method": "ara_reranker",
        "model": "Omartificial-Intelligence-Space/ARA-Reranker-V1",
        "description": "ARA-Reranker only"
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
# CUSTOM RERANKER CLASSES
# --------------------------------------------------
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import Optional, Any

class CustomCrossEncoderRetriever(BaseRetriever):
    """Retriever utilisant sentence_transformers.CrossEncoder"""
    
    base_retriever: Any
    model_name: str
    top_n: int
    model: Any = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, base_retriever: Any, model_name: str, top_n: int):
        super().__init__(base_retriever=base_retriever, model_name=model_name, top_n=top_n)
        print(f"Loading CrossEncoder from: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"✅ CrossEncoder loaded")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        if not docs:
            return []
        
        # Préparer les paires (query, document)
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Scoring
        scores = self.model.predict(pairs)
        
        # Trier et prendre top_n
        sorted_indices = scores.argsort()[::-1][:self.top_n]
        reranked_docs = [docs[i] for i in sorted_indices]
        
        return reranked_docs

# --------------------------------------------------
# UTILITAIRES
# --------------------------------------------------
def create_reranker_retriever(base_retriever, config: Dict):
    """Créer le retriever approprié selon la configuration"""
    method = config.get("method")
    model_name = config.get("model")
    
    try:
        if method == "cross_encoder":
            return CustomCrossEncoderRetriever(
                base_retriever=base_retriever,
                model_name=model_name,
                top_n=TOP_N_FINAL
            )
        
        elif method == "mini_reranker":
            return CustomCrossEncoderRetriever(
                base_retriever=base_retriever,
                model_name=model_name,
                top_n=TOP_N_FINAL
            )
        
        elif method == "ara_reranker":
            return CustomCrossEncoderRetriever(
                base_retriever=base_retriever,
                model_name=model_name,
                top_n=TOP_N_FINAL
            )
        
        else:
            return base_retriever
            
    except Exception as e:
        print(f"⚠️ Erreur lors de la création du retriever: {e}")
        raise e  # On propage l'erreur pour stopper l'exécution

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
class Step1RAGComparator:
    def __init__(self, qcm_data: Dict, text_file_path: str):
        self.qcm_data = qcm_data
        self.text_file_path = text_file_path

    def _save_summary_by_method(self, df: pd.DataFrame):
        """Sauvegarder un fichier résumé par méthode de re-ranking"""
        print("\n" + "="*70)
        print("💾 Création des fichiers résumés par méthode (ÉTAPE 1)...")
        print("="*70)
        
        df_valid = df[df['error_message'].isna()].copy()
        
        if len(df_valid) == 0:
            print("⚠️ Aucune donnée valide pour créer les résumés")
            return
        
        method_files = {
            "cross_encoder_only": "step1_summary_cross_encoder.csv",
            "mini_reranker_only": "step1_summary_mini_reranker.csv",
            "ara_reranker_only": "step1_summary_ara_reranker.csv"
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
        print(f"Re-ranking: {config['description']}")
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
        if config["use_reranking"]:
            retriever = create_reranker_retriever(base_retriever, config)
            print(f"✅ Retriever configured with {config['description']}")

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

        total_combinations = len(MODELS_TO_TEST) * len(RERANKER_CONFIGURATIONS)
        print(f"\n🚀 ÉTAPE 1: Comparaison des embeddings avec rerankers simples")
        print(f"📊 Total combinations: {total_combinations}")
        print(f"📊 Embedding models: {len(MODELS_TO_TEST)}")
        print(f"🔄 Reranking configs: {len(RERANKER_CONFIGURATIONS)}")

        for model_name, model_path in MODELS_TO_TEST.items():
            for config_name, config_params in RERANKER_CONFIGURATIONS.items():
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

                        smoothie = SmoothingFunction().method4
                        bleu = sentence_bleu([[correct_answer]], model_answer, smoothing_function=smoothie)
                        f1 = f1_score([1], [1 if is_correct else 0], zero_division=0)

                        entry = {
                            "model": model_name,
                            "configuration": config_name,
                            "question_id": qid,
                            "question": question,
                            "correct_answer": correct_answer,
                            "model_answer": model_answer,
                            "is_correct": is_correct,
                            "bleu": bleu,
                            "f1": f1,
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
                    mean_bleu = df_config["bleu"].mean()
                    mean_f1 = df_config["f1"].mean()
                    print(f"\n📊 Résumé {model_name} + {config_name}:")
                    print(f"   Accuracy       : {acc:.2%}")
                    print(f"   Mean Resp. Time: {mean_time:.2f} sec")
                    print(f"   Mean BLEU      : {mean_bleu:.3f}")
                    print(f"   Mean F1        : {mean_f1:.3f}")

        df = pd.DataFrame(results)
        df.to_csv("step1_results_full.csv", index=False, encoding="utf-8-sig")
        print("✅ Fichier complet sauvegardé: step1_results_full.csv")

        self._save_summary_by_method(df)

        print("\n" + "="*70)
        print("=== STEP 1 FINAL SUMMARY ===")
        print("="*70)
        
        if not df.empty:
            summary = df.groupby(["model", "configuration"]).agg({
                "is_correct": "mean",
                "bleu": "mean",
                "f1": "mean",
                "response_time": "mean"
            }).round(4)
            print(summary)
            
            print("\n🏆 Ranking by Accuracy:")
            ranking = summary.sort_values("is_correct", ascending=False)
            for i, (idx, row) in enumerate(ranking.iterrows(), 1):
                model, config = idx
                print(f"{i}. {model} ({config}): {row['is_correct']:.2%} accuracy | {row['response_time']:.2f}s avg")

        print(f"\n⏱️  Total execution time: {(time.time()-total_start_time)/60:.2f} minutes")
        
        print("\n" + "="*70)
        print("📁 FICHIERS GÉNÉRÉS (ÉTAPE 1):")
        print("="*70)
        print("1. step1_results_full.csv (données complètes)")
        print("2. step1_summary_cross_encoder.csv")
        print("3. step1_summary_mini_reranker.csv")
        print("4. step1_summary_ara_reranker.csv")
        print("\n➡️  Prochaine étape: Exécutez analyze_step1_results.py")
        print("="*70)
        
        return df

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    qcm_file = "qcm_test_140QA.json"
    text_file = "dataset_700QA.txt"

    if not os.path.exists(qcm_file):
        print(f"❌ File not found: {qcm_file}")
        return

    with open(qcm_file, 'r', encoding='utf-8') as f:
        qcm_data = json.load(f)

    print("\n" + "="*70)
    print("🚀 ÉTAPE 1: Comparaison Embeddings × Rerankers Simples")
    print("="*70)
    print(f"📊 Testing {len(MODELS_TO_TEST)} embedding models")
    print(f"❓ Processing {len(qcm_data)} questions")
    print(f"🔄 Reranking strategies: {len(RERANKER_CONFIGURATIONS)}")
    print("="*70)
    
    try:
        comparator = Step1RAGComparator(qcm_data, text_file)
        question_ids = list(qcm_data.keys())
        results = comparator.run_comparison(question_ids)
    except Exception as e:
        print(f"\n❌ SCRIPT TERMINATED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
