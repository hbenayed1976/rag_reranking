"""
pipeline_utils.py — Composants partagés du pipeline RAG Arabic Fiqh.

Contient :
- EmbeddingRetriever : construction de l'index FAISS + retrieval MMR (λ=0.7)
  avec filtre de dédoublonnage à 0.5, conformément à la Section 3.1 du papier
  (corrige l'écart entre la description méthodologique et le code original,
  qui n'implémentait qu'un similarity_search simple).
- CrossEncoderReranker : wrapper autour de sentence_transformers.CrossEncoder.
- HFGenerator : chargement et génération via un LLM HuggingFace, 4-bit par défaut pour tenir sur un GPU L4 24 Go.
- validate_answer : extraction robuste de la réponse (أ/ب/ج ↔ A/B/C).
- Utilitaires de checkpointing pour permettre l'arrêt/reprise d'un run long
  (essentiel : 27 720 générations au total, réparties sur plusieurs sessions).
- Utilitaires de mesure de ressources (RAM/VRAM) et de statistiques
  (bootstrap CI, test de McNemar).
"""

import os
import re
import gc
import json
import time
import platform
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import psutil

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import config


from answer_extraction_fixed import (
    validate_answer as _fixed_validate_answer,
    extract_answer_letter as _fixed_extract_answer_letter,
)


# ==================================================
# VALIDATION DE REPONSE
# ==================================================
def validate_answer(model_answer: str, correct_answer: str) -> bool:
    """
    Extraction de la lettre de réponse (أ/ب/ج/د ↔ A/B/C/D).

    Délègue à answer_extraction_fixed.validate_answer (regex corrigé :
    couvre les 4 lettres أبجد, priorise la dernière déclaration explicite
    "الإجابة الصحيحة هي: X" si présente, sinon la première lettre isolée
    valide). `correct_answer` peut être fourni en lettre arabe (أ/ب/ج/د)
    ou latine (A/B/C/D) ; les deux sont acceptés pour compatibilité avec
    les appelants existants.
    """
    if not model_answer or not correct_answer:
        return False

    arabic_to_latin = {'أ': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D'}
    latin_to_arabic = {v: k for k, v in arabic_to_latin.items()}

    correct_normalized = correct_answer.strip()
    if correct_normalized.upper() in latin_to_arabic:
        # déjà une lettre latine A/B/C/D
        correct_arabic = latin_to_arabic[correct_normalized.upper()]
    elif correct_normalized in arabic_to_latin:
        correct_arabic = correct_normalized
    else:
        return False

    return _fixed_validate_answer(model_answer, correct_arabic)


def validate_answer_legacy_buggy(model_answer: str, correct_answer: str) -> bool:
    
    if not model_answer or not correct_answer:
        return False

    arabic_to_latin = {'أ': 'A', 'ب': 'B', 'ج': 'C'}

    correct_normalized = correct_answer.strip().upper()
    if correct_normalized in arabic_to_latin:
        correct_normalized = arabic_to_latin[correct_normalized]

    primary_pattern = r'[ABCأبج](?=[.\s:،؛]|$)'
    matches = re.findall(primary_pattern, model_answer)

    if not matches:
        matches = re.findall(r'\b[ABCأبج]\b', model_answer)

    if not matches:
        matches = re.findall(r'[ABCأبج]', model_answer)  # tier 3 original, non borné

    if not matches:
        return False

    last_answer = matches[-1].upper()
    if last_answer in arabic_to_latin:
        last_answer = arabic_to_latin[last_answer]

    return last_answer == correct_normalized


def extract_answer_letter(model_answer: str):
    """Renvoie la lettre latine (A/B/C/D) extraite, ou None. Alias direct
    d'answer_extraction_fixed.extract_answer_letter, exposé ici pour que
    les scripts qui importent déjà pipeline_utils n'aient pas besoin d'un
    second import."""
    return _fixed_extract_answer_letter(model_answer)


# ==================================================
# RETRIEVAL : MMR + FILTRE DE DEDUPLICATION (Section 3.1)
# ==================================================
class EmbeddingRetriever:
    """
    Construit un index FAISS (inner-product, vecteurs L2-normalisés) pour un
    modèle d'embedding donné, et implémente le retrieval décrit dans la
    Section 3.1 : MMR (λ=0.7) sur les TOP_K=20 candidats, suivi d'un filtre
    de similarité à 0.5 pour retirer les passages quasi-dupliqués (le filtre
    compare chaque nouveau document sélectionné aux documents déjà retenus ;
    un document dont la similarité cosinus avec un document déjà retenu
    dépasse le seuil est écarté, sans re-normaliser les scores MMR restants).
    """

    def __init__(self, embedding_path: str, device: str):
        self.embedding_path = embedding_path
        self.device = device

        with open(config.TEXT_FILE, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("***")
        cleaned_chunks = [c.strip() for c in raw_chunks if c.strip()]

        # passage_id stable : position dans le fichier + hash court du contenu.
        # Le hash rend l'identifiant robuste à un réordonnancement du fichier
        # source (utile pour l'annotation de pertinence, qui doit référencer
        # les mêmes passages de façon non ambiguë quel que soit l'embedding).
        import hashlib
        self.documents = []
        for idx, chunk in enumerate(cleaned_chunks):
            passage_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:10]
            passage_id = f"{idx:04d}_{passage_hash}"
            self.documents.append(Document(page_content=chunk, metadata={"passage_id": passage_id}))

        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_path,
            model_kwargs={"trust_remote_code": True, "device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings_model)

    def retrieve(self, query: str, k: int = config.TOP_K,
                 method: str = config.RETRIEVAL_METHOD) -> List[Document]:
        """
        method="similarity" (défaut) : reproduit exactement le retrieval des
        scripts originaux (vectorstore.similarity_search(query, k=k)), sans
        MMR ni filtre de dédoublonnage. Utilisé pour la comparaison contrôlée
        avec les résultats du papier original (seule variable modifiée :
        l'infrastructure d'inférence).

        method="mmr" : implémentation du MMR (λ=0.7) + filtre de
        dédoublonnage à 0.5 décrits en Section 3.1 du papier. Conservé disponible pour une analyse ultérieure
        dédiée à l'écart méthodologie/code, mais non utilisé par défaut.
        """
        if method == "similarity":
            return self.vectorstore.similarity_search(query, k=k)

        if method != "mmr":
            raise ValueError(f"method inconnu: {method!r} (attendu: 'similarity' ou 'mmr')")

        # --- Implémentation MMR (non utilisée par défaut, cf. docstring) ---
        mmr_docs = self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=config.MMR_FETCH_K, lambda_mult=config.MMR_LAMBDA
        )

        if len(mmr_docs) <= 1:
            return mmr_docs

        doc_vectors = self.embeddings_model.embed_documents([d.page_content for d in mmr_docs])
        doc_vectors = np.array(doc_vectors)
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        doc_vectors = doc_vectors / norms

        kept_indices = []
        kept_vectors = []
        for i, vec in enumerate(doc_vectors):
            is_duplicate = False
            for kept_vec in kept_vectors:
                cosine_sim = float(np.dot(vec, kept_vec))
                if cosine_sim > config.DEDUP_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(i)
                kept_vectors.append(vec)

        return [mmr_docs[i] for i in kept_indices]

    def unload(self):
        del self.vectorstore
        del self.embeddings_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================================================
# RE-RANKING EN CASCADE (0, 1 ou 2 stages)
# ==================================================
class RerankerPipeline:
    def __init__(self, scenario: str, device: str, preloaded: Optional[Dict[str, CrossEncoder]] = None):
        self.scenario = scenario
        self.device = device
        self.stages_def = config.RERANKING_SCENARIOS_DEF[scenario]["stages"]
        self._models: Dict[str, CrossEncoder] = preloaded or {}
        self._owns_models = preloaded is None

        if self._owns_models:
            for reranker_name, _ in self.stages_def:
                if reranker_name not in self._models:
                    model_path = config.RERANKER_MODEL_PATHS[reranker_name]
                    self._models[reranker_name] = CrossEncoder(model_path, device=device)

    def apply(self, docs: List[Document], query: str) -> List[Document]:
        if not self.stages_def:
            return docs[:config.FINAL_N]

        current_docs = docs
        for reranker_name, top_n in self.stages_def:
            if not current_docs:
                return []
            model = self._models[reranker_name]
            pairs = [[query, d.page_content] for d in current_docs]
            scores = model.predict(pairs)
            sorted_idx = np.argsort(scores)[::-1][:top_n]
            current_docs = [current_docs[i] for i in sorted_idx]
        return current_docs

    def unload(self):
        if self._owns_models:
            for m in self._models.values():
                del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_all_rerankers(device: str) -> Dict[str, CrossEncoder]:
    """Charge les 3 CrossEncoder une seule fois, à réutiliser pour tous les
    scénarios et tous les embeddings lors de la construction du cache."""
    return {
        name: CrossEncoder(path, device=device)
        for name, path in config.RERANKER_MODEL_PATHS.items()
    }


def unload_rerankers(rerankers: Dict[str, CrossEncoder]):
    for m in rerankers.values():
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================
# GENERATION (HuggingFace, remplace Ollama)
# ==================================================
class HFGenerator:
    """
    Charge un modèle causal HuggingFace et génère du texte avec les
    paramètres de décodage du papier (temperature=0.05, top_k=40, top_p=0.95,
    max_new_tokens=512). 4-bit (nf4) par défaut pour tenir sur un GPU L4
    24 Go avec de la marge pour les embeddings/rerankers.
    """

    def __init__(self, model_id: str, device: str = config.DEFAULT_DEVICE,
                 quantize_4bit: bool = config.DEFAULT_QUANTIZE_4BIT):
        self.model_id = model_id
        self.device = device
        self.quantize_4bit = quantize_4bit and device == "cuda" and BITSANDBYTES_AVAILABLE

        if quantize_4bit and not self.quantize_4bit:
            print(f"   ⚠️ 4-bit demandé mais indisponible (device={device}, "
                  f"bitsandbytes_available={BITSANDBYTES_AVAILABLE}) → repli bf16/float32.")

        mode_label = "4-bit (nf4)" if self.quantize_4bit else ("bf16" if device == "cuda" else "float32")
        print(f"   Loading LLM '{model_id}' on {device} [{mode_label}]...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HF_TOKEN)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        load_kwargs = {"token": config.HF_TOKEN, "low_cpu_mem_usage": True}

        if self.quantize_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
            if device == "cuda":
                load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.GENERATION_PARAMS["max_new_tokens"],
                do_sample=True,
                temperature=config.GENERATION_PARAMS["temperature"],
                top_k=config.GENERATION_PARAMS["top_k"],
                top_p=config.GENERATION_PARAMS["top_p"],
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==================================================
# MONITORING RESSOURCES
# ==================================================
def get_resource_snapshot(device: str) -> Dict:
    process = psutil.Process(os.getpid())
    snapshot = {
        "ram_mb": process.memory_info().rss / (1024 ** 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }
    if device == "cuda" and torch.cuda.is_available():
        snapshot["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        snapshot["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        snapshot["gpu_allocated_mb"] = None
        snapshot["gpu_max_allocated_mb"] = None
    return snapshot


def print_environment_banner(device: str):
    print("=" * 70)
    print("🖥️  ENVIRONNEMENT D'EXECUTION")
    print("=" * 70)
    print(f"   Plateforme   : {platform.platform()}")
    print(f"   CPU logiques : {psutil.cpu_count(logical=True)}")
    print(f"   RAM totale   : {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if device == "cuda" and torch.cuda.is_available():
        print(f"   GPU          : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM totale  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("=" * 70)


# ==================================================
# CHECKPOINT / REPRISE
# ==================================================
def load_completed_keys(csv_path: str, key_cols: List[str]) -> Set[Tuple]:
    """Renvoie l'ensemble des combinaisons déjà traitées dans un CSV existant,
    pour permettre de reprendre un run interrompu sans dupliquer de travail."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    if not all(c in df.columns for c in key_cols):
        return set()
    return set(tuple(row) for row in df[key_cols].itertuples(index=False, name=None))


def append_row(csv_path: str, row: Dict):
    """Ajoute une ligne à un CSV, en créant l'en-tête si le fichier n'existe pas.
    Permet une sauvegarde incrémentale question par question (pas de perte de
    données en cas de crash / coupure)."""
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


# ==================================================
# STATISTIQUES (réponse au Reviewer #2)
# ==================================================
def bootstrap_ci(is_correct: pd.Series, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Intervalle de confiance bootstrap pour une accuracy binaire."""
    rng = np.random.default_rng(seed)
    values = is_correct.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return (float(values.mean()), float(lower), float(upper))


def mcnemar_test(correct_a: pd.Series, correct_b: pd.Series) -> Dict:
    """
    Test de McNemar pour deux configurations appariées sur les mêmes 140
    questions (ex. no-reranking vs meilleure config). Utilise la correction
    de continuité, adaptée aux petits effectifs discordants.
    """
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

    a = correct_a.to_numpy(dtype=bool)
    b = correct_b.to_numpy(dtype=bool)
    if len(a) != len(b):
        raise ValueError("Les deux séries doivent avoir la même longueur (mêmes questions).")

    both_correct = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    both_wrong = int(np.sum(~a & ~b))

    table = [[both_correct, only_a], [only_b, both_wrong]]
    result = _mcnemar(table, exact=(only_a + only_b) < 25, correction=True)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_discordant": only_a + only_b,
        "contingency_table": table,
    }


# ==================================================
# METRIQUES DE RETRIEVAL (réponse au Reviewer #2 : "retrieval quality is
# not directly measured because passage-level relevance annotations are
# missing")
# ==================================================
def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Proportion des passages pertinents connus qui figurent dans le top-k retrieved."""
    if not relevant_ids:
        return float("nan")
    top_k_ids = set(retrieved_ids[:k])
    return len(top_k_ids & relevant_ids) / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    """
    nDCG@k avec pertinence graduée (0/1/2...). relevance_map ne contient que
    les passages annotés (pool) ; tout passage retrieved absent du pool est
    traité comme pertinence 0 (convention standard en pooling IR : un
    document hors du pool annoté n'a jamais été jugé pertinent par personne).
    """
    def dcg(ordered_relevances: List[float]) -> float:
        score = 0.0
        for i, rel in enumerate(ordered_relevances[:k]):
            score += (2 ** rel - 1) / np.log2(i + 2)  # rang 1-indexé -> log2(rank+1)
        return score

    actual_relevances = [relevance_map.get(doc_id, 0.0) for doc_id in retrieved_ids]
    actual_dcg = dcg(actual_relevances)

    ideal_relevances = sorted(relevance_map.values(), reverse=True)
    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return float("nan")
    return actual_dcg / ideal_dcg


def parse_justification_output(raw_output: str) -> Tuple[str, str]:
    """
    Extrait (lettre_choisie, texte_justification) depuis une sortie générée
    avec JUSTIFICATION_TEMPLATE / BASELINE_JUSTIFICATION_TEMPLATE (format
    attendu : "الخيار الصحيح: <lettre>" puis "السبب: <texte>"). Répartition
    robuste si le modèle ne respecte pas exactement le format demandé :
    à défaut de trouver le marqueur "السبب", la justification est le texte
    complet (moins la ligne de la lettre si elle est identifiable).
    """
    if not raw_output:
        return "", ""

    letter_match = re.search(r'الخيار الصحيح\s*[:：]?\s*([ABCأبج])', raw_output)
    letter = letter_match.group(1) if letter_match else ""

    reason_match = re.search(r'السبب\s*[:：]?\s*(.+)', raw_output, flags=re.DOTALL)
    if reason_match:
        justification = reason_match.group(1).strip()
    else:
        justification = raw_output.strip()

    return letter, justification


def parse_judge_output(raw_judge_output: str) -> Dict[str, Optional[float]]:
    """
    Extrait (faithfulness, hallucination, justification_quality) depuis la
    sortie structurée demandée par JUDGE_PROMPT_TEMPLATE. Renvoie des valeurs
    None si un champ n'a pas pu être extrait (ex. le juge n'a pas respecté
    le format), pour distinguer "verdict = 0" d'un échec de parsing.
    """
    result = {"faithfulness": None, "hallucination": None, "justification_quality": None}
    if not raw_judge_output:
        return result

    m = re.search(r'الأمانة\s*[:：]?\s*(\d)', raw_judge_output)
    if m:
        result["faithfulness"] = float(m.group(1))

    m = re.search(r'الهلوسة\s*[:：]?\s*(\d)', raw_judge_output)
    if m:
        result["hallucination"] = float(m.group(1))

    m = re.search(r'جودة_التبرير\s*[:：]?\s*(\d)', raw_judge_output)
    if m:
        result["justification_quality"] = float(m.group(1))

    return result


def inter_annotator_kappa(annotations_a: pd.Series, annotations_b: pd.Series) -> float:
    """Cohen's kappa entre deux annotateurs sur les mêmes items (pertinence binaire)."""
    from sklearn.metrics import cohen_kappa_score
    mask = annotations_a.notna() & annotations_b.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score(annotations_a[mask].astype(int), annotations_b[mask].astype(int)))


def weighted_kappa_likert(annotations_a: pd.Series, annotations_b: pd.Series) -> float:
    """
    Kappa pondéré linéaire (Cohen's weighted kappa) pour deux annotateurs
    notant sur une échelle ordinale (Likert 1-5). Pénalise moins un
    désaccord de 1 point (ex. 4 vs 5) qu'un désaccord de 4 points (1 vs 5),
    contrairement au kappa non pondéré utilisé pour des catégories binaires.
    """
    from sklearn.metrics import cohen_kappa_score
    mask = annotations_a.notna() & annotations_b.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score(
        annotations_a[mask].astype(int), annotations_b[mask].astype(int), weights="linear"
    ))


# ==================================================
# PROXY AUTOMATIQUE DE FAITHFULNESS (NLI)
# ==================================================
class NLIFaithfulnessScorer:
    """
    Proxy automatique et scalable de faithfulness, à valider contre les
    jugements humains sur le sous-échantillon annoté (voir
    compute_faithfulness_metrics.py) avant toute extension à plus grande
    échelle. Traite le contexte récupéré comme prémisse et la justification
    générée par le LLM comme hypothèse ; un score de faithfulness élevé
    correspond à un score d'"entailment" élevé et un score de
    "contradiction" faible.
    """

    def __init__(self, model_id: str = None, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_id = model_id or config.NLI_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = device
        self.model.to(device)
        self.model.eval()
        # mDeBERTa-v3-mnli-xnli : labels dans l'ordre [entailment, neutral, contradiction]
        self.label_order = ["entailment", "neutral", "contradiction"]

    def score(self, premise_context: str, hypothesis_justification: str) -> Dict[str, float]:
        if not premise_context.strip() or not hypothesis_justification.strip():
            return {"entailment": float("nan"), "neutral": float("nan"),
                    "contradiction": float("nan"), "faithfulness_proxy": float("nan")}

        inputs = self.tokenizer(
            premise_context, hypothesis_justification,
            truncation=True, max_length=512, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        scores = dict(zip(self.label_order, probs.tolist()))
        scores["faithfulness_proxy"] = float(scores["entailment"] - scores["contradiction"])
        return scores

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
