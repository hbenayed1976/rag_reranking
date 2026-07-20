"""
config.py — Configuration centrale du pipeline d'évaluation Arabic Fiqh RAG.

Toute constante partagée entre les scripts (modèles, chemins, prompts,
paramètres de retrieval/reranking/génération) vit ici, pour qu'un seul
fichier serve de source de vérité et facilite la reproductibilité
(Reviewer #1) et la vérification des paramètres (Reviewer #2).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CHEMINS
# --------------------------------------------------
DATA_DIR = "data"
QCM_FILE = os.path.join(DATA_DIR, "qcm_test_140QA.json")
TEXT_FILE = os.path.join(DATA_DIR, "dataset_700QA.txt")

CACHE_DIR = "cache"
RETRIEVAL_CACHE_DIR = os.path.join(CACHE_DIR, "retrieval")     # top-K brut par embedding
CONTEXT_CACHE_DIR = os.path.join(CACHE_DIR, "context")         # top-N final par (embedding, scenario)

RESULTS_DIR = "results"
BASELINE_RESULTS_DIR = os.path.join(RESULTS_DIR, "baseline")
GENERATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "generation")
COST_RESULTS_DIR = os.path.join(RESULTS_DIR, "computational_cost")

ANNOTATION_DIR = os.path.join(RESULTS_DIR, "retrieval_annotation")
RETRIEVAL_METRICS_DIR = os.path.join(RESULTS_DIR, "retrieval_metrics")
HUMAN_EVAL_DIR = os.path.join(RESULTS_DIR, "human_eval")

for d in [DATA_DIR, CACHE_DIR, RETRIEVAL_CACHE_DIR, CONTEXT_CACHE_DIR,
          RESULTS_DIR, BASELINE_RESULTS_DIR, GENERATION_RESULTS_DIR, COST_RESULTS_DIR,
          ANNOTATION_DIR, RETRIEVAL_METRICS_DIR, HUMAN_EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------
# AUTHENTIFICATION HUGGINGFACE
# --------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # requis pour meta-llama/Meta-Llama-3-8B-Instruct (gated)

# --------------------------------------------------
# MODELES DE GENERATION (LLM), identifiants HuggingFace vérifiés
# --------------------------------------------------
LLM_MODEL_IDS = {
    "Allam-7B": "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "Fanar-9B": "QCRI/Fanar-1-9B-Instruct",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",  # gated
}

# Ordre d'exécution recommandé (du plus petit au plus gros en VRAM)
LLM_EXECUTION_ORDER = ["Allam-7B", "Llama3-8B", "Fanar-9B"]

# --------------------------------------------------
# MODELES D'EMBEDDING (11, cf. Table 1 du papier)
# --------------------------------------------------
EMBEDDING_MODEL_PATHS = {
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
    "IslamQA-BGE-M3": "IslamQA/bge-m3-finetuned",
}

# --------------------------------------------------
# MODELES DE RE-RANKING (cross-encoders)
# --------------------------------------------------
RERANKER_MODEL_PATHS = {
    "mini_reranker": "prithivida/miniReranker_arabic_v1",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ara_reranker": "Omartificial-Intelligence-Space/ARA-Reranker-V1",
}

# --------------------------------------------------
# PARAMETRES DE RETRIEVAL (cf. Section 3.1 du papier)
# --------------------------------------------------
TOP_K = 20            # candidats initiaux récupérés
RETRIEVAL_METHOD = "similarity"  # "similarity" (défaut, = scripts originaux) ou "mmr"
MMR_LAMBDA = 0.7       # compromis pertinence/diversité (utilisé seulement si method="mmr")
MMR_FETCH_K = 40       # pool candidat avant sélection MMR (utilisé seulement si method="mmr")
DEDUP_SIMILARITY_THRESHOLD = 0.5  # filtre post-MMR (utilisé seulement si method="mmr")

TOP_N_STAGE1 = 10     # après le 1er reranker, en configuration dual-stage
FINAL_N = 5           # documents finaux fournis au LLM

# --------------------------------------------------
# ANNOTATION DE PERTINENCE POUR LES METRIQUES DE RETRIEVAL
# (réponse au Reviewer #2 : "retrieval quality is not directly measured")
# --------------------------------------------------
N_QUESTIONS_TO_ANNOTATE = 40   # sous-échantillon annoté manuellement
POOL_DEPTH_PER_EMBEDDING = 10  # profondeur de pooling par embedding (top-N de chaque)
ANNOTATION_RANDOM_SEED = 42
RELEVANCE_LEVELS = {0: "not relevant", 1: "relevant", 2: "highly relevant"}

# --------------------------------------------------
# EVALUATION HUMAINE : FAITHFULNESS / HALLUCINATION / JUSTIFICATION QUALITY
# (réponse au Reviewer #2 : "should include stronger evaluation metrics such
# as faithfulness, hallucination rate, answer justification quality, and
# expert human assessment")
# --------------------------------------------------
# Prompt demandant explicitement une justification courte (2-3 phrases),
# fondée UNIQUEMENT sur le contexte fourni, en plus de la lettre. Utilisé
# uniquement pour le sous-échantillon d'évaluation humaine (pas pour le
# run de génération principal, qui suit le prompt FIQH_TEMPLATE d'origine).
JUSTIFICATION_TEMPLATE = """أنت خبير في الفقه المالكي. أجب عن السؤال التالي بالاعتماد فقط على السياق المعطى أدناه.

السياق:
{context}

السؤال: {question}

الخيارات:
{options}

أجب بالصيغة التالية بالضبط:
الخيار الصحيح: <{letters_or}>
السبب: <شرح قصير من 2 إلى 3 جمل، بالاعتماد فقط على السياق أعلاه، بدون إضافة معلومات غير موجودة فيه>
"""

BASELINE_JUSTIFICATION_TEMPLATE = """أنت خبير في الفقه المالكي. أجب عن السؤال التالي مباشرة.

السؤال: {question}

الخيارات:
{options}

أجب بالصيغة التالية بالضبط:
الخيار الصحيح: <{letters_or}>
السبب: <شرح قصير من 2 إلى 3 جمل>
"""

# Modèle NLI multilingue (couvre l'arabe via XNLI) — conservé disponible en
# option légère/rapide, mais l'évaluation de faithfulness demandée par le
# Reviewer #2 utilise désormais principalement le "LLM-as-judge" ci-dessous.
NLI_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# --------------------------------------------------
# LLM-AS-JUDGE : évaluation automatique de faithfulness / hallucination /
# justification quality par un LLM, en réponse au Reviewer #2 (préféré au
# proxy NLI ci-dessus, sur demande explicite).
# --------------------------------------------------
# Juge par défaut : le plus grand modèle (Fanar-9B), présumé plus capable
# pour une tâche de jugement/raisonnement que les deux autres.
# Juge de repli : utilisé UNIQUEMENT quand le modèle à juger est lui-même
# Fanar-9B (pour éviter le biais d'auto-évaluation).
JUDGE_MODEL = "Fanar-9B"
JUDGE_MODEL_FALLBACK = "Llama3-8B"

JUDGE_PROMPT_TEMPLATE = """أنت مقيّم خبير في الفقه المالكي. مهمتك تقييم جواب نموذج آخر بالاعتماد فقط على السياق المرجعي المعطى أدناه، دون إضافة معرفتك الخاصة.

السياق المرجعي:
{context}

السؤال:
{question}

جواب النموذج المُقيَّم (الخيار المختار وتبريره):
{model_output}

قيّم هذا الجواب حسب المعايير التالية، بالاعتماد فقط على السياق المرجعي أعلاه:

1. الأمانة (Faithfulness) من 1 إلى 5:
   1 = التبرير يخالف السياق أو يضيف معلومات غير موجودة فيه إطلاقًا
   5 = التبرير مطابق تمامًا للسياق، بدون أي إضافة

2. الهلوسة (Hallucination): هل يذكر التبرير معلومة أو مرجعًا أو اسمًا غير موجود في السياق؟
   0 = لا يوجد أي معلومة مختلقة
   1 = يوجد على الأقل معلومة أو مرجع غير موجود في السياق

3. جودة التبرير (Justification Quality) من 1 إلى 5:
   1 = تبرير غير واضح أو غير منطقي
   5 = تبرير واضح، منطقي، ومترابط بشكل جيد مع السياق

أجب بالصيغة التالية بالضبط، بدون أي شرح إضافي أو نص قبل أو بعد:
الأمانة: <رقم من 1 إلى 5>
الهلوسة: <0 أو 1>
جودة_التبرير: <رقم من 1 إلى 5>
"""

HUMAN_EVAL_LIKERT_SCALE = {1: "very poor", 2: "poor", 3: "acceptable", 4: "good", 5: "excellent"}

# --------------------------------------------------
# SCENARIOS DE RE-RANKING (6, cf. Section 3.4)
# --------------------------------------------------
# "stages": liste ordonnée de (nom_du_reranker, top_n_après_ce_stage)
RERANKING_SCENARIOS_DEF = {
    "No Re-ranking": {"stages": []},
    "mini_reranker": {"stages": [("mini_reranker", FINAL_N)]},
    "cross_encoder": {"stages": [("cross_encoder", FINAL_N)]},
    "ara_reranker": {"stages": [("ara_reranker", FINAL_N)]},
    "dual_mini_cross": {"stages": [("mini_reranker", TOP_N_STAGE1), ("cross_encoder", FINAL_N)]},
    "dual_mini_ara": {"stages": [("mini_reranker", TOP_N_STAGE1), ("ara_reranker", FINAL_N)]},
}

ALL_SCENARIOS = list(RERANKING_SCENARIOS_DEF.keys())
ALL_EMBEDDINGS = list(EMBEDDING_MODEL_PATHS.keys())
ALL_LLMS = list(LLM_MODEL_IDS.keys())

# --------------------------------------------------
# LETTRES DE REPONSE (CORRECTIF Risque #3)
# --------------------------------------------------
# Ordre canonique des lettres d'options en arabe. Certaines questions du
# jeu de 140 QCM ont 4 options (ex. د = "toutes les précédentes", cf.
# answer_extraction_fixed.py), pas seulement 3. Coder en dur "(أ، ب، ج)"
# dans les prompts empêchait structurellement le modèle de savoir qu'il
# pouvait répondre د sur ces questions-là — un biais en amont de
# l'extraction que même un regex parfait ne peut pas corriger. Les
# templates ci-dessous utilisent donc un placeholder {letters} /
# {letters_or}, rempli dynamiquement à partir des options réellement
# présentes dans CHAQUE question (cf. answer_letters_for_options ci-dessous).
LETTERS_ORDER = "أبجد"


def answer_letters_for_options(options: dict) -> list:
    """Renvoie les lettres arabes réellement présentes dans `options`
    (dict {lettre: texte_option}), dans l'ordre canonique أبجد. Ne suppose
    jamais qu'il n'y a que 3 options : si `options` contient 'د', elle est
    incluse."""
    return [l for l in LETTERS_ORDER if l in options]


def letters_comma(options: dict) -> str:
    """ex. {'أ':..,'ب':..,'ج':..} -> 'أ، ب، ج' ; avec 'د' -> 'أ، ب، ج، د'."""
    return "، ".join(answer_letters_for_options(options))


def letters_or(options: dict) -> str:
    """ex. {'أ':..,'ب':..,'ج':..} -> 'أ أو ب أو ج' ; avec 'د' -> 'أ أو ب أو ج أو د'."""
    return " أو ".join(answer_letters_for_options(options))


# --------------------------------------------------
# PROMPTS (identiques au papier / aux scripts originaux, à l'exception du
# placeholder {letters} — cf. CORRECTIF Risque #3 ci-dessus)
# --------------------------------------------------
BASELINE_TEMPLATE = """أنت خبير في الفقه المالكي. أجب عن السؤال التالي مباشرة.
قدم اجابة مختصرة مع تحديد الخيار الصحيح ({letters}).

السؤال: {question}

الخيارات:
{options}

الإجابة:"""

FIQH_TEMPLATE = """أنت خبير في الفقه المالكي. أجب عن السؤال التالي بالاعتماد على السياق المعطى.
 قدم اجابة مختصرة مع تحديد الخيار الصحيح ({letters}).

السياق:
{context}

السؤال: {question}

الخيارات:
{options}

"""

# --------------------------------------------------
# PARAMETRES DE DECODAGE (identiques au papier, Section 3.4)
# --------------------------------------------------
GENERATION_PARAMS = {
    "temperature": 0.05,
    "top_k": 40,
    "top_p": 0.95,
    "max_new_tokens": 512,
}

# --------------------------------------------------
# MATERIEL CIBLE
# --------------------------------------------------
# Environnement d'exécution pour cette phase de re-génération des résultats
# (à documenter dans la Section 3.5 révisée / le Data Availability Statement) :
#   GPU  : 1x NVIDIA L4 (24 Go VRAM)
#   RAM  : 32 Go
# Conséquence pratique : les LLM sont chargés en 4-bit (nf4, bitsandbytes) par
# défaut pour laisser de la marge VRAM aux embeddings/rerankers, et
# low_cpu_mem_usage=True est utilisé partout pour limiter le pic de RAM au
# chargement (cf. pipeline_utils.HFGenerator).
DEFAULT_DEVICE = "cuda"
DEFAULT_QUANTIZE_4BIT = True
