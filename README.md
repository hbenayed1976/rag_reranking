
# Arabic Islamic Jurisprudence RAG Benchmark — README
Evaluating Arabic Embeddings in Retrieval-Augmented Generation: A Comprehensive Multi-LLM Study with Advanced Re-ranking Strategies

This document describes how the benchmark was built and run: standalone LLM baselines,
retrieval-augmented generation (RAG) across 11 embedding models × 6 re-ranking scenarios ×
3 LLMs, and the human expert annotation campaigns used to validate the quantitative findings.

## 1. Task and Dataset

Two data files, both under `/data`:

| File | Role |
|---|---|
| `/data/dataset_700QA.txt` | Source corpus of 700 Q&A jurisprudence pairs, indexed by the 11 embedding models to build the retrieval pool that RAG configurations search over. Not used directly for evaluation. |
| `/data/qcm_test_140QA.json` | The 140-question multiple-choice evaluation set (`config.QCM_FILE`) — a JSON mapping `question_id -> {question, options, answer_letter}`. This is the held-out test set all accuracy figures in this benchmark are computed on. |

- **Task**: answer each of the 140 multiple-choice questions (QCM) on Islamic jurisprudence
  (fiqh), with or without retrieval augmentation from `dataset_700QA.txt`.
- **Answer extraction**: model outputs are free-text; the answer letter is extracted with a
  regex-based parser (`answer_extraction_fixed.validate_answer` /
  `extract_answer_letter`) and compared to the gold letter to compute `is_correct`.

## 2. Generation Models (LLMs)

Three instruction-tuned generation backbones, evaluated both standalone and under RAG:

| LLM | Role |
|---|---|
| Allam-7B | Arabic-centric generator |
| Fanar-9B | Arabic-centric generator |
| Llama3-8B | Multilingual generator (weaker native Arabic coverage) |

All models are loaded once per process (4-bit `nf4` quantization by default) and reused
across every question/configuration in a run, to avoid redundant reloads on constrained
hardware (single L4 GPU, 24GB VRAM).

## 3. Embedding Models (11)

Used for dense passage retrieval prior to (optional) re-ranking:

```
MiniLM-L12, AraModernBert-STS, MarBERTv2, Multilingual-E5,
Arabic-Triplet-Matryoshka-V2, Arabic-SBERT-100K, AraBERTv2,
CAMeLBERT, DistilBERT_Arabic, AraBERT_Large, IslamQA-BGE-M3
```

`CAMeLBERT` is the strongest embedding overall (highest average accuracy across all
LLM × scenario cells) and is used as the fixed embedding in most targeted annotation
campaigns below, to isolate the effect of re-ranking strategy without embedding
variability.

## 4. Re-ranking Scenarios (6)

Retrieval returns a pool of `TOP_K` candidate passages (20), which are optionally
re-ranked down to `FINAL_N` = 5 final passages before being inserted into the generation
prompt:

| Scenario | Description |
|---|---|
| `No Re-ranking` | Top-5 passages by embedding similarity alone |
| `mini_reranker` | Single-stage lightweight Arabic re-ranker |
| `cross_encoder` | Single-stage multilingual cross-encoder |
| `ara_reranker` | Single-stage stronger Arabic re-ranker |
| `dual_mini_ara` | Two-stage cascade: mini-reranker → Arabic second stage |
| `dual_mini_cross` | Two-stage cascade: mini-reranker → multilingual cross-encoder second stage |

11 embeddings × 6 scenarios × 3 LLMs = **198 experimental configurations**, plus the
3 no-RAG baselines.

## 5. Generation Pipeline

### 5.1 Context caching
Retrieval + re-ranking is decoupled from generation and cached once per
(embedding, scenario) pair, since it does not depend on which LLM will consume it:

```bash
python build_context_cache.py --embedding CAMeLBERT --scenario "mini_reranker"
```

Cache files: `results/context_cache/{scenario}/{embedding}.json`, containing the top-5
passages (rank, score, text) for all 140 questions.

### 5.2 Baseline generation (no RAG)

Run once per LLM (each loads the model once and evaluates all 140 questions from
`/data/qcm_test_140QA.json`):

```bash
python run_generation.py --llm Allam-7B  --mode baseline
python run_generation.py --llm Fanar-9B  --mode baseline
python run_generation.py --llm Llama3-8B --mode baseline
```

Output: `results/baseline/baseline_{llm}.csv` — one row per question with the generated
answer, `is_correct`, BLEU/F1, latency, and resource usage.

### 5.3 RAG generation

Each LLM is evaluated across all 11 embeddings × 6 re-ranking scenarios (66 combinations
per LLM, 198 total). A single LLM is loaded per process and reused across every
combination, so each block below corresponds to one full session per LLM:

```bash
# Allam-7B — all 66 (embedding, scenario) combinations
python run_generation.py --llm Allam-7B --mode rag --embedding MiniLM-L12 --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding MiniLM-L12 --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding MiniLM-L12 --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding MiniLM-L12 --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding MiniLM-L12 --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding MiniLM-L12 --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding AraModernBert-STS --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding AraModernBert-STS --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding AraModernBert-STS --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding AraModernBert-STS --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding AraModernBert-STS --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding AraModernBert-STS --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding MarBERTv2 --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding MarBERTv2 --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding MarBERTv2 --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding MarBERTv2 --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding MarBERTv2 --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding MarBERTv2 --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding Multilingual-E5 --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding Multilingual-E5 --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding Multilingual-E5 --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding Multilingual-E5 --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding Multilingual-E5 --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding Multilingual-E5 --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-Triplet-Matryoshka-V2 --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-Triplet-Matryoshka-V2 --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-Triplet-Matryoshka-V2 --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-Triplet-Matryoshka-V2 --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-Triplet-Matryoshka-V2 --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-Triplet-Matryoshka-V2 --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-SBERT-100K --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-SBERT-100K --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-SBERT-100K --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-SBERT-100K --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-SBERT-100K --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding Arabic-SBERT-100K --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERTv2 --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERTv2 --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERTv2 --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERTv2 --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERTv2 --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERTv2 --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding DistilBERT_Arabic --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding DistilBERT_Arabic --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding DistilBERT_Arabic --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding DistilBERT_Arabic --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding DistilBERT_Arabic --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding DistilBERT_Arabic --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERT_Large --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERT_Large --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERT_Large --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERT_Large --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERT_Large --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding AraBERT_Large --scenario "dual_mini_cross"
python run_generation.py --llm Allam-7B --mode rag --embedding IslamQA-BGE-M3 --scenario "No Re-ranking"
python run_generation.py --llm Allam-7B --mode rag --embedding IslamQA-BGE-M3 --scenario "mini_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding IslamQA-BGE-M3 --scenario "cross_encoder"
python run_generation.py --llm Allam-7B --mode rag --embedding IslamQA-BGE-M3 --scenario "ara_reranker"
python run_generation.py --llm Allam-7B --mode rag --embedding IslamQA-BGE-M3 --scenario "dual_mini_ara"
python run_generation.py --llm Allam-7B --mode rag --embedding IslamQA-BGE-M3 --scenario "dual_mini_cross"
```

The same 66 commands are repeated with `--llm Fanar-9B` and `--llm Llama3-8B` for the
other two generators (198 commands total across the three LLMs). Rather than typing all
198 by hand, use the loop form:

```bash
EMBEDDINGS=(
  "MiniLM-L12" "AraModernBert-STS" "MarBERTv2" "Multilingual-E5"
  "Arabic-Triplet-Matryoshka-V2" "Arabic-SBERT-100K" "AraBERTv2"
  "CAMeLBERT" "DistilBERT_Arabic" "AraBERT_Large" "IslamQA-BGE-M3"
)
SCENARIOS=(
  "No Re-ranking" "mini_reranker" "cross_encoder"
  "ara_reranker" "dual_mini_ara" "dual_mini_cross"
)

for llm in "Allam-7B" "Fanar-9B" "Llama3-8B"; do
  for emb in "${EMBEDDINGS[@]}"; do
    for scn in "${SCENARIOS[@]}"; do
      python run_generation.py --llm "$llm" --mode rag --embedding "$emb" --scenario "$scn"
    done
  done
done
```

Equivalent PowerShell form:

```powershell
$embeddings = "MiniLM-L12","AraModernBert-STS","MarBERTv2","Multilingual-E5",`
              "Arabic-Triplet-Matryoshka-V2","Arabic-SBERT-100K","AraBERTv2",`
              "CAMeLBERT","DistilBERT_Arabic","AraBERT_Large","IslamQA-BGE-M3"
$scenarios  = "No Re-ranking","mini_reranker","cross_encoder","ara_reranker","dual_mini_ara","dual_mini_cross"
$llms       = "Allam-7B","Fanar-9B","Llama3-8B"

foreach ($llm in $llms) {
    foreach ($emb in $embeddings) {
        foreach ($scn in $scenarios) {
            python run_generation.py --llm $llm --mode rag --embedding $emb --scenario "$scn"
        }
    }
}
```

Because `run_generation.py` is resumable (it skips already-completed
`(llm, embedding, scenario, question_id)` keys), the loop form can safely be interrupted
and re-run to continue from where it stopped — this is the recommended way to spread the
198-combination grid across multiple sessions.

Output: `results/generation/rag_{llm}.csv`, accumulating rows across every
(embedding, scenario) combination run for that LLM.

**Prerequisite**: each (embedding, scenario) pair must have a context cache built first
(Section 5.1) — `run_generation.py` raises a clear `FileNotFoundError` naming the missing
`build_context_cache.py` command if a cache is absent.

### 5.4 Accuracy auditing
Answer-extraction accuracy can be recomputed after the fact, without re-running any
generation, using the corrected regex-based extractor:
```bash
python recompute_accuracy_fixed.py --llm all --mode both
```
Output: `results/accuracy_corrected.csv`, comparing stored vs. corrected accuracy per
(llm, embedding, scenario) and flagging how many labels changed.

## 6. Human Expert Annotation

Automatic accuracy alone does not capture retrieval quality, answer grounding, or the
root cause of failures. Three complementary annotation efforts were designed, each
targeting a specific, pre-registered statistical finding rather than exhaustively
re-annotating all 198 configurations.

### 6.1 Protocol B — Baseline vs. Best RAG (H1)

For each LLM, contrasts the no-RAG baseline against that LLM's best-performing RAG
configuration:

| LLM | Best RAG configuration |
|---|---|
| Allam-7B | CAMeLBERT + mini_reranker |
| Fanar-9B | CAMeLBERT + mini_reranker |
| Llama3-8B | CAMeLBERT + dual_mini_ara |

- **Sampling**: a single stratified sample of 40 questions per LLM (all baseline errors +
  random complement of correct cases to reach n=40), reused identically for both the
  baseline and RAG condition (matched/paired design).
- **Fields**: `retrieval_relevance_1_5` and `retrieval_sufficiency_0_2` (RAG condition
  only, marked `N/A` for the no-retrieval baseline), plus `primary_error_source` — filled
  only when `is_correct = False`, using a category set that depends on whether a
  retrieval step exists:
  - **Baseline** (no retrieval): `Knowledge gap` / `Reasoning/selection error` /
    `Ambiguous question`
  - **RAG**: `Retrieval failure` / `Generator failure` / `Both` / `Ambiguous question`

Batch construction:
```bash
python build_annotation_batches_protocolB.py --llm all
```
Metrics aggregation (after annotation):
```bash
python compute_annotation_metrics_protocolB.py --llm all
```

### 6.2 Campaigns 2 & 3 — Targeted Re-ranking Contrasts (H2, H3)

Two additional single-factor contrasts, each isolating a specific, statistically
significant re-ranking effect identified in the quantitative analysis, with embedding
fixed at `CAMeLBERT`:

| Campaign | Hypothesis | LLM | Scenario A | Scenario B |
|---|---|---|---|---|
| 2 | Arabic re-rankers vs. multilingual cross-encoder | Fanar-9B | mini_reranker | cross_encoder |
| 3 | Arabic-only vs. mixed dual-stage re-ranking | Llama3-8B | dual_mini_ara | dual_mini_cross |

- **Sampling**: fixed 20 correct / 20 incorrect questions per campaign (not the
  "all-errors + complement" rule used elsewhere), based on Scenario B's correctness,
  and reused identically for Scenario A (matched design).
- **Fields**: same three fields as Protocol B's RAG condition (`retrieval_relevance_1_5`,
  `retrieval_sufficiency_0_2`, `primary_error_source` with the RAG category set — both
  scenarios involve retrieval here, so only one category set is needed).

Batch construction:
```bash
python build_annotation_batches_campaigns23.py --campaign all
```
Metrics aggregation (mean ± SD, sufficiency rate, error-source distribution, and a
paired Wilcoxon signed-rank test between Scenario A and B):
```bash
python compute_metrics_campaigns23.py --campaign all
```

### 6.3 Annotation mechanics (common to all protocols)

- **Double annotation**: every batch is generated once (single retrieval + generation
  pass per question) and duplicated into two identical `.xlsx` files, one per
  annotator (`annotatorA`, `annotatorB`), who work independently and in blind.
- **Resumability**: batch construction scripts track completed
  `(llm, scenario_type, question_id)` keys per annotator file and only regenerate what
  is missing.
- **Excel dropdowns**: categorical fields (`primary_error_source`, and numeric fields
  in some scripts) are constrained via Excel data validation lists, to keep entries
  consistent across annotators.
- **Inter-annotator agreement**: Cohen's κ, computed once both annotators have
  completed a batch, on numeric fields directly and on `primary_error_source` treated
  as categorical.

## 7. Directory Layout

```
results/
├── baseline/
│   └── baseline_{llm}.csv
├── generation/
│   └── rag_{llm}.csv
├── context_cache/
│   └── {scenario}/{embedding}.json
├── annotation/
│   ├── annotation_protocolB_batch{1-6}_{llm}_{S1|S2}_{annotator}.xlsx
│   ├── annotation_campaign{2|3}_batch{A|B}_{llm}_{embedding}_{scenario}_{annotator}.xlsx
│   ├── metrics_summary_protocolB.csv
│   └── metrics_summary_campaigns23.csv
├── accuracy_corrected.csv
```

## 8. Script Reference

| Script | Purpose |
|---|---|
| `build_context_cache.py` | Retrieval + re-ranking, cached per (embedding, scenario) |
| `run_generation.py` | Baseline and RAG generation for one or more LLMs |
| `recompute_accuracy_fixed.py` | Post-hoc accuracy audit with corrected answer extraction |
| `build_annotation_batches_protocolB.py` | Protocol B batch construction (H1) |
| `compute_annotation_metrics_protocolB.py` | Protocol B metrics aggregation |
| `build_annotation_batches_campaigns23.py` | Campaigns 2/3 batch construction (H2, H3) |
| `compute_metrics_campaigns23.py` | Campaigns 2/3 metrics aggregation + Wilcoxon test |

## 9. Reproducing the Full Benchmark (summary)

```bash
# 1. Context caches (11 embeddings x 6 scenarios = 66 combinations) -- see Section 5.1
#    for the full loop.

# 2. Baseline generation, 3 LLMs -- see Section 5.2 for exact commands.
python run_generation.py --llm Allam-7B  --mode baseline
python run_generation.py --llm Fanar-9B  --mode baseline
python run_generation.py --llm Llama3-8B --mode baseline

# 3. RAG generation, 3 LLMs x 11 embeddings x 6 scenarios (198 combinations) --
#    see Section 5.3 for the full loop form (bash/PowerShell).

# 4. Accuracy audit
python recompute_accuracy_fixed.py --llm all --mode both

# 5. Human annotation batches
python build_annotation_batches_protocolB.py --llm all
python build_annotation_batches_campaigns23.py --campaign all

# 6. (manual annotation step, two experts per batch)

# 7. Metrics aggregation
python compute_annotation_metrics_protocolB.py --llm all
python compute_metrics_campaigns23.py --campaign all
```

## 10. Known Limitations / Notes for Readers

- Protocol B and Campaigns 2/3 use **matched (paired) sampling**: within each
  comparison, both configurations are evaluated on the *same* 40 questions, drawn by
  oversampling errors in one reference configuration. This maximizes statistical power
  for paired tests but means the sample's error rate in the *other* configuration is
  not independently calibrated to its own true accuracy — see each script's docstring
  for the exact basis used.
- All qualitative annotation results are based on 2 independent human raters per batch;
  Cohen's κ should be inspected before treating any qualitative metric as reliable.
