"""
run_generation.py — Lance la génération (baseline sans RAG + toutes les
configurations RAG) pour un ou plusieurs LLM.

Conçu pour un GPU L4 (24 Go) + 32 Go de RAM :
- Chaque LLM est chargé UNE SEULE FOIS (pas 66 fois) et reste en mémoire
  pendant qu'on boucle sur les 66 combinaisons (embedding, scénario), ce qui
  élimine 65 rechargements de modèle par LLM par rapport aux scripts
  originaux (198 rechargements au total → 3).
- 4-bit (nf4) par défaut pour laisser de la marge VRAM.
- Reprise automatique : si le script est interrompu, le relancer reprend
  exactement là où il s'était arrêté (au niveau de la question), sans
  aucune perte ni duplication de travail — essentiel pour un run étalé sur
  plusieurs sessions dans le mois disponible.
- Un seul LLM chargé à la fois par processus : lancez le script une fois
  par LLM (éventuellement dans des sessions séparées / jours différents).

Usage:
    # Baseline (sans RAG) pour un LLM
    python run_generation.py --llm Allam-7B --mode baseline

    # RAG complet (66 combinaisons embedding x scénario) pour un LLM
    python run_generation.py --llm Allam-7B --mode rag

    # RAG restreint à certains embeddings/scénarios (utile pour tester vite)
    python run_generation.py --llm Allam-7B --mode rag --embedding CAMeLBERT --scenario mini_reranker

    # Les trois LLM à la suite (attention : très long, préférer un LLM par session)
    python run_generation.py --llm all --mode rag
"""

import os
import json
import time
import argparse

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score

import config
from pipeline_utils import (
    HFGenerator,
    validate_answer,
    print_environment_banner,
    get_resource_snapshot,
    load_completed_keys,
    append_row,
)


def baseline_results_path(llm_label: str) -> str:
    return os.path.join(config.BASELINE_RESULTS_DIR, f"baseline_{llm_label}.csv")


def rag_results_path(llm_label: str) -> str:
    return os.path.join(config.GENERATION_RESULTS_DIR, f"rag_{llm_label}.csv")


def load_context_cache(scenario: str, embedding_name: str) -> dict:
    scenario_dir = os.path.join(config.CONTEXT_CACHE_DIR, scenario.replace(" ", "_"))
    path = os.path.join(scenario_dir, f"{embedding_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache de contexte introuvable: {path}\n"
            f"Lancez d'abord: python build_context_cache.py --embedding {embedding_name} --scenario \"{scenario}\""
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(llm: HFGenerator, llm_label: str, qcm_data: dict, question_ids: list, device: str):
    out_path = baseline_results_path(llm_label)
    completed = load_completed_keys(out_path, ["question_id"])
    remaining = [qid for qid in question_ids if (qid,) not in completed]

    if not remaining:
        print(f"⏭️  Baseline {llm_label}: déjà complet ({len(completed)}/{len(question_ids)}), skip.")
        return

    print(f"\n{'=' * 70}")
    print(f"🚀 BASELINE (sans RAG) — {llm_label}  [{len(remaining)}/{len(question_ids)} questions restantes]")
    print(f"{'=' * 70}")

    smoothie = SmoothingFunction().method4

    for i, qid in enumerate(remaining, 1):
        q = qcm_data[qid]
        question = q["question"]
        options = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
        correct_answer = q["answer_letter"]

        prompt = config.BASELINE_TEMPLATE.format(
            question=question, options=options,
            letters=config.letters_comma(q["options"]),
        )

        try:
            t0 = time.time()
            model_answer = llm.invoke(prompt)
            gen_time = time.time() - t0

            is_correct = validate_answer(model_answer, correct_answer)
            bleu = sentence_bleu([[correct_answer]], model_answer, smoothing_function=smoothie)
            f1 = f1_score([1], [1 if is_correct else 0], zero_division=0)
            resources = get_resource_snapshot(device)

            row = {
                "llm": llm_label,
                "question_id": qid,
                "model_answer": model_answer,
                "is_correct": is_correct,
                "bleu": bleu,
                "f1": f1,
                "prompt_char_length": len(prompt),
                "generation_time_s": gen_time,
                "ram_mb": resources["ram_mb"],
                "gpu_max_allocated_mb": resources["gpu_max_allocated_mb"],
                "error_message": None,
            }
            append_row(out_path, row)

            status = "✅" if is_correct else "❌"
            print(f"   [{i}/{len(remaining)}] {status} gen={gen_time:.2f}s")

        except Exception as e:
            print(f"   ⚠️ Erreur Q{qid}: {e}")
            append_row(out_path, {
                "llm": llm_label, "question_id": qid, "error_message": str(e),
            })


def run_rag(llm: HFGenerator, llm_label: str, qcm_data: dict, question_ids: list,
            embeddings: list, scenarios: list, device: str):
    out_path = rag_results_path(llm_label)
    key_cols = ["llm", "embedding", "reranking_scenario", "question_id"]
    completed = load_completed_keys(out_path, key_cols)

    smoothie = SmoothingFunction().method4
    total_combos = len(embeddings) * len(scenarios)
    combo_idx = 0

    for embedding_name in embeddings:
        for scenario in scenarios:
            combo_idx += 1
            try:
                context_map = load_context_cache(scenario, embedding_name)
            except FileNotFoundError as e:
                print(f"⚠️ {e}")
                continue

            remaining = [
                qid for qid in question_ids
                if (llm_label, embedding_name, scenario, qid) not in completed
            ]
            if not remaining:
                print(f"⏭️  [{combo_idx}/{total_combos}] {llm_label} | {embedding_name} | {scenario}: "
                      f"déjà complet, skip.")
                continue

            print(f"\n{'=' * 70}")
            print(f"🔎 [{combo_idx}/{total_combos}] {llm_label} | {embedding_name} | {scenario} "
                  f"[{len(remaining)}/{len(question_ids)} questions restantes]")
            print(f"{'=' * 70}")

            for i, qid in enumerate(remaining, 1):
                q = qcm_data[qid]
                question = q["question"]
                options = "\n".join([f"{k}: {v}" for k, v in q["options"].items()])
                correct_answer = q["answer_letter"]
                context_entry = context_map.get(qid, {})
                context = context_entry.get("context", "") if isinstance(context_entry, dict) else context_entry

                prompt = config.FIQH_TEMPLATE.format(
                    context=context, question=question, options=options,
                    letters=config.letters_comma(q["options"]),
                )

                try:
                    t0 = time.time()
                    model_answer = llm.invoke(prompt)
                    gen_time = time.time() - t0

                    is_correct = validate_answer(model_answer, correct_answer)
                    bleu = sentence_bleu([[correct_answer]], model_answer, smoothing_function=smoothie)
                    f1 = f1_score([1], [1 if is_correct else 0], zero_division=0)
                    resources = get_resource_snapshot(device)

                    row = {
                        "llm": llm_label,
                        "embedding": embedding_name,
                        "reranking_scenario": scenario,
                        "question_id": qid,
                        "model_answer": model_answer,
                        "is_correct": is_correct,
                        "bleu": bleu,
                        "f1": f1,
                        "context_char_length": len(context),
                        "prompt_char_length": len(prompt),
                        "generation_time_s": gen_time,
                        "ram_mb": resources["ram_mb"],
                        "gpu_max_allocated_mb": resources["gpu_max_allocated_mb"],
                        "error_message": None,
                    }
                    append_row(out_path, row)

                    status = "✅" if is_correct else "❌"
                    print(f"   [{i}/{len(remaining)}] {status} gen={gen_time:.2f}s")

                except Exception as e:
                    print(f"   ⚠️ Erreur Q{qid}: {e}")
                    append_row(out_path, {
                        "llm": llm_label, "embedding": embedding_name,
                        "reranking_scenario": scenario, "question_id": qid,
                        "error_message": str(e),
                    })


def main():
    parser = argparse.ArgumentParser(description="Génération baseline/RAG pour un ou plusieurs LLM (HuggingFace, sans Ollama).")
    parser.add_argument("--llm", required=True,
                         choices=config.ALL_LLMS + ["all"],
                         help="LLM à exécuter, ou 'all' pour les 3 à la suite (déconseillé: préférer un LLM par session).")
    parser.add_argument("--mode", required=True, choices=["baseline", "rag", "both"])
    parser.add_argument("--embedding", choices=config.ALL_EMBEDDINGS, default=None,
                         help="Restreindre à un embedding (défaut: les 11).")
    parser.add_argument("--scenario", choices=config.ALL_SCENARIOS, default=None,
                         help="Restreindre à un scénario (défaut: les 6).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=config.DEFAULT_DEVICE)
    parser.add_argument("--no-4bit", action="store_true",
                         help="Désactive la quantification 4-bit (charge en bf16/float32).")
    args = parser.parse_args()

    if not os.path.exists(config.QCM_FILE):
        print(f"❌ Fichier non trouvé: {config.QCM_FILE}")
        return

    with open(config.QCM_FILE, "r", encoding="utf-8") as f:
        qcm_data = json.load(f)
    question_ids = list(qcm_data.keys())

    embeddings = [args.embedding] if args.embedding else config.ALL_EMBEDDINGS
    scenarios = [args.scenario] if args.scenario else config.ALL_SCENARIOS
    llms_to_run = config.ALL_LLMS if args.llm == "all" else [args.llm]
    quantize_4bit = not args.no_4bit

    print_environment_banner(args.device)

    for llm_label in llms_to_run:
        model_id = config.LLM_MODEL_IDS[llm_label]
        print(f"\n{'#' * 70}")
        print(f"# LLM: {llm_label} ({model_id})")
        print(f"{'#' * 70}")

        try:
            llm = HFGenerator(model_id, device=args.device, quantize_4bit=quantize_4bit)
        except Exception as e:
            print(f"❌ Échec du chargement de {llm_label}: {e}")
            import traceback
            traceback.print_exc()
            continue

        try:
            if args.mode in ("baseline", "both"):
                run_baseline(llm, llm_label, qcm_data, question_ids, args.device)
            if args.mode in ("rag", "both"):
                run_rag(llm, llm_label, qcm_data, question_ids, embeddings, scenarios, args.device)
        finally:
            llm.unload()

    print("\n✅ Génération terminée (ou interrompue proprement — relancez la même")
    print("   commande pour reprendre automatiquement là où ça s'est arrêté).")


if __name__ == "__main__":
    main()
