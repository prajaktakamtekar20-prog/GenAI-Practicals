import csv
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.config import OUTPUT_DIR, TOP_K
from backend.app.services.llm_service import get_llm_service
from backend.app.services.rag_service import RAGService
from backend.app.services.fine_tuning_service import FineTuningService

logger = logging.getLogger(__name__)


def compute_token_f1(prediction: str, reference: str) -> float:
    """Calculates word-level F1 score between prediction and reference answer."""
    pred_tokens = re.findall(r'\w+', prediction.lower())
    ref_tokens = re.findall(r'\w+', reference.lower())
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def compute_groundedness(answer: str, context: str) -> float:
    """Estimates the fraction of key terms in the answer supported by retrieved context."""
    if not context or not context.strip():
        return 0.0
    ans_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', answer.lower()))
    if not ans_words:
        return 1.0
    ctx_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', context.lower()))
    supported = ans_words & ctx_words
    return round(len(supported) / len(ans_words), 4)


class EvaluationService:
    def __init__(self, base_dir: Optional[str] = None):
        self.output_dir = Path(base_dir) / 'outputs' if base_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / 'evaluation_results.json'
        self.csv_path = self.output_dir / 'evaluation_results.csv'
        self.questions_path = Path(__file__).resolve().parent.parent.parent / 'evaluation' / 'test_questions.jsonl'

    def load_questions(self) -> List[Dict[str, Any]]:
        """Loads benchmark test questions."""
        if not self.questions_path.exists():
            return []
        questions = []
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    def get_results(self) -> Dict[str, Any]:
        """Returns existing evaluation results from disk, or pending status."""
        if self.results_path.exists():
            try:
                with open(self.results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading evaluation results: {e}")

        return {
            "status": "pending",
            "message": "Evaluation has not been executed yet. Click 'Run Evaluation' to compute real comparative benchmarks.",
            "modes_evaluated": [],
            "results": [],
            "summary": {}
        }

    def run_evaluation(self, modes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes repeatable benchmark across specified modes:
        - 'base': Base LLM without RAG
        - 'rag': Base LLM with RAG
        - 'fine_tuned': Fine-tuned model (if adapter available)
        - 'fine_tuned_rag': Fine-tuned model + RAG (if adapter available)
        """
        questions = self.load_questions()
        if not questions:
            raise ValueError(f"No test questions found in {self.questions_path}")

        llm = get_llm_service()
        llm.load()
        rag = RAGService()
        ft_service = FineTuningService()
        has_adapter = ft_service.has_trained_adapter()

        if modes is None:
            modes = ['base', 'rag']
            if has_adapter:
                modes.extend(['fine_tuned', 'fine_tuned_rag'])

        evaluation_start = time.time()
        mode_reports = []
        all_detail_rows = []

        refusal_phrases = [
            "cannot answer", "not covered", "insufficient", "not enough",
            "no evidence", "unsupported", "cannot be answered"
        ]

        for mode in modes:
            logger.info(f"Running evaluation for mode: {mode}")

            # Configure adapter state
            if 'fine_tuned' in mode and has_adapter:
                try:
                    llm.load_adapter()
                except Exception as e:
                    logger.warning(f"Could not load adapter for mode {mode}: {e}")
            else:
                llm.unload_adapter()

            use_rag = ('rag' in mode)
            mode_details = []
            latencies = []
            correctness_scores = []
            groundedness_scores = []
            citation_accuracies = []
            unsupported_counts = 0

            for q in questions:
                q_id = q.get("id")
                question_text = q["question"]
                ref_answer = q["reference_answer"]
                expected_docs = set(q.get("supporting_documents", []))
                category = q.get("category", "general")

                t0 = time.time()
                context = ""
                sources = []
                cited_docs = set()

                if use_rag:
                    hits = rag.retrieve(question_text, top_k=TOP_K)
                    context, sources = rag.build_rag_context(hits)
                    for s in sources:
                        cited_docs.add(s["filename"])
                    system_inst, user_prompt = rag.format_prompt(question_text, context)
                else:
                    system_inst = "You are an AI assistant. Answer the user's question directly and concisely."
                    user_prompt = question_text

                # Deterministic inference for repeatable evaluation
                answer = llm.generate(
                    prompt=user_prompt,
                    system_instruction=system_inst,
                    max_new_tokens=250,
                    temperature=0.0,
                    do_sample=False
                )
                t1 = time.time()
                latency_ms = round((t1 - t0) * 1000, 2)
                latencies.append(latency_ms)

                # Check refusal / unsupported behavior
                is_refusal = any(p in answer.lower() for p in refusal_phrases)
                if is_refusal:
                    unsupported_counts += 1

                # Correctness (Token F1 against reference)
                f1 = compute_token_f1(answer, ref_answer)
                correctness_scores.append(f1)

                # Groundedness
                g_score = compute_groundedness(answer, context) if use_rag else 0.0
                groundedness_scores.append(g_score)

                # Citation accuracy: Did retrieved sources include the expected doc?
                if expected_docs:
                    matched = expected_docs & cited_docs
                    cit_acc = round(len(matched) / len(expected_docs), 4) if use_rag else 0.0
                else:
                    # For out-of-domain questions, having 0 citations is correct
                    cit_acc = 1.0 if not cited_docs or is_refusal else 0.0
                citation_accuracies.append(cit_acc)

                row = {
                    "question_id": q_id,
                    "category": category,
                    "mode": mode,
                    "question": question_text,
                    "reference_answer": ref_answer,
                    "generated_answer": answer,
                    "latency_ms": latency_ms,
                    "correctness_f1": f1,
                    "groundedness": g_score,
                    "citation_accuracy": cit_acc,
                    "is_refusal": is_refusal,
                    "sources_count": len(sources),
                }
                mode_details.append(row)
                all_detail_rows.append(row)

            # Summarize mode metrics
            n_q = len(questions)
            avg_f1 = round(sum(correctness_scores) / max(1, n_q), 4)
            avg_latency = round(sum(latencies) / max(1, n_q), 2)
            avg_groundedness = round(sum(groundedness_scores) / max(1, n_q), 4)
            avg_cit_acc = round(sum(citation_accuracies) / max(1, n_q), 4)
            unsupported_rate = round(unsupported_counts / max(1, n_q), 4)

            mode_reports.append({
                "mode": mode,
                "label": {
                    "base": "Base LLM (No RAG)",
                    "rag": "RAG-Enhanced LLM",
                    "fine_tuned": "Fine-Tuned Model (No RAG)",
                    "fine_tuned_rag": "Fine-Tuned + RAG"
                }.get(mode, mode),
                "questions_evaluated": n_q,
                "average_correctness_f1": avg_f1,
                "average_groundedness": avg_groundedness,
                "average_citation_accuracy": avg_cit_acc,
                "average_latency_ms": avg_latency,
                "unsupported_refusal_rate": unsupported_rate,
                "details": mode_details,
            })

        # Ensure adapter is unloaded after evaluation
        llm.unload_adapter()

        total_duration = round(time.time() - evaluation_start, 2)

        results_data = {
            "status": "completed",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_duration_seconds": total_duration,
            "model_name": llm.model_name,
            "has_adapter": has_adapter,
            "modes_evaluated": modes,
            "mode_summaries": mode_reports,
            "all_detail_rows": all_detail_rows,
        }

        # Save JSON
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        # Save CSV
        self.export_csv(all_detail_rows)

        return results_data

    def export_csv(self, rows: Optional[List[Dict[str, Any]]] = None) -> str:
        """Exports evaluation rows to CSV file."""
        if rows is None:
            if not self.results_path.exists():
                return ""
            with open(self.results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("all_detail_rows", [])

        if not rows:
            return ""

        headers = [
            "question_id", "category", "mode", "question",
            "reference_answer", "generated_answer", "latency_ms",
            "correctness_f1", "groundedness", "citation_accuracy", "is_refusal", "sources_count"
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in headers})

        return str(self.csv_path)
