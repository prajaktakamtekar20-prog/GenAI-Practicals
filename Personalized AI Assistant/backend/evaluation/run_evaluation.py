import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.services.evaluation_service import EvaluationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run experimental evaluation across Base, RAG, and Fine-Tuned models.")
    parser.add_argument("--modes", nargs="+", default=None, help="Modes to evaluate: base, rag, fine_tuned, fine_tuned_rag")
    args = parser.parse_args()

    service = EvaluationService()
    logger.info("Starting experimental evaluation...")
    results = service.run_evaluation(modes=args.modes)

    print("\n" + "=" * 60)
    print("EXPERIMENTAL EVALUATION SUMMARY")
    print("=" * 60)
    for summary in results.get("mode_summaries", []):
        print(f"\nMode: {summary['label']}")
        print(f"  - Avg Correctness (Token F1): {summary['average_correctness_f1']:.4f}")
        print(f"  - Avg Groundedness:            {summary['average_groundedness']:.4f}")
        print(f"  - Avg Citation Accuracy:       {summary['average_citation_accuracy']:.4f}")
        print(f"  - Avg Latency:                 {summary['average_latency_ms']:.2f} ms")
        print(f"  - Unsupported Refusal Rate:    {summary['unsupported_refusal_rate']:.4f}")
    print("\n" + "=" * 60)
    print(f"Full results saved to: {results['modes_evaluated']}")
    print(f"CSV exported to: {service.csv_path}")


if __name__ == "__main__":
    main()
