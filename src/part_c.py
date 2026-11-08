from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.triage import run_pipeline


if __name__ == "__main__":
    result, summary = run_pipeline()

    print("\n===== PART C: ENHANCED TICKET TRIAGE =====")

    print("\nSTATUS COUNTS:")
    print(result["status"].value_counts())

    print("\nROUTING COUNTS:")
    print(result["route"].value_counts())

    print("\nTOTAL TOKENS:", summary["tokens"])

    print("\nPII REDACTION COUNTS:")
    print(summary["redactions"])

    print("\nTOTAL TICKETS:", len(result))

    print("\nSAVED:", ROOT / "outputs" / "triage_results.csv")

    print("\nPART C COMPLETED!")