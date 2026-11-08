
import sys
from pathlib import Path
import json
import re
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mock_llm
from src.triage import ZERO, FEW, redact, retry_chat, MAX_INPUT_TOKENS

ROOT = Path(__file__).resolve().parents[1]

tickets = pd.read_csv(
    ROOT / "data" / "raw" / "tickets.csv",
    keep_default_na=False
)

VALID_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
LABELS = ["billing", "delivery", "technical", "fraud"]


def evaluate(prompt):
    mock_llm.reset()

    predictions = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():
        text = str(ticket["text"])
        actual = str(ticket["true_category"])

        if not text.strip() or actual not in VALID_CATEGORIES:
            continue

        if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            continue

        safe_text, _ = redact(text)

        try:
            response = retry_chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": safe_text}
                ],
                ticket["ticket_id"]
            )

            total_tokens += response.usage.total_tokens

            content = response.choices[0].message.content or ""

            try:
                result = json.loads(content)
                predicted = result.get("category", "")

            except (json.JSONDecodeError, TypeError):
                found = [
                    word for word in re.findall(r"[a-z]+", content.lower())
                    if word in VALID_CATEGORIES
                ]

                predicted = found[0] if len(set(found)) == 1 else ""

            if predicted in VALID_CATEGORIES:
                predictions.append({
                    "ticket_id": ticket["ticket_id"],
                    "actual": actual,
                    "predicted": predicted
                })

        except Exception:
            continue

    result_df = pd.DataFrame(predictions)

    return result_df, total_tokens


zero_df, zero_tokens = evaluate(ZERO)
few_df, few_tokens = evaluate(FEW)

zero_accuracy = (
    (zero_df["actual"] == zero_df["predicted"]).mean() * 100
)

few_accuracy = (
    (few_df["actual"] == few_df["predicted"]).mean() * 100
)

print("\n===== ZERO-SHOT vs FEW-SHOT =====")

print("\nZERO-SHOT")
print("Accuracy:", round(zero_accuracy, 2), "%")
print("Classified tickets:", len(zero_df))
print("Total tokens:", zero_tokens)
print("Tokens per classified ticket:", round(zero_tokens / len(zero_df), 2))

print("\nFEW-SHOT")
print("Accuracy:", round(few_accuracy, 2), "%")
print("Classified tickets:", len(few_df))
print("Total tokens:", few_tokens)
print("Tokens per classified ticket:", round(few_tokens / len(few_df), 2))

print("\n===== ZERO-SHOT CONFUSION MATRIX =====")

cm = confusion_matrix(
    zero_df["actual"],
    zero_df["predicted"],
    labels=LABELS
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

print(cm_df)

print("\n===== FEW-SHOT TOKEN COST =====")
print("Additional tokens:", few_tokens - zero_tokens)
print(
    "Accuracy improvement:",
    round(few_accuracy - zero_accuracy, 2),
    "percentage points"
)