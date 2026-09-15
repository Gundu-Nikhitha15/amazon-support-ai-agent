import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

INPUT_FILE = "amazonhelp_escalation_review_240_final.csv"

DETAIL_FILE = "amazonhelp_escalation_evaluation.csv"
CONFUSION_FILE = "amazonhelp_escalation_confusion_matrix.csv"


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("1. Loading escalation evaluation file...", flush=True)

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print("Rows:", len(df), flush=True)


# ---------------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

print("\n2. Checking columns...", flush=True)

required_columns = [
    "expected_decision",
    "system_decision",
    "customer_message",
    "corrected_intent"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print("ERROR: Missing columns:", missing)
    print("Available columns:")
    print(list(df.columns))
    raise SystemExit

print("All required columns found.", flush=True)


# ---------------------------------------------------------
# 3. CLEAN DECISIONS
# ---------------------------------------------------------

print("\n3. Cleaning decisions...", flush=True)

df["expected_decision"] = (
    df["expected_decision"]
    .str.strip()
    .str.upper()
)

df["system_decision"] = (
    df["system_decision"]
    .str.strip()
    .str.upper()
)

print("Expected decisions:")
print(df["expected_decision"].value_counts())

print("\nSystem decisions:")
print(df["system_decision"].value_counts())


# ---------------------------------------------------------
# 4. CALCULATE METRICS
# ---------------------------------------------------------

print("\n4. Calculating escalation metrics...", flush=True)

y_true = df["expected_decision"]
y_pred = df["system_decision"]


accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    pos_label="ESCALATE",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label="ESCALATE",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label="ESCALATE",
    zero_division=0
)


# ---------------------------------------------------------
# 5. PRINT RESULTS
# ---------------------------------------------------------

print("\n==========================================")
print("ESCALATION EVALUATION RESULTS")
print("==========================================")

print(f"Accuracy :  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   :  {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score :  {f1:.4f} ({f1 * 100:.2f}%)")


# ---------------------------------------------------------
# 6. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_true,
        y_pred,
        labels=["AUTO_HANDLE", "ESCALATE"],
        zero_division=0
    )
)


# ---------------------------------------------------------
# 7. CONFUSION MATRIX
# ---------------------------------------------------------

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

labels = ["AUTO_HANDLE", "ESCALATE"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_AUTO_HANDLE", "Actual_ESCALATE"],
    columns=["Predicted_AUTO_HANDLE", "Predicted_ESCALATE"]
)

print(cm_df)


# ---------------------------------------------------------
# 8. SAVE CONFUSION MATRIX
# ---------------------------------------------------------

cm_df.to_csv(
    CONFUSION_FILE,
    encoding="utf-8-sig"
)

print("\nSaved:", CONFUSION_FILE)


# ---------------------------------------------------------
# 9. ADD CORRECT / WRONG COLUMN
# ---------------------------------------------------------

df["correct"] = (
    df["expected_decision"]
    ==
    df["system_decision"]
)


# ---------------------------------------------------------
# 10. SAVE DETAILED RESULTS
# ---------------------------------------------------------

df.to_csv(
    DETAIL_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", DETAIL_FILE)


# ---------------------------------------------------------
# 11. SHOW WRONG CASES
# ---------------------------------------------------------

wrong = df[df["correct"] == False]

print("\n==========================================")
print("WRONG ESCALATION DECISIONS")
print("==========================================")

print("Wrong:", len(wrong))
print("Correct:", len(df) - len(wrong))

if len(wrong) > 0:

    columns_to_show = [
        "customer_message",
        "corrected_intent",
        "system_decision",
        "expected_decision",
        "system_reason"
    ]

    available = [
        col for col in columns_to_show
        if col in wrong.columns
    ]

    print(
        wrong[available]
        .head(20)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------

print("\n==========================================")
print("DONE!")
print("==========================================")