import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# ============================================================
# 1. FILES
# ============================================================

TRAINING_FILE = "amazonhelp_training_final.csv"
GOLDEN_FILE = "amazonhelp_240_corrected.csv"

PREDICTIONS_FILE = "amazonhelp_classifier_evaluation.csv"
CONFUSION_FILE = "amazonhelp_confusion_matrix.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(TRAINING_FILE, dtype=str).fillna("")
golden_df = pd.read_csv(GOLDEN_FILE, dtype=str).fillna("")

print("Training records:", len(train_df))
print("Golden records:", len(golden_df))


# ============================================================
# 3. REMOVE ANY GOLDEN DATA FROM TRAINING
#    (Safety check to avoid data leakage)
# ============================================================

golden_ids = set(
    golden_df["customer_tweet_id"].astype(str).str.strip()
)

golden_messages = set(
    golden_df["customer_message"].astype(str).str.strip()
)

train_df = train_df[
    ~train_df["customer_tweet_id"].astype(str).str.strip().isin(golden_ids)
]

train_df = train_df[
    ~train_df["customer_message"].astype(str).str.strip().isin(golden_messages)
]

train_df = train_df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

golden_df = golden_df.drop_duplicates(
    subset=["customer_tweet_id"]
).reset_index(drop=True)

print("Training records after leakage check:", len(train_df))
print("Golden records:", len(golden_df))


# ============================================================
# 4. PREPARE TEXT AND LABELS
# ============================================================

X_train_text = train_df["customer_message"].tolist()
y_train = train_df["intent"].tolist()

X_golden_text = golden_df["customer_message"].tolist()
y_golden = golden_df["corrected_intent"].tolist()


# ============================================================
# 5. LOAD SENTENCE TRANSFORMER
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 6. CREATE EMBEDDINGS
# ============================================================

print("\nCreating training embeddings...")

X_train = embedding_model.encode(
    X_train_text,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("\nCreating Golden Set embeddings...")

X_golden = embedding_model.encode(
    X_golden_text,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ============================================================
# 7. TRAIN CLASSIFIER
# ============================================================

print("\nTraining Logistic Regression classifier...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

classifier.fit(X_train, y_train)


# ============================================================
# 8. PREDICT GOLDEN SET
# ============================================================

print("\nPredicting Golden Set...")

predicted_intents = classifier.predict(X_golden)

probabilities = classifier.predict_proba(X_golden)

confidence = probabilities.max(axis=1)


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_golden,
    predicted_intents
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_golden,
        predicted_intents,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_golden,
        predicted_intents,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# 10. PRINT MAIN RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GOLDEN SET CLASSIFIER EVALUATION")
print("=" * 60)

print(f"Golden Set Size : {len(golden_df)}")
print(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nMacro Metrics")
print(f"Precision       : {macro_precision:.4f}")
print(f"Recall          : {macro_recall:.4f}")
print(f"F1 Score        : {macro_f1:.4f}")

print("\nWeighted Metrics")
print(f"Precision       : {weighted_precision:.4f}")
print(f"Recall          : {weighted_recall:.4f}")
print(f"F1 Score        : {weighted_f1:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("PER-INTENT CLASSIFICATION REPORT")
print("=" * 60)

labels = sorted(
    set(y_golden) | set(predicted_intents)
)

report = classification_report(
    y_golden,
    predicted_intents,
    labels=labels,
    zero_division=0
)

print(report)


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

evaluation_df = pd.DataFrame({
    "customer_tweet_id": golden_df["customer_tweet_id"],
    "customer_message": golden_df["customer_message"],
    "true_intent": y_golden,
    "predicted_intent": predicted_intents,
    "confidence": confidence,
    "correct": np.array(y_golden) == np.array(predicted_intents)
})

evaluation_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print("\nSaved predictions to:", PREDICTIONS_FILE)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_golden,
    predicted_intents,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    CONFUSION_FILE
)

print("Saved confusion matrix to:", CONFUSION_FILE)


# ============================================================
# 14. WRONG PREDICTIONS
# ============================================================

wrong_df = evaluation_df[
    evaluation_df["correct"] == False
]

print("\n" + "=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

print("Wrong predictions:", len(wrong_df))
print("Correct predictions:", len(evaluation_df) - len(wrong_df))

print("\nSample wrong predictions:\n")

for _, row in wrong_df.head(20).iterrows():

    print("-" * 60)
    print("Customer :", row["customer_message"])
    print("True     :", row["true_intent"])
    print("Predicted:", row["predicted_intent"])
    print("Confidence:", round(row["confidence"], 4))


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Golden Set Size : {len(golden_df)}")
print(f"Correct         : {evaluation_df['correct'].sum()}")
print(f"Wrong           : {(~evaluation_df['correct']).sum()}")
print(f"Accuracy        : {accuracy * 100:.2f}%")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")

print("\nEvaluation completed successfully!")