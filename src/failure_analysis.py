import pandas as pd

print("==========================================")
print("FAILURE ANALYSIS")
print("==========================================\n")


# =========================================================
# FILES
# =========================================================

CLASSIFIER_FILE = "amazonhelp_classifier_evaluation.csv"
RETRIEVAL_FILE = "amazonhelp_retrieval_evaluation.csv"
ESCALATION_FILE = "amazonhelp_escalation_evaluation.csv"
RESPONSE_FILE = "amazonhelp_response_evaluation.csv"

OUTPUT_FILE = "amazonhelp_top5_failure_modes.csv"


# =========================================================
# 1. LOAD FILES
# =========================================================

print("1. Loading evaluation files...", flush=True)

classifier = pd.read_csv(CLASSIFIER_FILE, dtype=str).fillna("")
retrieval = pd.read_csv(RETRIEVAL_FILE, dtype=str).fillna("")
escalation = pd.read_csv(ESCALATION_FILE, dtype=str).fillna("")
response = pd.read_csv(RESPONSE_FILE, dtype=str).fillna("")

print("Classifier rows:", len(classifier))
print("Retrieval rows:", len(retrieval))
print("Escalation rows:", len(escalation))
print("Response rows:", len(response))


# =========================================================
# 2. CLASSIFICATION FAILURES
# =========================================================

print("\n2. Analysing classification failures...", flush=True)

classifier["correct"] = (
    classifier["true_intent"].str.strip()
    ==
    classifier["predicted_intent"].str.strip()
)

classification_wrong = classifier[
    classifier["correct"] == False
].copy()

print(
    "Classification errors:",
    len(classification_wrong)
)


# Find most common true → predicted errors

classification_pairs = (
    classification_wrong
    .groupby(
        ["true_intent", "predicted_intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\nTop classification confusions:")

print(
    classification_pairs.head(10).to_string(index=False)
)


# =========================================================
# 3. RETRIEVAL FAILURES
# =========================================================

print("\n3. Analysing retrieval failures...", flush=True)

retrieval_failures = retrieval.copy()

# Top-1 intent consistency
if "retrieved_1_intent" in retrieval_failures.columns:

    retrieval_failures["retrieval_correct"] = (
        retrieval_failures["corrected_intent"].str.strip()
        ==
        retrieval_failures["retrieved_1_intent"].str.strip()
    )

else:

    retrieval_failures["retrieval_correct"] = False


retrieval_wrong = retrieval_failures[
    retrieval_failures["retrieval_correct"] == False
].copy()

print(
    "Top-1 retrieval failures:",
    len(retrieval_wrong)
)


# =========================================================
# 4. ESCALATION FAILURES
# =========================================================

print("\n4. Analysing escalation failures...", flush=True)

escalation["escalation_correct"] = (
    escalation["expected_decision"].str.strip().str.upper()
    ==
    escalation["system_decision"].str.strip().str.upper()
)

escalation_wrong = escalation[
    escalation["escalation_correct"] == False
].copy()

print(
    "Escalation errors:",
    len(escalation_wrong)
)


# False escalation
false_escalation = escalation[
    (escalation["system_decision"].str.upper() == "ESCALATE")
    &
    (escalation["expected_decision"].str.upper() == "AUTO_HANDLE")
]

# Missed escalation
missed_escalation = escalation[
    (escalation["system_decision"].str.upper() == "AUTO_HANDLE")
    &
    (escalation["expected_decision"].str.upper() == "ESCALATE")
]

print(
    "False escalations:",
    len(false_escalation)
)

print(
    "Missed escalations:",
    len(missed_escalation)
)


# =========================================================
# 5. RESPONSE FAILURES
# =========================================================

print("\n5. Analysing response failures...", flush=True)

response["response_partial"] = (
    response["quality"] == "PARTIAL"
)

response_partial = response[
    response["response_partial"]
].copy()

print(
    "Partial responses:",
    len(response_partial)
)


# =========================================================
# 6. CREATE FAILURE SUMMARY
# =========================================================

print("\n6. Creating top failure modes...", flush=True)


failure_modes = []


# ---------------------------------------------------------
# Failure 1: Classification confusion
# ---------------------------------------------------------

if len(classification_pairs) > 0:

    row = classification_pairs.iloc[0]

    failure_modes.append({
        "rank": 1,
        "failure_mode":
            "Intent classification confusion",
        "evidence":
            f"{row['true_intent']} was predicted as "
            f"{row['predicted_intent']} "
            f"{row['count']} times.",
        "hypothesis":
            "The semantic classifier struggles with overlapping "
            "support topics and ambiguous customer wording."
    })


# ---------------------------------------------------------
# Failure 2: Retrieval mismatch
# ---------------------------------------------------------

failure_modes.append({
    "rank": 2,
    "failure_mode":
        "Retrieval returns a semantically similar but wrong-intent case",
    "evidence":
        f"{len(retrieval_wrong)} of {len(retrieval)} "
        "Golden queries had a top-1 retrieval whose predicted "
        "intent did not match the expected intent.",
    "hypothesis":
        "Semantic similarity alone can rank related delivery, "
        "tracking, refund and support cases above the correct issue."
})


# ---------------------------------------------------------
# Failure 3: Over-escalation
# ---------------------------------------------------------

failure_modes.append({
    "rank": 3,
    "failure_mode":
        "Low-confidence predictions cause over-escalation",
    "evidence":
        f"{len(false_escalation)} routine cases were escalated "
        "although the expected decision was AUTO_HANDLE.",
    "hypothesis":
        "Using low classifier confidence as a strong escalation "
        "trigger makes the system overly conservative."
})


# ---------------------------------------------------------
# Failure 4: Missed escalation
# ---------------------------------------------------------

failure_modes.append({
    "rank": 4,
    "failure_mode":
        "Some complex cases are incorrectly auto-handled",
    "evidence":
        f"{len(missed_escalation)} cases required human handling "
        "but the system selected AUTO_HANDLE.",
    "hypothesis":
        "A classifier confidence score does not fully capture "
        "fraud, delivery investigations, account issues or "
        "complex support failures."
})


# ---------------------------------------------------------
# Failure 5: Response relevance
# ---------------------------------------------------------

failure_modes.append({
    "rank": 5,
    "failure_mode":
        "Generated responses are not always tightly aligned "
        "with the customer intent",
    "evidence":
        f"{len(response_partial)} of {len(response)} responses "
        "were rated PARTIAL; intent relevance was only "
        "62.50% in the automated response evaluation.",
    "hypothesis":
        "The current response generator uses deterministic "
        "intent templates rather than directly grounding every "
        "sentence in the retrieved historical response."
})


# =========================================================
# 7. PRINT FINAL FAILURE MODES
# =========================================================

failure_df = pd.DataFrame(failure_modes)

print("\n==========================================")
print("TOP 5 FAILURE MODES")
print("==========================================\n")

for _, row in failure_df.iterrows():

    print(f"{row['rank']}. {row['failure_mode']}")
    print(f"   Evidence: {row['evidence']}")
    print(f"   Hypothesis: {row['hypothesis']}")
    print()


# =========================================================
# 8. SAVE
# =========================================================

print("7. Saving failure analysis...", flush=True)

failure_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", OUTPUT_FILE)

print("\n==========================================")
print("DONE!")
print("==========================================")