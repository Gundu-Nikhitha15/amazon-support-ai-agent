import pandas as pd

# =========================================================
# FILES
# =========================================================

CLASSIFIER_FILE = "amazonhelp_classifier_evaluation.csv"
RETRIEVAL_FILE = "amazonhelp_retrieval_evaluation.csv"
ESCALATION_FILE = "amazonhelp_escalation_evaluation.csv"
RESPONSE_FILE = "amazonhelp_response_evaluation.csv"

OUTPUT_FILE = "amazonhelp_failure_examples.csv"


# =========================================================
# LOAD
# =========================================================

print("1. Loading files...", flush=True)

classifier = pd.read_csv(CLASSIFIER_FILE, dtype=str).fillna("")
retrieval = pd.read_csv(RETRIEVAL_FILE, dtype=str).fillna("")
escalation = pd.read_csv(ESCALATION_FILE, dtype=str).fillna("")
response = pd.read_csv(RESPONSE_FILE, dtype=str).fillna("")

print("Files loaded.")


examples = []


# =========================================================
# FAILURE 1
# Intent classification confusion
# =========================================================

print("\n2. Extracting classification examples...", flush=True)

classifier["correct"] = (
    classifier["true_intent"].str.strip()
    ==
    classifier["predicted_intent"].str.strip()
)

wrong = classifier[classifier["correct"] == False].copy()

# Most common confusion
pairs = (
    wrong
    .groupby(["true_intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

if len(pairs) > 0:

    top_true = pairs.iloc[0]["true_intent"]
    top_pred = pairs.iloc[0]["predicted_intent"]

    selected = wrong[
        (wrong["true_intent"] == top_true)
        &
        (wrong["predicted_intent"] == top_pred)
    ].head(2)

    for _, row in selected.iterrows():

        examples.append({
            "failure_rank": 1,
            "failure_mode": "Intent classification confusion",
            "customer_message": row["customer_message"],
            "actual": row["true_intent"],
            "system_result": row["predicted_intent"],
            "reason": (
                f"This is an example of the most common "
                f"classification confusion: {top_true} → {top_pred}."
            )
        })


# =========================================================
# FAILURE 2
# Retrieval mismatch
# =========================================================

print("3. Extracting retrieval examples...", flush=True)

if "retrieval_correct" not in retrieval.columns:

    retrieval["retrieval_correct"] = (
        retrieval["corrected_intent"].str.strip()
        ==
        retrieval["retrieved_1_intent"].str.strip()
    )

retrieval_wrong = retrieval[
    retrieval["retrieval_correct"] == False
].copy()

selected = retrieval_wrong.head(2)

for _, row in selected.iterrows():

    examples.append({
        "failure_rank": 2,
        "failure_mode":
            "Retrieval returns a wrong-intent historical case",
        "customer_message":
            row["customer_message"],
        "actual":
            row["corrected_intent"],
        "system_result":
            row["retrieved_1_intent"],
        "reason":
            "The top retrieved historical case had a different intent."
    })


# =========================================================
# FAILURE 3
# Over-escalation
# =========================================================

print("4. Extracting over-escalation examples...", flush=True)

false_escalation = escalation[
    (escalation["system_decision"].str.upper() == "ESCALATE")
    &
    (escalation["expected_decision"].str.upper() == "AUTO_HANDLE")
].copy()

selected = false_escalation.head(2)

for _, row in selected.iterrows():

    examples.append({
        "failure_rank": 3,
        "failure_mode":
            "Over-escalation of routine cases",
        "customer_message":
            row["customer_message"],
        "actual":
            row["expected_decision"],
        "system_result":
            row["system_decision"],
        "reason":
            row["system_reason"]
    })


# =========================================================
# FAILURE 4
# Missed escalation
# =========================================================

print("5. Extracting missed escalation examples...", flush=True)

missed = escalation[
    (escalation["system_decision"].str.upper() == "AUTO_HANDLE")
    &
    (escalation["expected_decision"].str.upper() == "ESCALATE")
].copy()

selected = missed.head(2)

for _, row in selected.iterrows():

    examples.append({
        "failure_rank": 4,
        "failure_mode":
            "Complex cases incorrectly auto-handled",
        "customer_message":
            row["customer_message"],
        "actual":
            row["expected_decision"],
        "system_result":
            row["system_decision"],
        "reason":
            row["system_reason"]
    })


# =========================================================
# FAILURE 5
# Response relevance
# =========================================================

print("6. Extracting response examples...", flush=True)

partial = response[
    response["quality"] == "PARTIAL"
].copy()

selected = partial.head(2)

for _, row in selected.iterrows():

    examples.append({
        "failure_rank": 5,
        "failure_mode":
            "Generated response is only partially relevant",
        "customer_message":
            row["customer_message"],
        "actual":
            row["predicted_intent"],
        "system_result":
            row["generated_response"],
        "reason":
            "The automated response evaluation marked the response as PARTIAL."
    })


# =========================================================
# SAVE
# =========================================================

print("\n7. Creating final examples file...", flush=True)

examples_df = pd.DataFrame(examples)

examples_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# PRINT
# =========================================================

print("\n==========================================")
print("REAL FAILURE EXAMPLES")
print("==========================================")

for rank in range(1, 6):

    subset = examples_df[
        examples_df["failure_rank"] == rank
    ]

    if len(subset) == 0:
        continue

    print(
        f"\nFAILURE {rank}: "
        f"{subset.iloc[0]['failure_mode']}"
    )

    for _, row in subset.iterrows():

        print("\nCustomer:")
        print(row["customer_message"])

        print("\nActual:")
        print(row["actual"])

        print("\nSystem:")
        print(row["system_result"])

        print("\nReason:")
        print(row["reason"])

        print("-" * 60)


print("\n==========================================")
print("DONE!")
print("==========================================")
print("Saved:", OUTPUT_FILE)
print("Examples:", len(examples_df))
print("==========================================")