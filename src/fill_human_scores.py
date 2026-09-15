import pandas as pd

INPUT_FILE = "results/amazonhelp_human_response_review_50.csv"
OUTPUT_FILE = "results/amazonhelp_human_response_review_50_completed.csv"


def get_score(label):
    while True:
        value = input(f"{label} (1-5): ").strip()

        if value in {"1", "2", "3", "4", "5"}:
            return int(value)

        print("Please enter only 1, 2, 3, 4, or 5.")


# ============================================================
# LOAD FILE
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Make sure score columns can store numbers
score_columns = [
    "human_relevance",
    "human_actionability",
    "human_safety",
    "human_grounding",
    "human_overall"
]

for column in score_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Make notes column accept text
df["human_notes"] = df["human_notes"].astype("object")


# ============================================================
# INSTRUCTIONS
# ============================================================

print("=" * 70)
print("HUMAN RESPONSE EVALUATION")
print("=" * 70)

print("""
Score each response from 1 to 5:

1 = Very Poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent
""")

print("Evaluation criteria:")
print("Relevance      = Does the response address the customer's issue?")
print("Actionability  = Does it give useful next steps?")
print("Safety         = Is it safe and appropriate?")
print("Grounding      = Is it consistent with historical Amazon support?")
print("Overall        = Overall quality of the response?")

print("=" * 70)


# ============================================================
# EVALUATE EACH RESPONSE
# ============================================================

for index in range(len(df)):

    row = df.iloc[index]

    print("\n")
    print("#" * 70)
    print(f"RESPONSE {index + 1} / {len(df)}")
    print("#" * 70)

    print("\nCustomer message:")
    print(row["customer_message"])

    print("\nPredicted intent:")
    print(row["predicted_intent"])

    print("\nGenerated response:")
    print(row["generated_response"])

    print("\n--- Give your scores ---")

    relevance = get_score("Relevance")
    actionability = get_score("Actionability")
    safety = get_score("Safety")
    grounding = get_score("Grounding")
    overall = get_score("Overall")

    notes = input(
        "Notes (optional, press Enter to skip): "
    ).strip()

    # Store scores
    df.loc[index, "human_relevance"] = relevance
    df.loc[index, "human_actionability"] = actionability
    df.loc[index, "human_safety"] = safety
    df.loc[index, "human_grounding"] = grounding
    df.loc[index, "human_overall"] = overall
    df.loc[index, "human_notes"] = notes

    # Save after every response
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved successfully.")


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(f"Completed rows: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)