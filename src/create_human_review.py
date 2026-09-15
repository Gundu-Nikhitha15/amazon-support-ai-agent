import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "results/amazonhelp_rag_golden_240.csv"

OUTPUT_FILE = "results/amazonhelp_human_response_review_50.csv"


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 50
RANDOM_STATE = 42


# ============================================================
# LOAD RAG RESULTS
# ============================================================

print("=" * 70)
print("CREATE HUMAN RESPONSE EVALUATION SET")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    dtype=str
)

print()
print("Total RAG responses:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_message",
    "predicted_intent",
    "generated_response"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# SELECT 50 RANDOM RESPONSES
# ============================================================

if len(df) > SAMPLE_SIZE:

    review_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()

else:

    review_df = df.copy()


review_df = review_df.reset_index(drop=True)


# ============================================================
# CREATE HUMAN REVIEW COLUMNS
# ============================================================

output_df = review_df[
    [
        "customer_message",
        "predicted_intent",
        "generated_response"
    ]
].copy()


# Human will fill these columns.

output_df["human_relevance"] = ""

output_df["human_actionability"] = ""

output_df["human_safety"] = ""

output_df["human_grounding"] = ""

output_df["human_overall"] = ""

output_df["human_notes"] = ""


# ============================================================
# SAVE
# ============================================================

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT INSTRUCTIONS
# ============================================================

print()
print("=" * 70)
print("HUMAN REVIEW FILE CREATED")
print("=" * 70)

print()
print("File:")
print(OUTPUT_FILE)

print()
print("Rows selected:", len(output_df))

print()
print("Please manually review each response.")

print()
print("Use the following scale:")
print()
print("1 = Very poor")
print("2 = Poor")
print("3 = Acceptable")
print("4 = Good")
print("5 = Excellent")

print()
print("Evaluate these four criteria:")
print()
print("human_relevance")
print("human_actionability")
print("human_safety")
print("human_grounding")

print()
print("Then give one overall score:")
print("human_overall")

print()
print("Use human_notes for important observations.")

print()
print("=" * 70)
print("DONE")
print("=" * 70)