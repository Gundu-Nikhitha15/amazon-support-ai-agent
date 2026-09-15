import pandas as pd
import numpy as np


# ============================================================
# FILES
# ============================================================

RAG_FILE = "amazonhelp_rag_golden_240.csv"
GOLDEN_FILE = "amazonhelp_240_corrected.csv"

DETAIL_OUTPUT = "amazonhelp_retrieval_evaluation.csv"
SUMMARY_OUTPUT = "amazonhelp_retrieval_summary.csv"


# ============================================================
# LOAD FILES
# ============================================================

rag_df = pd.read_csv(RAG_FILE, dtype=str)
golden_df = pd.read_csv(GOLDEN_FILE, dtype=str)

rag_df.columns = rag_df.columns.str.strip()
golden_df.columns = golden_df.columns.str.strip()

print("RAG rows:", len(rag_df))
print("Golden rows:", len(golden_df))


# ============================================================
# CLEAN MESSAGE COLUMNS
# ============================================================

rag_df["customer_message"] = (
    rag_df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

golden_df["customer_message"] = (
    golden_df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

golden_df["corrected_intent"] = (
    golden_df["corrected_intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# MERGE WITH GOLDEN SET
# ============================================================

merged = rag_df.merge(
    golden_df[
        [
            "customer_tweet_id",
            "customer_message",
            "corrected_intent"
        ]
    ],
    on="customer_message",
    how="inner"
)

print("Matched Golden messages:", len(merged))


if len(merged) == 0:
    raise ValueError(
        "No Golden messages matched. Check customer_message values."
    )


# ============================================================
# FIND RETRIEVAL RANKS
# ============================================================

retrieval_ranks = []

for i in range(1, 6):

    intent_column = f"retrieved_{i}_intent"

    if intent_column in merged.columns:
        retrieval_ranks.append(i)


print(
    "Available retrieval ranks:",
    retrieval_ranks
)


if not retrieval_ranks:
    raise ValueError(
        "No retrieved intent columns found."
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

for rank in retrieval_ranks:

    column = f"retrieved_{rank}_intent"

    merged[f"rank_{rank}_intent_match"] = (
        merged[column].fillna("").astype(str).str.strip()
        ==
        merged["corrected_intent"].fillna("").astype(str).str.strip()
    )


# ============================================================
# TOP-K ACCURACY
# ============================================================

top1_accuracy = (
    merged["rank_1_intent_match"].mean()
)

top3_columns = [
    f"rank_{i}_intent_match"
    for i in retrieval_ranks
    if i <= 3
]

top5_columns = [
    f"rank_{i}_intent_match"
    for i in retrieval_ranks
    if i <= 5
]


top3_accuracy = (
    merged[top3_columns]
    .any(axis=1)
    .mean()
)

top5_accuracy = (
    merged[top5_columns]
    .any(axis=1)
    .mean()
)


print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

print(
    f"Top-1 intent consistency: "
    f"{top1_accuracy:.2%}"
)

print(
    f"Top-3 intent consistency: "
    f"{top3_accuracy:.2%}"
)

print(
    f"Top-5 intent consistency: "
    f"{top5_accuracy:.2%}"
)


# ============================================================
# FIRST CORRECT RETRIEVAL RANK
# ============================================================

def get_first_matching_rank(row):

    for rank in retrieval_ranks:

        value = row[f"rank_{rank}_intent_match"]

        if value:
            return rank

    return np.nan


merged["first_matching_rank"] = merged.apply(
    get_first_matching_rank,
    axis=1
)


print("\nFirst matching rank:")

print(
    merged["first_matching_rank"]
    .value_counts()
    .sort_index()
)


# ============================================================
# RETRIEVAL SIMILARITY
# ============================================================

similarity_columns = []

for rank in retrieval_ranks:

    column = f"retrieved_{rank}_similarity"

    if column in merged.columns:

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

        similarity_columns.append(column)


if similarity_columns:

    print("\nSimilarity statistics:")

    print(
        merged[similarity_columns]
        .describe()
    )


# ============================================================
# PER-INTENT RETRIEVAL PERFORMANCE
# ============================================================

summary_rows = []

for intent, group in merged.groupby(
    "corrected_intent"
):

    row = {
        "intent": intent,
        "support": len(group)
    }

    for rank in retrieval_ranks:

        match_column = (
            f"rank_{rank}_intent_match"
        )

        row[
            f"top_{rank}_intent_consistency"
        ] = group[
            match_column
        ].mean()

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)

summary_df = summary_df.sort_values(
    "support",
    ascending=False
)


print("\n" + "=" * 60)
print("PER-INTENT RETRIEVAL RESULTS")
print("=" * 60)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# WORST RETRIEVAL CASES
# ============================================================

print("\n" + "=" * 60)
print("WORST RETRIEVAL CASES")
print("=" * 60)

wrong_top1 = merged[
    ~merged["rank_1_intent_match"]
].copy()


if len(wrong_top1) > 0:

    display_columns = [
        "customer_message",
        "corrected_intent",
        "retrieved_1_intent",
        "retrieved_1_similarity"
    ]

    available_columns = [
        col
        for col in display_columns
        if col in wrong_top1.columns
    ]

    print(
        wrong_top1[
            available_columns
        ].head(20).to_string(
            index=False
        )
    )

else:

    print(
        "No Top-1 retrieval mismatches."
    )


# ============================================================
# BEST RETRIEVAL CASES
# ============================================================

print("\n" + "=" * 60)
print("BEST RETRIEVAL CASES")
print("=" * 60)

best_cases = merged[
    merged["rank_1_intent_match"]
].copy()


if len(best_cases) > 0:

    display_columns = [
        "customer_message",
        "corrected_intent",
        "retrieved_1_intent",
        "retrieved_1_similarity"
    ]

    available_columns = [
        col
        for col in display_columns
        if col in best_cases.columns
    ]

    print(
        best_cases[
            available_columns
        ].head(10).to_string(
            index=False
        )
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

merged.to_csv(
    DETAIL_OUTPUT,
    index=False
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    "Detailed results saved:",
    DETAIL_OUTPUT
)

print(
    "Summary saved:",
    SUMMARY_OUTPUT
)