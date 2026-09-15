import pandas as pd

# Files
selected_file = "amazonhelp_training_selected.csv"
verified_file = "amazonhelp_600_verified.csv"
golden_file = "amazonhelp_240_corrected.csv"

output_file = "amazonhelp_combined_training.csv"

# -----------------------------
# 1. Load files
# -----------------------------
selected = pd.read_csv(selected_file, dtype=str)
verified = pd.read_csv(verified_file, dtype=str)
golden = pd.read_csv(golden_file, dtype=str)

print("Selected training data:", len(selected))
print("Verified 600 data:", len(verified))
print("Golden Set:", len(golden))

# -----------------------------
# 2. Prepare verified data
# -----------------------------
verified = verified.rename(
    columns={
        "corrected_intent": "intent"
    }
)

verified = verified[
    ["customer_tweet_id", "customer_message", "intent", "confidence", "reason"]
]

selected = selected[
    ["customer_tweet_id", "customer_message", "intent", "confidence", "reason"]
]

# -----------------------------
# 3. Remove Golden Set
#    Just to be completely safe
# -----------------------------
golden_ids = set(
    golden["customer_tweet_id"].astype(str)
)

selected = selected[
    ~selected["customer_tweet_id"].astype(str).isin(golden_ids)
]

verified = verified[
    ~verified["customer_tweet_id"].astype(str).isin(golden_ids)
]

# -----------------------------
# 4. Combine
# -----------------------------
combined = pd.concat(
    [verified, selected],
    ignore_index=True
)

# -----------------------------
# 5. Remove duplicate tweet IDs
# -----------------------------
combined = combined.drop_duplicates(
    subset=["customer_tweet_id"],
    keep="first"
)

# -----------------------------
# 6. Remove duplicate messages
# -----------------------------
combined = combined.drop_duplicates(
    subset=["customer_message"],
    keep="first"
)

combined = combined.reset_index(drop=True)

# -----------------------------
# 7. Save
# -----------------------------
combined.to_csv(
    output_file,
    index=False
)

# -----------------------------
# 8. Validation
# -----------------------------
print("\n====================================")
print("COMBINED TRAINING DATA")
print("====================================")

print("Total rows:", len(combined))

print("\nIntent distribution:")
print(
    combined["intent"]
    .value_counts()
    .sort_index()
)

print("\nConfidence distribution:")
print(
    combined["confidence"]
    .value_counts()
)

print("\nDuplicate tweet IDs:",
      combined["customer_tweet_id"].duplicated().sum())

print("\nDuplicate messages:",
      combined["customer_message"].duplicated().sum())

print("\nSaved as:", output_file)