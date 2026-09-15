import pandas as pd
import re

# ============================================================
# 1. FILES
# ============================================================

INPUT_FILE = "amazonhelp_cleaned_v2.csv"
GOLDEN_FILE = "amazonhelp_240_corrected.csv"
OUTPUT_FILE = "training_sample_600.csv"


# ============================================================
# 2. LOAD CLEANED AMAZON DATA
# ============================================================

print("Loading cleaned Amazon conversations...")

df = pd.read_csv(INPUT_FILE, dtype=str)

print("Total cleaned conversations:", len(df))


# ============================================================
# 3. KEEP CUSTOMER MESSAGES
# ============================================================

df = df[
    ["customer_tweet_id", "customer_message"]
].dropna()

df = df[
    df["customer_message"].str.strip() != ""
]

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

print("Unique customer messages:", len(df))


# ============================================================
# 4. LOAD GOLDEN SET
# ============================================================

golden = pd.read_csv(
    GOLDEN_FILE,
    dtype=str
)

golden_ids = set(
    golden["customer_tweet_id"].dropna()
)

print("Golden Set messages:", len(golden_ids))


# ============================================================
# 5. REMOVE GOLDEN SET FROM TRAINING SAMPLE
# ============================================================

df = df[
    ~df["customer_tweet_id"].isin(golden_ids)
].reset_index(drop=True)

print(
    "Messages available after removing Golden Set:",
    len(df)
)


# ============================================================
# 6. CREATE SIMPLE TEXT CLUSTERS
# ============================================================

def clean_for_clustering(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["processed_message"] = df[
    "customer_message"
].apply(clean_for_clustering)


# ============================================================
# 7. TF-IDF
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_features=5000
)

X = vectorizer.fit_transform(
    df["processed_message"]
)

print(
    "TF-IDF features:",
    X.shape[1]
)


# ============================================================
# 8. CREATE 12 CLUSTERS
# ============================================================

NUMBER_OF_CLUSTERS = 12

print(
    "\nCreating",
    NUMBER_OF_CLUSTERS,
    "clusters..."
)

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)


# ============================================================
# 9. TAKE 50 REAL MESSAGES FROM EACH CLUSTER
# ============================================================

print("\nSelecting messages...")

samples = []

for cluster_number in sorted(
    df["cluster"].unique()
):

    cluster_data = df[
        df["cluster"] == cluster_number
    ]

    sample_size = min(
        50,
        len(cluster_data)
    )

    sample = cluster_data.sample(
        n=sample_size,
        random_state=42
    )

    samples.append(sample)


# Combine all samples
training_sample = pd.concat(
    samples,
    ignore_index=True
)


# ============================================================
# 10. SHUFFLE
# ============================================================

training_sample = training_sample.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 11. ADD LABEL COLUMNS
# ============================================================

training_sample["intent"] = ""
training_sample["notes"] = ""


# Keep only useful columns
training_sample = training_sample[
    [
        "customer_tweet_id",
        "customer_message",
        "cluster",
        "intent",
        "notes"
    ]
]


# ============================================================
# 12. SAVE
# ============================================================

training_sample.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. RESULTS
# ============================================================

print("\n========================================")
print("DONE!")
print("========================================")

print(
    "Training sample created:",
    len(training_sample)
)

print(
    "Saved as:",
    OUTPUT_FILE
)

print("\nMessages per cluster:")

print(
    training_sample[
        "cluster"
    ].value_counts().sort_index()
)

print("\nNext step:")
print(
    "Open training_sample_600.csv and fill the 'intent' column."
)