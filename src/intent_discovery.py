import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ==========================================
# 1. LOAD DATA
# ==========================================

input_file = "amazonhelp_cleaned_v2.csv"
output_file = "intent_discovery_messages.csv"

df = pd.read_csv(input_file, dtype=str)

print("Total conversations:", len(df))


# ==========================================
# 2. GET UNIQUE CUSTOMER MESSAGES
# ==========================================

customers = df[
    ["customer_tweet_id", "customer_message"]
].dropna()

customers = customers.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

print("Unique customer messages:", len(customers))


# ==========================================
# 3. PREPARE TEXT FOR CLUSTERING
# ==========================================

def clean_for_clustering(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


customers["processed_message"] = (
    customers["customer_message"]
    .apply(clean_for_clustering)
)


# ==========================================
# 4. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_features=5000
)

X = vectorizer.fit_transform(
    customers["processed_message"]
)

print("TF-IDF created.")
print(
    "Features:",
    len(vectorizer.get_feature_names_out())
)


# ==========================================
# 5. CLUSTERING
# ==========================================

NUMBER_OF_CLUSTERS = 12

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

customers["cluster"] = kmeans.fit_predict(X)


# ==========================================
# 6. SAVE ALL MESSAGES + CLUSTERS
# ==========================================

result = customers[
    [
        "customer_tweet_id",
        "customer_message",
        "cluster"
    ]
]

result.to_csv(
    output_file,
    index=False
)


# ==========================================
# 7. PRINT SUMMARY
# ==========================================

print("\n========================================")
print("INTENT DISCOVERY COMPLETE")
print("========================================")

print("Total messages:", len(result))

print("Saved file:", output_file)


# ==========================================
# 8. SHOW SAMPLE FROM EACH CLUSTER
# ==========================================

for cluster in range(NUMBER_OF_CLUSTERS):

    cluster_data = result[
        result["cluster"] == cluster
    ]

    print("\n----------------------------------------")
    print("CLUSTER", cluster)
    print("Messages:", len(cluster_data))
    print("----------------------------------------")

    for message in cluster_data[
        "customer_message"
    ].head(10):

        print("-", message)