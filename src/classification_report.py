import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =========================================================
# 1. Load final training dataset
# =========================================================

training_file = "amazonhelp_training_final.csv"

df = pd.read_csv(training_file, dtype=str)

print("Total training examples:", len(df))

# Remove any missing values
df = df.dropna(subset=["customer_message", "intent"]).copy()

X = df["customer_message"]
y = df["intent"]

print("Usable training examples:", len(df))
print("Number of intents:", y.nunique())

# =========================================================
# 2. Split training data
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Internal test examples:", len(X_test))

# =========================================================
# 3. Convert text into TF-IDF features
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)

# =========================================================
# 4. Train Logistic Regression classifier
# =========================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed.")

# =========================================================
# 5. Evaluate on internal test set
# =========================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("INTERNAL TEST RESULTS")
print("====================================")

print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# =========================================================
# 6. Confusion Matrix
# =========================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)

# =========================================================
# 7. Save predictions
# =========================================================

results = pd.DataFrame({
    "customer_message": X_test.values,
    "actual_intent": y_test.values,
    "predicted_intent": y_pred
})

results.to_csv(
    "amazonhelp_internal_test_predictions.csv",
    index=False
)

print("\nSaved:")
print("amazonhelp_internal_test_predictions.csv")