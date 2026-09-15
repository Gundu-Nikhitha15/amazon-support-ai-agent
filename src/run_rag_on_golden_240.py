import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


# ============================================================
# FILES
# ============================================================

TRAINING_FILE = "amazonhelp_training_final.csv"
GOLDEN_FILE = "amazonhelp_240_corrected.csv"
HISTORY_FILE = "amazonhelp_cleaned_final.csv"

OUTPUT_FILE = "amazonhelp_rag_golden_240.csv"


# ============================================================
# SETTINGS
# ============================================================

HISTORY_LIMIT = 10000

SEMANTIC_CANDIDATES = 50

TOP_K = 5

SIMILARITY_WEIGHT = 0.80
INTENT_WEIGHT = 0.20


# ============================================================
# CHECK FILES
# ============================================================

for file in [TRAINING_FILE, GOLDEN_FILE, HISTORY_FILE]:

    if not Path(file).exists():
        raise FileNotFoundError(
            f"\nFile not found: {file}\n"
            f"Make sure it is in the same folder as this Python file."
        )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    TRAINING_FILE,
    dtype=str
)

train_df.columns = train_df.columns.str.strip()

train_df = train_df.dropna(
    subset=["customer_message", "intent"]
)

train_df["customer_message"] = (
    train_df["customer_message"]
    .astype(str)
    .str.strip()
)

train_df["intent"] = (
    train_df["intent"]
    .astype(str)
    .str.strip()
)


# Remove duplicate messages
train_df = train_df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

print("Training messages:", len(train_df))


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("\nLoading Golden set...")

golden_df = pd.read_csv(
    GOLDEN_FILE,
    dtype=str
)

golden_df.columns = golden_df.columns.str.strip()

golden_df = golden_df.dropna(
    subset=["customer_message", "corrected_intent"]
)

golden_df["customer_message"] = (
    golden_df["customer_message"]
    .astype(str)
    .str.strip()
)

golden_df["corrected_intent"] = (
    golden_df["corrected_intent"]
    .astype(str)
    .str.strip()
)

print("Golden messages:", len(golden_df))


# ============================================================
# IMPORTANT LEAKAGE CHECK
# ============================================================

golden_messages = set(
    golden_df["customer_message"]
)

before = len(train_df)

train_df = train_df[
    ~train_df["customer_message"].isin(golden_messages)
].reset_index(drop=True)

removed = before - len(train_df)

print(
    "Removed Golden messages from training:",
    removed
)

print(
    "Final training messages:",
    len(train_df)
)


# ============================================================
# TRAIN SEMANTIC CLASSIFIER
# ============================================================

print("\nLoading SentenceTransformer...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print("Creating training embeddings...")

X_train = model.encode(
    train_df["customer_message"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

y_train = train_df["intent"].values


print("\nTraining Logistic Regression classifier...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

classifier.fit(
    X_train,
    y_train
)

print("Classifier trained.")


# ============================================================
# LOAD HISTORICAL AMAZON CONVERSATIONS
# ============================================================

print("\nLoading historical Amazon conversations...")

history_df = pd.read_csv(
    HISTORY_FILE,
    dtype=str,
    nrows=HISTORY_LIMIT
)

history_df.columns = history_df.columns.str.strip()

required_columns = [
    "customer_tweet_id",
    "customer_message",
    "amazon_tweet_id",
    "amazon_response"
]

missing = [
    col
    for col in required_columns
    if col not in history_df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in history file: {missing}"
    )


history_df = history_df.dropna(
    subset=[
        "customer_message",
        "amazon_response"
    ]
)

history_df["customer_message"] = (
    history_df["customer_message"]
    .astype(str)
    .str.strip()
)

history_df["amazon_response"] = (
    history_df["amazon_response"]
    .astype(str)
    .str.strip()
)


# Remove duplicate customer messages
history_df = history_df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)


# ============================================================
# REMOVE GOLDEN MESSAGES FROM RETRIEVAL HISTORY
# ============================================================

before = len(history_df)

history_df = history_df[
    ~history_df["customer_message"].isin(
        golden_messages
    )
].reset_index(drop=True)

removed = before - len(history_df)

print(
    "Removed Golden messages from retrieval history:",
    removed
)

print(
    "Historical conversations available:",
    len(history_df)
)


# ============================================================
# CREATE HISTORY EMBEDDINGS
# ============================================================

print("\nCreating historical embeddings...")

history_embeddings = model.encode(
    history_df["customer_message"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Historical embeddings created.")


# ============================================================
# PREDICT INTENT FOR HISTORICAL DATA
# ============================================================

print("\nPredicting intents for historical conversations...")

history_predictions = classifier.predict(
    history_embeddings
)

history_probabilities = classifier.predict_proba(
    history_embeddings
)

history_confidences = (
    history_probabilities.max(axis=1)
)

history_df["predicted_intent"] = history_predictions

history_df["intent_confidence"] = (
    history_confidences
)

print("Historical intents predicted.")


# ============================================================
# RESPONSE GENERATOR
# ============================================================

def generate_response(intent):

    if intent == "Delivered but Not Received":

        return (
            "I'm sorry that your order is showing as delivered "
            "but you haven't received it. Please check around "
            "the delivery location and with household members "
            "or others who may have accepted the package. If it "
            "is still missing, please contact Amazon support so "
            "the delivery can be investigated."
        )

    elif intent == "Delivery Delay":

        return (
            "I'm sorry that your order has been delayed. "
            "Please check the latest tracking information and "
            "estimated delivery date. If the order is already "
            "past the expected date, please contact Amazon "
            "support so they can look into the delivery."
        )

    elif intent == "Order Tracking / Status":

        return (
            "You can check the latest order status and tracking "
            "information from your Amazon orders page. This will "
            "show the current shipment status and estimated "
            "delivery date."
        )

    elif intent == "Wrong / Damaged / Defective Item":

        return (
            "I'm sorry that you received an incorrect or damaged "
            "item. Please check the return or replacement options "
            "available for the order. If you need further help, "
            "please contact Amazon support."
        )

    elif intent == "Return / Refund":

        return (
            "Please check the return or refund options available "
            "for your order from your Amazon orders page. If the "
            "refund or return is not progressing as expected, "
            "please contact Amazon support."
        )

    elif intent == "Order Cancellation":

        return (
            "Please check your Amazon orders page to see whether "
            "the order can still be cancelled. If the cancellation "
            "option is unavailable, please contact Amazon support."
        )

    elif intent == "Payment / Billing":

        return (
            "Please review the payment and billing details for "
            "your order or membership. If you do not recognize "
            "the charge or need further clarification, please "
            "contact Amazon support."
        )

    elif intent == "Prime Membership / Benefits":

        return (
            "Please check your Amazon Prime membership details "
            "and available benefits from your account. If you "
            "need help with your membership, please contact "
            "Amazon support."
        )

    elif intent == "Account / Login / Security":

        return (
            "For account or login issues, please use Amazon's "
            "official account support options. For your security, "
            "do not share passwords, verification codes, or other "
            "sensitive information in a public message."
        )

    elif intent == "Product / App / Website Issue":

        return (
            "I'm sorry you're experiencing an issue with the "
            "Amazon website or app. Please try refreshing the "
            "page or restarting the app. If the problem continues, "
            "please contact Amazon support."
        )

    elif intent == "Customer Support / Complaint":

        return (
            "I'm sorry about your experience with our support. "
            "Your concern needs further assistance from our "
            "support team, so please contact Amazon support "
            "for additional help."
        )

    elif intent == "Non-actionable / Exclude":

        return ""

    return ""


# ============================================================
# ESCALATION LOGIC
# ============================================================

def decide_escalation(
    predicted_intent,
    confidence,
    best_similarity,
    customer_message
):

    message = customer_message.lower()

    # Low classifier confidence
    if confidence < 0.50:

        return (
            "ESCALATE",
            "Low classification confidence"
        )

    # Weak retrieval
    if best_similarity < 0.45:

        return (
            "ESCALATE",
            "Weak historical retrieval similarity"
        )

    # Account/security issues
    security_patterns = [
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "password",
        "hacked",
        "account locked",
        "account hacked",
        "security"
    ]

    if any(
        pattern in message
        for pattern in security_patterns
    ):

        return (
            "ESCALATE",
            "Account/security issue"
        )

    # Explicit human/support request
    human_patterns = [
        "human",
        "agent",
        "representative",
        "someone help me",
        "speak to someone",
        "talk to someone",
        "call me"
    ]

    if any(
        pattern in message
        for pattern in human_patterns
    ):

        return (
            "ESCALATE",
            "Customer requests human assistance"
        )

    # Customer support complaints
    if predicted_intent == "Customer Support / Complaint":

        return (
            "ESCALATE",
            "Customer support complaint"
        )

    # Non-actionable
    if predicted_intent == "Non-actionable / Exclude":

        return (
            "ESCALATE",
            "Non-actionable message"
        )

    return (
        "AUTO_HANDLE",
        "Sufficient confidence and retrieval"
    )


# ============================================================
# PROCESS GOLDEN MESSAGES
# ============================================================

results = []

print("\n" + "=" * 70)
print("RUNNING RAG ON 240 GOLDEN MESSAGES")
print("=" * 70)


for index, row in golden_df.iterrows():

    customer_message = row["customer_message"]

    true_intent = row["corrected_intent"]

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    probabilities = classifier.predict_proba(
        query_embedding
    )[0]

    predicted_intent = classifier.predict(
        query_embedding
    )[0]

    confidence = probabilities.max()


    # --------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------------------------

    similarities = np.dot(
        history_embeddings,
        query_embedding[0]
    )

    # Get top 50 semantic candidates
    candidate_indices = np.argsort(
        similarities
    )[::-1][:SEMANTIC_CANDIDATES]


    # --------------------------------------------------------
    # INTENT-AWARE RERANKING
    # --------------------------------------------------------

    candidates = []

    for candidate_index in candidate_indices:

        similarity = similarities[candidate_index]

        historical_intent = (
            history_df.iloc[
                candidate_index
            ]["predicted_intent"]
        )

        historical_confidence = (
            history_df.iloc[
                candidate_index
            ]["intent_confidence"]
        )

        intent_match = int(
            historical_intent == predicted_intent
        )

        score = (
            SIMILARITY_WEIGHT * similarity
            +
            INTENT_WEIGHT * intent_match
        )

        candidates.append(
            {
                "index": candidate_index,
                "similarity": similarity,
                "intent": historical_intent,
                "intent_confidence": historical_confidence,
                "intent_match": intent_match,
                "score": score
            }
        )


    # Sort by final score
    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )


    top_candidates = candidates[:TOP_K]


    # --------------------------------------------------------
    # BEST SIMILARITY
    # --------------------------------------------------------

    best_similarity = max(
        candidate["similarity"]
        for candidate in top_candidates
    )


    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    decision, escalation_reason = (
        decide_escalation(
            predicted_intent,
            confidence,
            best_similarity,
            customer_message
        )
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    if decision == "AUTO_HANDLE":

        generated_response = generate_response(
            predicted_intent
        )

    else:

        generated_response = (
            "This request should be reviewed by a human "
            "support agent."
        )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    result = {

        "customer_tweet_id":
            row["customer_tweet_id"],

        "customer_message":
            customer_message,

        "true_intent":
            true_intent,

        "predicted_intent":
            predicted_intent,

        "confidence":
            confidence,

        "decision":
            decision,

        "escalation_reason":
            escalation_reason,

        "best_similarity":
            best_similarity,

        "generated_response":
            generated_response
    }


    # --------------------------------------------------------
    # STORE TOP 5 RETRIEVALS
    # --------------------------------------------------------

    for rank, candidate in enumerate(
        top_candidates,
        start=1
    ):

        history_row = history_df.iloc[
            candidate["index"]
        ]

        result[
            f"retrieved_{rank}_customer"
        ] = history_row[
            "customer_message"
        ]

        result[
            f"retrieved_{rank}_response"
        ] = history_row[
            "amazon_response"
        ]

        result[
            f"retrieved_{rank}_intent"
        ] = candidate[
            "intent"
        ]

        result[
            f"retrieved_{rank}_intent_confidence"
        ] = candidate[
            "intent_confidence"
        ]

        result[
            f"retrieved_{rank}_similarity"
        ] = candidate[
            "similarity"
        ]

        result[
            f"retrieved_{rank}_score"
        ] = candidate[
            "score"
        ]

        result[
            f"retrieved_{rank}_intent_match"
        ] = candidate[
            "intent_match"
        ]


    results.append(result)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (index + 1) % 10 == 0:

        print(
            f"Processed {index + 1}/{len(golden_df)}"
        )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RAG GOLDEN EVALUATION COMPLETE")
print("=" * 70)

print(
    "Golden messages processed:",
    len(results_df)
)

print(
    "Saved:",
    OUTPUT_FILE
)


print("\nDecision counts:")

print(
    results_df["decision"]
    .value_counts()
)


print("\nPredicted intent counts:")

print(
    results_df["predicted_intent"]
    .value_counts()
)


print("\nAverage best similarity:")

print(
    results_df["best_similarity"].mean()
)


print("\nFirst 5 results:")

print(
    results_df[
        [
            "customer_message",
            "true_intent",
            "predicted_intent",
            "confidence",
            "decision",
            "best_similarity"
        ]
    ].head()
)