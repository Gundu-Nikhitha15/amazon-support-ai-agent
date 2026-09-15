import pandas as pd
import numpy as np
import re
import html

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. FILE PATHS
# ============================================================

TRAINING_FILE = "data/amazonhelp_training_final.csv"
HISTORY_FILE = "data/amazonhelp_cleaned_final.csv"
GOLDEN_FILE = "data/amazonhelp_240_corrected.csv"
OUTPUT_FILE = "results/amazonhelp_rag_results_v3.csv"


# ============================================================
# 2. SETTINGS
# ============================================================

# Number of historical records used
HISTORY_LIMIT = 10000

# Retrieve this many semantic candidates first
SEMANTIC_CANDIDATES = 50

# Final number of historical cases
TOP_K = 5

# Retrieval scoring weights
SEMANTIC_WEIGHT = 0.80
INTENT_WEIGHT = 0.20


# ============================================================
# 3. INTENT LIST
# ============================================================

INTENTS = [
    "Order Tracking / Status",
    "Delivery Delay",
    "Delivered but Not Received",
    "Wrong / Damaged / Defective Item",
    "Return / Refund",
    "Order Cancellation",
    "Payment / Billing",
    "Prime Membership / Benefits",
    "Account / Login / Security",
    "Product / App / Website Issue",
    "Customer Support / Complaint",
    "Non-actionable / Exclude"
]


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove suffixes such as ^JS
    text = re.sub(
        r"\^[A-Z]{1,3}\b",
        " ",
        text
    )

    # Remove escaped newlines and tabs
    text = text.replace(
        "\\n",
        " "
    )

    text = text.replace(
        "\\t",
        " "
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 5. ESCALATION DECISION
# ============================================================

def decide_escalation(
    predicted_intent,
    confidence,
    best_similarity,
    customer_message
):

    text = customer_message.lower()

    reasons = []


    # --------------------------------------------------------
    # Rule 1: Low classification confidence
    # --------------------------------------------------------

    if confidence < 0.50:

        reasons.append(
            f"low classification confidence ({confidence:.2f})"
        )


    # --------------------------------------------------------
    # Rule 2: Weak historical retrieval
    # --------------------------------------------------------

    if best_similarity < 0.45:

        reasons.append(
            f"weak historical similarity ({best_similarity:.2f})"
        )


    # --------------------------------------------------------
    # Rule 3: Account / security issues
    # --------------------------------------------------------

    security_patterns = [

        "can't log in",
        "cannot log in",
        "unable to log in",
        "cannot login",
        "can't login",

        "password",
        "hacked",
        "account hacked",
        "account stolen",

        "unauthorized access",
        "security",
        "phishing",
        "fraud"
    ]

    if any(
        pattern in text
        for pattern in security_patterns
    ):

        reasons.append(
            "account/security issue requires human review"
        )


    # --------------------------------------------------------
    # Rule 4: Customer wants human support
    # --------------------------------------------------------

    human_patterns = [

        "speak to someone",
        "talk to someone",
        "speak with someone",
        "talk with someone",

        "human",
        "representative",
        "agent",

        "call me",
        "customer service"
    ]

    if any(
        pattern in text
        for pattern in human_patterns
    ):

        reasons.append(
            "customer requests or indicates need for human support"
        )


    # --------------------------------------------------------
    # Rule 5: Customer support complaint
    # --------------------------------------------------------

    if predicted_intent == "Customer Support / Complaint":

        reasons.append(
            "support complaint should be handled by a human"
        )


    # --------------------------------------------------------
    # Rule 6: Non-actionable message
    # --------------------------------------------------------

    if predicted_intent == "Non-actionable / Exclude":

        reasons.append(
            "message is non-actionable"
        )


    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if len(reasons) > 0:

        return (
            "ESCALATE",
            "; ".join(reasons)
        )

    else:

        return (
            "AUTO_HANDLE",
            "sufficient confidence, retrieval strength, and no escalation trigger"
        )


# ============================================================
# 6. LOCAL RESPONSE GENERATOR
# ============================================================

def generate_local_response(
    intent,
    customer_message
):


    # --------------------------------------------------------
    # Order Tracking
    # --------------------------------------------------------

    if intent == "Order Tracking / Status":

        return (
            "I understand you'd like an update on your order. "
            "Please check the latest tracking information and estimated "
            "delivery date in your Amazon order details. "
            "If you still need help, please contact Amazon support."
        )


    # --------------------------------------------------------
    # Delivery Delay
    # --------------------------------------------------------

    elif intent == "Delivery Delay":

        return (
            "I'm sorry that your order has been delayed. "
            "Please check the latest tracking information and updated "
            "delivery date in your Amazon order details. "
            "If the order remains delayed, please contact Amazon support "
            "so the delivery can be investigated."
        )


    # --------------------------------------------------------
    # Delivered but Not Received
    # --------------------------------------------------------

    elif intent == "Delivered but Not Received":

        return (
            "I'm sorry that your order is showing as delivered but "
            "you haven't received it. Please check around the delivery "
            "location and with household members or others who may have "
            "accepted the package. If it is still missing, please contact "
            "Amazon support so the delivery can be investigated."
        )


    # --------------------------------------------------------
    # Wrong / Damaged / Defective
    # --------------------------------------------------------

    elif intent == "Wrong / Damaged / Defective Item":

        return (
            "I'm sorry that you received an incorrect, damaged, or "
            "defective item. Please check the order details for the "
            "available return or replacement options. If you need further "
            "assistance, please contact Amazon support."
        )


    # --------------------------------------------------------
    # Return / Refund
    # --------------------------------------------------------

    elif intent == "Return / Refund":

        return (
            "I understand you're requesting a return or refund. "
            "Please check the order details for the available return "
            "and refund options. If the refund or return is not progressing "
            "as expected, please contact Amazon support."
        )


    # --------------------------------------------------------
    # Cancellation
    # --------------------------------------------------------

    elif intent == "Order Cancellation":

        return (
            "I understand that you'd like to cancel your order. "
            "Please check your order details to see whether cancellation "
            "is still available. If you need further assistance, please "
            "contact Amazon support."
        )


    # --------------------------------------------------------
    # Payment / Billing
    # --------------------------------------------------------

    elif intent == "Payment / Billing":

        return (
            "I understand your concern about the charge. "
            "Please review the payment and order details associated with "
            "your Amazon account. If the charge is unexpected or incorrect, "
            "please contact Amazon support so they can review the billing "
            "information."
        )


    # --------------------------------------------------------
    # Prime
    # --------------------------------------------------------

    elif intent == "Prime Membership / Benefits":

        return (
            "I understand your question about Amazon Prime. "
            "Please check your Prime membership details and available "
            "benefits in your Amazon account. If you need clarification "
            "about your membership, please contact Amazon support."
        )


    # --------------------------------------------------------
    # Account / Security
    # --------------------------------------------------------

    elif intent == "Account / Login / Security":

        return (
            "I'm sorry you're having trouble with your Amazon account. "
            "For account or security issues, please use Amazon's official "
            "account support options. Please do not share passwords, "
            "payment details, or other sensitive information here."
        )


    # --------------------------------------------------------
    # Product / App / Website
    # --------------------------------------------------------

    elif intent == "Product / App / Website Issue":

        return (
            "I'm sorry you're experiencing an issue with Amazon's "
            "website, app, or product functionality. Please try refreshing "
            "or restarting the affected service. If the issue continues, "
            "please contact Amazon support for further assistance."
        )


    # --------------------------------------------------------
    # Customer Support
    # --------------------------------------------------------

    elif intent == "Customer Support / Complaint":

        return (
            "I'm sorry about your experience with our support service. "
            "Your concern should be reviewed by a member of the support "
            "team. Please contact Amazon support so they can look into "
            "the issue and assist you further."
        )


    # --------------------------------------------------------
    # Non-actionable
    # --------------------------------------------------------

    elif intent == "Non-actionable / Exclude":

        return (
            "No automated response generated because the message "
            "does not contain a clear actionable support request."
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return (
        "Please contact Amazon support for further assistance."
    )


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

print()
print("============================================================")
print("STEP 1 - LOADING TRAINING DATA")
print("============================================================")


train_df = pd.read_csv(
    TRAINING_FILE,
    dtype=str
)


# Clean messages

train_df["customer_message"] = (
    train_df["customer_message"]
    .fillna("")
    .apply(clean_text)
)


# Clean intent column

train_df["intent"] = (
    train_df["intent"]
    .fillna("")
    .str.strip()
)


# Remove empty rows

train_df = train_df[
    (train_df["customer_message"] != "") &
    (train_df["intent"] != "")
].copy()


print(
    "Training records:",
    len(train_df)
)


print()
print("Training intent distribution:")

print(
    train_df["intent"].value_counts()
)


# ============================================================
# 8. LOAD SENTENCE TRANSFORMER
# ============================================================

print()
print("============================================================")
print("STEP 2 - LOADING SENTENCE TRANSFORMER")
print("============================================================")


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Sentence Transformer loaded."
)


# ============================================================
# 9. CREATE TRAINING EMBEDDINGS
# ============================================================

print()
print("============================================================")
print("STEP 3 - CREATING TRAINING EMBEDDINGS")
print("============================================================")


X_train = model.encode(
    train_df["customer_message"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)


y_train = train_df["intent"].values


print(
    "Training embedding shape:",
    X_train.shape
)


# ============================================================
# 10. TRAIN INTENT CLASSIFIER
# ============================================================

print()
print("============================================================")
print("STEP 4 - TRAINING INTENT CLASSIFIER")
print("============================================================")


classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


classifier.fit(
    X_train,
    y_train
)


print(
    "Intent classifier trained successfully."
)


# ============================================================
# 11. LOAD GOLDEN SET
# ============================================================

print()
print("============================================================")
print("STEP 5 - LOADING GOLDEN SET")
print("============================================================")


golden_df = pd.read_csv(
    GOLDEN_FILE,
    dtype=str
)


golden_ids = set(
    golden_df["customer_tweet_id"]
    .dropna()
    .astype(str)
)


print(
    "Golden records:",
    len(golden_ids)
)


# ============================================================
# 12. LOAD HISTORICAL DATA
# ============================================================

print()
print("============================================================")
print("STEP 6 - LOADING HISTORICAL AMAZONHELP DATA")
print("============================================================")


history_parts = []

remaining = HISTORY_LIMIT


for chunk in pd.read_csv(
    HISTORY_FILE,
    dtype=str,
    chunksize=5000
):


    # Clean customer messages

    chunk["customer_message"] = (
        chunk["customer_message"]
        .fillna("")
        .apply(clean_text)
    )


    # Clean Amazon responses

    chunk["amazon_response"] = (
        chunk["amazon_response"]
        .fillna("")
        .apply(clean_text)
    )


    # Remove empty messages

    chunk = chunk[
        (chunk["customer_message"] != "") &
        (chunk["amazon_response"] != "")
    ]


    # --------------------------------------------------------
    # IMPORTANT:
    # Golden evaluation examples must never enter retrieval.
    # --------------------------------------------------------

    chunk = chunk[
        ~chunk["customer_tweet_id"]
        .astype(str)
        .isin(golden_ids)
    ]


    if len(chunk) > 0:

        take = min(
            remaining,
            len(chunk)
        )

        history_parts.append(
            chunk.iloc[:take].copy()
        )

        remaining -= take


    if remaining <= 0:
        break


# Combine chunks

history_df = pd.concat(
    history_parts,
    ignore_index=True
)


# Remove duplicate customer messages

history_df = history_df.drop_duplicates(
    subset=["customer_message"]
).reset_index(
    drop=True
)


print(
    "Historical records loaded:",
    len(history_df)
)


# ============================================================
# 13. CREATE HISTORICAL EMBEDDINGS
# ============================================================

print()
print("============================================================")
print("STEP 7 - CREATING HISTORICAL EMBEDDINGS")
print("============================================================")


history_embeddings = model.encode(
    history_df["customer_message"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)


print(
    "Historical embedding shape:",
    history_embeddings.shape
)


# ============================================================
# 14. PREDICT HISTORICAL INTENTS
# ============================================================

print()
print("============================================================")
print("STEP 8 - PREDICTING HISTORICAL INTENTS")
print("============================================================")


history_probabilities = classifier.predict_proba(
    history_embeddings
)


history_predictions = classifier.classes_[
    np.argmax(
        history_probabilities,
        axis=1
    )
]


history_confidences = np.max(
    history_probabilities,
    axis=1
)


history_df["predicted_intent"] = (
    history_predictions
)


history_df["intent_confidence"] = (
    history_confidences
)


print(
    "Historical intent prediction completed."
)


# ============================================================
# 15. INTENT-AWARE RETRIEVAL FUNCTION
# ============================================================

def retrieve_cases(
    customer_message,
    predicted_intent
):


    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True
    )[0]


    # --------------------------------------------------------
    # Calculate semantic similarity
    #
    # Because embeddings are normalized,
    # dot product = cosine similarity.
    # --------------------------------------------------------

    similarities = np.dot(
        history_embeddings,
        query_embedding
    )


    # --------------------------------------------------------
    # Select top semantic candidates
    # --------------------------------------------------------

    candidate_count = min(
        SEMANTIC_CANDIDATES,
        len(similarities)
    )


    candidate_indices = np.argsort(
        similarities
    )[-candidate_count:][::-1]


    candidates = history_df.iloc[
        candidate_indices
    ].copy()


    # Add semantic similarity

    candidates["semantic_similarity"] = [

        similarities[index]

        for index in candidate_indices

    ]


    # --------------------------------------------------------
    # Intent compatibility
    # --------------------------------------------------------

    candidates["intent_match"] = (

        candidates["predicted_intent"]
        ==
        predicted_intent

    ).astype(float)


    # --------------------------------------------------------
    # Combined retrieval score
    #
    # 80% semantic similarity
    # 20% intent compatibility
    # --------------------------------------------------------

    candidates["retrieval_score"] = (

        SEMANTIC_WEIGHT
        *
        candidates["semantic_similarity"]

        +

        INTENT_WEIGHT
        *
        candidates["intent_match"]

    )


    # --------------------------------------------------------
    # Sort by final retrieval score
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        by="retrieval_score",
        ascending=False
    )


    # Return top K

    return candidates.head(
        TOP_K
    ).copy()


# ============================================================
# 16. PROCESS ONE CUSTOMER MESSAGE
# ============================================================

def process_message(customer_message):

    """
    Process one customer message through the complete
    Amazon support AI pipeline.

    Returns:
        dictionary containing:
        - predicted intent
        - confidence
        - escalation decision
        - escalation reason
        - retrieval similarity
        - generated response
        - retrieved historical cases
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if customer_message is None:
        raise ValueError(
            "Customer message cannot be empty."
        )


    # --------------------------------------------------------
    # Clean message
    # --------------------------------------------------------

    cleaned_message = clean_text(
        customer_message
    )


    if not cleaned_message:
        raise ValueError(
            "Customer message cannot be empty."
        )


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [cleaned_message],
        normalize_embeddings=True
    )


    # --------------------------------------------------------
    # Predict intent
    # --------------------------------------------------------

    probabilities = classifier.predict_proba(
        query_embedding
    )[0]


    predicted_index = np.argmax(
        probabilities
    )


    predicted_intent = classifier.classes_[
        predicted_index
    ]


    confidence = float(
        probabilities[predicted_index]
    )


    # --------------------------------------------------------
    # Retrieve historical cases
    # --------------------------------------------------------

    retrieved = retrieve_cases(
        cleaned_message,
        predicted_intent
    )


    if retrieved.empty:

        best_similarity = 0.0

    else:

        best_similarity = float(
            retrieved[
                "semantic_similarity"
            ].iloc[0]
        )


    # --------------------------------------------------------
    # Escalation decision
    # --------------------------------------------------------

    decision, escalation_reason = (
        decide_escalation(
            predicted_intent,
            confidence,
            best_similarity,
            cleaned_message
        )
    )


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    if decision == "AUTO_HANDLE":

        response = generate_local_response(
            predicted_intent,
            cleaned_message
        )

    else:

        response = (
            "Human review required before sending "
            "an automated response."
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "customer_message":
            customer_message,

        "predicted_intent":
            predicted_intent,

        "confidence":
            round(confidence, 4),

        "decision":
            decision,

        "escalation_reason":
            escalation_reason,

        "best_similarity":
            round(best_similarity, 4),

        "generated_response":
            response,

        "retrieved":
            retrieved
    }


# ============================================================
# 17. DEMO FUNCTION
# ============================================================

def run_demo():

    test_messages = [

        "My package was supposed to arrive yesterday but it still hasn't arrived.",

        "The tracking says my package was delivered but I never received it.",

        "I was charged $50 for Prime and I don't understand why.",

        "The item I received is damaged and I want a refund.",

        "I cannot log into my Amazon account.",

        "Your customer service is useless and nobody is helping me.",

        "The Amazon website is not working."

    ]


    results = []


    print()
    print()
    print("============================================================")
    print("AMAZONHELP AI SUPPORT AGENT")
    print("============================================================")


    for number, customer_message in enumerate(
        test_messages,
        start=1
    ):


        print()
        print()
        print("------------------------------------------------------------")

        print(
            f"TEST CASE {number}"
        )

        print("------------------------------------------------------------")


        print()
        print("Customer:")

        print(
            customer_message
        )


        result = process_message(
            customer_message
        )


        predicted_intent = result[
            "predicted_intent"
        ]

        confidence = result[
            "confidence"
        ]

        decision = result[
            "decision"
        ]

        escalation_reason = result[
            "escalation_reason"
        ]

        best_similarity = result[
            "best_similarity"
        ]

        response = result[
            "generated_response"
        ]

        retrieved = result[
            "retrieved"
        ]


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print("Predicted Intent:")

        print(
            predicted_intent
        )


        print()
        print("Confidence:")

        print(
            confidence
        )


        print()
        print("Decision:")

        print(
            decision
        )


        print()
        print("Reason:")

        print(
            escalation_reason
        )


        print()
        print("Best Retrieval Similarity:")

        print(
            best_similarity
        )


        print()
        print("Generated Response:")

        print(
            response
        )


        # ----------------------------------------------------
        # Print retrieved cases
        # ----------------------------------------------------

        print()
        print("Top Retrieved Historical Cases:")


        for rank, (_, row) in enumerate(
            retrieved.iterrows(),
            start=1
        ):

            print()

            print(
                f"{rank}. "
                f"Semantic={row['semantic_similarity']:.4f} | "
                f"IntentMatch={int(row['intent_match'])} | "
                f"Score={row['retrieval_score']:.4f}"
            )


            print(
                "Historical Intent:",
                row["predicted_intent"]
            )


            print(
                "Intent Confidence:",
                round(
                    float(
                        row["intent_confidence"]
                    ),
                    4
                )
            )


            print(
                "Customer:",
                row["customer_message"]
            )


            print(
                "Amazon:",
                row["amazon_response"]
            )


        # ----------------------------------------------------
        # Create output row
        # ----------------------------------------------------

        result_row = {

            "customer_message":
                customer_message,

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
                response
        }


        # ----------------------------------------------------
        # Store top 5 retrieval results
        # ----------------------------------------------------

        for rank, (_, row) in enumerate(
            retrieved.iterrows(),
            start=1
        ):


            result_row[
                f"retrieved_{rank}_customer"
            ] = row[
                "customer_message"
            ]


            result_row[
                f"retrieved_{rank}_response"
            ] = row[
                "amazon_response"
            ]


            result_row[
                f"retrieved_{rank}_intent"
            ] = row[
                "predicted_intent"
            ]


            result_row[
                f"retrieved_{rank}_similarity"
            ] = round(
                float(
                    row[
                        "semantic_similarity"
                    ]
                ),
                4
            )


            result_row[
                f"retrieved_{rank}_score"
            ] = round(
                float(
                    row[
                        "retrieval_score"
                    ]
                ),
                4
            )


        results.append(
            result_row
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("============================================================")
    print("RAG RUN COMPLETED")
    print("============================================================")


    print()
    print(
        "Results saved to:",
        OUTPUT_FILE
    )


    print()
    print(
        "Total test cases:",
        len(results_df)
    )


    print()
    print(
        "Auto-handle:",
        (
            results_df["decision"]
            ==
            "AUTO_HANDLE"
        ).sum()
    )


    print(
        "Escalated:",
        (
            results_df["decision"]
            ==
            "ESCALATE"
        ).sum()
    )


    print()
    print(
        "Predicted Intent Distribution:"
    )


    print(
        results_df[
            "predicted_intent"
        ].value_counts()
    )


    print()
    print("============================================================")
    print("DONE")
    print("============================================================")


# ============================================================
# 18. RUN DEMO ONLY WHEN RAG.PY IS EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_demo()