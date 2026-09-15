import pandas as pd
import re

INPUT_FILE = "amazonhelp_rag_golden_240.csv"
OUTPUT_FILE = "amazonhelp_response_evaluation.csv"
SUMMARY_FILE = "amazonhelp_response_summary.csv"


print("1. Loading RAG Golden results...", flush=True)

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print("Rows:", len(df), flush=True)


# ---------------------------------------------------------
# Required columns
# ---------------------------------------------------------

required = [
    "customer_message",
    "generated_response",
    "predicted_intent"
]

missing = [c for c in required if c not in df.columns]

if missing:
    print("ERROR - Missing columns:", missing)
    print("Available columns:", list(df.columns))
    raise SystemExit

print("Required columns found.", flush=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def has_response(text):
    return len(text.strip()) >= 20


def has_action(text):
    action_words = [
        "please",
        "check",
        "contact",
        "review",
        "verify",
        "visit",
        "return",
        "refund",
        "track",
        "update",
        "restart",
        "try",
        "support"
    ]

    text = text.lower()

    return any(word in text for word in action_words)


def is_safe(text):
    dangerous_patterns = [
        "password",
        "otp",
        "one time password",
        "credit card number",
        "debit card number",
        "cvv"
    ]

    text = text.lower()

    return not any(x in text for x in dangerous_patterns)


def intent_relevant(intent, response):

    intent = intent.lower()
    response = response.lower()

    keyword_map = {

        "delivery delay": [
            "delivery",
            "arrive",
            "shipment",
            "tracking",
            "late"
        ],

        "delivered but not received": [
            "delivered",
            "received",
            "delivery location",
            "household",
            "support"
        ],

        "order tracking / status": [
            "tracking",
            "order",
            "delivery",
            "status"
        ],

        "return / refund": [
            "return",
            "refund",
            "money"
        ],

        "payment / billing": [
            "payment",
            "charge",
            "billing",
            "account",
            "support"
        ],

        "prime membership / benefits": [
            "prime",
            "membership",
            "benefits"
        ],

        "account / login / security": [
            "account",
            "login",
            "password",
            "support"
        ],

        "product / app / website issue": [
            "website",
            "app",
            "technical",
            "restart",
            "browser",
            "support"
        ],

        "wrong / damaged / defective item": [
            "damaged",
            "defective",
            "wrong",
            "replacement",
            "return"
        ],

        "order cancellation": [
            "cancel",
            "cancellation",
            "order"
        ],

        "customer support / complaint": [
            "support",
            "help",
            "customer service",
            "contact"
        ],

        "non-actionable / exclude": [
            "support"
        ]
    }

    keywords = keyword_map.get(intent, [])

    if not keywords:
        return True

    return any(word in response for word in keywords)


# ---------------------------------------------------------
# Evaluate each response
# ---------------------------------------------------------

print("\n2. Evaluating responses...", flush=True)

results = []

for _, row in df.iterrows():

    message = row["customer_message"]
    response = row["generated_response"]
    intent = row["predicted_intent"]

    # Basic response check
    useful = has_response(response)

    # Actionability
    actionable = has_action(response)

    # Safety
    safe = is_safe(response)

    # Intent relevance
    relevant = intent_relevant(intent, response)

    # Overall score
    score = (
        int(useful)
        + int(actionable)
        + int(safe)
        + int(relevant)
    )

    if score == 4:
        quality = "GOOD"
    elif score >= 2:
        quality = "PARTIAL"
    else:
        quality = "POOR"

    results.append({
        "customer_message": message,
        "predicted_intent": intent,
        "generated_response": response,
        "response_present": useful,
        "actionable": actionable,
        "safe": safe,
        "intent_relevant": relevant,
        "quality_score": score,
        "quality": quality
    })


evaluation = pd.DataFrame(results)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n3. Calculating summary...", flush=True)

total = len(evaluation)

response_present_pct = (
    evaluation["response_present"].mean() * 100
)

actionable_pct = (
    evaluation["actionable"].mean() * 100
)

safe_pct = (
    evaluation["safe"].mean() * 100
)

relevant_pct = (
    evaluation["intent_relevant"].mean() * 100
)

good_pct = (
    (evaluation["quality"] == "GOOD").mean() * 100
)

partial_pct = (
    (evaluation["quality"] == "PARTIAL").mean() * 100
)

poor_pct = (
    (evaluation["quality"] == "POOR").mean() * 100
)


print("\n==========================================")
print("RESPONSE EVALUATION RESULTS")
print("==========================================")

print("Total responses:", total)

print(
    f"Response present : {response_present_pct:.2f}%"
)

print(
    f"Actionable       : {actionable_pct:.2f}%"
)

print(
    f"Safe             : {safe_pct:.2f}%"
)

print(
    f"Intent relevant  : {relevant_pct:.2f}%"
)

print(
    f"GOOD             : {good_pct:.2f}%"
)

print(
    f"PARTIAL          : {partial_pct:.2f}%"
)

print(
    f"POOR             : {poor_pct:.2f}%"
)


# ---------------------------------------------------------
# Quality counts
# ---------------------------------------------------------

print("\nQuality distribution:")

print(
    evaluation["quality"].value_counts()
)


# ---------------------------------------------------------
# Save detailed evaluation
# ---------------------------------------------------------

print("\n4. Saving detailed evaluation...", flush=True)

evaluation.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", OUTPUT_FILE)


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

summary = pd.DataFrame([
    {
        "metric": "Total Responses",
        "value": total
    },
    {
        "metric": "Response Present %",
        "value": round(response_present_pct, 2)
    },
    {
        "metric": "Actionable %",
        "value": round(actionable_pct, 2)
    },
    {
        "metric": "Safe %",
        "value": round(safe_pct, 2)
    },
    {
        "metric": "Intent Relevant %",
        "value": round(relevant_pct, 2)
    },
    {
        "metric": "GOOD %",
        "value": round(good_pct, 2)
    },
    {
        "metric": "PARTIAL %",
        "value": round(partial_pct, 2)
    },
    {
        "metric": "POOR %",
        "value": round(poor_pct, 2)
    }
])

summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", SUMMARY_FILE)


# ---------------------------------------------------------
# Show poor responses
# ---------------------------------------------------------

print("\n==========================================")
print("POOR RESPONSE EXAMPLES")
print("==========================================")

poor = evaluation[
    evaluation["quality"] == "POOR"
]

if len(poor) > 0:

    print(
        poor[
            [
                "customer_message",
                "predicted_intent",
                "generated_response"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

else:

    print("No POOR responses found.")


print("\n==========================================")
print("DONE!")
print("==========================================")