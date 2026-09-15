import pandas as pd
import re
import html
from langdetect import detect_langs, DetectorFactory

# Make language detection reproducible
DetectorFactory.seed = 0


# ============================================================
# FILES
# ============================================================

input_file = "amazonhelp_cleaned.csv"
output_file = "amazonhelp_cleaned_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_file, dtype=str)

print("Original records:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_tweet_id",
    "customer_message",
    "amazon_tweet_id",
    "amazon_response"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# ============================================================
# STEP 1: REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=[
        "customer_message",
        "amazon_response"
    ]
)

print("After removing missing values:", len(df))


# ============================================================
# STEP 2: CLEAN TEXT
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    # Example: &amp; becomes &
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    # Example: @AmazonHelp
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove agent suffixes
    # Example: ^JS, ^AG
    text = re.sub(
        r"\s*\^[A-Za-z]{1,4}\s*$",
        "",
        text
    )

    # Replace escaped new lines and tabs
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["customer_message"] = (
    df["customer_message"].apply(clean_text)
)

df["amazon_response"] = (
    df["amazon_response"].apply(clean_text)
)

print("Text cleaning completed.")


# ============================================================
# STEP 3: REMOVE EMPTY MESSAGES
# ============================================================

df = df[
    (df["customer_message"] != "") &
    (df["amazon_response"] != "")
]

print("After removing empty messages:", len(df))


# ============================================================
# STEP 4: REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset=[
        "customer_message",
        "amazon_response"
    ]
)

print("After removing duplicates:", len(df))


# ============================================================
# STEP 5: REMOVE VERY SHORT MESSAGES
# ============================================================

df["customer_length"] = (
    df["customer_message"].str.len()
)

df["response_length"] = (
    df["amazon_response"].str.len()
)

df = df[
    (df["customer_length"] >= 10) &
    (df["response_length"] >= 10)
]

print("After removing very short messages:", len(df))


# ============================================================
# STEP 6: REMOVE ACKNOWLEDGEMENT-ONLY MESSAGES
# ============================================================

acknowledgements = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "thankyou",
    "thx",
    "ty",
    "got it",
    "great",
    "yes",
    "no",
    "sure",
    "perfect",
    "awesome",
    "appreciate it"
}


def is_acknowledgement(text):

    text_lower = text.lower().strip()

    # Remove punctuation
    text_lower = re.sub(
        r"[^\w\s]",
        "",
        text_lower
    )

    return text_lower in acknowledgements


df = df[
    ~df["customer_message"].apply(
        is_acknowledgement
    )
]

print(
    "After removing acknowledgement-only messages:",
    len(df)
)


# ============================================================
# STEP 7: DETECT ENGLISH LANGUAGE
# ============================================================

def detect_english_probability(text):

    try:

        results = detect_langs(text)

        for result in results:

            if result.lang == "en":
                return result.prob

        return 0.0

    except Exception:

        return 0.0


print("\nDetecting customer message languages...")

df["customer_english_probability"] = (
    df["customer_message"]
    .apply(detect_english_probability)
)


print("Detecting AmazonHelp response languages...")

df["response_english_probability"] = (
    df["amazon_response"]
    .apply(detect_english_probability)
)


# ============================================================
# STEP 8: KEEP ONLY ENGLISH CONVERSATIONS
# ============================================================

ENGLISH_THRESHOLD = 0.80

df = df[
    (df["customer_english_probability"] >= ENGLISH_THRESHOLD) &
    (df["response_english_probability"] >= ENGLISH_THRESHOLD)
]

print(
    "After English language filtering:",
    len(df)
)


# ============================================================
# STEP 9: REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "customer_length",
        "response_length",
        "customer_english_probability",
        "response_english_probability"
    ]
)


# ============================================================
# STEP 10: FINAL DUPLICATE CHECK
# ============================================================

df = df.drop_duplicates(
    subset=[
        "customer_message",
        "amazon_response"
    ]
)


# ============================================================
# STEP 11: SAVE FINAL DATASET
# ============================================================

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n======================================")
print("CLEANING COMPLETE")
print("======================================")

print(
    "Final usable conversations:",
    len(df)
)

print(
    "Saved file:",
    output_file
)


# ============================================================
# SHOW SAMPLE
# ============================================================

print("\nSample cleaned conversations:\n")

print(
    df[
        [
            "customer_message",
            "amazon_response"
        ]
    ]
    .head(10)
    .to_string(index=False)
)