# Amazon Customer Support AI Agent

An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment.

The system uses real Amazon customer-support conversations from Twitter to:

1. Classify an incoming customer message into a support intent.
2. Retrieve similar historical Amazon support conversations.
3. Decide whether the request can be auto-handled or should be escalated to a human.
4. Generate a safe response based on the predicted intent and historical support context.

The project focuses on evaluating the complete pipeline rather than optimizing only for a single headline metric.

---

## 1. Problem Statement

Customer-support conversations are noisy, informal, and often ambiguous.

The goal of this project is to build a support-agent pipeline for **AmazonHelp** that can understand a new customer message, identify the type of issue, find similar historical support cases, and determine whether the request should be handled automatically or sent to a human agent.

The system is designed as a support-agent backend/pipeline rather than a full production chatbot UI.

### What the system does

```text
Customer Message
       |
       v
Text Cleaning
       |
       v
Intent Classification
       |
       v
Historical Case Retrieval
       |
       v
Escalation Decision
       |
       v
Response Generation
       |
       v
Final Support Response
What is not built

This project does not attempt to build:

A production customer-support platform
A complete web/mobile chatbot interface
A production-grade authentication system
A fully autonomous agent
A paid LLM API dependency

The focus is on demonstrating a reproducible AI-support pipeline and evaluating its strengths and weaknesses.

2. Dataset

The project uses the Customer Support on Twitter dataset from Kaggle.

The original dataset contains millions of customer-support tweets from multiple brands.

For this project, the selected brand is:

AmazonHelp

AmazonHelp was selected because it provides a large and diverse collection of real customer-support interactions.

Dataset processing

The original dataset was processed to:

Select AmazonHelp conversations
Reconstruct customer → Amazon support pairs
Remove missing messages
Remove empty messages
Remove duplicate customer/response pairs
Remove extremely short messages
Remove acknowledgement-only messages
Clean URLs, mentions, HTML and whitespace
Detect and retain English conversations

The cleaned historical dataset contains approximately 58K usable customer-support conversations.

A smaller historical subset of approximately 10K conversations is used by the runtime retrieval pipeline to keep execution practical.

3. Intent Taxonomy

The project defines 12 intents based on patterns found in the AmazonHelp data.

#	Intent	Description
1	Order Tracking / Status	Questions about order or shipment status
2	Delivery Delay	Late, delayed or missed expected delivery
3	Delivered but Not Received	Tracking says delivered but customer did not receive it
4	Wrong / Damaged / Defective Item	Wrong, damaged, defective or problematic product
5	Return / Refund	Return or refund requests
6	Order Cancellation	Requests to cancel an order
7	Payment / Billing	Charges, billing, payment and unexpected charges
8	Prime Membership / Benefits	Prime membership and benefit-related questions
9	Account / Login / Security	Login, account access and security issues
10	Product / App / Website Issue	Technical problems with Amazon products, apps or website
11	Customer Support / Complaint	Complaints about the support experience
12	Non-actionable / Exclude	Greetings, thanks, vague or non-support messages
4. Labeled Training Data

A manually reviewed golden set of 240 real Amazon customer messages was created for evaluation.

The golden set was kept separate from classifier training to reduce evaluation leakage.

Additional labeled training examples were created using semantic labeling and quality filtering.

The final training set contains:

1,317 labeled examples

with all 12 intents represented.

The training set contains:

1,295 HIGH-confidence examples
22 MEDIUM-confidence examples
0 LOW-confidence examples
5. System Architecture

The system contains four main stages.

Stage 1 — Intent Classification

The incoming customer message is converted into a sentence embedding using:

SentenceTransformer: all-MiniLM-L6-v2

A Logistic Regression classifier then predicts:

Intent
Classification confidence

The classifier uses balanced class weights because the intent distribution is uneven.

Stage 2 — Historical Retrieval

The system searches historical Amazon customer-support conversations using semantic similarity.

The historical customer messages are embedded using the same SentenceTransformer model.

The pipeline:

Retrieves the top 50 semantic candidates.
Predicts the intent of each historical candidate.
Gives a small additional score to candidates whose predicted intent matches the new message.
Selects the top 5 historical cases.

The final ranking uses:

Final Score =
0.80 × Semantic Similarity
+
0.20 × Intent Match

This intent-aware reranking was added because pure semantic similarity sometimes retrieved cases from a related but incorrect support category.

6. Escalation Decision

The system decides between:

AUTO_HANDLE

and

ESCALATE

Escalation can occur when:

Classification confidence is below the configured threshold.
Historical retrieval similarity is weak.
The issue involves account/security concerns.
The customer explicitly requests human assistance.
The predicted intent is Customer Support / Complaint.
The message is Non-actionable / Exclude.

The goal is to avoid automatically responding to cases where the system is uncertain or where human investigation is more appropriate.

7. Response Generation

The current implementation does not depend on a paid LLM API.

For automatically handled cases, the system uses intent-specific response templates.

Examples include:

Delivery Delay
Delivered but Not Received
Return / Refund
Payment / Billing
Account / Security
Product / App / Website
Customer Support

For escalated cases, the system returns:

This request should be reviewed by a human support agent.

This conservative behavior prioritizes safety over unsupported automated answers.

Current limitation

The response generator is deterministic rather than being a generative LLM.

Therefore, historical conversations are currently used primarily for retrieval and evidence/context, while the final automated wording comes from intent-specific templates.

A future version would use an open or paid LLM to generate a response directly from the retrieved historical resolutions.

8. Evaluation

The system was evaluated using the manually labeled 240-message golden set.

8.1 Classification Results

Three approaches were compared.

Method	Golden Accuracy
Majority-class baseline	29.17%
TF-IDF + Logistic Regression	52.08%
Sentence Embeddings + Logistic Regression	55.83%

The semantic classifier improved over:

Majority baseline by 26.67 percentage points
TF-IDF baseline by 3.75 percentage points
Headline result

55.83% accuracy on 240 unseen golden messages.

This result should be interpreted as an evaluation of the intent classification component, not as the percentage of customer conversations successfully resolved end-to-end.

8.2 Retrieval Evaluation

Retrieval was evaluated by comparing the predicted intent of retrieved historical cases with the golden intent.

Exact-rank intent consistency:

Rank	Consistency
Top 1	55.42%
Top 2	55.83%
Top 3	55.83%
Top 4	55.42%
Top 5	54.58%

These numbers measure intent consistency at each retrieval rank rather than standard "any correct item in top-k" retrieval accuracy.

8.3 Escalation Evaluation

The escalation policy was evaluated against the reviewed escalation labels.

Metric	Result
Accuracy	50.00%
ESCALATE Precision	41.36%
ESCALATE Recall	90.80%
ESCALATE F1	56.83%

The system has high recall for escalation but also produces many false escalations.

This shows that the current policy is conservative but can over-escalate routine customer requests.

8.4 Automated Response Screening

All 240 golden messages produced a response.

Metric	Result
Response present	100%
Actionable	100%
Safe	97.92%
Intent relevant	62.50%
GOOD	60.42%
PARTIAL	39.58%
POOR	0%

These are automated screening results and should not be interpreted as an independent human or LLM-as-judge evaluation.

9. Human Response Evaluation

A separate sample of 50 generated responses was manually reviewed.

Each response was evaluated on:

Relevance
Actionability
Safety
Grounding
Overall quality
Results
Metric	Average
Relevance	4.52 / 5
Actionability	2.12 / 5
Safety	5.00 / 5
Grounding	3.86 / 5
Overall Quality	3.64 / 5

38 out of 50 responses (76%) received an overall score of 4 or 5.

Interpretation

The strongest areas were:

Relevance
Safety

The main weakness was:

Actionability

The lower actionability score is largely caused by the system's conservative escalation behavior. When the system is uncertain, it often produces a human-review message instead of giving the customer an immediate automated resolution.

These scores are from human review and are not LLM-as-judge scores.

10. Top 5 Failure Modes
1. Intent Classification Confusion

The classifier struggles with overlapping delivery-related intents.

The largest observed confusion was:

Delivery Delay
        ↓
Delivered but Not Received

This occurred 11 times in the 240-message golden evaluation.

Example

Customer:

"my order is still not fucking delivered , I want a fucking proper service"

The system can interpret this as a delivered-but-not-received issue even though there is no evidence that the order was marked delivered.

Hypothesis

Short and informal customer messages often do not contain enough explicit information to distinguish related delivery intents.

2. Wrong-Intent Historical Retrieval

Semantic similarity can retrieve a historically similar message belonging to a different support intent.

There were 107 top-1 retrieval intent mismatches in the evaluation.

Delivery, tracking, refund and support complaints often share similar language, making semantic retrieval alone insufficient.

Hypothesis

Messages from different support categories can use the same words such as:

order
delivery
problem
received
help

Intent-aware reranking reduces this problem but does not eliminate it.

3. Over-Escalation of Routine Cases

The escalation policy produced many false escalations.

There were 112 false escalations against the reviewed escalation labels.

A major cause was the low-confidence rule.

Example

A routine delivery-delay message may receive a classification confidence below the threshold and therefore be sent to a human even when the issue is suitable for automated handling.

Hypothesis

Classification confidence alone is not a reliable indicator of whether human intervention is necessary.

4. Missed Escalation on Complex Cases

The system incorrectly auto-handled some cases that should receive human investigation.

There were 8 missed escalations.

Example

A customer reported that a package was marked as delivered even though it was missing and described the situation as possible fraud.

The classifier confidence and retrieval similarity were sufficient for automatic handling, but the case should receive human investigation.

Hypothesis

Risk and complexity are not fully captured by classification confidence and semantic similarity.

5. Limited Response Grounding

The current response generator uses deterministic templates.

This makes the responses safe and consistent, but they do not always incorporate specific information from the retrieved historical resolution.

For example, a retrieved Amazon support response may contain a specific troubleshooting or support action, while the generated response may only provide a generic template.

Hypothesis

A generative model grounded directly in retrieved historical responses could produce more specific and useful responses.

11. What Is Misleading About My Headline Number?

The headline classification accuracy is:

55.83%

This does not mean that the AI support agent successfully resolves 55.83% of all customer conversations.

The number measures only whether the predicted intent matches the manually labeled intent on the 240-message golden set.

The complete support pipeline also depends on:

Retrieval quality
Escalation decisions
Response quality
Safety
The complexity of the customer request

The evaluation therefore intentionally reports multiple metrics instead of presenting classification accuracy as an end-to-end success rate.

12. What I Would Do With One More Week

If given another week, I would focus on improving reliability rather than adding more features.

1. Improve the golden evaluation set

Increase the golden set from 240 to approximately 500 examples, especially for difficult boundaries such as:

Delivery Delay vs Delivered but Not Received
Order Tracking vs Delivery Delay
Prime vs Payment/Billing
Customer Support vs Non-actionable
2. Improve classifier training

Add more verified examples for the weakest intents and hard boundary cases.

3. Improve retrieval

Experiment with:

Larger retrieval candidate pools
Cross-encoder reranking
Intent-aware retrieval
Deduplication of near-identical historical cases
4. Improve escalation policy

Replace the single low-confidence rule with a combined risk score using:

Classification confidence
Retrieval similarity
Intent
Fraud/security indicators
Explicit human request
Message complexity
5. Improve response grounding

Use a local/open LLM or an approved API to generate responses from the top historical support cases, while enforcing safety and escalation rules.

6. Improve evaluation

Add:

Larger human evaluation set
Independent second reviewer
LLM-as-judge with validated agreement against human labels
More detailed response-quality metrics
13. Key Design Decisions
Decision 1 — Selected AmazonHelp

AmazonHelp had a large number of support interactions and provided enough variety for intent discovery, classification and retrieval.

Decision 2 — Defined intents from the data

The original dataset does not contain an intent column.

The intent taxonomy was therefore created from observed customer-support patterns.

Decision 3 — Used a manually labeled golden set

A separate 240-message golden set was created so that model performance could be measured on examples not used for training.

Decision 4 — Used semantic embeddings

Sentence embeddings performed better than the TF-IDF baseline on the golden set.

Decision 5 — Added intent-aware retrieval

Pure semantic similarity sometimes retrieved the wrong support category, so predicted intent was added as a soft reranking signal.

Decision 6 — Used conservative escalation

When the system is uncertain or the issue is sensitive, it prefers human review instead of generating an unsupported response.

Decision 7 — Avoided paid API dependency

The final implementation does not require a paid LLM API.

This makes the repository easier to reproduce and avoids dependency on API credits.

Decision 8 — Kept the response generator deterministic

A deterministic response layer was chosen to keep behavior predictable and safe.

The trade-off is lower response specificity.

Decision 9 — Evaluated failures instead of hiding them

The project reports classification, retrieval, escalation and response weaknesses instead of optimizing only for the best-looking metric.

Decision 10 — Kept the golden set separate

Golden evaluation messages were excluded from training and historical retrieval where applicable to reduce evaluation leakage.

14. Project Structure
amazon-support-ai-agent/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_agent.py
│
├── src/
│   ├── Cleaning.py
│   ├── Combined_set.py
│   ├── classification_report.py
│   ├── evaluate_classifier.py
│   ├── evaluate_escalation.py
│   ├── evaluate_responses.py
│   ├── evaluate_retrieval.py
│   ├── failure_analysis.py
│   ├── RAG.py
│   ├── run_rag_on_golden_240.py
│   ├── extract_failure_examples.py
│   ├── intent_discovery.py
│   └── create_better_training_sample.py
│
├── data/
│   ├── amazonhelp_240_corrected.csv
│   ├── amazonhelp_cleaned_final.csv
│   └── amazonhelp_training_final.csv
│
├── results/
│   ├── amazonhelp_baseline_comparison.csv
│   ├── amazonhelp_classifier_evaluation.csv
│   ├── amazonhelp_escalation_evaluation.csv
│   ├── amazonhelp_failure_examples.csv
│   ├── amazonhelp_response_evaluation.csv
│   ├── amazonhelp_response_summary.csv
│   ├── amazonhelp_retrieval_evaluation.csv
│   ├── amazonhelp_rag_golden_240.csv
│   ├── amazonhelp_top5_failure_modes.csv
│   └── amazonhelp_human_response_review_50.csv
│
└── report/
    └── Hiver_SDE_Assignment_Report.pdf
15. Installation

Clone the repository:

git clone https://github.com/Gundu-Nikhitha15/amazon-support-ai-agent.git
cd amazon-support-ai-agent

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
16. Run the AI Support Agent

Run:

python run_agent.py

The program accepts customer messages interactively.

Example:

Customer: My package is two days late.

Example output:

Predicted Intent:
Delivery Delay

Confidence:
0.6869

Decision:
AUTO_HANDLE

Response:
I'm sorry that your order has been delayed. Please check the latest tracking information and estimated delivery date. If the order is already past the expected date, please contact Amazon support so they can look into the delivery.

Another example:

Customer: I cannot log into my Amazon account.

The system can classify the issue as:

Account / Login / Security

and escalate it for human review because account/security issues are treated conservatively.

17. Reproducibility

The main runtime entry point is:

python run_agent.py

The repository contains the processed datasets and evaluation outputs required to understand and reproduce the reported results.

The full original Kaggle dataset is not included in the repository because of its size.

The project is designed to keep the main runtime practical by using a bounded historical retrieval subset.

18. Limitations

The current system has several limitations:

Intent classification accuracy is only 55.83% on the golden set.
Some support intents have overlapping language.
Retrieval can return semantically similar but incorrect support cases.
The escalation policy is conservative and over-escalates.
Some complex cases can still be auto-handled incorrectly.
Response generation is template-based rather than fully generative.
Human evaluation was performed on a 50-response sample.
The current human evaluation is not an LLM-as-judge evaluation.
The project is a prototype rather than a production support system.
19. Final Takeaway

The main objective of this project was not to build a perfect customer-support chatbot.

The objective was to demonstrate a complete, measurable pipeline for turning noisy real-world customer-support data into an AI support agent and to clearly identify where that system succeeds and fails.

The current system demonstrates:

Real-world data cleaning
Intent discovery
Supervised intent classification
Semantic retrieval
Intent-aware reranking
Escalation logic
Safe response generation
Baseline comparison
Golden-set evaluation
Human response evaluation
Failure analysis

The evaluation shows that the semantic classifier improves substantially over simple baselines, while also revealing important weaknesses in intent boundaries, retrieval, escalation and response specificity.

The most important next improvement is not simply adding more features, but improving classification reliability, escalation precision and response grounding.