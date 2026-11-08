# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains a four-part synthetic-data capstone: a trusted data foundation, customer segmentation and lapse prediction, guarded support-ticket triage, and a policy RAG assistant with a resumable refund agent. All customer, order, product, ticket, and policy data are fictional. All monetary values are INR. The supplied `generate_data.py` and `mock_llm.py` files are used offline and must remain unmodified.

## Reproduce

Run these commands from the repository root in Windows PowerShell. The project requires Python 3.11 or newer.

1. `py -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `py -m pip install -r requirements.txt`
4. `py generate_data.py`
5. `py src/audit_raw.py`
6. `py src/part_a.py`
7. `py src/validate_data.py`
8. `py src/run_sql.py`
9. `py src/pandas_analysis.py`
10. `py src/part_b.py`
11. `py src/lapse_prediction.py`
12. `py src/part_c.py`
13. `py -m src.triage`
14. `py src/evaluate_triage.py`
15. `py src/part_c_reliability.py`
16. `py -m src.part_d`
17. `py -c "from src.policy_rag import PolicyRAG,evaluate; r=PolicyRAG(chunk_size=40,overlap=10); d=evaluate(r); a=d[d.expected_doc!='NONE']; u=d[d.expected_doc=='NONE']; print('answerable_doc',int(a.retrieved_doc.sum()),'/',len(a),'facts',int(a.fact_found.sum()),'/',len(a),'fallback',int(u.fallback.sum()),'/',len(u))"

For Linux/macOS, replace the environment creation and activation commands with `python3 -m venv .venv` and `source .venv/bin/activate`; use `python` instead of `py` for the remaining commands.

## Part A

### Raw data audit

`src/audit_raw.py` audits the raw customer, product, order, and ticket files before cleaning. It reports each table's shape, column data types, per-column missing counts and percentages. Tickets are read with `keep_default_na=False`; therefore the four intentionally empty ticket texts remain empty strings and are counted separately rather than being converted to NA.

| Raw table | Shape | Column data types |
|---|---:|---|
| customers | 1,200 × 5 | customer_id object; city object; signup_date object; age float64; gender object |
| products | 40 × 4 | product_id object; product_name object; category object; price_inr int64 |
| orders | 13,084 × 9 | order_id object; customer_id object; product_id object; quantity int64; order_ts object; status object; discount_pct int64; order_amount object; delivery_minutes float64 |
| tickets | 164 × 5 | ticket_id object; customer_id object; order_id object; text object; true_category object |

| Raw column | Missing count | Missing % |
|---|---:|---:|
| customers.customer_id / city / signup_date | 0 each | 0.00% each |
| customers.age | 61 | 5.08% |
| customers.gender | 30 | 2.50% |
| products.product_id / product_name / category / price_inr | 0 each | 0.00% each |
| orders.order_id / customer_id / product_id / quantity / order_ts / status / discount_pct | 0 each | 0.00% each |
| orders.order_amount | 53 | 0.41% |
| orders.delivery_minutes | 2,541 | 19.42% |
| tickets.ticket_id / customer_id / order_id / text / true_category | 0 NA values each | 0.00% each |

For tickets, empty strings are intentionally preserved rather than parsed as NA: there are 4 empty ticket texts and 8 blank order IDs.

| Required raw audit checkpoint | Observed | Reference |
|---|---:|---:|
| Raw order rows | 13,084 | 13,084 |
| Exact duplicate order rows | 119 | 119 |
| Pending rows | 656 | 656 |
| Distinct order IDs | 12,309 | 12,309 |
| Non-numeric order amounts (`to_numeric(errors='coerce')`) | 120 | 120 |
| Negative delivery minutes | 106 | 106 |
| Blank delivery minutes | 2,541 | 2,541 |
| Blank customer ages | 61 | 61 |
| Customer ages outside 10–100 | 6 | 6 |
| Distinct raw city spellings | 24 | 24 |
| Distinct non-blank raw gender spellings | 10 | 10 |
| Blank genders | 30 | 30 |

### Cleaning rules and results

Customer cleaning:

1. Trim city strings and apply title case; the result contains exactly six standard city names.
2. Map gender values to `male`, `female`, or `unknown`; blank values become `unknown`.
3. Convert age to numeric; values outside the inclusive 10–100 range become missing. Fill missing ages with the median age of that customer's city using `groupby('city').transform('median')`.
4. Parse `signup_date` as a datetime.

| Customer checkpoint | Result |
|---|---:|
| Customers | 1,200 |
| Pune / Delhi / Bengaluru / Chennai / Hyderabad / Mumbai | 227 / 210 / 197 / 191 / 191 / 184 |
| Female / male / unknown | 610 / 560 / 30 |
| Missing ages after cleaning | 0 |
| Cleaned age range | 18–60 |
| Missing ages filled after blank and invalid values | 67 (61 blank + 6 outside 10–100) |

Order cleaning is performed in the required order:

1. Parse `order_ts` as datetime.
2. Remove 119 exact duplicate rows.
3. Sort by timestamp and keep the latest row for each `order_id`; 656 older duplicate-order records are removed, leaving one row per order ID.
4. Convert `order_amount` to numeric and fill the 112 remaining missing/unparseable amounts after de-duplication from product price × quantity × (1 − discount percentage), rounded to 2 decimals.
5. Set the 105 negative delivery values remaining after de-duplication to missing. Fill the 774 missing delivery times for completed orders using the median delivery time for the corresponding product category among completed orders. Leave all cancelled-order delivery times missing.

| Clean order checkpoint | Result | Reference |
|---|---:|---:|
| Clean orders | 12,309 | 12,309 |
| Completed / cancelled / pending | 11,110 / 1,199 / 0 | 11,110 / 1,199 / 0 |
| Missing order amounts | 0 | 0 |
| Missing completed-order delivery times | 0 | 0 |
| Missing cancelled-order delivery times | 1,199 | 1,199 |
| Duplicate order IDs | 0 | 0 |
| Completed-order revenue | INR 2,998,668.80 | INR 2,998,668.80 |

The three clean CSVs are written to `data/clean/`; `data/instamart.db` contains the `customers`, `products`, and `orders` tables. The validation script checks 21 conditions and prints `ALL CHECKPOINTS PASSED` when they pass (21/21 on the current data).

### SQL analysis

`sql/queries.sql` contains exactly the seven required business queries. `src/run_sql.py` runs them with `pandas.read_sql_query` and checks their expected row counts.

| Query | Requirement | Result |
|---|---|---:|
| Q1 | Completed orders over INR 500; customer total over INR 8,000 | 30 customers |
| Q2 | City × product-category revenue, three-table join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 | Products above average catalogue price | 11 products |
| Q4 | Customers without a completed order | 25 customers |
| Q5 | Spend above average using two CTEs | 550 customers; average spend INR 2,552.06 |
| Q6 | RANK, DENSE_RANK and ROW_NUMBER by city | 1,175 customers; 8 rows have RANK different from ROW_NUMBER |
| Q7 | Daily completed revenue and running total | 183 days; final cumulative revenue INR 2,998,668.80 |

Top customer by city in Q6: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10). Ties receive the same RANK and DENSE_RANK, while ROW_NUMBER assigns a different sequential number to each row, so tied rows can make RANK differ from ROW_NUMBER.

### Pandas cross-check and EDA

`src/pandas_analysis.py` rebuilds Q2 with `pd.pivot_table(..., margins=True)`. The pivot grand total is INR 2,998,668.80, matching SQL. City-wise completed revenue (sum / mean / count): Bengaluru INR 523,643.40 / INR 276.47 / 1,894; Chennai INR 463,629.15 / INR 272.72 / 1,700; Delhi INR 502,829.80 / INR 243.62 / 2,064; Hyderabad INR 528,589.90 / INR 284.95 / 1,855; Mumbai INR 479,058.50 / INR 297.00 / 1,613; Pune INR 500,918.05 / INR 252.48 / 1,984.

| Completed order amount statistic | Result |
|---|---:|
| Count | 11,110 |
| Mean / median | INR 269.91 / INR 135.00 |
| Standard deviation | 432.07 |
| Minimum / Q1 / Q2 / Q3 / maximum | INR 12.00 / INR 64.80 / INR 135.00 / INR 297.00 / INR 3,998.00 |
| Skewness | 4.61 |
| IQR / lower fence / upper fence | INR 232.20 / INR -283.50 / INR 645.30 |
| Outliers below / above fences | 0 / 897 |

**EDA summary (under 200 words):** The data describes completed Instamart orders across six cities and five product categories. Completed order amounts are strongly right-skewed: the mean is INR 269.91, the median is INR 135.00, and skewness is 4.61. The middle 50% of orders lie between INR 64.80 and INR 297.00. The IQR upper fence is INR 645.30; 897 orders exceed it and none fall below the lower fence. These high values are outliers by the IQR rule, but they may represent genuine electronics purchases rather than data errors, so they should be investigated before removal. In the correlation matrix, order amount and discount percentage have the largest absolute correlation (about -0.095); order amount and quantity are about +0.087. Both relationships are very weak, and correlation of this size does not establish causation.

## Part B

### 1. RFM Customer Analysis

The RFM analysis contains 1,175 customers using the snapshot date 2025-10-01.

| Metric | Result |
|---|---:|
| Recency median | 19 days |
| Recency minimum | 0 days |
| Recency maximum | 180 days |
| Frequency mean | 9.46 |
| Frequency maximum | 33 |
| Monetary mean | INR 2,552.06 |
| Monetary maximum | INR 14,766.55 |

RFM represents Recency (how recently a customer ordered), Frequency (how often a customer ordered), and Monetary (how much a customer spent).

### 2. K-Means Clustering

The three RFM features were standardized using StandardScaler.
K-Means was evaluated for K values from 2 to 8 with
`n_init=10` and `random_state=42`.

| K | Inertia | Silhouette Score |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 was selected because it has the highest silhouette score (0.526)
among the tested values and provides four useful business segments.
The inertia decreases as K increases because more clusters allow
customers to be grouped closer to their cluster centres. Therefore,
a lower inertia at K=8 does not automatically mean K=8 is the best
choice; silhouette score and business interpretability must also
be considered.

### 3. Customer Segment Profiles

| Segment ID | Segment Name | Customers | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) |
|---:|---|---:|---:|---:|---:|
| 0 | Regular Customers | 436 | 28.86 | 3.77 | 1,057.62 |
| 1 | Premium Customers | 138 | 24.12 | 6.77 | 7,129.70 |
| 2 | Loyal Customers | 419 | 8.77 | 19.40 | 3,410.02 |
| 3 | At-Risk Customers | 182 | 109.04 | 2.20 | 685.99 |

Recommended business actions:

- Regular Customers: Send personalized offers and reorder reminders
  to increase purchase frequency.
- Premium Customers: Provide priority support, exclusive rewards,
  and personalized retention offers.
- Loyal Customers: Maintain engagement through loyalty rewards
  and regular product recommendations.
- At-Risk Customers: Use win-back campaigns and targeted discounts
  to encourage another purchase.

Segment 1 is marked as Premium because it has the highest mean
monetary value (INR 7,129.70).

Cluster numbers are only labels. They do not represent an inherent
ranking or order.

### 4. PCA

PCA was applied to the standardized RFM matrix using
`PCA(n_components=2)`.

| Component | Explained Variance Ratio |
|---|---:|
| PC1 | 0.634 |
| PC2 | 0.218 |
| Total | 0.852 |

The first two principal components explain approximately 85.2%
of the variance in the standardized RFM data.

### 5. Lapse Prediction Problem

The prediction cutoff date is 2025-09-01.
The label period is September 1 to September 30, 2025.

A customer is labelled as lapsed (1) if they have no completed order
during September. Otherwise, the customer is labelled as not lapsed (0).

| Metric | Result |
|---|---:|
| Cohort customers | 1,156 |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training rows | 924 |
| Test rows | 232 |

Only historical orders dated before 2025-09-01 were used to construct
prediction features. September outcomes were used only to create
the target label.

The features include recency, frequency, monetary, average discount,
cancellation rate, average delivery time, favourite product category,
city, age, gender, and tenure.

Customer IDs and Part B task 3 segment labels were not used as model
features.

### 6. Data Preprocessing and Model Tuning

Numeric features were processed using median imputation and
StandardScaler. Categorical features were processed using
most-frequent imputation and OneHotEncoder.

Preprocessing was included inside each model Pipeline and
GridSearchCV, so the transformations were fitted on training data
within the cross-validation process.

The train/test split used:

- `test_size=0.20`
- `stratify=y`
- `random_state=42`

The tuning results were:

| Model | Best Parameters | Best CV F1 |
|---|---|---:|
| Logistic Regression | C=0.01 | 0.7057 |
| Random Forest | max_depth=3, min_samples_leaf=5 | 0.7140 |

### 7. Baseline and Test Evaluation

The baseline predicts every test customer as not lapsed.

For the lapsed class, the baseline precision, recall, and F1-score
are all 0.00.

Both tuned models were evaluated on the same untouched test set.

| Metric (Lapsed Class) | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 0.58 | 0.56 |
| Recall | 0.81 | 0.85 |
| F1-score | 0.68 | 0.68 |

Confusion matrices (rows = actual class, columns = predicted class):

Logistic Regression:
[[91 52]
 [17 72]]

Random Forest:
[[84 59]
 [13 76]]
 ### 8. Class Imbalance and Oversampling

| Metric (Lapsed Class) | Original Logistic Regression | Oversampled Logistic Regression |
|---|---:|---:|
| Precision | 0.58 | 0.58 |
| Recall | 0.81 | 0.80 |
| F1-score | 0.68 | 0.67 |

Oversampling did not improve the results on this test set.
Precision remained the same, while recall and F1-score decreased slightly.
Oversampling was applied only to training data.

### 9. Random Forest Feature Importance

| Feature | Importance |
|---|---:|
| Frequency | 0.402343 |
| Monetary | 0.232677 |
| Cancellation Rate | 0.096855 |
| Recency | 0.095323 |
| Average Discount | 0.049102 |

Frequency and monetary value are the two most important features
for predicting customer lapse.

### 10. Logistic Regression Coefficients

Positive coefficients:

| Feature | Coefficient |
|---|---:|
| Recency | +0.191131 |
| Personal Care | +0.064788 |
| Average Delivery Time | +0.056866 |

Negative coefficients:

| Feature | Coefficient |
|---|---:|
| Grocery | -0.091842 |
| Monetary | -0.133093 |
| Frequency | -0.681779 |

Positive coefficients are associated with higher lapse odds,
while negative coefficients are associated with lower lapse odds.
These are model associations, not proof of causation.

### 11. Model Saving and Reload Verification

The selected Random Forest model was saved as
`outputs/lapse_model.joblib`.

Model reload check: `True`

The reloaded model produced exactly the same predictions
on the test data.

### 12. Premium Customer Retention List

| Customer ID | Lapse Probability | Actually Lapsed |
|---|---:|---:|
| C0060 | 0.588887 | 0 |
| C1112 | 0.585386 | 0 |
| C0942 | 0.556644 | 0 |
| C0267 | 0.553274 | 0 |
| C0021 | 0.552894 | 0 |
| C0188 | 0.549645 | 1 |
| C0862 | 0.548030 | 0 |
| C0831 | 0.546835 | 0 |
| C1156 | 0.546422 | 0 |
| C0133 | 0.545540 | 0 |

Out of the 10 Premium test customers, 1 was actually lapsed.

Limitation: The Premium segment label comes from whole-window RFM,
which includes September outcomes. Therefore, this retention list
is illustrative and is not a leakage-free evaluation. The lapse
prediction model itself uses only pre-cutoff features.

### 13. Leakage Prevention

- No orders dated on or after 2025-09-01 were used to construct prediction features.
- September outcomes were used only to create the target label.
- RFM segment labels were not used as prediction features.
- Imputers, scaler and encoder were fitted within the training pipeline.
- Oversampling was applied only to training data.
- The test set remained untouched during training and oversampling.

## Part C

### 1. Ticket Triage Pipeline

The enhanced ticket triage pipeline processes all 164 support tickets using the offline deterministic `mock_llm.py` simulator. The pipeline is implemented in `src/triage.py` and executed through `src/part_c.py`.

### 2. Input Guardrails

- 4 empty or whitespace-only tickets were skipped with status `skipped_empty`.
- 4 long tickets were flagged with status `flagged_long` because their estimated input exceeded the 400-token limit.
- Both checks are performed before any model call.

### 3. PII Redaction

PII is redacted in the required order before sending ticket text to the model.

| PII type | Redactions |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order ID | 132 |
| **Total** | **278** |

The model receives only the redacted ticket text as its user message. An assertion verifies that the required patterns do not remain after redaction.

### 4. Retry and Reliability

The pipeline handles simulated transient LLM errors using exponential backoff with jitter. All 19 simulated failures were recovered. Retry wait times were within the required 0.045–0.0575-second range.

### 5. Output Validation

| Classification status | Tickets |
|---|---:|
| Valid JSON classifications | 135 |
| Keyword-extracted classifications | 12 |
| Invalid outputs | 9 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| **Total** | **164** |

JSON outputs are validated against the four allowed categories. Chatty responses are parsed using regular-expression keyword extraction. Invalid outputs are sent for human review.

### 6. Zero-shot vs Few-shot Evaluation

Both prompts were evaluated on 147 eligible tickets.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct predictions | 126 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 7,372 | 12,520 |
| Tokens per classified ticket | 50.15 | 85.17 |

Few-shot used 5,148 additional tokens and improved accuracy by 14.29 percentage points.

The zero-shot confusion matrix is:

| Actual / Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

Few-shot is selected because the additional token cost is justified by the improved classification accuracy, particularly for fraud and technical tickets.

### 7. Token Budget Demonstration

A separate zero-shot run used a 6,000-token budget.

| Status | Tickets |
|---|---:|
| JSON classified | 110 |
| Keyword classified | 9 |
| Invalid output | 7 |
| Budget exhausted | 30 |
| Empty skipped | 4 |
| Long flagged | 4 |
| **Total** | **164** |

Actual token usage was 5,963, within the 6,000-token budget. Tickets blocked by the budget were routed to human review.

The output is saved as `outputs/triage_budget_demo.csv`.

### 8. Customer-based Routing

The final few-shot run used the customer segments generated in Part B.

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| **Total** | **164** |

Fraud tickets are routed to the security team with high priority. Premium customers are routed to the premium desk with high priority. Empty, long, invalid, and budget-exhausted tickets are routed to human review.

### 9. Final Output

The final output is `outputs/triage_results.csv`, containing 164 rows and these columns:

`ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, `tokens_used`.

Rule-validated JSON and keyword classifications are routed automatically according to category and customer segment. Invalid, empty, long, and budget-exhausted tickets are handed to human review.

### 10. Implementation Files

- `src/triage.py` — guardrails, PII redaction, prompts, retry handling, token budget, validation, and routing.
- `src/part_c.py` — executes the final Part C pipeline.
- `src/evaluate_triage.py` — zero-shot/few-shot accuracy, token comparison, and confusion matrix.
- `mock_llm.py` — offline deterministic LLM simulator.
- `outputs/triage_results.csv` — final triage results.
- `outputs/triage_budget_demo.csv` — zero-shot budget demonstration.

## Part D

### D1: Chunk-Based Policy RAG

The policy assistant uses chunk-based Retrieval-Augmented Generation (RAG). It splits five policy documents into 40-word chunks with 10-word overlap, producing 15 chunks in total. TF-IDF and cosine similarity are used to retrieve relevant chunks. The retrieved context is passed to the mock LLM to generate grounded answers.

**Policy evaluation results:**

| Metric | Result |
|---|---:|
| Total evaluation questions | 14 |
| Expected documents retrieved | 12/14 |
| Expected facts found | 12/14 |
| Unanswerable questions receiving fallback | 2/2 |

Among the 12 answerable questions, document retrieval and expected fact matching both achieved 12/12.

With zero overlap, document retrieval remains 12/12 for answerable questions, while expected facts are found for 10/12. Fallback is returned for 2/2 unanswerable questions. The questions about the refund threshold and failed-payment return time break because the expected answer phrase is split across chunk boundaries.

### D2: ReAct Refund Agent

The refund agent uses five tools:

- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

The agent uses a SQLite database to verify orders and enforce refund eligibility. It supports bounded execution using iteration, token, and time limits. Agent runs and refund actions are recorded as JSONL, while checkpoints are stored in the tracked `state/` folder.

**Agent scenario results:**

| Scenario | Expected outcome | Actual result | Model calls | Tokens used |
|---|---|---|---:|---:|
| S1 | REFUND INR 128.25 | REFUND | 5 | 897 |
| S2 | REFUND INR 135.00 | REFUND | 5 | 897 |
| S3 | ESCALATE INR 1,999.00 | ESCALATE | 5 | 884 |
| S4 | DENIED | DENIED | 4 | 605 |
| S5 | DENIED | DENIED | 3 | 337 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 72 |
| S7 | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 6 | 2976 |

All seven scenario outcomes and refund amounts matched their expected values (7/7).

### D3: Reliability and Recovery

The agent supports the following reliability tests:

| Test | Configuration | Result |
|---|---|---|
| Token limit | `token_limit=200` | FALLBACK_TOKEN_LIMIT |
| Timeout | `time_limit_seconds=0` | FALLBACK_TIMEOUT |
| Maximum iterations | S7 loop scenario | FALLBACK_MAX_ITERATIONS |
| Crash and resume | Crash after second tool call | REFUND |

During the crash-and-resume test, the agent saved its checkpoint after the second tool observation. On resumption, it continued with the remaining tools. Across the original run and resume, the four tools executed exactly once each in this order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`.

### Part D Evaluation

The integrated Part D evaluation is run using:

```bash
py -m src.part_d
```

The evaluation reuses existing agent execution logs instead of executing refund transactions again.

**Generated evaluation files:**

- `outputs/policy_rag_evaluation_results.csv`
- `outputs/react_agent_scenario_results.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

### ReAct versus Plan-and-Execute

ReAct alternates model calls and tool observations, so the agent can change its next action after learning that an order is missing, cancelled, late, or above the automatic refund threshold.

A plan-and-execute design may use fewer model calls for a predictable path, but it must validate every tool observation and re-plan on unexpected results.

ReAct is chosen here because refund eligibility depends on database observations and the consequences of issuing an incorrect refund are significant. Iteration, token, and time limits provide safe handoff paths.

## Limitations and Next Steps

- All data and policy documents are synthetic; this is not real Instamart data or policy.
- `mock_llm.py` is an offline deterministic simulator, not a hosted language model.
- The refund ledger simulates actions and does not move real money.
- The lapse model and retention list require validation on future real-world data; the Premium flag in the retention list is a post-hoc whole-window label.
- Before production, add authenticated access, audit controls, human approval, monitoring, and integration testing against verified source systems.
