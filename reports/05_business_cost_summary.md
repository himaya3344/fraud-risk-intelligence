# Cost-sensitive fraud decision policy — policy-partition analysis

## Purpose

Choose Approve / Manual Review / Block risk-score cutoffs from the reserved 360-row policy partition. Use the already fitted calibrated model. Do not evaluate the final test partition at this stage.

## Hypothetical cost matrix (currency units, CU)

- **Approve + legitimate:** 0 CU
- **Approve + fraud:** transaction amount x fraud-loss multiplier + fraud extra loss
- **Review + legitimate:** review cost + legitimate-review friction cost
- **Review + fraud:** review cost + (1 - reviewer capture fraction) x fraud loss
- **Block + legitimate:** false-block fixed cost + false-block fraction x transaction amount
- **Block + fraud:** fraud-block handling cost

**All cost values and review capture rates are analyst assumptions, not bank evidence.**

```json
{
  "fraud_loss_multiplier": 1.0,
  "fraud_extra_loss": 30.0,
  "review_cost": 8.0,
  "legit_review_friction": 3.0,
  "review_fraud_capture": 0.8,
  "legit_block_fixed": 15.0,
  "legit_block_fraction": 0.1,
  "fraud_block_handling": 1.0
}
```

## Constraints and policy chosen

- Review capacity <= 25% of policy transactions.
- Legitimate-transaction block rate <= 2% on policy partition.
- Review threshold = 0.17180.
- Block threshold = 0.44042.
- Transactions: 360; actual fraud labels in policy partition: 43.
- Approve=266, Review=86, Block=8.
- Fraud flagged for review or block=21 / 43.
- Legitimate reviewed=67, legitimate blocked=6.
- Fraud captured by automatic block=2. Expected fraud also stopped by review=15.20 (assumed).
- Total hypothetical cost=6045.07 CU.

## Policy comparison (same assumptions and policy partition)

| Policy | Total cost (CU) | Reviews | Blocks | Fraud flagged |
|---|---:|---:|---:|---:|
| Chosen base 3-way policy | 6045.07 | 86 | 8 | 21 |
| Approve all | 7529.85 | 0 | 0 | 0 |
| Binary 0.50 (block positive) | 7212.31 | 0 | 2 | 2 |
| Review at 0.50 (never block) | 7290.22 | 2 | 0 | 2 |

## Sensitivity

Saved `reports/05_policy_sensitivity.csv` compares reoptimized thresholds across six cost scenarios to the base thresholds held fixed.

## Interpretation and limitations

- 0.50 is a default reference threshold, not an economically optimal threshold.
- Review queue and false-block constraints are **hypothetical** safeguards.
- This simulation has labels, but real-time systems must not access actual fraud labels when deciding.
- Review effectiveness is an assumed expected capture probability, not measured fraud recovery.
- Threshold-search results are optimistic because they are optimized on only 360 policy records; assess uncertainty later.
- The chosen calibrated approach did not beat the original uncalibrated Brier score on the earlier small selection subset.
- Transaction currency and operational friction costs are unspecified; all values here are CU.
- No chronological or customer-based holdout is possible because identifiers/timestamps are missing.
- DO NOT tune on the held-out final TEST subset; evaluate it once near final submission.
