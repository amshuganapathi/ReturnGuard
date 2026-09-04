# ReturnGuard

## AI-Powered Return Risk Manager

ReturnGuard is an AI-powered e-commerce return risk management system that predicts whether a transaction is likely to result in a product return.

The system combines historical customer behavior, product behavior, pricing information, temporal features, and transaction attributes to produce an actionable risk assessment.

Instead of only predicting probability, ReturnGuard converts the prediction into an operational decision:

- **LOW** → APPROVE
- **MEDIUM** → VERIFY
- **HIGH** → MANUAL REVIEW

---

# Problem

Product returns create operational and financial costs for e-commerce businesses.

A return-risk system should identify potentially high-risk transactions before additional return costs are incurred, while avoiding unnecessary intervention for legitimate customers.

ReturnGuard addresses this problem using:

- Point-in-time historical features
- Machine learning
- Probability calibration
- Risk-based policy thresholds
- New-user safeguards
- Human-readable explanations

---

# Solution

A transaction passes through the following pipeline:

1. Transaction details are submitted.
2. Point-in-time historical features are generated using information available before the transaction.
3. A frozen CatBoost model predicts raw return probability.
4. Platt scaling calibrates the probability.
5. The calibrated probability is converted into a risk score.
6. A policy layer assigns a risk band and operational decision.
7. New users receive a conservative VERIFY decision.
8. The web dashboard displays the result and risk explanation.

### Production Flow

```text
Transaction
     ↓
Point-in-Time Feature Generation
     ↓
CatBoost Classifier
     ↓
Platt Probability Calibration
     ↓
Risk Policy
     ↓
Risk Score + Risk Band + Decision
     ↓
Explanation
     ↓
Web Dashboard
```markdown
---

# Dataset

ReturnGuard was trained and evaluated using the **BADS 2020/2021 e-commerce returns dataset**.

The full training dataset is intentionally excluded from this public repository.

For reproducible demonstration and serving from a fresh GitHub clone, this repository includes a small demo history file:

`data/demo_historical_data.csv`

The application automatically uses the full BADS dataset when it is available locally, and falls back to the demo history when the full dataset is unavailable.

The demo history is provided only for demonstrating the prediction pipeline. The reported model metrics are based on the complete chronological evaluation described in this README and are not calculated from the demo history.
