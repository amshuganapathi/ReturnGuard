# ReturnGuard - ML Methodology

## 1. Project Overview

ReturnGuard is a return-risk prediction system designed to estimate the probability that an e-commerce order item will be returned.

The system is designed as a risk-decisioning pipeline rather than only a binary classifier.

The pipeline consists of:

1. Point-in-time feature engineering
2. CatBoost risk prediction
3. Probability calibration using Platt scaling
4. Capacity-constrained decision policy
5. Human-readable risk explanations
6. Audit logging
7. Fail-safe handling for missing information

---

## 2. Dataset

Dataset:

BADS WS2021 return prediction dataset.

File:

`data/BADS_WS2021_known.csv`

Dataset size:

- 100,000 rows
- 14 original columns

Target:

`return`

Class distribution:

- Return = 0: 54.182%
- Return = 1: 45.818%

The dataset contains information about:

- orders
- products
- users
- brands
- item characteristics
- prices
- customer information
- return outcomes

---

## 3. Temporal Evaluation Strategy

Random train/test splitting was not used for the final evaluation.

The dataset is sorted chronologically using:

```text
order_date
order_item_id