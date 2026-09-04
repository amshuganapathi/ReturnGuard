import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostClassifier, Pool


# ============================================================
# RETURNGUARD - EXPLAINABLE PREDICTION
# Uses the FINAL frozen production model
# ============================================================

MODEL_PATH = "models/returnguard_final_temporal_no_item_id.cbm"
CALIBRATOR_PATH = "models/returnguard_calibrator_no_item_id.pkl"

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.63


print("=" * 70)
print("RETURNGUARD - EXPLAINABLE PREDICTION")
print("=" * 70)


# ============================================================
# 1. LOAD FINAL PRODUCTION MODEL
# ============================================================

print("\nLoading final production model...")

model = CatBoostClassifier()

model.load_model(MODEL_PATH)

print("Model loaded.")
print(f"Model iterations: {model.tree_count_}")
print(f"Model features: {len(model.feature_names_)}")


# ============================================================
# 2. LOAD FINAL CALIBRATOR
# ============================================================

print("\nLoading final calibrator...")

calibrator = joblib.load(CALIBRATOR_PATH)

print("Calibrator loaded.")


# ============================================================
# 3. FINAL PRODUCTION FEATURE ORDER
# ============================================================

features = [
    "order_year",
    "order_month",
    "order_day",
    "order_dayofweek",

    "customer_age",
    "account_age_days",

    "item_price",
    "item_price_log",

    "user_historical_frequency",
    "user_historical_return_rate",

    "item_historical_frequency",
    "item_historical_return_rate",

    "brand_historical_frequency",
    "brand_historical_return_rate",

    "size_historical_frequency",
    "size_historical_return_rate",

    "state_historical_frequency",
    "state_historical_return_rate",

    "item_price_vs_item_avg",
    "item_price_vs_brand_avg",

    "item_size",
    "item_color",
    "brand_id",
    "user_title",
    "user_state"
]


# ============================================================
# 4. VERIFY MODEL FEATURE COUNT
# ============================================================

if len(features) != 25:
    raise ValueError(
        f"Expected 25 production features, got {len(features)}"
    )

if len(model.feature_names_) != 25:
    raise ValueError(
        f"Expected model to contain 25 features, "
        f"but model contains {len(model.feature_names_)}"
    )

if model.feature_names_ != features:
    raise ValueError(
        "Production feature order does not match the frozen model."
    )

print("\nFeature verification passed.")
print("25 production features confirmed.")
print("Feature order matches frozen model.")


# ============================================================
# 5. DEMO ORDER
# ============================================================

order = {

    "order_year": 2016,
    "order_month": 9,
    "order_day": 5,
    "order_dayofweek": 0,

    "customer_age": 30.0,
    "account_age_days": 400,

    "item_price": 59.90,
    "item_price_log": np.log1p(59.90),

    # Customer history
    "user_historical_frequency": 3,
    "user_historical_return_rate": 0.33,

    # Product history
    "item_historical_frequency": 10,
    "item_historical_return_rate": 0.40,

    # Brand history
    "brand_historical_frequency": 50,
    "brand_historical_return_rate": 0.45,

    # Size history
    "size_historical_frequency": 100,
    "size_historical_return_rate": 0.42,

    # State history
    "state_historical_frequency": 1000,
    "state_historical_return_rate": 0.46,

    # Price behavior
    "item_price_vs_item_avg": 1.05,
    "item_price_vs_brand_avg": 1.02,

    # Categorical features
    "item_size": "M",
    "item_color": "Black",
    "brand_id": "10",
    "user_title": "Mr",
    "user_state": "California"
}


# ============================================================
# 6. CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "item_size",
    "item_color",
    "brand_id",
    "user_title",
    "user_state"
]


# ============================================================
# 7. CREATE INPUT DATAFRAME
# ============================================================

input_df = pd.DataFrame(
    [order],
    columns=features
)


# ============================================================
# 8. FORMAT CATEGORICAL FEATURES
# ============================================================

for col in categorical_columns:
    input_df[col] = (
        input_df[col]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# 9. CREATE CATBOOST POOL
# ============================================================

prediction_pool = Pool(
    data=input_df,
    cat_features=categorical_columns
)


# ============================================================
# 10. RAW MODEL PREDICTION
# ============================================================

print("\nGenerating prediction...")

raw_probability = model.predict_proba(
    prediction_pool
)[0][1]


# ============================================================
# 11. CALIBRATE PROBABILITY
# ============================================================

epsilon = 1e-6

p = np.clip(
    raw_probability,
    epsilon,
    1 - epsilon
)

log_odds = np.log(
    p / (1 - p)
)

calibrated_probability = calibrator.predict_proba(
    [[log_odds]]
)[0][1]


# ============================================================
# 12. RISK SCORE
# ============================================================

risk_score = calibrated_probability * 100


# ============================================================
# 13. RISK LEVEL AND DECISION
# ============================================================

if calibrated_probability < LOW_THRESHOLD:

    risk_level = "LOW"
    decision = "APPROVE"

elif calibrated_probability < HIGH_THRESHOLD:

    risk_level = "MEDIUM"
    decision = "VERIFY"

else:

    risk_level = "HIGH"
    decision = "MANUAL REVIEW"


# ============================================================
# 14. SHAP EXPLANATION
# ============================================================

print("\nCalculating SHAP explanation...")

shap_values = model.get_feature_importance(
    type="ShapValues",
    data=prediction_pool
)

# CatBoost returns one additional value:
# the expected/base prediction.

shap_for_features = shap_values[0][:-1]


# ============================================================
# 15. CREATE EXPLANATION TABLE
# ============================================================

explanation = pd.DataFrame({

    "feature": features,

    "value": [
        input_df.iloc[0][feature]
        for feature in features
    ],

    "shap_contribution": shap_for_features

})


# Absolute SHAP contribution
explanation["absolute_contribution"] = (
    explanation["shap_contribution"].abs()
)


# Sort from most important to least important
explanation = explanation.sort_values(
    "absolute_contribution",
    ascending=False
)


# ============================================================
# 16. DISPLAY DECISION
# ============================================================

print("\n" + "=" * 70)
print("RETURNGUARD DECISION")
print("=" * 70)

print(
    f"\nRaw Model Probability : "
    f"{raw_probability:.2%}"
)

print(
    f"Calibrated Probability: "
    f"{calibrated_probability:.2%}"
)

print(
    f"Risk Score            : "
    f"{risk_score:.2f}/100"
)

print(
    f"Risk Level            : "
    f"{risk_level}"
)

print(
    f"Decision              : "
    f"{decision}"
)


# ============================================================
# 17. DISPLAY TOP RISK FACTORS
# ============================================================

print("\n" + "=" * 70)
print("TOP RISK FACTORS")
print("=" * 70)

top_features = explanation.head(10)


for _, row in top_features.iterrows():

    contribution = row["shap_contribution"]

    if contribution > 0:
        direction = "INCREASES return risk"
    else:
        direction = "DECREASES return risk"

    print(f"\n{row['feature']}")

    print(
        f"  Value: {row['value']}"
    )

    print(
        f"  SHAP contribution: "
        f"{contribution:+.4f}"
    )

    print(
        f"  Effect: {direction}"
    )


# ============================================================
# 18. SAVE EXPLANATION
# ============================================================

output_file = "data/latest_prediction_explanation.csv"

explanation.to_csv(
    output_file,
    index=False
)

print("\nExplanation saved to:")
print(output_file)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"\nReturn probability: "
    f"{calibrated_probability:.2%}"
)

print(
    f"Risk score: "
    f"{risk_score:.2f}/100"
)

print(
    f"Risk level: "
    f"{risk_level}"
)

print(
    f"Decision: "
    f"{decision}"
)

print("\nRisk policy:")

print(
    f"LOW    : probability < "
    f"{LOW_THRESHOLD:.2f}"
)

print(
    f"MEDIUM : "
    f"{LOW_THRESHOLD:.2f} <= probability < "
    f"{HIGH_THRESHOLD:.2f}"
)

print(
    f"HIGH   : probability >= "
    f"{HIGH_THRESHOLD:.2f}"
)

print("\n" + "=" * 70)
print("EXPLAINABLE PREDICTION COMPLETE")
print("=" * 70)