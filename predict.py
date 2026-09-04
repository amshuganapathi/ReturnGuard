import os
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from prediction_features import (
    build_prediction_features,
    is_new_user,
    FEATURES,
)


# ============================================================
# RETURNGUARD - FINAL PREDICTION ENGINE
# ============================================================
#
# Flow:
#
# Transaction
#      ↓
# Point-in-time feature generation
#      ↓
# Frozen CatBoost model
#      ↓
# Platt calibration
#      ↓
# Risk policy
#      ↓
# APPROVE / VERIFY / MANUAL REVIEW
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = (
    "models/returnguard_final_temporal_no_item_id.cbm"
)

CALIBRATOR_FILE = (
    "models/returnguard_calibrator_no_item_id.pkl"
)

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.63

MODEL_NAME = (
    "returnguard_final_temporal_no_item_id"
)

MODEL_ITERATIONS = 424


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )


if not os.path.exists(CALIBRATOR_FILE):

    raise FileNotFoundError(
        f"Calibrator not found: {CALIBRATOR_FILE}"
    )


model = CatBoostClassifier()

model.load_model(
    MODEL_FILE
)


calibrator = joblib.load(
    CALIBRATOR_FILE
)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_COLUMNS = [

    "item_size",
    "item_color",
    "brand_id",
    "user_title",
    "user_state",
]


# ============================================================
# RISK POLICY
# ============================================================

def determine_risk_band(
    probability,
    is_new_user=False
):
    """
    Convert calibrated probability into a ReturnGuard
    risk band and operational decision.

    New users are always sent to VERIFY because they
    have no historical behavioral information.
    """

    # --------------------------------------------------------
    # NEW USER FAIL-SAFE
    # --------------------------------------------------------

    if is_new_user:

        return "MEDIUM", "VERIFY"

    # --------------------------------------------------------
    # NORMAL POLICY
    # --------------------------------------------------------

    if probability < LOW_THRESHOLD:

        return "LOW", "APPROVE"

    elif probability < HIGH_THRESHOLD:

        return "MEDIUM", "VERIFY"

    else:

        return "HIGH", "MANUAL_REVIEW"


# ============================================================
# CALIBRATE PROBABILITY
# ============================================================

def calibrate_probability(
    raw_probability
):
    """
    Apply the saved Platt calibration model.
    """

    epsilon = 1e-6

    clipped = np.clip(
        raw_probability,
        epsilon,
        1 - epsilon
    )

    log_odds = np.log(
        clipped /
        (1 - clipped)
    )

    calibrated = calibrator.predict_proba(
        np.array([
            [log_odds]
        ])
    )[0, 1]

    return float(calibrated)


# ============================================================
# VALIDATE MODEL FEATURES
# ============================================================

def validate_model_features(
    input_features
):
    """
    Ensure that prediction_features.py generated
    exactly the features required by the frozen model.
    """

    missing = [
        feature
        for feature in FEATURES
        if feature not in input_features
    ]

    if missing:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )

    if len(input_features) != len(FEATURES):

        raise ValueError(
            "Unexpected number of model features. "
            f"Expected {len(FEATURES)}, "
            f"received {len(input_features)}."
        )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    input_features
):
    """
    Convert generated features into the exact
    DataFrame structure expected by CatBoost.
    """

    validate_model_features(
        input_features
    )

    input_df = pd.DataFrame(
        [input_features]
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        input_df[column] = (
            input_df[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # --------------------------------------------------------
    # Exact feature order
    # --------------------------------------------------------

    input_df = input_df[
        FEATURES
    ]

    return input_df


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_return_risk(
    transaction
):
    """
    Generate a complete ReturnGuard prediction.

    The caller supplies transaction information only.

    Historical behavioral features are calculated
    automatically using prediction_features.py.
    """

    # ========================================================
    # STEP 1
    # DETERMINE WHETHER USER IS NEW
    # ========================================================

    new_user = is_new_user(
        transaction
    )

    # ========================================================
    # STEP 2
    # BUILD POINT-IN-TIME FEATURES
    # ========================================================

    generated_features = (
        build_prediction_features(
            transaction
        )
    )

    # ========================================================
    # STEP 3
    # PREPARE MODEL INPUT
    # ========================================================

    input_df = prepare_model_input(
        generated_features
    )

    # ========================================================
    # STEP 4
    # RAW MODEL PREDICTION
    # ========================================================

    raw_probability = model.predict_proba(
        input_df
    )[0, 1]

    raw_probability = float(
        raw_probability
    )

    # ========================================================
    # STEP 5
    # CALIBRATION
    # ========================================================

    calibrated_probability = (
        calibrate_probability(
            raw_probability
        )
    )

    # ========================================================
    # STEP 6
    # RISK SCORE
    # ========================================================

    risk_score = (
        calibrated_probability
        * 100
    )

    # ========================================================
    # STEP 7
    # RISK POLICY
    # ========================================================

    risk_band, decision = (
        determine_risk_band(
            calibrated_probability,
            new_user
        )
    )

    # ========================================================
    # STEP 8
    # RETURN RESULT
    # ========================================================
    # ========================================================
    # STEP 8
    # RISK EXPLANATION
    # ========================================================

    explanation = []

    user_history = generated_features.get(
        "user_historical_frequency",
        0
    )

    user_return_rate = generated_features.get(
        "user_historical_return_rate",
        0.5
    )

    item_history = generated_features.get(
        "item_historical_frequency",
        0
    )

    item_return_rate = generated_features.get(
        "item_historical_return_rate",
        0.5
    )

    price_vs_item = generated_features.get(
        "item_price_vs_item_avg",
        1.0
    )

    price_vs_brand = generated_features.get(
        "item_price_vs_brand_avg",
        1.0
    )


    if new_user:
        explanation.append(
            "New customer with no prior purchase history"
        )
    else:
        explanation.append(
            f"Customer has {int(user_history)} prior transactions"
        )

        if user_return_rate >= 0.50:
            explanation.append(
                f"Customer historical return rate is "
                f"{user_return_rate:.1%}"
            )
        else:
            explanation.append(
                f"Customer historical return rate is "
                f"{user_return_rate:.1%}"
            )


    if item_history > 0:
        explanation.append(
            f"Product historical return rate is "
            f"{item_return_rate:.1%}"
        )
    else:
        explanation.append(
            "Product has no prior transaction history"
        )


    if price_vs_item > 1.10:
        explanation.append(
            "Item price is above its historical product average"
        )
    elif price_vs_item < 0.90:
        explanation.append(
            "Item price is below its historical product average"
        )
    else:
        explanation.append(
            "Item price is close to its historical product average"
        )


    if price_vs_brand > 1.10:
        explanation.append(
            "Item price is above the historical brand average"
        )
    elif price_vs_brand < 0.90:
        explanation.append(
            "Item price is below the historical brand average"
        )
    else:
        explanation.append(
            "Item price is close to the historical brand average"
        )
    return {

        "raw_probability":
            round(
                raw_probability,
                4
            ),

        "calibrated_probability":
            round(
                calibrated_probability,
                4
            ),

        "risk_score":
            round(
                risk_score,
                2
            ),

        "risk_band":
            risk_band,

        "decision":
            decision,

        "is_new_user":
            bool(new_user),

        "model":
            MODEL_NAME,

        "model_iterations":
            MODEL_ITERATIONS,

       "feature_count":
    len(generated_features),

"risk_explanation":
    explanation,
    }


# ============================================================
# DEMO TRANSACTION
# ============================================================

if __name__ == "__main__":

    demo_transaction = {

        "order_item_id":
            100001,

        "order_date":
            "2016-09-12",

        "user_id":
            30822,

        "item_id":
            643,

        "item_size":
            "38",

        "item_color":
            "navy",

        "brand_id":
            30,

        "item_price":
            59.90,

        "user_title":
            "Mrs",

        "user_dob":
            "1969-01-01",

        "user_state":
            "Saxony",

        "user_reg_date":
            "2015-07-01",
    }


    print("=" * 70)
    print("RETURNGUARD - FINAL MODEL PREDICTION")
    print("=" * 70)


    try:

        result = predict_return_risk(
            demo_transaction
        )


        print(
            "\nRaw Model Probability : "
            f"{result['raw_probability']:.2%}"
        )

        print(
            "Calibrated Probability: "
            f"{result['calibrated_probability']:.2%}"
        )

        print(
            "Risk Score            : "
            f"{result['risk_score']:.2f}/100"
        )

        print(
            "Risk Band             : "
            f"{result['risk_band']}"
        )

        print(
            "Decision              : "
            f"{result['decision']}"
        )

        print(
            "New User              : "
            f"{result['is_new_user']}"
        )

        print(
            "Model                 : "
            f"{result['model']}"
        )

        print(
            "Iterations            : "
            f"{result['model_iterations']}"
        )

        print(
            "Features Used         : "
            f"{result['feature_count']}"
        )


        print(
            "\n" + "=" * 70
        )

    except Exception as error:

        print(
            "\nPrediction failed:"
        )

        print(
            str(error)
        )

        raise