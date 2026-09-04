import os
import numpy as np
import pandas as pd


# ============================================================
# RETURNGUARD - PREDICTION FEATURE BUILDER
# ============================================================
#
# Purpose:
#   Convert a new transaction into the exact 25 features
#   expected by the frozen ReturnGuard model.
#
# Important:
#   Historical features use ONLY transactions that occurred
#   before the new transaction.
#
#   The current transaction is NEVER included in its own
#   historical statistics.
#
# ============================================================


DATA_FILE = "data/BADS_WS2021_known.csv"

INITIAL_GLOBAL_PRIOR = 0.50


# ============================================================
# MODEL FEATURE LIST
# ============================================================

FEATURES = [

    # Time
    "order_year",
    "order_month",
    "order_day",
    "order_dayofweek",

    # Customer
    "customer_age",
    "account_age_days",

    # Price
    "item_price",
    "item_price_log",

    # User history
    "user_historical_frequency",
    "user_historical_return_rate",

    # Item history
    "item_historical_frequency",
    "item_historical_return_rate",

    # Brand history
    "brand_historical_frequency",
    "brand_historical_return_rate",

    # Size history
    "size_historical_frequency",
    "size_historical_return_rate",

    # State history
    "state_historical_frequency",
    "state_historical_return_rate",

    # Historical price
    "item_price_vs_item_avg",
    "item_price_vs_brand_avg",

    # Categorical
    "item_size",
    "item_color",
    "brand_id",
    "user_title",
    "user_state",
]


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
# REQUIRED NEW-TRANSACTION FIELDS
# ============================================================

REQUIRED_INPUT_FIELDS = [

    "order_item_id",
    "order_date",
    "user_id",
    "item_id",
    "item_size",
    "item_color",
    "brand_id",
    "item_price",
    "user_title",
    "user_dob",
    "user_state",
    "user_reg_date",
]


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_historical_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    df["user_dob"] = pd.to_datetime(
        df["user_dob"],
        errors="coerce"
    )

    df["user_reg_date"] = pd.to_datetime(
        df["user_reg_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Sort exactly like training
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "order_date",
            "order_item_id"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# VALIDATE NEW TRANSACTION
# ============================================================

def validate_transaction(transaction):

    missing = [

        field

        for field in REQUIRED_INPUT_FIELDS

        if field not in transaction
    ]

    if missing:

        raise ValueError(
            "Missing required transaction fields: "
            + ", ".join(missing)
        )


# ============================================================
# GET HISTORICAL STATISTICS
# ============================================================

def get_history_stats(
    count_dict,
    return_dict,
    key,
    global_rate
):

    count = count_dict.get(
        key,
        0
    )

    # --------------------------------------------------------
    # Cold start
    # --------------------------------------------------------

    if count == 0:

        rate = global_rate

    else:

        rate = (
            return_dict.get(key, 0)
            / count
        )

    return count, rate


# ============================================================
# BUILD HISTORICAL STATE
# ============================================================

def build_historical_state(
    historical_df,
    prediction_date,
    prediction_order_item_id
):

    # --------------------------------------------------------
    # Only use transactions before the new transaction.
    #
    # For the same date, use order_item_id to preserve the
    # exact chronological ordering used during training.
    # --------------------------------------------------------

    historical_df = historical_df[
        (
            historical_df["order_date"]
            < prediction_date
        )
        |
        (
            (
                historical_df["order_date"]
                == prediction_date
            )
            &
            (
                historical_df["order_item_id"]
                < prediction_order_item_id
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Initialize history
    # --------------------------------------------------------

    user_count = {}
    user_returns = {}

    item_count = {}
    item_returns = {}

    brand_count = {}
    brand_returns = {}

    size_count = {}
    size_returns = {}

    state_count = {}
    state_returns = {}

    item_price_sum = {}
    brand_price_sum = {}

    global_count = 0
    global_returns = 0

    # ========================================================
    # PROCESS HISTORICAL TRANSACTIONS
    # ========================================================

    for _, row in historical_df.iterrows():

        # Normalize dictionary keys to strings so training/history
        # lookups and serving-time lookups use the same representation.

        user = str(row["user_id"])
        item = str(row["item_id"])
        brand = str(row["brand_id"])
        size = str(row["item_size"])
        state = str(row["user_state"])

        price = float(
            row["item_price"]
        )

        target = int(
            row["return"]
        )

        # ----------------------------------------------------
        # Global
        # ----------------------------------------------------

        global_count += 1

        global_returns += target

        # ----------------------------------------------------
        # User
        # ----------------------------------------------------

        user_count[user] = (
            user_count.get(user, 0)
            + 1
        )

        user_returns[user] = (
            user_returns.get(user, 0)
            + target
        )

        # ----------------------------------------------------
        # Item
        # ----------------------------------------------------

        item_count[item] = (
            item_count.get(item, 0)
            + 1
        )

        item_returns[item] = (
            item_returns.get(item, 0)
            + target
        )

        # ----------------------------------------------------
        # Brand
        # ----------------------------------------------------

        brand_count[brand] = (
            brand_count.get(brand, 0)
            + 1
        )

        brand_returns[brand] = (
            brand_returns.get(brand, 0)
            + target
        )

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        size_count[size] = (
            size_count.get(size, 0)
            + 1
        )

        size_returns[size] = (
            size_returns.get(size, 0)
            + target
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state_count[state] = (
            state_count.get(state, 0)
            + 1
        )

        state_returns[state] = (
            state_returns.get(state, 0)
            + target
        )

        # ----------------------------------------------------
        # Price history
        # ----------------------------------------------------

        item_price_sum[item] = (
            item_price_sum.get(item, 0.0)
            + price
        )

        brand_price_sum[brand] = (
            brand_price_sum.get(brand, 0.0)
            + price
        )

    return {

        "user_count": user_count,
        "user_returns": user_returns,

        "item_count": item_count,
        "item_returns": item_returns,

        "brand_count": brand_count,
        "brand_returns": brand_returns,

        "size_count": size_count,
        "size_returns": size_returns,

        "state_count": state_count,
        "state_returns": state_returns,

        "item_price_sum": item_price_sum,
        "brand_price_sum": brand_price_sum,

        "global_count": global_count,
        "global_returns": global_returns,
    }


# ============================================================
# BUILD FEATURES FOR NEW TRANSACTION
# ============================================================

def build_prediction_features(
    transaction
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_transaction(
        transaction
    )

    # --------------------------------------------------------
    # Load historical data
    # --------------------------------------------------------

    historical_df = load_historical_data()

    # --------------------------------------------------------
    # Convert new transaction values
    # --------------------------------------------------------

    prediction_date = pd.to_datetime(
        transaction["order_date"],
        errors="coerce"
    )

    if pd.isna(prediction_date):

        raise ValueError(
            "Invalid order_date."
        )

    prediction_order_item_id = int(
        transaction["order_item_id"]
    )

    # --------------------------------------------------------
    # Build history BEFORE current transaction
    # --------------------------------------------------------

    history = build_historical_state(
        historical_df,
        prediction_date,
        prediction_order_item_id
    )

    # --------------------------------------------------------
    # Extract values
    # --------------------------------------------------------

    # Normalize transaction lookup keys to strings.
    # This must match the representation used in historical dictionaries.

    user = str(transaction["user_id"])
    item = str(transaction["item_id"])
    brand = str(transaction["brand_id"])
    size = str(transaction["item_size"])
    state = str(transaction["user_state"])

    price = float(
        transaction["item_price"]
    )

    # ========================================================
    # GLOBAL HISTORICAL RATE
    # ========================================================

    if history["global_count"] == 0:

        global_rate = INITIAL_GLOBAL_PRIOR

    else:

        global_rate = (
            history["global_returns"]
            /
            history["global_count"]
        )

    # ========================================================
    # USER HISTORY
    # ========================================================

    user_freq, user_rate = get_history_stats(

        history["user_count"],
        history["user_returns"],
        user,
        global_rate
    )

    # ========================================================
    # ITEM HISTORY
    # ========================================================

    item_freq, item_rate = get_history_stats(

        history["item_count"],
        history["item_returns"],
        item,
        global_rate
    )

    # ========================================================
    # BRAND HISTORY
    # ========================================================

    brand_freq, brand_rate = get_history_stats(

        history["brand_count"],
        history["brand_returns"],
        brand,
        global_rate
    )

    # ========================================================
    # SIZE HISTORY
    # ========================================================

    size_freq, size_rate = get_history_stats(

        history["size_count"],
        history["size_returns"],
        size,
        global_rate
    )

    # ========================================================
    # STATE HISTORY
    # ========================================================

    state_freq, state_rate = get_history_stats(

        history["state_count"],
        history["state_returns"],
        state,
        global_rate
    )

    # ========================================================
    # ITEM HISTORICAL AVERAGE PRICE
    # ========================================================

    item_price_history = history[
        "item_price_sum"
    ].get(
        item,
        0.0
    )

    if item_freq > 0:

        item_avg_price = (
            item_price_history
            /
            item_freq
        )

    else:

        item_avg_price = price

    # ========================================================
    # BRAND HISTORICAL AVERAGE PRICE
    # ========================================================

    brand_price_history = history[
        "brand_price_sum"
    ].get(
        brand,
        0.0
    )

    if brand_freq > 0:

        brand_avg_price = (
            brand_price_history
            /
            brand_freq
        )

    else:

        brand_avg_price = price

    # ========================================================
    # PRICE VS ITEM AVERAGE
    # ========================================================

    if item_avg_price == 0:

        item_price_ratio = 1.0

    else:

        item_price_ratio = (
            price
            /
            item_avg_price
        )

    # ========================================================
    # PRICE VS BRAND AVERAGE
    # ========================================================

    if brand_avg_price == 0:

        brand_price_ratio = 1.0

    else:

        brand_price_ratio = (
            price
            /
            brand_avg_price
        )

    # ========================================================
    # CUSTOMER AGE
    # ========================================================

    user_dob = pd.to_datetime(
        transaction["user_dob"],
        errors="coerce"
    )

    if pd.isna(user_dob):
           # Frozen training-time median used by the deployed model.
        customer_age = 51.51813826146475

    else:

        customer_age = (

            (
                prediction_date
                -
                user_dob
            ).days

            /
            365.25
        )

    # ========================================================
    # ACCOUNT AGE
    # ========================================================

    user_reg_date = pd.to_datetime(
        transaction["user_reg_date"],
        errors="coerce"
    )

    if pd.isna(user_reg_date):

        account_age_days = 0

    else:

        account_age_days = (

            prediction_date
            -
            user_reg_date
        ).days

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if account_age_days < 0:

            account_age_days = 0

    # ========================================================
    # CATEGORICAL VALUES
    # ========================================================

    # NOTE: `size`, `brand`, `state` above are already str()-cast
    # (never NaN, since str(nan) == "nan", not an actual missing marker).
    # The pd.isna() checks below therefore look at the ORIGINAL
    # transaction fields, not the normalized lookup keys.

    item_size = (

        str(transaction["item_size"])

        if not pd.isna(transaction["item_size"])

        else "UNKNOWN"
    )

    item_color = (

        str(transaction["item_color"])

        if not pd.isna(
            transaction["item_color"]
        )

        else "UNKNOWN"
    )

    brand_id = (

        str(transaction["brand_id"])

        if not pd.isna(transaction["brand_id"])

        else "UNKNOWN"
    )

    user_title = (

        str(transaction["user_title"])

        if not pd.isna(
            transaction["user_title"]
        )

        else "UNKNOWN"
    )

    user_state = (

        str(transaction["user_state"])

        if not pd.isna(transaction["user_state"])

        else "UNKNOWN"
    )

    # ========================================================
    # CREATE EXACT MODEL INPUT
    # ========================================================

    prediction_features = {

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        "order_year":
            prediction_date.year,

        "order_month":
            prediction_date.month,

        "order_day":
            prediction_date.day,

        "order_dayofweek":
            prediction_date.dayofweek,

        # ----------------------------------------------------
        # Customer
        # ----------------------------------------------------

        "customer_age":
            customer_age,

        "account_age_days":
            account_age_days,

        # ----------------------------------------------------
        # Price
        # ----------------------------------------------------

        "item_price":
            price,

        "item_price_log":
            np.log1p(price),

        # ----------------------------------------------------
        # User history
        # ----------------------------------------------------

        "user_historical_frequency":
            user_freq,

        "user_historical_return_rate":
            user_rate,

        # ----------------------------------------------------
        # Item history
        # ----------------------------------------------------

        "item_historical_frequency":
            item_freq,

        "item_historical_return_rate":
            item_rate,

        # ----------------------------------------------------
        # Brand history
        # ----------------------------------------------------

        "brand_historical_frequency":
            brand_freq,

        "brand_historical_return_rate":
            brand_rate,

        # ----------------------------------------------------
        # Size history
        # ----------------------------------------------------

        "size_historical_frequency":
            size_freq,

        "size_historical_return_rate":
            size_rate,

        # ----------------------------------------------------
        # State history
        # ----------------------------------------------------

        "state_historical_frequency":
            state_freq,

        "state_historical_return_rate":
            state_rate,

        # ----------------------------------------------------
        # Historical price
        # ----------------------------------------------------

        "item_price_vs_item_avg":
            item_price_ratio,

        "item_price_vs_brand_avg":
            brand_price_ratio,

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        "item_size":
            item_size,

        "item_color":
            item_color,

        "brand_id":
            brand_id,

        "user_title":
            user_title,

        "user_state":
            user_state,
    }

    return prediction_features


# ============================================================
# DETECT NEW USER
# ============================================================

def is_new_user(
    transaction
):

    validate_transaction(
        transaction
    )

    historical_df = load_historical_data()

    prediction_date = pd.to_datetime(
        transaction["order_date"],
        errors="coerce"
    )

    if pd.isna(prediction_date):

        raise ValueError(
            "Invalid order_date."
        )

    prediction_order_item_id = int(
        transaction["order_item_id"]
    )

    user = str(transaction["user_id"])

    # --------------------------------------------------------
    # Look only at history before this transaction
    # --------------------------------------------------------

    previous_transactions = historical_df[
        (
            historical_df["order_date"]
            < prediction_date
        )
        |
        (
            (
                historical_df["order_date"]
                == prediction_date
            )
            &
            (
                historical_df["order_item_id"]
                < prediction_order_item_id
            )
        )
    ]

    user_history = previous_transactions[
        previous_transactions["user_id"].astype(str)
        == user
    ]

    return len(user_history) == 0


# ============================================================
# DISPLAY FEATURES
# ============================================================

def print_prediction_features(
    transaction
):

    print("=" * 70)
    print("RETURNGUARD - PREDICTION FEATURE BUILDER")
    print("=" * 70)

    new_user = is_new_user(
        transaction
    )

    features = build_prediction_features(
        transaction
    )

    print(
        "\nNew user:",
        new_user
    )

    print(
        "\nGenerated model features:"
    )

    print("-" * 70)

    for index, feature in enumerate(
        FEATURES,
        start=1
    ):

        value = features[
            feature
        ]

        if isinstance(
            value,
            float
        ):

            print(
                f"{index:2}. "
                f"{feature:40} "
                f"{value:.6f}"
            )

        else:

            print(
                f"{index:2}. "
                f"{feature:40} "
                f"{value}"
            )

    print("-" * 70)

    print(
        "\nTotal features:",
        len(features)
    )

    print(
        "Expected features:",
        len(FEATURES)
    )

    print("\n" + "=" * 70)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 70)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example transaction
    #
    # order_item_id = 100001 is after the dataset's
    # existing order_item_ids, so all historical dataset
    # transactions can be considered prior history for this
    # demonstration date.
    # --------------------------------------------------------

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

    print_prediction_features(
        demo_transaction
    )