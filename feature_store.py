import os
import numpy as np
import pandas as pd


# ============================================================
# RETURNGUARD FEATURE STORE
# ============================================================
#
# Purpose:
#   Create point-in-time historical features for ReturnGuard.
#
# Important:
#   The current transaction is NOT added to history until
#   after its historical features have been calculated.
#
# ============================================================


DATA_FILE = "data/BADS_WS2021_known.csv"

INITIAL_GLOBAL_PRIOR = 0.50


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "order_item_id",
    "order_date",
    "item_id",
    "item_size",
    "item_color",
    "brand_id",
    "item_price",
    "user_id",
    "user_title",
    "user_dob",
    "user_state",
    "user_reg_date",
    "return",
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

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Historical dataset is missing columns: "
            + ", ".join(missing)
        )

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
        ["order_date", "order_item_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# BUILD POINT-IN-TIME FEATURE STORE
# ============================================================

def build_feature_store(df):

    # --------------------------------------------------------
    # Historical counters
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

    # --------------------------------------------------------
    # Historical price sums
    # --------------------------------------------------------

    item_price_sum = {}
    brand_price_sum = {}

    # --------------------------------------------------------
    # Global history
    # --------------------------------------------------------

    global_count = 0
    global_returns = 0

    feature_rows = []

    # ========================================================
    # PROCESS TRANSACTIONS CHRONOLOGICALLY
    # ========================================================

    for _, row in df.iterrows():

        # Normalize dictionary keys to strings so training/history
        # lookups and serving-time lookups use the same representation.

        user = str(row["user_id"])
        item = str(row["item_id"])
        brand = str(row["brand_id"])
        size = str(row["item_size"])
        state = str(row["user_state"])

        price = float(row["item_price"])

        # ----------------------------------------------------
        # GLOBAL HISTORICAL RETURN RATE
        # ----------------------------------------------------

        if global_count == 0:

            global_rate = INITIAL_GLOBAL_PRIOR

        else:

            global_rate = (
                global_returns /
                global_count
            )

        # ----------------------------------------------------
        # USER HISTORY
        # ----------------------------------------------------

        user_freq = user_count.get(
            user,
            0
        )

        if user_freq == 0:

            user_rate = global_rate

        else:

            user_rate = (
                user_returns.get(user, 0)
                / user_freq
            )

        # ----------------------------------------------------
        # ITEM HISTORY
        # ----------------------------------------------------

        item_freq = item_count.get(
            item,
            0
        )

        if item_freq == 0:

            item_rate = global_rate

        else:

            item_rate = (
                item_returns.get(item, 0)
                / item_freq
            )

        # ----------------------------------------------------
        # BRAND HISTORY
        # ----------------------------------------------------

        brand_freq = brand_count.get(
            brand,
            0
        )

        if brand_freq == 0:

            brand_rate = global_rate

        else:

            brand_rate = (
                brand_returns.get(brand, 0)
                / brand_freq
            )

        # ----------------------------------------------------
        # SIZE HISTORY
        # ----------------------------------------------------

        size_freq = size_count.get(
            size,
            0
        )

        if size_freq == 0:

            size_rate = global_rate

        else:

            size_rate = (
                size_returns.get(size, 0)
                / size_freq
            )

        # ----------------------------------------------------
        # STATE HISTORY
        # ----------------------------------------------------

        state_freq = state_count.get(
            state,
            0
        )

        if state_freq == 0:

            state_rate = global_rate

        else:

            state_rate = (
                state_returns.get(state, 0)
                / state_freq
            )

        # ----------------------------------------------------
        # HISTORICAL ITEM PRICE
        # ----------------------------------------------------

        item_price_history = item_price_sum.get(
            item,
            0.0
        )

        if item_freq > 0:

            item_avg_price = (
                item_price_history /
                item_freq
            )

        else:

            item_avg_price = price

        # ----------------------------------------------------
        # HISTORICAL BRAND PRICE
        # ----------------------------------------------------

        brand_price_history = brand_price_sum.get(
            brand,
            0.0
        )

        if brand_freq > 0:

            brand_avg_price = (
                brand_price_history /
                brand_freq
            )

        else:

            brand_avg_price = price

        # ----------------------------------------------------
        # PRICE VS ITEM AVERAGE
        # ----------------------------------------------------

        if item_avg_price == 0:

            item_price_ratio = 1.0

        else:

            item_price_ratio = (
                price /
                item_avg_price
            )

        # ----------------------------------------------------
        # PRICE VS BRAND AVERAGE
        # ----------------------------------------------------

        if brand_avg_price == 0:

            brand_price_ratio = 1.0

        else:

            brand_price_ratio = (
                price /
                brand_avg_price
            )

        # ----------------------------------------------------
        # CUSTOMER AGE
        # ----------------------------------------------------

        if pd.isna(row["user_dob"]):

            customer_age = np.nan

        else:

            customer_age = (
                (
                    row["order_date"]
                    - row["user_dob"]
                ).days
                / 365.25
            )

        # ----------------------------------------------------
        # ACCOUNT AGE
        # ----------------------------------------------------

        if pd.isna(row["user_reg_date"]):

            account_age_days = 0

        else:

            account_age_days = (
                row["order_date"]
                - row["user_reg_date"]
            ).days

            # -----------------------------------------------
            # Safety check:
            # Registration cannot logically be after order.
            # -----------------------------------------------

            if account_age_days < 0:

                account_age_days = 0

        # ----------------------------------------------------
        # CATEGORICAL VALUES
        # ----------------------------------------------------

        item_size = (
            str(row["item_size"])
            if not pd.isna(row["item_size"])
            else "UNKNOWN"
        )

        item_color = (
            str(row["item_color"])
            if not pd.isna(row["item_color"])
            else "UNKNOWN"
        )

        brand_id = (
            str(row["brand_id"])
            if not pd.isna(row["brand_id"])
            else "UNKNOWN"
        )

        user_title = (
            str(row["user_title"])
            if not pd.isna(row["user_title"])
            else "UNKNOWN"
        )

        user_state = (
            str(row["user_state"])
            if not pd.isna(row["user_state"])
            else "UNKNOWN"
        )

        # ====================================================
        # SAVE POINT-IN-TIME FEATURES
        # ====================================================

        feature_rows.append({

            # ------------------------------------------------
            # Time
            # ------------------------------------------------

            "order_year":
                row["order_date"].year,

            "order_month":
                row["order_date"].month,

            "order_day":
                row["order_date"].day,

            "order_dayofweek":
                row["order_date"].dayofweek,

            # ------------------------------------------------
            # Customer
            # ------------------------------------------------

            "customer_age":
                customer_age,

            "account_age_days":
                account_age_days,

            # ------------------------------------------------
            # Price
            # ------------------------------------------------

            "item_price":
                price,

            "item_price_log":
                np.log1p(price),

            # ------------------------------------------------
            # User history
            # ------------------------------------------------

            "user_historical_frequency":
                user_freq,

            "user_historical_return_rate":
                user_rate,

            # ------------------------------------------------
            # Item history
            # ------------------------------------------------

            "item_historical_frequency":
                item_freq,

            "item_historical_return_rate":
                item_rate,

            # ------------------------------------------------
            # Brand history
            # ------------------------------------------------

            "brand_historical_frequency":
                brand_freq,

            "brand_historical_return_rate":
                brand_rate,

            # ------------------------------------------------
            # Size history
            # ------------------------------------------------

            "size_historical_frequency":
                size_freq,

            "size_historical_return_rate":
                size_rate,

            # ------------------------------------------------
            # State history
            # ------------------------------------------------

            "state_historical_frequency":
                state_freq,

            "state_historical_return_rate":
                state_rate,

            # ------------------------------------------------
            # Historical price behavior
            # ------------------------------------------------

            "item_price_vs_item_avg":
                item_price_ratio,

            "item_price_vs_brand_avg":
                brand_price_ratio,

            # ------------------------------------------------
            # Categorical features
            # ------------------------------------------------

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
        })

        # ====================================================
        # ADD CURRENT TRANSACTION TO HISTORY
        # ====================================================
        #
        # This happens AFTER feature calculation.
        #
        # Therefore the current return label can never
        # influence its own historical features.
        #
        # ====================================================

        target = int(row["return"])

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

    # ========================================================
    # CREATE FEATURE DATAFRAME
    # ========================================================

    feature_df = pd.DataFrame(
        feature_rows
    )

    return feature_df


# ============================================================
# CREATE AND SAVE FEATURE STORE
# ============================================================

def create_feature_store():

    print("=" * 70)
    print("RETURNGUARD - BUILDING POINT-IN-TIME FEATURE STORE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading historical dataset...")

    df = load_historical_data()

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['order_date'].min().date()} "
        f"-> "
        f"{df['order_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    print(
        "\nCalculating historical features..."
    )

    feature_df = build_feature_store(
        df
    )

    print(
        f"Feature rows created: "
        f"{len(feature_df):,}"
    )

    # ========================================================
    # IDENTIFIERS / TARGET
    # ========================================================

    identifiers = df[
        [
            "order_item_id",
            "order_date",
            "user_id",
            "item_id",
            "item_price",
            "return"
        ]
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Remove item_price from feature dataframe if present
    #
    # item_price is already included in identifiers.
    # This prevents duplicate item_price columns.
    # --------------------------------------------------------

    feature_df = feature_df.drop(
        columns=["item_price"],
        errors="ignore"
    )

    # ========================================================
    # COMBINE
    # ========================================================

    output = pd.concat(
        [
            identifiers,
            feature_df.reset_index(drop=True)
        ],
        axis=1
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    # --------------------------------------------------------
    # Check duplicate column names
    # --------------------------------------------------------

    duplicate_columns = (
        output.columns[
            output.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            "Duplicate columns detected: "
            + ", ".join(duplicate_columns)
        )

    # --------------------------------------------------------
    # Check row count
    # --------------------------------------------------------

    if len(output) != len(df):

        raise ValueError(
            "Feature store row count does not "
            "match historical dataset."
        )

    # --------------------------------------------------------
    # Check negative account ages
    # --------------------------------------------------------

    negative_account_age = (
        output["account_age_days"] < 0
    ).sum()

    if negative_account_age > 0:

        raise ValueError(
            "Negative account ages remain: "
            + str(negative_account_age)
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        "data/returnguard_feature_store.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        f"\nFeature store saved to:"
        f"\n{output_file}"
    )

    print(
        "\nFeature store shape:"
    )

    print(
        output.shape
    )

    print(
        "\nNumber of columns:",
        len(output.columns)
    )

    print(
        "\nDuplicate columns:",
        len(duplicate_columns)
    )

    print(
        "Negative account ages:",
        negative_account_age
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        output.head().to_string()
    )

    print("\n" + "=" * 70)
    print("FEATURE STORE COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_feature_store()