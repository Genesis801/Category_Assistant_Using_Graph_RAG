import json
import textwrap
from pathlib import Path


NOTEBOOK_PATH = Path("price_optimization.ipynb")


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


cells = [
    md_cell(
        """
        # Price Recommendation Engine

        This notebook builds a price recommendation workflow for Amazon-style catalog data.

        We will:
        - clean noisy fields such as `ratings` and `no_of_ratings`
        - run memory-aware EDA on a large dataset
        - engineer structured features from product names and category metadata
        - perform lightweight feature selection for the structured features
        - train a simple tree-based baseline
        - compare it with advanced gradient-boosted models (`XGBoost` and `LightGBM` when available)
        - expose a helper to recommend a price for a new product given its name and department

        The main prediction target is `discount_price`, which acts as the market-facing recommended price.
        """
    ),
    md_cell(
        """
        ## Notes on scalability

        The raw dataset is large, so this notebook is designed to stay efficient:
        - only required columns are loaded
        - string columns are cast intentionally
        - feature engineering is vectorized where possible
        - training can be capped with a configurable sample size
        - GPU acceleration is used automatically by `XGBoost` / `LightGBM` if the libraries are installed and CUDA is available

        If `xgboost` or `lightgbm` are not installed yet, you can install them manually in a separate cell:

        ```python
        %pip install xgboost lightgbm
        ```
        """
    ),
    code_cell(
        """
        import re
        import warnings
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.tree import DecisionTreeRegressor

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid")
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

        DATA_PATH = Path("Amazon_data_category_wise.csv")
        TARGET_COL = "discount_price"
        RANDOM_STATE = 42
        MAX_TRAIN_ROWS = 200_000
        VALID_SIZE = 0.2

        try:
            import torch
            GPU_AVAILABLE = torch.cuda.is_available()
        except Exception:
            GPU_AVAILABLE = False

        try:
            from xgboost import XGBRegressor
            XGBOOST_AVAILABLE = True
        except Exception:
            XGBRegressor = None
            XGBOOST_AVAILABLE = False

        try:
            from lightgbm import LGBMRegressor
            LIGHTGBM_AVAILABLE = True
        except Exception:
            LGBMRegressor = None
            LIGHTGBM_AVAILABLE = False

        print(
            {
                "gpu_available": GPU_AVAILABLE,
                "xgboost_available": XGBOOST_AVAILABLE,
                "lightgbm_available": LIGHTGBM_AVAILABLE,
            }
        )
        """
    ),
    code_cell(
        """
        usecols = [
            "name",
            "main_category",
            "sub_category",
            "ratings",
            "no_of_ratings",
            "discount_price",
            "actual_price",
        ]

        dtype_map = {
            "name": "string",
            "main_category": "string",
            "sub_category": "string",
            "ratings": "string",
            "no_of_ratings": "string",
            "discount_price": "float64",
            "actual_price": "float64",
        }

        raw_df = pd.read_csv(
            DATA_PATH,
            usecols=usecols,
            dtype=dtype_map,
            low_memory=False,
        )

        print(f"Shape: {raw_df.shape}")
        print(f"Approx memory usage: {raw_df.memory_usage(deep=True).sum() / 1024**2:,.2f} MB")
        raw_df.head()
        """
    ),
    code_cell(
        """
        NUMBER_PATTERN = re.compile(r"(\\d+[\\d,]*\\.?\\d*)")

        def safe_float(text):
            if pd.isna(text):
                return np.nan
            text = str(text).strip().lower()
            match = NUMBER_PATTERN.search(text.replace(",", ""))
            return float(match.group(1)) if match else np.nan

        def extract_count(text):
            if pd.isna(text):
                return np.nan
            text = str(text)
            match = re.search(r"[\\d,]+", text)
            if match:
                return float(match.group().replace(",", ""))
            return np.nan

        def extract_inventory_status(text):
            text = "" if pd.isna(text) else str(text).lower()
            if "out of stock" in text:
                return "out_of_stock"
            if "unavailable" in text:
                return "unavailable"
            if "left in stock" in text:
                return "low_stock"
            if text.strip() == "":
                return "missing"
            return "available"

        def extract_inventory_count(text):
            text = "" if pd.isna(text) else str(text).lower()
            if "left in stock" in text:
                return extract_count(text)
            return np.nan

        def extract_brand(name):
            text = "" if pd.isna(name) else str(name).strip()
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9&+\\-]*)", text)
            return match.group(1).lower() if match else "unknown"

        def extract_measure(text, pattern):
            text = "" if pd.isna(text) else str(text).lower()
            match = re.search(pattern, text)
            return float(match.group(1)) if match else np.nan

        def build_features(df):
            out = df.copy()

            out["name"] = out["name"].fillna("").astype("string")
            out["main_category"] = out["main_category"].fillna("unknown").astype("string")
            out["sub_category"] = out["sub_category"].fillna("unknown").astype("string")

            out["rating_value"] = out["ratings"].apply(safe_float)
            out["rating_count"] = out["no_of_ratings"].apply(extract_count)
            out["inventory_status"] = out["no_of_ratings"].apply(extract_inventory_status)
            out["inventory_count_hint"] = out["no_of_ratings"].apply(extract_inventory_count)
            out["brand"] = out["name"].apply(extract_brand)

            name_lower = out["name"].str.lower()
            out["name_char_len"] = out["name"].str.len()
            out["name_word_count"] = out["name"].str.split().str.len()
            out["digit_count"] = out["name"].str.count(r"\\d")
            out["has_numbers"] = out["digit_count"].gt(0).astype(int)
            out["has_combo"] = name_lower.str.contains("combo|pack of|set of", regex=True, na=False).astype(int)
            out["has_wireless"] = name_lower.str.contains("wireless|bluetooth|wifi", regex=True, na=False).astype(int)
            out["has_smart"] = name_lower.str.contains("smart|ai|pro|max|plus", regex=True, na=False).astype(int)

            out["first_number"] = name_lower.str.extract(r"(\\d+(?:\\.\\d+)?)").astype(float)
            out["size_inch"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*(?:inch|inches|\\\")"))
            out["size_cm"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*cm"))
            out["weight_kg"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*kg"))
            out["volume_l"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*l(?:itre|iter)?\\b"))
            out["volume_ml"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*ml"))
            out["storage_gb"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*gb"))
            out["storage_tb"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*tb"))
            out["power_w"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*w\\b"))
            out["capacity_ton"] = name_lower.apply(lambda x: extract_measure(x, r"(\\d+(?:\\.\\d+)?)\\s*ton"))

            out["actual_price"] = pd.to_numeric(out["actual_price"], errors="coerce")
            out["discount_price"] = pd.to_numeric(out["discount_price"], errors="coerce")
            out["actual_price"] = out["actual_price"].fillna(out["discount_price"])
            out["price_gap"] = out["actual_price"] - out["discount_price"]
            out["discount_pct_vs_list"] = np.where(
                out["actual_price"].gt(0),
                1 - (out["discount_price"] / out["actual_price"]),
                np.nan,
            )

            out["log_actual_price"] = np.log1p(out["actual_price"].clip(lower=0))
            out["log_rating_count"] = np.log1p(out["rating_count"].clip(lower=0))
            out["log_inventory_count_hint"] = np.log1p(out["inventory_count_hint"].clip(lower=0))

            return out
        """
    ),
    code_cell(
        """
        model_df = build_features(raw_df)

        model_df = model_df.loc[model_df[TARGET_COL].notna() & model_df[TARGET_COL].gt(0)].copy()
        model_df["log_target"] = np.log1p(model_df[TARGET_COL])

        print(f"Rows after target filtering: {len(model_df):,}")
        print(model_df[[TARGET_COL, "actual_price", "rating_value", "rating_count", "inventory_status"]].head())
        """
    ),
    md_cell(
        """
        ## EDA

        The next few cells inspect category coverage, target distribution, missingness, and a few business-facing relationships that usually matter for pricing:
        - department and sub-category mix
        - relationship between actual price and selling price
        - review signal as a proxy for demand / product maturity
        - stock-status anomalies embedded inside the ratings-count column
        """
    ),
    code_cell(
        """
        eda_summary = pd.DataFrame(
            {
                "rows": [len(model_df)],
                "main_categories": [model_df["main_category"].nunique()],
                "sub_categories": [model_df["sub_category"].nunique()],
                "brands": [model_df["brand"].nunique()],
                "missing_rating_value_pct": [model_df["rating_value"].isna().mean() * 100],
                "missing_rating_count_pct": [model_df["rating_count"].isna().mean() * 100],
                "missing_actual_price_pct": [model_df["actual_price"].isna().mean() * 100],
            }
        )
        eda_summary
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        (
            model_df["main_category"]
            .value_counts()
            .head(12)
            .sort_values()
            .plot(kind="barh", ax=axes[0, 0], color="#4c78a8")
        )
        axes[0, 0].set_title("Top Departments by Row Count")
        axes[0, 0].set_xlabel("Products")

        sns.histplot(np.log1p(model_df[TARGET_COL]), bins=60, ax=axes[0, 1], color="#f58518")
        axes[0, 1].set_title("Log Selling Price Distribution")
        axes[0, 1].set_xlabel("log1p(discount_price)")

        sample_for_plot = model_df.sample(min(30_000, len(model_df)), random_state=RANDOM_STATE)
        sns.scatterplot(
            data=sample_for_plot,
            x="actual_price",
            y=TARGET_COL,
            hue="main_category",
            alpha=0.35,
            s=20,
            legend=False,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Actual Price vs Selling Price")
        axes[1, 0].set_xlim(left=0)
        axes[1, 0].set_ylim(bottom=0)

        (
            model_df["inventory_status"]
            .value_counts()
            .sort_values()
            .plot(kind="barh", ax=axes[1, 1], color="#54a24b")
        )
        axes[1, 1].set_title("Inventory Signals Extracted from `no_of_ratings`")
        axes[1, 1].set_xlabel("Rows")

        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        category_price_view = (
            model_df.groupby(["main_category", "sub_category"], observed=True)[TARGET_COL]
            .agg(["count", "median", "mean"])
            .sort_values("count", ascending=False)
            .head(20)
        )
        category_price_view
        """
    ),
    md_cell(
        """
        ## Feature engineering

        We use three feature families:
        - text signal from product `name` through TF-IDF
        - categorical signal from `main_category`, `sub_category`, `brand`, and inventory state
        - structured numeric features such as ratings, review counts, actual price, extracted sizes, and simple text-derived attributes

        The advanced models benefit from richer interactions, while the baseline gives us a simpler benchmark.
        """
    ),
    code_cell(
        """
        candidate_numeric_features = [
            "rating_value",
            "rating_count",
            "inventory_count_hint",
            "name_char_len",
            "name_word_count",
            "digit_count",
            "has_numbers",
            "has_combo",
            "has_wireless",
            "has_smart",
            "first_number",
            "size_inch",
            "size_cm",
            "weight_kg",
            "volume_l",
            "volume_ml",
            "storage_gb",
            "storage_tb",
            "power_w",
            "capacity_ton",
            "actual_price",
            "log_actual_price",
            "log_rating_count",
            "log_inventory_count_hint",
        ]

        fs_sample = model_df.sample(min(120_000, len(model_df)), random_state=RANDOM_STATE).copy()
        fs_matrix = fs_sample[candidate_numeric_features].copy()
        fs_matrix = fs_matrix.replace([np.inf, -np.inf], np.nan)
        fs_matrix = fs_matrix.fillna(fs_matrix.median(numeric_only=True))

        mi_scores = mutual_info_regression(
            fs_matrix,
            fs_sample["log_target"],
            random_state=RANDOM_STATE,
        )

        mi_series = pd.Series(mi_scores, index=candidate_numeric_features).sort_values(ascending=False)
        selected_numeric_features = mi_series.head(12).index.tolist()

        print("Selected numeric features:")
        print(selected_numeric_features)
        mi_series.to_frame("mutual_information").style.background_gradient(cmap="Blues")
        """
    ),
    code_cell(
        """
        feature_columns = [
            "name",
            "main_category",
            "sub_category",
            "brand",
            "inventory_status",
            *selected_numeric_features,
        ]

        train_df = model_df[feature_columns + [TARGET_COL, "log_target"]].copy()

        if len(train_df) > MAX_TRAIN_ROWS:
            train_df = train_df.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

        X = train_df[feature_columns]
        y_log = train_df["log_target"]
        y_actual = train_df[TARGET_COL]

        X_train, X_valid, y_train_log, y_valid_log, y_train_actual, y_valid_actual = train_test_split(
            X,
            y_log,
            y_actual,
            test_size=VALID_SIZE,
            random_state=RANDOM_STATE,
            stratify=train_df["main_category"],
        )

        print(f"Training rows: {len(X_train):,}")
        print(f"Validation rows: {len(X_valid):,}")
        """
    ),
    code_cell(
        """
        text_feature = "name"
        categorical_features = ["main_category", "sub_category", "brand", "inventory_status"]
        numeric_features = selected_numeric_features

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=5_000,
                        ngram_range=(1, 2),
                        min_df=5,
                        stop_words="english",
                        sublinear_tf=True,
                    ),
                    text_feature,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                ),
            ],
            remainder="drop",
        )

        models = {
            "DecisionTreeBaseline": DecisionTreeRegressor(
                max_depth=18,
                min_samples_leaf=40,
                random_state=RANDOM_STATE,
            )
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda" if GPU_AVAILABLE else "cpu",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                device="gpu" if GPU_AVAILABLE else "cpu",
            )

        list(models.keys())
        """
    ),
    code_cell(
        """
        def evaluate_model(y_true, y_pred):
            return {
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "R2": r2_score(y_true, y_pred),
            }

        trained_models = {}
        rows = []

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train_log)
            valid_pred = np.expm1(pipeline.predict(X_valid))
            valid_pred = np.clip(valid_pred, a_min=0, a_max=None)

            metrics = evaluate_model(y_valid_actual, valid_pred)
            metrics["model"] = model_name
            rows.append(metrics)
            trained_models[model_name] = pipeline

            print(model_name, metrics)

        results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
        results_df
        """
    ),
    code_cell(
        """
        if not results_df.empty:
            baseline_rmse = results_df.loc[results_df["model"] == "DecisionTreeBaseline", "RMSE"].iloc[0]
            results_df["rmse_improvement_vs_baseline_pct"] = (
                (baseline_rmse - results_df["RMSE"]) / baseline_rmse
            ) * 100

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.barplot(data=results_df, x="model", y="MAE", ax=axes[0], palette="Blues_d")
        axes[0].set_title("MAE Comparison")
        axes[0].tick_params(axis="x", rotation=20)

        sns.barplot(data=results_df, x="model", y="RMSE", ax=axes[1], palette="Greens_d")
        axes[1].set_title("RMSE Comparison")
        axes[1].tick_params(axis="x", rotation=20)

        sns.barplot(data=results_df, x="model", y="R2", ax=axes[2], palette="Oranges_d")
        axes[2].set_title("R2 Comparison")
        axes[2].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plt.show()

        results_df
        """
    ),
    code_cell(
        """
        best_model_name = results_df.iloc[0]["model"]
        best_model = trained_models[best_model_name]

        perm_sample = X_valid.sample(min(5_000, len(X_valid)), random_state=RANDOM_STATE)
        perm_target = y_valid_actual.loc[perm_sample.index]

        permutation = permutation_importance(
            best_model,
            perm_sample,
            np.log1p(perm_target),
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )

        raw_feature_importance = (
            pd.Series(permutation.importances_mean, index=feature_columns)
            .sort_values(ascending=False)
            .to_frame("importance")
        )

        raw_feature_importance
        """
    ),
    md_cell(
        """
        ## Price recommendation helper

        The helper below scores a new product using the best available trained model.

        Inputs:
        - `name`: required
        - `main_category`: required
        - `sub_category`: optional but recommended
        - `actual_price`: optional but highly valuable
        - `rating_value`, `rating_count`, `inventory_status`: optional context features

        If optional fields are missing, the function backs off to category-level medians.
        """
    ),
    code_cell(
        """
        numeric_defaults_global = model_df[selected_numeric_features].median(numeric_only=True)
        numeric_defaults_by_main = model_df.groupby("main_category", observed=True)[selected_numeric_features].median(numeric_only=True)

        def recommend_price(
            name,
            main_category,
            sub_category="unknown",
            actual_price=None,
            rating_value=None,
            rating_count=None,
            inventory_status="available",
            inventory_count_hint=None,
        ):
            if main_category in numeric_defaults_by_main.index:
                base_numeric = numeric_defaults_by_main.loc[main_category].copy()
            else:
                base_numeric = numeric_defaults_global.copy()

            request_df = pd.DataFrame(
                [
                    {
                        "name": name,
                        "main_category": main_category,
                        "sub_category": sub_category,
                        "ratings": rating_value,
                        "no_of_ratings": rating_count,
                        "discount_price": np.nan,
                        "actual_price": actual_price,
                    }
                ]
            )

            enriched = build_features(request_df)
            enriched["inventory_status"] = inventory_status
            if inventory_count_hint is not None:
                enriched["inventory_count_hint"] = inventory_count_hint
                enriched["log_inventory_count_hint"] = np.log1p(max(inventory_count_hint, 0))

            for col in selected_numeric_features:
                if col not in enriched.columns or enriched[col].isna().all():
                    enriched[col] = base_numeric.get(col, np.nan)
                enriched[col] = enriched[col].fillna(base_numeric.get(col, np.nan))

            score_row = enriched[feature_columns]
            predicted_price = np.expm1(best_model.predict(score_row))[0]
            return round(float(predicted_price), 2)

        recommend_price(
            name="LG 1.5 Ton 5 Star AI Dual Inverter Split AC",
            main_category="appliances",
            sub_category="Air Conditioners",
            actual_price=65000,
            rating_value=4.2,
            rating_count=2500,
            inventory_status="available",
        )
        """
    ),
    md_cell(
        """
        ## Interpretation and next steps

        Typical follow-up improvements:
        - tune the boosted models with Optuna or randomized search
        - use time-aware validation if you add timestamped pricing history later
        - enrich the product-name parser with domain-specific units for each department
        - move from point prediction to price bands or uncertainty intervals
        - incorporate margin, inventory cost, seasonality, and competitor data to turn this into a true optimization engine
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
