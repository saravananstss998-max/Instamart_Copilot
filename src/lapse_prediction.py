
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

# 1. Load data
orders = pd.read_csv(DATA / "orders.csv")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# 2. Historical data only: before September 1
history = orders[orders["order_ts"] < CUTOFF].copy()
completed = history[history["status"] == "completed"].copy()

# Every customer with at least one completed historical order
cohort = completed["customer_id"].unique()

# 3. Build historical customer features
features = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_time=("delivery_minutes", "mean")
)

features["recency"] = (
    CUTOFF - features["last_order"]
).dt.days.clip(lower=0)

# Cancellation rate uses all historical final-status orders
status_counts = history.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda s: (s == "cancelled").sum()
    )
)

features["cancellation_rate"] = (
    status_counts["cancelled_orders"]
    / status_counts["total_orders"].replace(0, pd.NA)
).fillna(0)

# Favourite product category from completed historical orders
completed_products = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    completed_products.groupby("customer_id")["category"]
    .agg(lambda s: s.dropna().mode().iloc[0]
         if not s.dropna().mode().empty else "Unknown")
)

features["favourite_category"] = favourite_category

# Customer profile and tenure
features = features.join(
    customers.set_index("customer_id"),
    how="left"
)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days.clip(lower=0)

features = features.drop(
    columns=["last_order", "signup_date"],
    errors="ignore"
)

# 4. Create lapse label using September outcomes
september_completed = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"] == "completed")
]

active_ids = set(september_completed["customer_id"])

features["lapsed"] = (
    ~features.index.isin(active_ids)
).astype(int)

# 5. Prepare X and y
X = features.drop(columns=["lapsed"])
y = features["lapsed"]

# Remove identifiers from model features
X = X.drop(columns=["customer_id"], errors="ignore")

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nCOHORT SUMMARY")
print("Customers:", len(X))
print("Lapsed:", int(y.sum()))
print("Lapse rate:", round(y.mean() * 100, 2), "%")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# Baseline: predict every customer as not lapsed
baseline_predictions = [0] * len(y_test)
print("\nBASELINE")
print(classification_report(
    y_test,
    baseline_predictions,
    zero_division=0
))

# 7. Logistic Regression tuning
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

# 8. Random Forest tuning
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nMODEL TUNING RESULTS")
print("Logistic Regression best params:", lr_grid.best_params_)
print("Logistic Regression CV F1:", round(lr_grid.best_score_, 4))
print("Random Forest best params:", rf_grid.best_params_)
print("Random Forest CV F1:", round(rf_grid.best_score_, 4))
# Evaluate both tuned models on untouched test data

lr_test_predictions = lr_grid.best_estimator_.predict(X_test)
rf_test_predictions = rf_grid.best_estimator_.predict(X_test)

print("\nLOGISTIC REGRESSION TEST RESULTS")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_test_predictions))
print(classification_report(
    y_test,
    lr_test_predictions,
    labels=[1],
    target_names=["Lapsed"],
    zero_division=0
))

print("\nRANDOM FOREST TEST RESULTS")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_test_predictions))
print(classification_report(
    y_test,
    rf_test_predictions,
    labels=[1],
    target_names=["Lapsed"],
    zero_division=0
))
# 9. Select using cross-validation score only
if lr_grid.best_score_ >= rf_grid.best_score_:
    selected_model = lr_grid.best_estimator_
    selected_name = "Logistic Regression"
else:
    selected_model = rf_grid.best_estimator_
    selected_name = "Random Forest"

print("\nSELECTED MODEL:", selected_name)

# 10. Evaluate selected model on untouched test data
test_predictions = selected_model.predict(X_test)

print("\nSELECTED MODEL TEST RESULTS")
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# 11. Oversample training data only for Logistic Regression
train_data = X_train.copy()
train_data["_target"] = y_train

majority_count = train_data["_target"].value_counts().max()

balanced_parts = []
for class_value in sorted(train_data["_target"].unique()):
    class_rows = train_data[
        train_data["_target"] == class_value
    ]
    balanced_parts.append(
        class_rows.sample(
            n=majority_count,
            replace=len(class_rows) < majority_count,
            random_state=42
        )
    )

balanced_train = pd.concat(balanced_parts).sample(
    frac=1,
    random_state=42
)

X_balanced = balanced_train.drop(columns=["_target"])
y_balanced = balanced_train["_target"]

balanced_lr = clone(lr_grid.best_estimator_)
balanced_lr.fit(X_balanced, y_balanced)

balanced_predictions = balanced_lr.predict(X_test)

print("\nOVERSAMPLED LOGISTIC REGRESSION TEST RESULTS")
print(confusion_matrix(y_test, balanced_predictions))
print(classification_report(
    y_test,
    balanced_predictions,
    zero_division=0
))

# 12. Feature analysis
fitted_preprocessor = selected_model.named_steps["preprocessor"]
feature_names = fitted_preprocessor.get_feature_names_out()

if selected_name == "Random Forest":
    importances = selected_model.named_steps["model"].feature_importances_
else:
    # Use the tuned Random Forest for feature importance analysis
    rf_model = rf_grid.best_estimator_
    importances = rf_model.named_steps["model"].feature_importances_
    feature_names = rf_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(
    importance_table.head(5).to_string(index=False)
)

# Logistic Regression coefficient signs
lr_model = lr_grid.best_estimator_
lr_names = lr_model.named_steps[
    "preprocessor"
].get_feature_names_out()
lr_coefficients = lr_model.named_steps["model"].coef_[0]

coefficient_table = pd.DataFrame({
    "feature": lr_names,
    "coefficient": lr_coefficients
}).sort_values("coefficient", ascending=False)

print("\nTOP POSITIVE LOGISTIC REGRESSION FEATURES")
print(coefficient_table.head(3).to_string(index=False))

print("\nTOP NEGATIVE LOGISTIC REGRESSION FEATURES")
print(coefficient_table.tail(3).to_string(index=False))

importance_table.to_csv(
    OUTPUT / "feature_importance.csv",
    index=False
)
coefficient_table.to_csv(
    OUTPUT / "logistic_coefficients.csv",
    index=False
)

# 13. Save selected model and verify reload
joblib.dump(selected_model, OUTPUT / "lapse_model.joblib")

loaded_model = joblib.load(OUTPUT / "lapse_model.joblib")
reloaded_predictions = loaded_model.predict(X_test)

print("\nMODEL RELOAD CHECK:",
      (test_predictions == reloaded_predictions).all())

# 14. Premium customer retention list
segments = pd.read_csv(OUTPUT / "customer_segments.csv")

premium_ids = set(
    segments.loc[
        segments["is_premium"] == True,
        "customer_id"
    ]
)

test_customers = X_test.index
probabilities = selected_model.predict_proba(X_test)[:, 1]

retention = pd.DataFrame({
    "customer_id": test_customers,
    "lapse_probability": probabilities,
    "actually_lapsed": y_test.loc[test_customers].astype(int).values
})

retention = retention[
    retention["customer_id"].isin(premium_ids)
].sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention.to_csv(
    OUTPUT / "retention_list.csv",
    index=False
)

print("\nPREMIUM RETENTION LIST")
print(retention.to_string(index=False))
print("Premium test customers listed:", len(retention))

print("\nLAPSE PREDICTION COMPLETED")
