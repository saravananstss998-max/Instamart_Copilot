from pathlib import Path
import pandas as pd

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input file
ORDERS_FILE = PROJECT_DIR / "data" / "clean" / "orders.csv"

# Read orders data
orders = pd.read_csv(ORDERS_FILE)

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep completed orders only
completed = orders[orders["status"] == "completed"].copy()

# Reference date for RFM analysis
REFERENCE_DATE = pd.Timestamp("2025-10-01")

# Calculate RFM metrics
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
)

# Recency in days
rfm["recency"] = (
    REFERENCE_DATE - rfm["last_order"]
).dt.days

# Keep required columns
rfm = rfm[["recency", "frequency", "monetary"]]

# Round values
rfm = rfm.round(2)

# Display results
print("\n===== RFM ANALYSIS =====")
print("Customers analyzed:", len(rfm))
print("\nFirst 10 customers:")
print(rfm.head(10))

print("\n===== RFM SUMMARY =====")
print(rfm.describe().round(2))

# Save RFM results
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "customer_rfm.csv"
rfm.to_csv(OUTPUT_FILE)

print("\nRFM file saved to:", OUTPUT_FILE)

# ===== K-MEANS CLUSTERING EVALUATION =====

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Select RFM features
X = rfm[["recency", "frequency", "monetary"]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n===== K-MEANS EVALUATION =====")

results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": round(inertia, 2),
        "silhouette": round(silhouette, 3)
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# ===== CUSTOMER SEGMENTATION WITH K=4 =====

# Train the final K-Means model
final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(X_scaled)

# Summarize the segments
segment_summary = rfm.groupby("segment_id").agg(
    customer_count=("recency", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\n===== CUSTOMER SEGMENT SUMMARY =====")
print(segment_summary)

# Name the four customer segments
segment_names = {
    0: "Regular Customers",
    1: "Premium Customers",
    2: "Loyal Customers",
    3: "At-Risk Customers"
}

# Assign a name to each customer
rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Mark premium customers with 1, others with 0
rfm["is_premium"] = (
    rfm["segment_name"] == "Premium Customers"
).astype(int)

# Save customer segments
segment_output = rfm.reset_index()

SEGMENT_FILE = OUTPUT_DIR / "customer_segments.csv"

segment_output.to_csv(SEGMENT_FILE, index=False)

print("\n===== SEGMENT COUNTS =====")
print(rfm["segment_name"].value_counts())

print("\nCustomer segments saved to:", SEGMENT_FILE)

# ===== PCA ANALYSIS =====

from sklearn.decomposition import PCA

# Reduce the three RFM features to two dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n===== PCA ANALYSIS =====")

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_.round(3)
)

print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

# Save PCA coordinates with customer segments
pca_output = rfm.reset_index()[
    ["customer_id", "segment_id", "segment_name"]
].copy()

pca_output["PC1"] = X_pca[:, 0]
pca_output["PC2"] = X_pca[:, 1]

PCA_FILE = OUTPUT_DIR / "customer_pca.csv"
pca_output.to_csv(PCA_FILE, index=False)

print("PCA results saved to:", PCA_FILE)