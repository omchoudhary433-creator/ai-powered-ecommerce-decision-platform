"""
AI-Powered Intelligent E-Commerce Decision Platform
Single-file Streamlit application integrating the analytical workflow from
ecommerce_analysis(1).ipynb. The notebook does not need to be opened to run
this app: analyses are recalculated from the selected CSV at app startup.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import sqlite3
import warnings

import pandas as pd
import streamlit as st

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ecommerce.db"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="AI-Powered Intelligent E-Commerce Decision Platform",
    page_icon="🛍️",
    layout="wide",
)

# ----------------------------- Helpers ---------------------------------
def get_column(df, candidates):
    lookup = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


def connect_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK(price >= 0),
                stock INTEGER NOT NULL CHECK(stock >= 0)
            )
        """)


def read_inventory():
    with connect_db() as conn:
        return pd.read_sql_query("SELECT * FROM inventory ORDER BY id DESC", conn)


def save_csv(df, filename):
    if df is not None and not df.empty:
        df.to_csv(OUTPUT_DIR / filename, index=False)


def safe_qcut(values, labels):
    # rank(method='first') mirrors the notebook and avoids duplicate-edge failures.
    return pd.qcut(values.rank(method="first"), q=4, labels=labels).astype(int)


def money(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "—"


init_db()

# ----------------------------- Data input -------------------------------
st.sidebar.title("🛍️ E-Commerce Platform")
st.sidebar.caption("Notebook analytics integrated in this Streamlit app")
st.sidebar.subheader("1) Choose your data source")
source_mode = st.sidebar.radio(
    "Data source",
    ["Upload CSV", "Upload SQLite Database", "Use project CSV automatically"],
    index=2,
    help="CSV: upload sales rows. SQLite: upload a .db/.sqlite file and choose its table."
)

candidate_paths = [
    BASE_DIR / "sales_data.csv",
    BASE_DIR / "data" / "ecommerce_cleaned.csv",
    BASE_DIR / "online_retail.csv",
    BASE_DIR / "OnlineRetail.csv",
    BASE_DIR / "data.csv",
]
existing = next((p for p in candidate_paths if p.exists()), None)

@st.cache_data(show_spinner="Loading CSV...")
def read_csv_path(path_string):
    return pd.read_csv(path_string, low_memory=False)

@st.cache_data(show_spinner="Reading uploaded CSV...")
def read_uploaded_csv(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

@st.cache_data(show_spinner="Inspecting SQLite database...")
def sqlite_table_names(db_bytes):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        db_file = Path(folder) / "uploaded_database.sqlite"
        db_file.write_bytes(db_bytes)
        with sqlite3.connect(str(db_file)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
    return [row[0] for row in rows]

@st.cache_data(show_spinner="Loading selected database table...")
def read_sqlite_table(db_bytes, table_name):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        db_file = Path(folder) / "uploaded_database.sqlite"
        db_file.write_bytes(db_bytes)
        with sqlite3.connect(str(db_file)) as conn:
            safe_table = '"' + table_name.replace('"', '""') + '"'
            return pd.read_sql_query(f"SELECT * FROM {safe_table}", conn)

raw_df = pd.DataFrame()
load_error = None
data_source_label = "No data selected"

try:
    if source_mode == "Upload CSV":
        uploaded_csv = st.sidebar.file_uploader("2) Select sales CSV", type=["csv"], key="sales_csv_upload")
        if uploaded_csv is not None:
            raw_df = read_uploaded_csv(uploaded_csv.getvalue())
            data_source_label = f"CSV: {uploaded_csv.name}"
        else:
            st.sidebar.info("CSV file select karo. Example: ecommerce_cleaned.csv")

    elif source_mode == "Upload SQLite Database":
        uploaded_db = st.sidebar.file_uploader(
            "2) Select SQLite database (.db / .sqlite)", type=["db", "sqlite", "sqlite3"], key="sales_db_upload"
        )
        if uploaded_db is not None:
            db_bytes = uploaded_db.getvalue()
            tables = sqlite_table_names(db_bytes)
            if tables:
                selected_table = st.sidebar.selectbox("3) Select data table", tables)
                raw_df = read_sqlite_table(db_bytes, selected_table)
                data_source_label = f"SQLite: {uploaded_db.name} → {selected_table}"
            else:
                st.sidebar.error("Is SQLite database mein koi normal data table nahi mila.")
        else:
            st.sidebar.info("Apni .db ya .sqlite file select karo.")

    else:
        if existing is not None:
            raw_df = read_csv_path(str(existing))
            data_source_label = f"Project CSV: {existing.name}"
        else:
            st.sidebar.warning("Project folder mein supported CSV nahi mili. Upload CSV option choose karo.")
except Exception as exc:
    load_error = str(exc)
    st.sidebar.error(f"Data load error: {load_error}")

if not raw_df.empty:
    st.sidebar.success(f"Loaded: {len(raw_df):,} rows × {len(raw_df.columns)} columns")

if not raw_df.empty:
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

# Prepare normalized working sales copy, without silently discarding raw rows.
sales_df = raw_df.copy()
qty_col = get_column(sales_df, ["Quantity"])
price_col = get_column(sales_df, ["UnitPrice", "Unit Price", "unit_price", "Price"])
date_col = get_column(sales_df, ["InvoiceDate", "Invoice Date", "Order Date", "Date"])
invoice_col = get_column(sales_df, ["InvoiceNo", "Invoice No", "Invoice", "OrderID", "Order ID"])
customer_col = get_column(sales_df, ["CustomerID", "Customer ID", "Customer"])
stock_col = get_column(sales_df, ["StockCode", "Stock Code", "ProductID", "Product ID", "SKU"])
description_col = get_column(sales_df, ["Description", "Product", "Product Name"])
country_col = get_column(sales_df, ["Country"])

if not sales_df.empty:
    if qty_col:
        sales_df[qty_col] = pd.to_numeric(sales_df[qty_col], errors="coerce")
    if price_col:
        sales_df[price_col] = pd.to_numeric(sales_df[price_col], errors="coerce")
    if date_col:
        sales_df[date_col] = pd.to_datetime(sales_df[date_col], errors="coerce", dayfirst=True)
    if qty_col and price_col:
        sales_df["Revenue"] = sales_df[qty_col] * sales_df[price_col]
    else:
        rev_col = get_column(sales_df, ["Revenue", "Sales", "TotalSales", "Amount", "CalculatedRevenue"])
        if rev_col:
            sales_df["Revenue"] = pd.to_numeric(sales_df[rev_col], errors="coerce")
        else:
            sales_df["Revenue"] = pd.NA

# --------------------- Run all notebook analytics -----------------------
monthly_sales = pd.DataFrame()
rfm = pd.DataFrame()
segment_summary = pd.DataFrame()
cluster_summary = pd.DataFrame()
product_summary = pd.DataFrame()
country_summary = pd.DataFrame()
recommendations_df = pd.DataFrame()
forecast_results = pd.DataFrame()
future_df = pd.DataFrame()
analysis_df = pd.DataFrame()

if not sales_df.empty and qty_col and price_col:
    analysis_df = sales_df.dropna(subset=[qty_col, price_col]).copy()
    analysis_df = analysis_df.dropna(subset=[date_col]).copy() if date_col else analysis_df.copy()
    if date_col:
        analysis_df = analysis_df[analysis_df[date_col].notna()].copy()
        analysis_df["Month"] = analysis_df[date_col].dt.to_period("M").dt.to_timestamp()
        tx_agg = { "TotalRevenue": ("Revenue", "sum"), "TotalQuantity": (qty_col, "sum") }
        if invoice_col:
            tx_agg["TotalTransactions"] = (invoice_col, "nunique")
        else:
            tx_agg["TotalTransactions"] = (qty_col, "size")
        monthly_sales = analysis_df.groupby("Month", as_index=False).agg(**tx_agg).sort_values("Month")

    # RFM: positive quantity and price, valid customer/date/order fields.
    if customer_col and date_col and invoice_col:
        customer_df = analysis_df[
            analysis_df[customer_col].notna()
            & (analysis_df[qty_col] > 0)
            & (analysis_df[price_col] > 0)
            & analysis_df[invoice_col].notna()
        ].copy()
        if not customer_df.empty:
            reference_date = customer_df[date_col].max() + pd.Timedelta(days=1)
            rfm = customer_df.groupby(customer_col).agg(
                Recency=(date_col, lambda x: (reference_date - x.max()).days),
                Frequency=(invoice_col, "nunique"),
                Monetary=("Revenue", "sum"),
            ).reset_index().rename(columns={customer_col: "CustomerID"})
            if len(rfm) >= 4:
                rfm["R_Score"] = safe_qcut(rfm["Recency"], [4, 3, 2, 1])
                rfm["F_Score"] = safe_qcut(rfm["Frequency"], [1, 2, 3, 4])
                rfm["M_Score"] = safe_qcut(rfm["Monetary"], [1, 2, 3, 4])
                rfm["RFM_Score"] = (
                    rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
                )
                def assign_segment(row):
                    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
                    if r >= 3 and f >= 3 and m >= 3:
                        return "Champions"
                    if r >= 2 and f >= 3:
                        return "Loyal Customers"
                    if r >= 3 and f >= 2:
                        return "Potential Loyalists"
                    if r <= 2 and f >= 2:
                        return "At Risk"
                    return "Needs Attention"
                rfm["CustomerSegment"] = rfm.apply(assign_segment, axis=1)
                segment_summary = rfm.groupby("CustomerSegment", as_index=False).agg(
                    Customers=("CustomerID", "count"),
                    AvgRecency=("Recency", "mean"),
                    AvgFrequency=("Frequency", "mean"),
                    TotalMonetary=("Monetary", "sum"),
                ).round(2).sort_values("Customers", ascending=False)

                # K-Means clustering: clip extreme frequency/monetary values, scale and fit.
                if SKLEARN_AVAILABLE and len(rfm) >= 4:
                    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
                    for c in ["Frequency", "Monetary"]:
                        features[c] = features[c].clip(upper=features[c].quantile(.99))
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(features)
                    kmeans = KMeans(n_clusters=min(4, len(rfm)), random_state=42, n_init=10)
                    rfm["Cluster"] = kmeans.fit_predict(scaled)
                    cluster_summary = rfm.groupby("Cluster", as_index=False).agg(
                        Customers=("CustomerID", "count"),
                        AvgRecency=("Recency", "mean"),
                        AvgFrequency=("Frequency", "mean"),
                        AvgMonetary=("Monetary", "mean"),
                    ).round(2)

    # Product-level sales aggregation.
    if stock_col or description_col:
        group_cols = [c for c in [stock_col, description_col] if c]
        product_summary = analysis_df.groupby(group_cols, dropna=False).agg(
            TotalQuantity=(qty_col, "sum"),
            TotalRevenue=("Revenue", "sum"),
            Transactions=(invoice_col, "nunique") if invoice_col else (qty_col, "size"),
        ).reset_index().sort_values("TotalRevenue", ascending=False)

    # Country sales summary.
    if country_col:
        country_summary = analysis_df.groupby(country_col, dropna=False).agg(
            TotalRevenue=("Revenue", "sum"),
            TotalQuantity=(qty_col, "sum"),
            Transactions=(invoice_col, "nunique") if invoice_col else (qty_col, "size"),
        ).reset_index().sort_values("TotalRevenue", ascending=False)

    # Basket co-purchase recommendations (same logic as notebook).
    if invoice_col and stock_col:
        basket_data = analysis_df[
            analysis_df[invoice_col].notna() & analysis_df[stock_col].notna()
        ].copy()
        basket_data[stock_col] = basket_data[stock_col].astype(str)
        baskets = basket_data.groupby(invoice_col)[stock_col].apply(lambda x: sorted(set(x)))
        pair_counts = Counter()
        for items in baskets:
            if 1 < len(items) <= 100:
                pair_counts.update(combinations(items, 2))
        rows = []
        for (a, b), count in pair_counts.items():
            rows.extend([
                {"Product": a, "RecommendedProduct": b, "CoPurchaseCount": count},
                {"Product": b, "RecommendedProduct": a, "CoPurchaseCount": count},
            ])
        recommendations_df = pd.DataFrame(rows)
        if not recommendations_df.empty:
            recommendations_df = recommendations_df.sort_values(
                ["Product", "CoPurchaseCount"], ascending=[True, False]
            )

    # Random Forest time-based forecast: chronological holdout and next 3 months.
    if SKLEARN_AVAILABLE and len(monthly_sales) >= 6:
        fdf = monthly_sales.sort_values("Month").reset_index(drop=True).copy()
        fdf["MonthNumber"] = range(len(fdf))
        fdf["MonthOfYear"] = fdf["Month"].dt.month
        split = max(1, int(len(fdf) * .8))
        X = fdf[["MonthNumber", "MonthOfYear"]]
        y = fdf["TotalRevenue"]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X.iloc[:split], y.iloc[:split])
        pred = model.predict(X.iloc[split:])
        forecast_results = pd.DataFrame({
            "Month": fdf["Month"].iloc[split:].values,
            "ActualRevenue": y.iloc[split:].values,
            "PredictedRevenue": pred,
        })
        mae = mean_absolute_error(y.iloc[split:], pred)
        next_months = pd.date_range(
            fdf["Month"].max() + pd.offsets.MonthBegin(1), periods=3, freq="MS"
        )
        future_X = pd.DataFrame({
            "MonthNumber": range(len(fdf), len(fdf) + 3),
            "MonthOfYear": next_months.month,
        })
        future_df = pd.DataFrame({
            "Month": next_months,
            "PredictedRevenue": model.predict(future_X),
        })
    else:
        mae = None
else:
    mae = None

# Save notebook-style outputs automatically. No Jupyter run/export step required.
for frame, name in [
    (rfm, "customer_segments.csv"),
    (segment_summary, "customer_segment_summary.csv"),
    (rfm, "customer_clusters.csv"),
    (cluster_summary, "kmeans_cluster_summary.csv"),
    (product_summary, "top_products.csv"),
    (country_summary, "country_sales_summary.csv"),
    (recommendations_df, "product_recommendations.csv"),
    (monthly_sales, "monthly_sales.csv"),
]:
    save_csv(frame, name)

# ----------------------------- Navigation -------------------------------
pages = [
    "Executive Dashboard", "Data Overview & Cleaning", "Monthly Sales Analysis",
    "Customer Intelligence (RFM)", "K-Means Customer Clusters",
    "Product Analysis & Recommendations", "Country Analysis", "Sales Forecasting",
    "Inventory Management", "Sentiment Analysis", "Export & Project Status",
]
page = st.sidebar.radio("Navigate modules", pages)

st.title("🛍️ AI-Powered Intelligent E-Commerce Decision Platform")
st.caption("Sales analytics • Customer segmentation • Product co-purchase • Forecasting • Inventory")
st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: {data_source_label}")
if raw_df.empty:
    st.warning("Data load nahi hua. Sidebar mein Upload CSV ya Upload SQLite Database choose karke file select karein. Project CSV mode ke liye data/ecommerce_cleaned.csv ya sales_data.csv app.py ke paas rakhein.")
    if load_error:
        st.error(f"CSV read error: {load_error}")
    st.stop()

# ----------------------------- Page modules ------------------------------
if page == "Executive Dashboard":
    st.subheader("Executive Overview")
    revenue = pd.to_numeric(sales_df["Revenue"], errors="coerce").sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", money(revenue))
    c2.metric("Transaction Rows", f"{len(sales_df):,}")
    c3.metric("Unique Customers", f"{sales_df[customer_col].nunique():,}" if customer_col else "N/A")
    c4.metric("Unique Products", f"{sales_df[stock_col].nunique():,}" if stock_col else "N/A")
    if not monthly_sales.empty:
        st.subheader("Monthly Revenue")
        st.line_chart(monthly_sales.set_index("Month")["TotalRevenue"])
    left, right = st.columns(2)
    with left:
        st.subheader("Top Products by Revenue")
        if not product_summary.empty:
            st.dataframe(product_summary.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Product identifier/description columns unavailable.")
    with right:
        st.subheader("Customer Segment Summary")
        if not segment_summary.empty:
            st.dataframe(segment_summary, use_container_width=True, hide_index=True)
        else:
            st.info("RFM requires CustomerID, InvoiceDate and InvoiceNo and at least four customers.")
    st.subheader("Data Preview")
    st.dataframe(sales_df.head(10), use_container_width=True)

elif page == "Data Overview & Cleaning":
    st.subheader("Dataset Profiling")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(raw_df):,}")
    b.metric("Columns", f"{len(raw_df.columns):,}")
    c.metric("Duplicate Rows", f"{raw_df.duplicated().sum():,}")
    st.write("Column data types and missing values")
    profile = pd.DataFrame({
        "Column": raw_df.columns,
        "Data Type": [str(t) for t in raw_df.dtypes],
        "Missing Values": [int(raw_df[col].isna().sum()) for col in raw_df.columns],
        "Missing %": [round(float(raw_df[col].isna().mean() * 100), 2) for col in raw_df.columns],
    })
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.write("Customer ID availability")
    if customer_col:
        missing_cid = int(raw_df[customer_col].isna().sum())
        st.metric("Rows with missing Customer ID", f"{missing_cid:,}")
        if qty_col and price_col:
            miss_revenue = (sales_df.loc[sales_df[customer_col].isna(), "Revenue"]).sum()
            st.metric("Revenue in rows without Customer ID", money(miss_revenue))
    st.info("The app preserves the original uploaded rows. Numeric/date conversion and invalid-row filtering are applied to analysis copies only.")
    st.subheader("Prepared Data Preview")
    st.dataframe(sales_df.head(30), use_container_width=True)

elif page == "Monthly Sales Analysis":
    st.subheader("Monthly Sales Analysis")
    if monthly_sales.empty:
        st.warning("Valid date, quantity and unit-price columns are required.")
    else:
        st.metric("Months in dataset", len(monthly_sales))
        st.line_chart(monthly_sales.set_index("Month")[["TotalRevenue", "TotalQuantity"]])
        st.dataframe(monthly_sales, use_container_width=True, hide_index=True)
        st.download_button("Download monthly sales CSV", monthly_sales.to_csv(index=False), "monthly_sales.csv", "text/csv")

elif page == "Customer Intelligence (RFM)":
    st.subheader("RFM Customer Segmentation")
    if rfm.empty or "CustomerSegment" not in rfm.columns:
        st.warning("RFM could not be calculated. Required: Customer ID, invoice/order ID, invoice date, quantity and unit price; at least four valid customers.")
    else:
        x1, x2, x3 = st.columns(3)
        x1.metric("Customers", f"{len(rfm):,}")
        x2.metric("Segments", f"{rfm['CustomerSegment'].nunique()}")
        x3.metric("Total Customer Monetary Value", money(rfm["Monetary"].sum()))
        st.bar_chart(rfm["CustomerSegment"].value_counts())
        st.subheader("Average Spending by Segment")
        st.bar_chart(rfm.groupby("CustomerSegment")["Monetary"].mean().sort_values(ascending=False))
        st.dataframe(segment_summary, use_container_width=True, hide_index=True)
        st.subheader("Customer-level RFM Details")
        st.dataframe(rfm, use_container_width=True, hide_index=True)
        st.download_button("Download RFM customers", rfm.to_csv(index=False), "customer_segments.csv", "text/csv")

elif page == "K-Means Customer Clusters":
    st.subheader("K-Means Clustering")
    if cluster_summary.empty:
        st.warning("K-Means requires scikit-learn and at least four valid RFM customers.")
    else:
        st.bar_chart(cluster_summary.set_index("Cluster")["Customers"])
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        st.dataframe(rfm, use_container_width=True, hide_index=True)
        st.download_button("Download clustered customers", rfm.to_csv(index=False), "customer_clusters.csv", "text/csv")

elif page == "Product Analysis & Recommendations":
    st.subheader("Top Product Analysis")
    if product_summary.empty:
        st.warning("Product code or description and valid sales columns are required.")
    else:
        st.bar_chart(product_summary.head(15).set_index(description_col or stock_col)["TotalRevenue"])
        st.dataframe(product_summary.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download top products", product_summary.to_csv(index=False), "top_products.csv", "text/csv")
    st.divider()
    st.subheader("Frequently Bought Together")
    if recommendations_df.empty:
        st.info("Co-purchase suggestions require invoice/order ID and product code. Only invoices with multiple distinct products contribute.")
    else:
        selected_product = st.selectbox("Select product code", sorted(recommendations_df["Product"].unique()))
        st.dataframe(
            recommendations_df[recommendations_df["Product"] == selected_product].head(10),
            use_container_width=True, hide_index=True
        )
        st.download_button("Download co-purchase recommendations", recommendations_df.to_csv(index=False), "product_recommendations.csv", "text/csv")

elif page == "Country Analysis":
    st.subheader("Country-wise Sales Performance")
    if country_summary.empty:
        st.warning("Country column is not available in the uploaded dataset.")
    else:
        st.bar_chart(country_summary.head(10).set_index(country_col)["TotalRevenue"])
        st.dataframe(country_summary, use_container_width=True, hide_index=True)
        st.download_button("Download country summary", country_summary.to_csv(index=False), "country_sales_summary.csv", "text/csv")

elif page == "Sales Forecasting":
    st.subheader("Random Forest Revenue Forecast")
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Run: pip install scikit-learn")
    elif monthly_sales.empty or len(monthly_sales) < 6:
        st.warning("Random Forest forecasting requires at least 6 valid monthly records.")
    else:
        st.metric("Chronological holdout MAE", money(mae))
        st.caption("Notebook-style time-based holdout: earlier months train the model, later months are used for evaluation. Future estimates are experimental, not guaranteed.")
        if not forecast_results.empty:
            st.line_chart(forecast_results.set_index("Month")[["ActualRevenue", "PredictedRevenue"]])
            st.dataframe(forecast_results, use_container_width=True, hide_index=True)
        st.subheader("Next 3 Months Forecast")
        st.line_chart(future_df.set_index("Month")["PredictedRevenue"])
        st.dataframe(future_df, use_container_width=True, hide_index=True)
        st.download_button("Download future forecast", future_df.to_csv(index=False), "revenue_forecast_next_3_months.csv", "text/csv")

elif page == "Inventory Management":
    st.subheader("Inventory Management (SQLite)")
    with st.form("add_inventory_form", clear_on_submit=True):
        st.markdown("**Add New Product**")
        p1, p2 = st.columns(2)
        with p1:
            name = st.text_input("Product Name")
            category = st.text_input("Category")
        with p2:
            price = st.number_input("Price", min_value=0.0, step=1.0)
            stock = st.number_input("Stock Quantity", min_value=0, step=1)
        submitted = st.form_submit_button("➕ Add Product")
        if submitted:
            if not name.strip() or not category.strip():
                st.error("Product Name aur Category blank nahi ho sakte.")
            else:
                with connect_db() as conn:
                    conn.execute(
                        "INSERT INTO inventory(product_name, category, price, stock) VALUES(?,?,?,?)",
                        (name.strip(), category.strip(), float(price), int(stock)),
                    )
                st.success("Product successfully added.")
                st.rerun()
    inventory = read_inventory()
    st.subheader("Current Inventory")
    if inventory.empty:
        st.info("Inventory is empty. Add a product above.")
    else:
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        selected_id = st.selectbox("Select product ID to edit/delete", inventory["id"].tolist())
        product = inventory.loc[inventory["id"] == selected_id].iloc[0]
        with st.form("edit_inventory_form"):
            e1, e2 = st.columns(2)
            with e1:
                edit_name = st.text_input("Product Name", value=str(product["product_name"]))
                edit_category = st.text_input("Category", value=str(product["category"]))
            with e2:
                edit_price = st.number_input("Price", min_value=0.0, value=float(product["price"]), step=1.0)
                edit_stock = st.number_input("Stock Quantity", min_value=0, value=int(product["stock"]), step=1)
            update = st.form_submit_button("💾 Save Changes")
            if update:
                if not edit_name.strip() or not edit_category.strip():
                    st.error("Product Name aur Category blank nahi ho sakte.")
                else:
                    with connect_db() as conn:
                        conn.execute(
                            "UPDATE inventory SET product_name=?, category=?, price=?, stock=? WHERE id=?",
                            (edit_name.strip(), edit_category.strip(), float(edit_price), int(edit_stock), int(selected_id)),
                        )
                    st.success("Product updated.")
                    st.rerun()
        if st.button("🗑️ Delete Selected Product"):
            with connect_db() as conn:
                conn.execute("DELETE FROM inventory WHERE id=?", (int(selected_id),))
            st.success("Product deleted.")
            st.rerun()

elif page == "Sentiment Analysis":
    st.subheader("Customer Review Sentiment")
    review = st.text_area("Enter a customer review", height=130)
    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter review text.")
        elif VADER_AVAILABLE:
            score = SentimentIntensityAnalyzer().polarity_scores(review)["compound"]
            label = "Positive" if score >= .05 else ("Negative" if score <= -.05 else "Neutral")
            st.metric("Sentiment", label)
            st.write(f"VADER compound score: {score:.3f}")
            st.caption("VADER is a lexicon/rule-based sentiment tool; inspect results for sarcasm, context and non-English text.")
        else:
            positive = ["good", "great", "excellent", "amazing", "love", "fast", "happy", "best", "awesome"]
            negative = ["bad", "poor", "worst", "hate", "slow", "broken", "terrible", "disappointed"]
            text = review.lower()
            score = sum(w in text for w in positive) - sum(w in text for w in negative)
            label = "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral / Unclear")
            st.metric("Sentiment", label)
            st.caption("Fallback keyword demonstration; it is not a trained NLP classifier.")

elif page == "Export & Project Status":
    st.subheader("Notebook Outputs — Generated Automatically")
    output_files = [
        "customer_segments.csv", "customer_segment_summary.csv", "customer_clusters.csv",
        "kmeans_cluster_summary.csv", "top_products.csv", "country_sales_summary.csv",
        "product_recommendations.csv", "monthly_sales.csv",
    ]
    status = pd.DataFrame({
        "Output file": output_files,
        "Status": ["Generated" if (OUTPUT_DIR / f).exists() else "Not generated (required columns/data unavailable)" for f in output_files],
    })
    st.dataframe(status, use_container_width=True, hide_index=True)
    st.success(f"Outputs are stored in: {OUTPUT_DIR}")
    st.info("Notebook ke main analysis steps (monthly sales, Random Forest forecast, RFM segmentation, K-Means, product/country analysis aur basket co-purchase) app ke andar dobara calculate hote hain. Notebook ko alag se run karna zaroori nahi. Notebook ke file-management/debugging cells app mein copy nahi kiye gaye.")
    st.write(f"**Currently loaded source:** {data_source_label}")
    st.write("**Inventory database:** local SQLite file `ecommerce.db` (sales dataset se separate). Uploaded SQLite database ko analysis input ke roop mein read-only load kiya jata hai; inventory database overwrite nahi hota.")
    st.subheader("Dataset and module readiness")
    readiness = pd.DataFrame([
        ["Monthly analysis", "Ready" if not monthly_sales.empty else "Needs valid date + quantity + price"],
        ["RFM segmentation", "Ready" if not segment_summary.empty else "Needs customer, invoice, date fields and 4+ customers"],
        ["K-Means", "Ready" if not cluster_summary.empty else "Needs scikit-learn and 4+ RFM customers"],
        ["Product analysis", "Ready" if not product_summary.empty else "Needs product code/description"],
        ["Co-purchase recommendations", "Ready" if not recommendations_df.empty else "Needs invoice ID + product code"],
        ["Random Forest forecast", "Ready" if not future_df.empty else "Needs scikit-learn and 6+ months"],
    ], columns=["Module", "Status"])
    st.dataframe(readiness, use_container_width=True, hide_index=True)
    st.download_button("Download all generated outputs (CSV bundle not zipped)", "\n".join(output_files), "generated_output_file_list.txt", "text/plain")

st.sidebar.markdown("---")
st.sidebar.caption("Local prototype • Outputs recalculate from the currently selected dataset")
