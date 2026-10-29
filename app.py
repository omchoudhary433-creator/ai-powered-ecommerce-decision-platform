import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------- Paths and page setup --------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = next((p for p in [BASE_DIR / "sales_data.csv", BASE_DIR / "data" / "ecommerce_cleaned.csv", BASE_DIR / "online_retail.csv", BASE_DIR / "OnlineRetail.csv", BASE_DIR / "data.csv"] if p.exists()), BASE_DIR / "sales_data.csv")
DB_PATH = BASE_DIR / "ecommerce.db"

st.set_page_config(
    page_title="AI-Powered E-Commerce Decision Platform",
    page_icon="🛍️",
    layout="wide",
)

# -------------------- Database --------------------
def connect_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )
            """
        )


init_db()


# -------------------- Load and prepare sales data --------------------
@st.cache_data(show_spinner="Loading sales dataset...")
def load_sales_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data.columns = [str(c).strip() for c in data.columns]

    # Convert common numeric/date fields if they exist.
    for col in ("Quantity", "quantity", "UnitPrice", "unit_price", "Price", "price"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ("InvoiceDate", "Invoice Date", "invoice_date", "Order Date", "order_date"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce", dayfirst=True)

    # Derive revenue for the common UCI Online Retail dataset.
    quantity_col = next(
        (column for column in data.columns if column.lower() == "quantity"),
        None,
    )
    price_col = next(
        (
            column
            for column in data.columns
            if column.lower() in ("unitprice", "unit_price", "price")
        ),
        None,
    )
    if quantity_col and price_col:
        data["CalculatedRevenue"] = data[quantity_col].fillna(0) * data[price_col].fillna(0)
    return data


sales_data = load_sales_data()


def get_column(df, candidates):
    lookup = {
        str(column).strip().lower().replace(" ", "").replace("_", ""): column
        for column in df.columns
    }
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]
    return None


def read_inventory():
    with connect_db() as conn:
        return pd.read_sql_query("SELECT * FROM inventory ORDER BY id DESC", conn)


def show_sales_chart():
    if sales_data.empty:
        st.warning(f"Sales CSV not found or empty. Expected file: {DATA_PATH}")
        return
    revenue_col = get_column(
        sales_data,
        ["CalculatedRevenue", "Revenue", "Sales", "TotalSales", "Amount"],
    )
    date_col = get_column(
        sales_data,
        ["InvoiceDate", "Invoice Date", "Order Date", "Date"],
    )
    if revenue_col and date_col:
        monthly = sales_data.dropna(subset=[date_col]).copy()
        monthly["Month"] = monthly[date_col].dt.to_period("M").astype(str)
        monthly_sales = monthly.groupby("Month")[revenue_col].sum().sort_index()
        if not monthly_sales.empty:
            st.subheader("Monthly Revenue Trend")
            st.line_chart(monthly_sales)
    elif revenue_col:
        st.subheader("Revenue Trend (row order)")
        st.line_chart(sales_data[revenue_col].dropna().head(500))
    else:
        st.info("Revenue calculate nahi ho paayi. CSV mein Quantity aur UnitPrice columns check karein.")


st.sidebar.title("🛍️ E-Commerce Platform")
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Inventory Management",
        "Customer Intelligence",
        "Product Recommendations",
        "Sales Forecasting",
        "Sentiment Analysis",
    ],
)
st.sidebar.caption("Local development prototype")

st.title("AI-Powered Intelligent E-Commerce Platform")

# -------------------- Executive Dashboard --------------------
if page == "Executive Dashboard":
    st.subheader("Executive Overview")
    if sales_data.empty:
        st.error(f"Sales dataset load nahi hua. CSV yahan honi chahiye: {DATA_PATH}")
    else:
        revenue_col = get_column(
        sales_data,
        ["CalculatedRevenue", "Revenue", "Sales", "TotalSales", "Amount"],
    )
        quantity_col = get_column(sales_data, ["Quantity"])
        invoice_col = get_column(
            sales_data, ["InvoiceNo", "Invoice No", "OrderID", "Order ID"]
        )
        customer_col = get_column(
            sales_data, ["CustomerID", "Customer ID", "Customer"]
        )
        product_col = get_column(
            sales_data, ["StockCode", "ProductID", "Product ID", "Description"]
        )
        date_col = get_column(
        sales_data,
        ["InvoiceDate", "Invoice Date", "Order Date", "Date"],
    )

        if revenue_col:
            total_revenue = pd.to_numeric(sales_data[revenue_col], errors="coerce").sum()
        else:
            total_revenue = 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Historical Revenue", f"{total_revenue:,.2f}" if revenue_col else "N/A")
        k2.metric("Transaction Rows", f"{len(sales_data):,}")
        k3.metric("Unique Customers", f"{sales_data[customer_col].nunique():,}" if customer_col else "N/A")
        k4.metric("Products / Items", f"{sales_data[product_col].nunique():,}" if product_col else "N/A")

        show_sales_chart()

        left, right = st.columns(2)
        if product_col and revenue_col:
            top_products = sales_data.groupby(product_col)[revenue_col].sum().nlargest(10).sort_values()
            with left:
                st.subheader("Top 10 Products by Revenue")
                st.bar_chart(top_products)
        if date_col and quantity_col:
            daily = sales_data.dropna(subset=[date_col]).copy()
            daily["Month"] = daily[date_col].dt.to_period("M").astype(str)
            with right:
                st.subheader("Monthly Quantity Sold")
                st.bar_chart(daily.groupby("Month")[quantity_col].sum())

        with st.expander("Preview first 10 rows of source dataset"):
            st.dataframe(sales_data.head(10), use_container_width=True)
        st.caption(f"Source: {DATA_PATH.name} · {len(sales_data):,} rows loaded")

    inventory = read_inventory()
    st.divider()
    st.subheader("Inventory Snapshot")
    a, b, c = st.columns(3)
    a.metric("Inventory Products", len(inventory))
    b.metric("Total Stock Units", int(inventory["stock"].sum()) if not inventory.empty else 0)
    c.metric(
        "Inventory Value",
        f"{(inventory['price'] * inventory['stock']).sum():,.2f}" if not inventory.empty else "0.00",
    )
    if not inventory.empty:
        st.bar_chart(inventory.groupby("category")["stock"].sum())
    else:
        st.info("Inventory abhi empty hai. Inventory Management page se product add karein.")

# -------------------- Inventory Management --------------------
elif page == "Inventory Management":
    st.subheader("Inventory Management")
    st.write("Yahan manually products add, update aur delete kar sakte ho. Ye data local `ecommerce.db` mein save hota hai.")

    with st.form("add_product_form", clear_on_submit=True):
        st.markdown("### Add New Product")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Product Name")
            category = st.text_input("Category")
        with col2:
            price = st.number_input("Price", min_value=0.0, value=0.0, step=1.0)
            stock = st.number_input("Stock Quantity", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("➕ Add Product", use_container_width=True)
        if submitted:
            if not name.strip() or not category.strip():
                st.error("Product Name aur Category dono fill karo.")
            else:
                with connect_db() as conn:
                    conn.execute(
                        "INSERT INTO inventory (product_name, category, price, stock) VALUES (?, ?, ?, ?)",
                        (name.strip(), category.strip(), float(price), int(stock)),
                    )
                st.success(f"{name.strip()} successfully add ho gaya.")
                st.rerun()

    inventory = read_inventory()
    st.divider()
    st.markdown("### Current Inventory")
    if inventory.empty:
        st.info("Abhi inventory mein koi product nahi hai. Upar form fill karke Add Product dabao.")
    else:
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        st.markdown("### Edit or Delete Product")
        selected_id = st.selectbox(
            "Select product ID",
            inventory["id"].tolist(),
            format_func=lambda value: f"ID {value} — {inventory.loc[inventory['id'] == value, 'product_name'].iloc[0]}",
        )
        product = inventory.loc[inventory["id"] == selected_id].iloc[0]
        with st.form("edit_product_form"):
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
                    st.success("Product update ho gaya.")
                    st.rerun()
        if st.button("🗑️ Delete Selected Product"):
            with connect_db() as conn:
                conn.execute("DELETE FROM inventory WHERE id=?", (int(selected_id),))
            st.success("Product delete ho gaya.")
            st.rerun()

# -------------------- Customer Intelligence --------------------
elif page == "Customer Intelligence":
    st.subheader("Customer Intelligence")
    if sales_data.empty:
        st.warning("Sales dataset load nahi hua.")
    else:
        customer_col = get_column(
            sales_data, ["CustomerID", "Customer ID", "Customer"]
        )
        revenue_col = get_column(
        sales_data,
        ["CalculatedRevenue", "Revenue", "Sales", "TotalSales", "Amount"],
    )
        invoice_col = get_column(
            sales_data, ["InvoiceNo", "Invoice No", "OrderID", "Order ID"]
        )
        if not customer_col:
            st.info("Customer ID column nahi mili. Dataset columns neeche check karo.")
        else:
            customers = sales_data.dropna(subset=[customer_col]).groupby(customer_col).agg(
                Transactions=(invoice_col, "nunique") if invoice_col else (customer_col, "size"),
                Revenue=(revenue_col, "sum") if revenue_col else (customer_col, "size"),
            ).reset_index()
            customers = customers.sort_values("Revenue", ascending=False)
            m1, m2 = st.columns(2)
            m1.metric("Customers", f"{len(customers):,}")
            m2.metric("Average Revenue / Customer", f"{customers['Revenue'].mean():,.2f}")
            st.subheader("Top Customers")
            st.dataframe(customers.head(20), use_container_width=True, hide_index=True)
            st.caption("Ye basic customer summary hai; advanced ML segmentation abhi implement nahi hui hai.")

# -------------------- Product Recommendations --------------------
elif page == "Product Recommendations":
    st.subheader("Product Recommendations")
    if sales_data.empty:
        st.warning("Sales dataset load nahi hua.")
    else:
        invoice_col = get_column(
            sales_data, ["InvoiceNo", "Invoice No", "OrderID", "Order ID"]
        )
        product_col = get_column(sales_data, ["Description", "StockCode", "ProductID", "Product ID"])
        if not invoice_col or not product_col:
            st.info("Recommendations ke liye invoice/order ID aur product/description columns chahiye.")
        else:
            top = sales_data[product_col].dropna().astype(str).value_counts().head(15)
            st.subheader("Most Frequently Purchased Products")
            st.bar_chart(top.sort_values())
            st.dataframe(
                top.rename_axis("Product").reset_index(name="Transaction Rows"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Ye popularity-based starter recommendation hai, personalized ML recommender nahi.")

# -------------------- Sales Forecasting --------------------
elif page == "Sales Forecasting":
    st.subheader("Sales Forecasting")
    if sales_data.empty:
        st.warning("Sales dataset load nahi hua.")
    else:
        revenue_col = get_column(
        sales_data,
        ["CalculatedRevenue", "Revenue", "Sales", "TotalSales", "Amount"],
    )
        date_col = get_column(
        sales_data,
        ["InvoiceDate", "Invoice Date", "Order Date", "Date"],
    )
        if revenue_col and date_col:
            temp = sales_data.dropna(subset=[date_col]).copy()
            temp["Month"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
            monthly = temp.groupby("Month")[revenue_col].sum().sort_index()
            st.subheader("Historical Monthly Revenue")
            st.line_chart(monthly)
            if len(monthly) >= 3:
                forecast = monthly.rolling(3, min_periods=1).mean().iloc[-3:]
                next_month = monthly.index.max() + pd.offsets.MonthBegin(1)
                forecast_value = float(forecast.iloc[-1])
                st.metric("Simple next-month baseline estimate", f"{forecast_value:,.2f}")
                st.caption(
                f"Estimated next month: {next_month.strftime('%B %Y')}. "
                "This is a simple moving-average baseline, not an advanced ML forecast."
            )
        else:
            st.info("Forecast ke liye date aur revenue (ya Quantity + UnitPrice) columns chahiye.")

# -------------------- Sentiment Analysis --------------------
elif page == "Sentiment Analysis":
    st.subheader("AI-Powered Sentiment Analysis")
    st.write(
        "Product review enter karo. VADER NLP sentiment analyzer "
        "review ko Positive, Negative ya Neutral classify karega."
    )

    review = st.text_area(
        "Enter a product review",
        placeholder="Example: The product quality is excellent and delivery was fast."
    )

    if st.button("Analyze Review"):
        if not review.strip():
            st.warning("Pehle review type karo.")
        else:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(review)
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = "Positive"
                st.success(f"Sentiment: {sentiment}")
            elif compound <= -0.05:
                sentiment = "Negative"
                st.error(f"Sentiment: {sentiment}")
            else:
                sentiment = "Neutral"
                st.info(f"Sentiment: {sentiment}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive score", f'{scores["pos"]:.2f}')
            col2.metric("Neutral score", f'{scores["neu"]:.2f}')
            col3.metric("Negative score", f'{scores["neg"]:.2f}')

            st.caption(
                f"Compound score: {compound:.3f} "
                "| VADER is a pretrained, lexicon-based NLP sentiment analyzer."
            )
