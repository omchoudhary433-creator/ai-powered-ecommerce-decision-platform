# AI-Powered Intelligent E-Commerce Decision Platform

A local Streamlit-based e-commerce analytics and inventory-management prototype developed as a BCA Summer Training project. It combines sales insights, customer analysis, product and country summaries, experimental forecasting, review-sentiment demonstration, and local inventory operations.

> **Project status:** Academic prototype. It is not a production e-commerce system. Results depend on the selected dataset, its quality, and the fields available.

## Project Information

| Field | Details |
|---|---|
| Project title | AI-Powered Intelligent E-Commerce Decision Platform |
| Student | Hariom |
| Enrollment number | 02913402024 |
| Programme | Bachelor of Computer Applications (BCA) |
| Course | BCA 311 – Summer Training |
| Institute | Ideal Institute of Management and Technology, Delhi |
| Project Guide | Prof. Dr. Parminder Kaur |
| Academic session | 2026–27 |

## Problem Statement

E-commerce transaction data can contain hundreds of thousands of records. Reviewing sales trends, customer behaviour, and product performance manually takes time. This project provides an interface to explore supported sales data, analyse customer and product patterns, and maintain a separate local inventory list.

## Implemented Application Modules

The Streamlit application (`app.py`) includes:

1. **Executive Dashboard** – overview of available business indicators.
2. **Data Overview & Cleaning** – data preview, recognized fields, and data preparation.
3. **Monthly Sales Analysis** – monthly revenue and sales trends when valid dates are available.
4. **Customer Intelligence (RFM)** – recency, frequency, and monetary customer analysis.
5. **K-Means Customer Clusters** – customer grouping using suitable features.
6. **Product Analysis & Recommendations** – product performance and co-purchase patterns; not a personalized recommender.
7. **Country Analysis** – country-level summaries when the source contains a country field.
8. **Sales Forecasting** – an experimental baseline forecast that requires further validation.
9. **Inventory Management** – local add, edit, and delete operations using SQLite.
10. **Sentiment Analysis** – review-text sentiment demonstration using VADER when available, with a keyword-based fallback.
11. **Export & Project Status** – project information and downloadable analytical summaries when generated.

Some modules may show limited results or data-readiness messages if required columns, valid dates, sufficient history, or optional dependencies are unavailable.

## Jupyter Analysis Notebook

The repository also includes `ecommerce_analysis.ipynb`, a development and analysis notebook that is separate from the Streamlit application.

### Notebook workflow

1. Run the library-import cell.
2. Run the **Select your CSV dataset** cell. A Windows file-selection dialog opens.
3. Select the transaction-level `sales_data.csv` file from your computer.
4. Run the following cells in order to inspect data quality and perform:
   - overall business KPI analysis
   - monthly sales analysis
   - experimental Random Forest forecasting
   - RFM customer analysis
   - K-Means customer clustering
   - product performance analysis
   - country-level analysis
   - product co-purchase analysis
5. Run the export cell to save generated analysis summaries.

The notebook expects a **CSV file**, not the ZIP archive. If the dataset is supplied as `sales_data.zip`, extract it first and select the resulting `sales_data.csv` file. The notebook uses the selected file path (`csv_path`); it does not depend on a hard-coded `C:\Users\...` path.

The notebook is for data exploration and analysis. The Streamlit app does **not** execute or import the notebook, so `app.py` can run independently.

## Dataset

The project analysis was prepared for an e-commerce transaction CSV with fields such as:

| Column | Purpose |
|---|---|
| `InvoiceNo` | Invoice / transaction identifier |
| `StockCode` | Product identifier |
| `Description` | Product description |
| `Quantity` | Units purchased |
| `InvoiceDate` | Transaction date and time |
| `UnitPrice` | Unit price |
| `CustomerID` | Customer identifier |
| `Country` | Customer/order country |
| `Revenue` | Revenue value, when included in the source or derived during preparation |

The notebook validates the required transaction columns and prepares a working dataset. Customer-based analysis excludes records without a usable `CustomerID` where that identifier is required.

The previously analysed dataset produced an observed load of **530,104 rows × 9 columns**, with **4,338 known customers** and **3,922 products**. These are reference results for that dataset, not guaranteed values for every CSV selected by a user.

## Technology Stack

| Technology | Role |
|---|---|
| Python | Main programming language |
| Streamlit | Local interactive application |
| pandas | CSV loading, cleaning, grouping, and summaries |
| NumPy | Numerical operations |
| scikit-learn | K-Means clustering and experimental forecasting |
| Matplotlib | Analysis visualizations |
| SQLite / `sqlite3` | Local inventory storage |
| VADER Sentiment | Optional lexicon-based sentiment demonstration |
| Jupyter Notebook | Exploratory data analysis and analysis outputs |

## Data Sources and Storage

- The notebook user selects a local sales CSV through a file dialog.
- The Streamlit app uses the sales-data source selected in its interface (supported CSV or SQLite-table input, depending on the app configuration).
- The local `ecommerce.db` file stores inventory records.
- Sales analytics and inventory are separate in this version.
- Sales transactions are **not automatically synchronized** with inventory stock.

## Run the Streamlit Application

### Requirements

- Python installed
- The project files
- A supported sales CSV or SQLite input for analytics
- Dependencies listed in `requirements.txt`

### Install and launch

Open a terminal in the project folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

The notebook is optional for launching the app. If you want to run the notebook, open it in Jupyter, select the extracted `sales_data.csv` when prompted, and run its cells in order.

## Suggested Repository Structure

```text
ai-powered-ecommerce-decision-platform/
├── app.py
├── requirements.txt
├── README.md
├── ecommerce.db
├── sales_data.zip                 # optional, if permitted to share
├── notebooks/
│   └── ecommerce_analysis.ipynb   # optional analysis notebook
├── outputs/                       # optional generated CSV summaries
└── reports/                       # optional report and presentation
```

Only include files and folders that actually exist. Generated CSV summaries in `outputs/` are optional; the notebook can regenerate them when run. Do not commit private customer data, credentials, API keys, or datasets unless you have permission to share them. If redistribution is not permitted, document the dataset source and how to obtain it instead.

## Current Limitations

- Results depend on the selected file's schema, data quality, and available fields.
- The notebook file picker uses a desktop GUI; it is intended for a local Windows/Jupyter environment, not a headless or hosted notebook environment.
- Sales analytics and inventory are separate data stores; there is no automatic stock reconciliation.
- Product suggestions are based on product frequency/co-purchase patterns, not personalized recommendations.
- Sentiment analysis is a VADER/keyword demonstration, not a newly trained and fully evaluated NLP classifier.
- Forecasting is experimental and needs backtesting and further validation before business use.
- The application runs locally; authentication, role-based access, cloud deployment, monitoring, and production security controls are outside this version.
- A connected Power BI report, deep-learning model, LLM assistant, and RAG system are not claimed as completed features.

## Future Scope

Possible future enhancements include automated data refresh, forecast backtesting, evaluated personalized recommendations, a trained sentiment model, a document-grounded RAG assistant, Power BI integration, authentication, backups, monitoring, and secure deployment. These are proposed extensions, not completed features.

## References

- Python Documentation – https://docs.python.org/
- Streamlit Documentation – https://docs.streamlit.io/
- pandas Documentation – https://pandas.pydata.org/docs/
- NumPy Documentation – https://numpy.org/doc/
- scikit-learn User Guide – https://scikit-learn.org/stable/user_guide.html
- SQLite Documentation – https://www.sqlite.org/docs.html
- VADER Sentiment – https://github.com/cjhutto/vaderSentiment

---

**Developed by Hariom**  
BCA 311 – Summer Training  
Ideal Institute of Management and Technology, Delhi  
Academic Session 2026–27
