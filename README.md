# AI-Powered Intelligent E-Commerce Decision Platform

A Streamlit-based e-commerce analytics dashboard that helps explore sales performance, customer segments, product performance, inventory information, and review sentiment.

## Features

- **Executive Dashboard:** Overview of e-commerce sales and business performance.
- **Inventory Management:** View and manage inventory information supported by the app.
- **Customer Intelligence:** Explore customer segments and related insights.
- **Product Recommendations:** Identify products based on the recommendation logic implemented in the application.
- **Sales Forecasting:** Explore the sales forecast or baseline provided by the application.
- **Sentiment Analysis:** Analyze review text using VADER sentiment analysis.

## Technologies Used

- Python
- Streamlit
- Pandas
- SQLite
- VADER Sentiment Analysis
- Data visualization libraries used by the application

## Dataset

The project uses transaction-level e-commerce data containing fields such as invoice number, product code, description, quantity, invoice date, unit price, customer ID, country, and revenue.

## Run Locally

1. Install Python.
2. Open a terminal in the project directory.
3. Install the required dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

## Project Purpose

This project demonstrates practical skills in data analysis, dashboard development, customer segmentation, inventory data handling, and text sentiment analysis.

## Notes

The exact behavior of forecasting, recommendations, and inventory features depends on the implementation in the source code. VADER is a pretrained, lexicon-based sentiment analyzer; it is not a custom model trained on this project's reviews.
