# Workforce Retention AI Dashboard

A data-driven application to predict employee attrition and identify strategic retention opportunities.

## 1. Overview

This project uses Machine Learning (Random Forest) to analyze workforce metrics and predict the likelihood of an employee leaving. It provides two main interfaces:
1.  **Workforce Overview**: A strategic dashboard for HR leaders to view aggregate risk, financial impact of turnover, and "Burnout" zones.
2.  **Employee Prediction**: A tactical tool to assess risk for individual employees based on their specific workload, tenure, and department.

## 2. Key Features
*   **Real-time Risk scoring** based on 3-5 year tenure and workload (>250h/month).
*   **Financial Modeling**: Estimate ROI of retention programs.
*   **Burnout Analysis**: Visualizing the impact of overwork on turnover.

## 3. Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/workforce-retention-dashboard.git
    cd workforce-retention-dashboard
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 4. Usage

Run the Streamlit app:
```bash
streamlit run Workforce_Overview.py
```

## 5. Model Info
*   **Algorithm**: Random Forest Classifier
*   **Training Data**: HR Analytics Dataset (15k records)
*   **Key Insight**: Work accidents are negatively correlated with turnover (injured employees stay longer), while burnout is the #1 driver of exit.
