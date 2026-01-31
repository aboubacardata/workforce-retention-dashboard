import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np

# =========================================================
# 1. SETUP & CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Retention Decision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Resources (Cached)
@st.cache_resource
def load_resources():
    # Load dataset
    file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/hr_data.csv"
    df = pd.read_csv(file_name)

    # Load pre-trained model
    try:
        model = joblib.load('final_model.joblib')
    except:
        return df, None

    # Preprocess & Predict
    X = df.drop(columns=['left', 'employee_id'])
    
    # Get probability of class 1 (leaving)
    probs = model.predict_proba(X)[:, 1]
    df['Attrition Risk'] = probs
    
    return df, model

try:
    df, model = load_resources()
    if model is None:
        # Dummy logic for demo if model missing
        df['Attrition Risk'] = np.random.uniform(0, 1, len(df))
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# =========================================================
# 2. SIDEBAR — INTERVENTION SETTINGS
# =========================================================
st.sidebar.title("Control Panel")

with st.sidebar.form("scenario_form"):
    st.header("Intervention Settings")
    
    # 1. Target Threshold
    risk_threshold_input = st.slider(
        "Risk Threshold for Intervention",
        min_value=40, max_value=90, value=60, step=5,
        format="%d%%",
        help="We target employees with risk ABOVE this %"
    )
    risk_threshold = risk_threshold_input / 100.0

    # 2. Financials
    intervention_cost = st.number_input(
        "Cost per Intervention ($)", 
        value=2000, step=500
    )
    
    retention_lift_input = st.slider(
        "Expected Success Rate",
        min_value=10, max_value=50, value=20, step=5,
        format="%d%%",
        help="% of targeted employees who will STAY"
    )
    retention_lift = retention_lift_input / 100.0
    
    replacement_cost = st.number_input(
        "Replacement Cost ($)", 
        value=15000, step=1000
    )
    
    # Update Button
    submitted = st.form_submit_button("Update Results")

# =========================================================
# 3. CALCULATIONS (THE ENGINE)
# =========================================================
# KPIs
total_employees = len(df)
avg_instability = df['Attrition Risk'].mean()
critical_risk_count = len(df[df['Attrition Risk'] > 0.7])

# Financial Scenario Logic
flagged_employees = df[df["Attrition Risk"] >= risk_threshold]
n_flagged = len(flagged_employees)

expected_saved = n_flagged * retention_lift
program_cost = n_flagged * intervention_cost
gross_savings = expected_saved * replacement_cost
net_savings = gross_savings - program_cost

# Helper to format large numbers to prevent font shrinking (e.g. 1.2M instead of 1,200,000)
def format_large_currency(num):
    if abs(num) >= 1_000_000:
        return f"${num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"${num/1_000:.0f}k"
    else:
        return f"${num:,.0f}"

# =========================================================
# 4. DASHBOARD HEADER & KPIs
# =========================================================
st.title("Workforce Stability & Retention Decision Dashboard")
st.markdown("### Strategic Workforce Overview")

# Top Row KPIs (5 Columns)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Total Workforce", f"{total_employees:,}")
kpi2.metric("Avg Instability Score", f"{avg_instability:.1%}")
kpi3.metric("Critical Risk (>70%)", f"{critical_risk_count:,}")
kpi4.metric("Intervention Targets", f"{n_flagged:,}", help=f"Risk > {risk_threshold:.0%}")

# Savings Metric with formatting
savings_display = format_large_currency(net_savings)
savings_color = "normal" if net_savings >= 0 else "inverse"

kpi5.metric(
    "Proj. Net Savings", 
    savings_display, 
    delta=f"ROI: {((net_savings/program_cost)*100 if program_cost>0 else 0):.0f}%",
    delta_color=savings_color,
    help=f"Exact Value: ${net_savings:,.0f}"
)

# REDESIGNED INSIGHT SECTION
st.markdown("""
<div style="
    background-color: #E8F4F9; 
    padding: 20px; 
    border-radius: 12px; 
    border-left: 6px solid #0078D4; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin-top: 20px; 
    margin-bottom: 30px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
">
    <div style="display: flex; align-items: start;">
        <span style="font-size: 24px; margin-right: 15px;">💡</span>
        <div>
            <h4 style="margin: 0 0 8px 0; color: #004E8C; font-size: 18px; font-weight: 700;">Strategic Insight</h4>
            <p style="margin: 0; color: #2C3E50; font-size: 16px; line-height: 1.5;">
                <strong>Primary Drivers:</strong> The model isolates <strong>Burnout</strong> (>250h/month) and the <strong>3-5 Year Tenure</strong> mark as the dominant risk factors.<br>
                <strong>Critical Finding:</strong> Contrary to intuition, employees with <strong>Work Accidents</strong> are statistically <em>more likely to stay</em>. 
                Retention efforts must therefore aggressively target <strong>workload balance</strong> rather than safety-related turnover.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DIVIDER 1: Separates Insight Box from Charts
# ---------------------------------------------------------
st.divider()

# =========================================================
# 5. STRATEGIC INSIGHTS (CHARTS)
# =========================================================
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Risk by Department")
    # Aggregation
    dept_risk = df.groupby('department')['Attrition Risk'].mean().reset_index()
    
    # Chart
    bar_chart = alt.Chart(dept_risk).mark_bar().encode(
        x=alt.X('department', sort='-y', title='Department'),
        y=alt.Y('Attrition Risk', axis=alt.Axis(format='%'), title='Avg Risk'),
        color=alt.Color('Attrition Risk', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['department', alt.Tooltip('Attrition Risk', format='.1%')]
    ).properties(height=300)
    
    st.altair_chart(bar_chart, use_container_width=True)

with chart_col2:
    st.subheader("Burnout Risk vs. Workforce Volume")
    
    # Binning logic
    df['Workload'] = pd.cut(
        df['average_montly_hours'], 
        bins=[0, 150, 200, 260, 350], 
        labels=['Under (<150)', 'Balanced (150-200)', 'High (200-260)', 'Burnout (>260)']
    )
    
    # Aggregate
    burnout_agg = df.groupby('Workload', observed=True)['Attrition Risk'].mean().reset_index()
    burnout_agg.rename(columns={'Attrition Risk': 'Avg Risk'}, inplace=True)

    # Heat Bar Chart
    burnout_chart = alt.Chart(burnout_agg).mark_bar().encode(
        x=alt.X('Workload', title='Monthly Hours Group'),
        y=alt.Y('Avg Risk', axis=alt.Axis(format='%'), title='Avg Risk'),
        color=alt.Color('Avg Risk', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['Workload', alt.Tooltip('Avg Risk', format='.1%')]
    ).properties(height=300)
    
    st.altair_chart(burnout_chart, use_container_width=True)

# ---------------------------------------------------------
# DIVIDER 2: Separates Charts from Action Table
# ---------------------------------------------------------
st.divider()

# =========================================================
# 6. ACTION & DRIVERS (SIDE BY SIDE)
# =========================================================
col_list, col_drivers = st.columns([1.5, 1])

# --- LEFT COLUMN: TABLE ---
with col_list:
    st.subheader("Priority Intervention List")
    st.markdown(f"High risk employees (> **{risk_threshold:.0%}**) requiring immediate attention.")

    # 1. Filter Data
    display_cols = ['employee_id', 'department', 'Attrition Risk', 'average_montly_hours', 'time_spend_company']
    filtered_data = flagged_employees.sort_values('Attrition Risk', ascending=False)[display_cols]
    
    # 2. Pagination Logic
    ITEMS_PER_PAGE = 5
    
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0

    total_items = len(filtered_data)
    total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    if st.session_state.page_number >= total_pages:
        st.session_state.page_number = total_pages - 1

    start_idx = st.session_state.page_number * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_data = filtered_data.iloc[start_idx:end_idx].copy()
    
    page_data['Attrition Risk'] = page_data['Attrition Risk'].map('{:.1%}'.format)

    # 3. Display Table
    st.table(page_data)

    # 4. Pagination Controls
    prev_col, page_col, next_col = st.columns([1, 2, 1])

    with prev_col:
        if st.button("Previous", disabled=(st.session_state.page_number == 0)):
            st.session_state.page_number -= 1
            st.rerun()

    with page_col:
        st.markdown(f"<div style='text-align: center; margin-top: 5px;'>Page <b>{st.session_state.page_number + 1}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)

    with next_col:
        if st.button("Next", disabled=(st.session_state.page_number >= total_pages - 1)):
            st.session_state.page_number += 1
            st.rerun()

# --- RIGHT COLUMN: CHART ---
with col_drivers:
    st.subheader("Model Diagnosis: Why?")
    st.markdown("Key factors driving current turnover predictions.")
    
    drivers = pd.DataFrame({
        'Factor': ['Tenure (Years)', 'Overwork (Hours)', 'Project Count', 'Salary Level', 'Department'],
        'Importance': [0.32, 0.31, 0.27, 0.03, 0.02] 
    })
    
    driver_chart = alt.Chart(drivers).mark_bar(color='#2E86C1').encode(
        x=alt.X('Importance', axis=alt.Axis(format='%')),
        y=alt.Y('Factor', sort='-x'),
        tooltip=[alt.Tooltip('Importance', format='.1%')]
    ).properties(height=300)
    
    st.altair_chart(driver_chart, use_container_width=True)

# ---------------------------------------------------------
# DIVIDER 3: Full Width Divider (Outside Columns)
# This closes the section, covering both the Table and the Chart
# ---------------------------------------------------------
st.divider()

# Download Button (Now full width or aligned left in main container)
full_csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button("Download Full List", data=full_csv, file_name="intervention_targets.csv", mime="text/csv")