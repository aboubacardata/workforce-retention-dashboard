import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Page Setup ---
st.set_page_config(page_title="Attrition Risk", layout="wide")

# --- 2. Load Resources ---
@st.cache_resource
def load_resources():
    # Helper to find file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Go up one level to hr_pred
    model_path = os.path.join(project_root, 'final_model.joblib')
    
    # Load Model
    model = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"Model not found at: {model_path}")
    
    # Load Data (for comparisons)
    df = None
    try:
        url = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/hr_data.csv"
        df = pd.read_csv(url)
    except:
        pass
        
    return model, df

model, df_ref = load_resources()

# Helper for percentile calculation
def get_percentile(val, col, df):
    if df is None: return 50
    return np.sum(df[col] < val) / len(df) * 100

# --- 3. Dashboard Layout ---

# Create two main columns
left_col, right_col = st.columns([1, 1.2], gap="large")

# ================= LEFT PANEL: SNAPSHOT INPUTS =================
with left_col:
    with st.container(border=True):
        st.subheader("Employee Situation Snapshot")
        
        # Section 1: Workload
        st.caption("Workload & Engagement")
        
        # Changed to Number Input for easier typing
        c1, c2 = st.columns(2)
        with c1:
            avg_hours = st.number_input("Avg Monthly Hours", min_value=90, max_value=310, value=250, step=5)
        with c2:
            num_projects = st.number_input("Number of Projects", min_value=2, max_value=7, value=4, step=1)
            
        st.divider()

        # Section 3: Tenure & Context
        st.caption("Tenure & Context")
        
        c5, c6 = st.columns(2)
        with c5:
            tenure = st.number_input("Time at Company (Years)", min_value=2, max_value=10, value=3)
        with c6:
             st.write("") # Spacer
             promoted = st.checkbox("Promoted in last 5 years?")
        
        c7, c8 = st.columns(2)
        with c7:
            dept = st.selectbox("Department", ['sales', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting'])
        with c8:
            salary = st.selectbox("Salary Band", ['low', 'medium', 'high'])
            
        accident = st.checkbox("Experienced a Work Accident?")
        
        st.divider()
        
        # Clean button text
        analyze_btn = st.button("Assess Attrition Risk", type="primary", use_container_width=True)


# ================= RIGHT PANEL: RISK REPORT =================
with right_col:
    st.subheader("Assess Attrition Risk")
    
    with st.container(border=True):
        st.markdown("#### Attrition Risk Assessment")
        
        if analyze_btn and model:
            # 1. Prediction Logic
            # IMPORTANT: Matches columns used in train_model.py
            input_df = pd.DataFrame({
                'number_project': [num_projects], 
                'average_montly_hours': [avg_hours],
                'time_spend_company': [tenure], 
                'Work_accident': [1 if accident else 0],
                'promotion_last_5years': [1 if promoted else 0], 
                'department': [dept], 
                'salary': [salary]
            })
            
            prob = model.predict_proba(input_df)[0][1]
            
            # 2. Risk Status (Native Streamlit Alerts)
            if prob > 0.7:
                st.error(f"Risk: HIGH")
            elif prob > 0.3:
                st.warning(f"Risk: MEDIUM")
            else:
                st.success(f"Risk: LOW")
                
            st.markdown(f"**Risk Score: {prob:.0%}** (6-month horizon)")
            st.divider()
            
            # 3. Comparisons
            st.markdown("#### How this compares")
            if df_ref is not None:
                dept_data = df_ref[df_ref['department'] == dept]
                pct_hours = get_percentile(avg_hours, 'average_montly_hours', dept_data)
                pct_tenure = get_percentile(tenure, 'time_spend_company', df_ref)
                
                st.markdown(f"""
                * Hours are higher than **{pct_hours:.0f}%** of employees in **{dept}**
                * Tenure is higher than **{pct_tenure:.0f}%** of employees with similar roles
                """)
            
            st.divider()
            
            # 4. Key Risk Signals
            st.markdown("#### Key Risk Signals")
            
            has_signals = False
            if avg_hours > 240:
                st.markdown("🔴 **Sustained high monthly workload**")
                has_signals = True
            if not promoted and tenure > 4:
                st.markdown("🟠 **Limited recent role progression**")
                has_signals = True
            if num_projects > 5:
                st.markdown("🟠 **Project overload (Burnout risk)**")
                has_signals = True
            if num_projects < 3:
                st.markdown("🟡 **Under-utilization risk**")
                has_signals = True
            
            if not has_signals:
                st.markdown("🟢 No critical risk factors identified.")

            st.divider()
            st.caption("This risk assessment supports prioritization and review, not automated decision-making.")
            
        elif not analyze_btn:
            st.info("Adjust parameters on the left and click 'Assess Attrition Risk' to see the report.")