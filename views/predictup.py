import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random

# Global variables for model caching
_pipeline = None
_model_time = None
_model_cost = None
_model_oar = None
_scaler_cluster = None
_scaler_optimal = None
_kmeans = None
_ohe_columns = None
_preprocess_columns = None
_feature_cols_original = None
_cluster_features_columns = None
_feature_cols_model = None

def load_model():
    """Lazy load the model pipeline"""
    global _pipeline, _model_time, _model_cost, _model_oar
    global _scaler_cluster, _scaler_optimal, _kmeans
    global _ohe_columns, _preprocess_columns, _feature_cols_original
    global _cluster_features_columns, _feature_cols_model
    
    if _pipeline is None:
        try:
            _pipeline = joblib.load("models/pipeline_final.pkl")
            _model_time = _pipeline["model_time"]
            _model_cost = _pipeline["model_cost"]
            _model_oar = _pipeline["model_oar"]
            _scaler_cluster = _pipeline["scaler_cluster"]
            _scaler_optimal = _pipeline["scaler_optimal"]
            _kmeans = _pipeline["kmeans"]
            _ohe_columns = _pipeline["ohe_columns"]
            _preprocess_columns = _pipeline["preprocess_columns"]
            _feature_cols_original = _pipeline["feature_cols_original"]
            _cluster_features_columns = _pipeline["cluster_features_columns"]
            _feature_cols_model = _pipeline["feature_cols_model"]
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    return {
        "model_time": _model_time,
        "model_cost": _model_cost,
        "model_oar": _model_oar,
        "scaler_cluster": _scaler_cluster,
        "scaler_optimal": _scaler_optimal,
        "kmeans": _kmeans,
        "ohe_columns": _ohe_columns,
        "preprocess_columns": _preprocess_columns,
        "feature_cols_original": _feature_cols_original,
        "cluster_features_columns": _cluster_features_columns,
        "feature_cols_model": _feature_cols_model
    }

# FEATURE GENERATOR
def generate_features(dept, job, source, num_applicants, time_to_hire, cost_per_hire, oar):
    pipeline_data = load_model()
    preprocess_columns = pipeline_data["preprocess_columns"]
    cluster_features_columns = pipeline_data["cluster_features_columns"]
    scaler_cluster = pipeline_data["scaler_cluster"]
    kmeans = pipeline_data["kmeans"]
    feature_cols_model = pipeline_data["feature_cols_model"]

    df = pd.DataFrame([{col: 0 for col in preprocess_columns}])

    df["num_applicants"] = num_applicants
    df["time_to_hire_days"] = time_to_hire
    df["cost_per_hire"] = cost_per_hire
    df["offer_acceptance_rate"] = oar

    df["efficiency_score"] = oar / time_to_hire
    df["time_cost_interaction"] = cost_per_hire * time_to_hire

    dcol = f"department_{dept}"
    jcol = f"job_title_{job}"
    scol = f"source_{source}"

    if dcol in df.columns: df[dcol] = 1
    if jcol in df.columns: df[jcol] = 1
    if scol in df.columns: df[scol] = 1

    df_cluster_input = df[cluster_features_columns]

    X_scaled = scaler_cluster.transform(df_cluster_input)
    df["cluster"] = kmeans.predict(X_scaled)

    df_final = df.reindex(columns=feature_cols_model, fill_value=0)

    return df_final


def run():

    st.markdown("<h1 style='text-align:center;font-size:45px;'>OPTIMAL SCORE PREDICTION</h1>",
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1,2.5])
    with col_left:

        st.subheader("Upload Dataset")
        uploaded = st.file_uploader("Upload .csv dataset", type=["csv"])

        num_applicants = None
        time_to_hire   = None
        cost_per_hire  = None
        oar            = None


        if uploaded:
            user_df = pd.read_csv(uploaded)

            required = ["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]
            if not all(col in user_df.columns for col in required):
                st.error("❌ Dataset harus memiliki kolom: time_to_hire_days, cost_per_hire, offer_acceptance_rate")
                return

            num_applicants = len(user_df)
            time_to_hire   = user_df["time_to_hire_days"].mean()
            cost_per_hire  = user_df["cost_per_hire"].mean()
            oar            = user_df["offer_acceptance_rate"].mean()

            st.success("✔ Dataset berhasil dibaca")
            st.dataframe(user_df.head())
            st.write(f"Total Applicants: **{num_applicants}**")
            st.write(f"Avg Time to Hire: **{time_to_hire:.2f}**")
            st.write(f"Avg Cost per Hire: **{cost_per_hire:.2f}**")
            st.write(f"Avg OAR: **{oar:.4f}**")

        
        department_jobs = {
            "Engineering": ["Software Engineer","DevOps Engineer","Backend Developer","Data Engineer"],
            "Sales": ["Account Executive","Business Development Manager","Sales Associate","Sales Representative"],
            "Product": ["Product Manager","Product Analyst","UX Designer","UI Designer"],
            "HR": ["HR Coordinator","Recruitment Specialist","Talent Acquisition","HR Manager","Payroll Specialist"],
            "Marketing": ["Marketing Specialist","Social Media Manager","Content Strategist","SEO Analyst"],
            "Finance": ["Accountant","Finance Manager","Financial Analyst","Payroll Specialist"]
        }

        st.subheader("Department")
        dept = st.selectbox("", list(department_jobs.keys()),  key="department_select")
        st.subheader("Job Title")
        job  = st.selectbox("", department_jobs[dept], key="job_select")

        st.write("----")

        time_w = st.number_input("Time Weight", 0.0,1.0,0.3,step=0.01, key="w_time")
        cost_w = st.number_input("Cost Weight", 0.0,1.0,0.3,step=0.01, key="w_cost")
        oar_w  = st.number_input("OAR Weight",  0.0,1.0,0.3,step=0.01, key="w_oar")

        valid = abs((time_w+cost_w+oar_w)-1) < 0.001

        if valid: st.success("✔ Total weight valid")
        else:     st.error("❌ Total harus = 1")

        predict = st.button("PREDICT", disabled=not(uploaded and valid), key="button")

        if not (predict and uploaded and valid):
            return

    with col_right:
        pipeline_data = load_model()
        model_time = pipeline_data["model_time"]
        model_cost = pipeline_data["model_cost"]
        model_oar = pipeline_data["model_oar"]
        scaler_optimal = pipeline_data["scaler_optimal"]

        sources = ["Referral","LinkedIn","Job Portal","Recruiter"]
        full = []

        for src in sources:

            X = generate_features(
                dept, job, src,
                num_applicants, time_to_hire,
                cost_per_hire, oar
            )

            pred_time = model_time.predict(X)[0]
            pred_cost = model_cost.predict(X)[0]
            pred_oar  = model_oar.predict(X)[0]

            scaled = scaler_optimal.transform([[pred_time,pred_cost,pred_oar]])
            optimal = (scaled[0][0]*time_w) + (scaled[0][1]*cost_w) + (scaled[0][2]*oar_w)

            full.append([
            src,
            round(pred_time),                      
            round(pred_cost),                      
            round(pred_oar, 2),                    
            round(optimal, 2)                      
        ])



        df = pd.DataFrame(full, columns=["Source","Pred Time","Pred Cost","Pred OAR","Optimal Score"])
        df = df.sort_values(by="Optimal Score", ascending=False)

 # TABLE CSS
        table_css = """
        <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 14px;
            overflow: hidden;
            font-family: 'Arial';
            margin-top: 10px;
        }
        .custom-table thead {
            background-color: #6C8CFF;
            color: #000000 !important;
            font-weight: bold;
            text-align: left;
        }
        .custom-table th, .custom-table td {
            padding: 14px 20px;
            font-size: 16px;
            color: #000000 !important;
            text-align: center;
        }
        .custom-table tbody tr:nth-child(even) {
            background-color: #f5f7ff;
        }
        .custom-table tbody tr:nth-child(odd) {
            background-color: #ffffff;
        }
        .custom-table tbody tr:hover {
            background-color: #e9edff;
            cursor: pointer;
        }
        </style>
        """

        st.markdown(table_css, unsafe_allow_html=True)
        html_table = df.to_html(index=False, classes="custom-table")
        st.markdown(html_table, unsafe_allow_html=True)



        # BEST RECOMMENDATION 
        best = df.iloc[0]
        best_source = best["Source"]
        time_val = round(best["Pred Time"], 2)
        cost_val = round(best["Pred Cost"], 2)
        oar_val = round(best["Pred OAR"], 2)
        score_val = round(best["Optimal Score"], 4)

        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            st.markdown("### **Recommendation**")
            st.markdown(
                f"<h3 style='color:#b00000;'>{best_source}</h3>",unsafe_allow_html=True
                )
            st.markdown("### **For**")
            st.markdown(f"<h4>Department: {dept}</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>Job Title: {job}</h4>", unsafe_allow_html=True)

        with c2:
            st.markdown("## Prediction Values")
            metric_style = """
                    <div style="
                        background-color:#e8f4fb;
                        padding:20px;
                        border-radius:25px;
                        margin-bottom:20px;
                        text-align:center;
                        font-size:20px;
                        font-weight:600;
                        color:#1f4e79;
                    ">
                        {title}<br><span style="font-size:26px; font-weight:800;">{value}</span>
                    </div>
                """

            st.markdown(metric_style.format(title="Predicted Time",  value=f"{time_val:.0f}"), unsafe_allow_html=True)
            st.markdown(metric_style.format(title="Predicted Cost",  value=f"{cost_val:.0f}"), unsafe_allow_html=True)
            st.markdown(metric_style.format(title="Predicted OAR",   value=f"{oar_val:.2f}"), unsafe_allow_html=True)
            st.markdown(metric_style.format(title="Optimal Score",   value=f"{score_val:.2f}"), unsafe_allow_html=True)