import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("recruitment_efficiency_improved.csv")

def run():
    st.title("Recruitment Efficiency Dashboard")

    # LAYOUT: 2 KOLOM 
    left, right = st.columns([1, 3])

    # LEFT SIDE – KPI CARDS
    with left:
        st.markdown("## Key Metrics")

        st.container().metric(
            label="Total Recruitments",
            value=len(df)
        )

        st.container().metric(
            label="Avg Time to Hire (days)",
            value=f"{df['time_to_hire_days'].mean():.1f}"
        )

        st.container().metric(
            label="Avg Cost per Hire",
            value=f"${df['cost_per_hire'].mean():.1f}"
        )

        st.container().metric(
            label="Avg Offer Acceptance Rate",
            value=f"{df['offer_acceptance_rate'].mean():.2f}"
        )

        st.container().metric(
            label="Total Departments",
            value=df["department"].nunique()
        )

    # RIGHT SIDE – ALL CHARTS
    with right:
        st.subheader("Average Time to Hire per Department")
        time_dept = df.groupby("department")["time_to_hire_days"].mean().reset_index()
        fig1 = px.bar(time_dept, x="department", y="time_to_hire_days",
                      labels={"time_to_hire_days": "Avg Days"},
                      color="department")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(" Average Cost per Hire per Department")
        cost_dept = df.groupby("department")["cost_per_hire"].mean().reset_index()
        fig2 = px.bar(cost_dept, x="department", y="cost_per_hire",
                      labels={"cost_per_hire": "Avg Cost"},
                      color="department")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recruitment Source Distribution")
        source_count = df["source"].value_counts().reset_index()
        source_count.columns = ["source", "count"]
        fig3 = px.pie(source_count, names="source", values="count")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Time to Hire vs Cost per Hire")
        fig4 = px.scatter(df, x="time_to_hire_days", y="cost_per_hire",
                          color="department",
                          hover_data=["job_title"])
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Offer Acceptance Rate per Job Title")
        offer_job = df.groupby("job_title")["offer_acceptance_rate"].mean().reset_index()
        fig5 = px.bar(offer_job, x="job_title", y="offer_acceptance_rate",
                      labels={"offer_acceptance_rate": "Avg Offer Acceptance"},
                      color="job_title")
        st.plotly_chart(fig5, use_container_width=True)