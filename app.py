# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Disable PyArrow for dataframes to avoid compatibility issues
import os
os.environ["STREAMLIT_SERVER_ENABLE_ARROW_TABLES"] = "false"

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Tutor Retention Dashboard",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for better table styling
st.markdown("""
<style>
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    table th {
        background-color: #262730;
        color: white;
        padding: 12px 8px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #444;
    }
    table td {
        padding: 10px 8px;
        border-bottom: 1px solid #333;
    }
    table tr:hover {
        background-color: #1e1e1e;
    }
    .alert-red {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-amber {
        background-color: #ffaa00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Data loading & cleaning
# -----------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Parse dates - handle DD/MM/YYYY format
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors='coerce')
    
    # Filter to only "Attended" lessons if Status column exists
    if "Status" in df.columns:
        df = df[df["Status"] == "Attended"].copy()

    # Build identifiers
    df["student"] = df["Student First Name"].str.strip() + " " + df["Student Last Name"].str.strip()
    df["tutor"] = df["Tutor First Name"].str.strip() + " " + df["Tutor Last Name"].str.strip()

    # Parse hours
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")

    # Parse cost: e.g. "S$130.00" -> 130.00
    df["Cost_numeric"] = (
        df["Cost"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    # Month-level key for retention / churn
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    return df


# -----------------------------
# Helper functions for KPIs
# -----------------------------
def monthly_retention_churn(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Compute month-over-month retention and churn for a given ID
    (id_col = 'student' or 'tutor').
    """
    active = (
        df.groupby(["month"])[id_col]
        .nunique()
        .rename("active_count")
        .reset_index()
        .sort_values("month")
    )

    # Set of ids per month
    ids_by_month = (
        df.groupby("month")[id_col]
        .apply(lambda s: set(s.unique()))
        .reset_index()
        .rename(columns={id_col: "ids"})
        .sort_values("month")
    )

    active["ids"] = ids_by_month["ids"]

    # Compute retention vs previous month
    retained_list = [np.nan]
    retention_rate_list = [np.nan]
    churn_rate_list = [np.nan]

    for i in range(1, len(active)):
        prev_ids = active.loc[i - 1, "ids"]
        curr_ids = active.loc[i, "ids"]
        retained = len(curr_ids.intersection(prev_ids))
        prev_active = active.loc[i - 1, "active_count"]

        retained_list.append(retained)
        if prev_active == 0:
            retention_rate_list.append(np.nan)
            churn_rate_list.append(np.nan)
        else:
            rr = retained / prev_active
            retention_rate_list.append(rr)
            churn_rate_list.append(1 - rr)

    active["retained_from_prev"] = retained_list
    active["retention_rate"] = retention_rate_list
    active["churn_rate"] = churn_rate_list
    active = active.drop(columns=["ids"])

    return active


def tutor_student_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (tutor, student) relationship statistics:
    - first & last lesson
    - number of lessons and months
    - tenure in days
    """
    g = df.groupby(["tutor", "student"])

    pair = g.agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        n_lessons=("Date", "count"),
        n_months=("month", lambda x: x.nunique()),
        total_hours=("Hours", "sum"),
        total_revenue=("Cost_numeric", "sum"),
    ).reset_index()

    pair["tenure_days"] = (pair["last_date"] - pair["first_date"]).dt.days + 1

    # Month of last lesson (useful as "churn month" for this pair)
    pair["last_month"] = pair["last_date"].dt.to_period("M").dt.to_timestamp()

    return pair


def calculate_student_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate student alerts:
    - Red Alert: No lesson in past 30 days
    - Amber Alert: Lesson frequency dropped (gap increased significantly)
    """
    alerts = []
    current_date = df['Date'].max()
    
    for student in df['student'].unique():
        student_df = df[df['student'] == student].sort_values('Date')
        
        if len(student_df) < 2:
            continue  # Need at least 2 lessons to calculate frequency
        
        last_lesson = student_df['Date'].max()
        days_since_last = (current_date - last_lesson).days
        
        # Red Alert: No lesson in past 30 days (but had lessons before)
        red_alert = days_since_last > 30
        
        # Calculate average gap between lessons (excluding current gap)
        dates = student_df['Date'].values
        gaps = []
        for i in range(len(dates) - 1):
            gap = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
            gaps.append(gap)
        
        avg_gap = np.mean(gaps) if gaps else 0
        
        # Amber Alert: Current gap is 2x the average historical gap (and > 14 days)
        amber_alert = False
        if not red_alert and avg_gap > 0 and days_since_last > 14:
            if days_since_last > (avg_gap * 2):
                amber_alert = True
        
        if red_alert or amber_alert:
            # Get student's tutor(s)
            tutors = student_df['tutor'].unique()
            tutor_str = ", ".join(tutors)
            
            alerts.append({
                'Student': student,
                'Tutor(s)': tutor_str,
                'Last Lesson': last_lesson,
                'Days Since Last': days_since_last,
                'Avg Gap (days)': avg_gap,
                'Total Lessons': len(student_df),
                'Alert Type': '🔴 Red' if red_alert else '🟠 Amber',
                'Action': 'Follow up immediately' if red_alert else 'Check on engagement'
            })
    
    return pd.DataFrame(alerts)


def tutor_retention_metrics(pair: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates relationship stats to the tutor level, focusing on retention quality.
    """
    g = pair.groupby("tutor")

    tutor_stats = g.agg(
        n_students=("student", "nunique"),
        n_pairs=("student", "count"),
        avg_lessons_per_student=("n_lessons", "mean"),
        median_lessons_per_student=("n_lessons", "median"),
        avg_months_per_student=("n_months", "mean"),
        median_months_per_student=("n_months", "median"),
        avg_tenure_days=("tenure_days", "mean"),
        total_revenue=("total_revenue", "sum"),
        total_hours=("total_hours", "sum"),
    ).reset_index()

    # "Dropped quickly" = pairs with only one lesson
    quick_drop = pair.assign(is_quick_drop=pair["n_lessons"] == 1)
    quick_drop_counts = quick_drop.groupby("tutor")["is_quick_drop"].sum().rename(
        "n_quick_drops"
    )
    tutor_stats = tutor_stats.merge(quick_drop_counts, on="tutor", how="left")
    tutor_stats["n_quick_drops"] = tutor_stats["n_quick_drops"].fillna(0)

    tutor_stats["share_quick_drop"] = tutor_stats["n_quick_drops"] / tutor_stats[
        "n_pairs"
    ].replace(0, np.nan)

    # Simple retention score: higher is better retention
    # (you can tweak weights)
    tutor_stats["retention_score"] = (
        (1 - tutor_stats["share_quick_drop"]).fillna(0) * 0.6
        + (tutor_stats["avg_months_per_student"] / tutor_stats["avg_months_per_student"].max()).fillna(0)
        * 0.4
    )

    return tutor_stats


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Tutor & Student Retention Dashboard")

uploaded = st.file_uploader("Upload your lesson CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV exported from your lesson system to begin.")
    st.markdown("""
    ### Expected CSV format:
    - **Date** (DD/MM/YYYY format, e.g., 02/03/2025)
    - **Student First Name**, **Student Last Name**
    - **Tutor First Name**, **Tutor Last Name**
    - **Service** (lesson type)
    - **Status** (Attended, Cancelled, etc.)
    - **Hours** (decimal)
    - **Cost** (e.g., S$130.00)
    """)
    st.stop()

# Load data
try:
    df = load_data(uploaded)
    
    if len(df) == 0:
        st.error("No 'Attended' lessons found in the CSV. Please check your data.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading CSV: {str(e)}")
    st.stop()

# Date filter
min_date, max_date = df["Date"].min(), df["Date"].max()
st.sidebar.header("Filters")
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(start_date, tuple):  # if user clears and reselects
    start_date, end_date = start_date

mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
df = df.loc[mask].copy()

st.caption(
    f"Showing data from **{df['Date'].min().date()}** to **{df['Date'].max().date()}** "
    f"({len(df)} attended lessons)."
)

# -----------------------------
# Overview Section
# -----------------------------
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Lessons", f"{len(df):,}")
col2.metric("Unique Students", df["student"].nunique())
col3.metric("Unique Tutors", df["tutor"].nunique())
col4.metric(
    "Total Revenue",
    f"S${df['Cost_numeric'].sum():,.0f}",
)

# -----------------------------
# Retention & Churn
# -----------------------------
student_retention = monthly_retention_churn(df, "student")
tutor_retention = monthly_retention_churn(df, "tutor")

tab_overview, tab_tutors, tab_students, tab_churn = st.tabs(
    ["Business KPIs", "Tutor Retention", "Student Retention", "MoM Churn"]
)

# ----- Business KPIs -----
with tab_overview:
    st.markdown("### Revenue & Load over Time")

    monthly_rev = (
        df.groupby("month")
        .agg(
            revenue=("Cost_numeric", "sum"),
            lessons=("Date", "count"),
            students=("student", "nunique"),
            tutors=("tutor", "nunique"),
        )
        .reset_index()
    )

    fig = px.bar(
        monthly_rev,
        x="month",
        y="revenue",
        labels={"month": "Month", "revenue": "Revenue (S$)"},
        hover_data=["lessons", "students", "tutors"]
    )
    fig.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Active Students & Tutors")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_rev["month"],
        y=monthly_rev["students"],
        mode='lines+markers',
        name='Students',
        line=dict(color='#1f77b4')
    ))
    fig.add_trace(go.Scatter(
        x=monthly_rev["month"],
        y=monthly_rev["tutors"],
        mode='lines+markers',
        name='Tutors',
        line=dict(color='#ff7f0e')
    ))
    fig.update_layout(
        height=220,
        xaxis_title="Month",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- Tutor Retention -----
with tab_tutors:
    st.markdown("### Tutor Retention Quality")

    pair_stats = tutor_student_pair_stats(df)
    tutor_stats = tutor_retention_metrics(pair_stats)

    min_students = st.slider(
        "Minimum number of students to include tutor in ranking",
        min_value=1,
        max_value=int(tutor_stats["n_students"].max()),
        value=3,
    )
    filtered_tutors = tutor_stats[tutor_stats["n_students"] >= min_students].copy()

    st.write(
        f"Showing tutors who have taught at least **{min_students}** distinct students "
        f"within the selected date range."
    )

    # Best & worst tables
    top_n = st.slider("How many tutors to show in each list?", 3, 20, 5)

    best = (
        filtered_tutors.sort_values("retention_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    worst = (
        filtered_tutors.sort_values("retention_score", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Best tutors table
    st.markdown("#### 🟢 Best Tutors (High Retention)")
    best_display = best[
        [
            "tutor",
            "retention_score",
            "n_students",
            "share_quick_drop",
            "avg_months_per_student",
            "avg_lessons_per_student",
            "total_revenue",
        ]
    ].copy()
    best_display["retention_score"] = best_display["retention_score"].apply(lambda x: f"{x:.2f}")
    best_display["share_quick_drop"] = best_display["share_quick_drop"].apply(lambda x: f"{x:.1%}")
    best_display["avg_months_per_student"] = best_display["avg_months_per_student"].apply(lambda x: f"{x:.2f}")
    best_display["avg_lessons_per_student"] = best_display["avg_lessons_per_student"].apply(lambda x: f"{x:.2f}")
    best_display["total_revenue"] = best_display["total_revenue"].apply(lambda x: f"S${x:,.0f}")
    st.markdown(best_display.to_html(index=False, escape=False), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Worst tutors table
    st.markdown("#### 🔴 Red-Flag Tutors (High Drop Rate)")
    worst_display = worst[
        [
            "tutor",
            "retention_score",
            "n_students",
            "share_quick_drop",
            "avg_months_per_student",
            "avg_lessons_per_student",
            "total_revenue",
        ]
    ].copy()
    worst_display["retention_score"] = worst_display["retention_score"].apply(lambda x: f"{x:.2f}")
    worst_display["share_quick_drop"] = worst_display["share_quick_drop"].apply(lambda x: f"{x:.1%}")
    worst_display["avg_months_per_student"] = worst_display["avg_months_per_student"].apply(lambda x: f"{x:.2f}")
    worst_display["avg_lessons_per_student"] = worst_display["avg_lessons_per_student"].apply(lambda x: f"{x:.2f}")
    worst_display["total_revenue"] = worst_display["total_revenue"].apply(lambda x: f"S${x:,.0f}")
    st.markdown(worst_display.to_html(index=False, escape=False), unsafe_allow_html=True)

    st.markdown("---")
    
    # Detailed dropout analysis
    st.markdown("### 📋 Detailed Student Dropout Analysis")
    st.markdown("See which students 'dropped' (had only 1 lesson) with each tutor")
    
    # Create dropout details
    dropout_details = pair_stats[pair_stats['n_lessons'] == 1].copy()
    dropout_details = dropout_details.sort_values('tutor')
    
    # Let user select a tutor to see details
    tutor_list = sorted(tutor_stats['tutor'].unique())
    selected_tutor = st.selectbox("Select a tutor to see dropout details:", tutor_list)
    
    if selected_tutor:
        tutor_dropouts = dropout_details[dropout_details['tutor'] == selected_tutor]
        tutor_all_students = pair_stats[pair_stats['tutor'] == selected_tutor]
        
        # Get tutor stats
        tutor_info = tutor_stats[tutor_stats['tutor'] == selected_tutor].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", int(tutor_info['n_students']))
        col2.metric("Dropped Students", len(tutor_dropouts))
        col3.metric("Drop Rate", f"{tutor_info['share_quick_drop']:.1%}")
        col4.metric("Retention Score", f"{tutor_info['retention_score']:.2f}")
        
        # Show dropped students
        if len(tutor_dropouts) > 0:
            st.markdown(f"#### 🚨 Students who dropped after 1 lesson ({len(tutor_dropouts)} students)")
            
            dropout_display = tutor_dropouts[['student', 'first_date', 'total_revenue']].copy()
            dropout_display.columns = ['Student', 'Lesson Date', 'Revenue']
            dropout_display['Lesson Date'] = dropout_display['Lesson Date'].dt.strftime('%Y-%m-%d')
            dropout_display['Revenue'] = dropout_display['Revenue'].apply(lambda x: f"S${x:,.0f}")
            dropout_display = dropout_display.reset_index(drop=True)
            
            st.markdown(dropout_display.to_html(index=False, escape=False), unsafe_allow_html=True)
        else:
            st.success("✅ No students dropped after just 1 lesson!")
        
        # Show retained students (2+ lessons)
        retained_students = tutor_all_students[tutor_all_students['n_lessons'] >= 2]
        if len(retained_students) > 0:
            st.markdown(f"#### ✅ Students with 2+ lessons ({len(retained_students)} students)")
            
            retained_display = retained_students[['student', 'n_lessons', 'n_months', 'total_revenue', 'last_date']].copy()
            retained_display.columns = ['Student', 'Lessons', 'Months Active', 'Total Revenue', 'Last Lesson']
            retained_display = retained_display.sort_values('Lessons', ascending=False)
            retained_display['Total Revenue'] = retained_display['Total Revenue'].apply(lambda x: f"S${x:,.0f}")
            retained_display['Last Lesson'] = retained_display['Last Lesson'].dt.strftime('%Y-%m-%d')
            retained_display = retained_display.reset_index(drop=True)
            
            st.markdown(retained_display.to_html(index=False, escape=False), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Summary table: All tutors with dropout lists
    st.markdown("### 📊 Complete Dropout Summary (All Tutors)")
    
    # Create summary with dropout student names
    dropout_summary = []
    for tutor in tutor_stats['tutor'].unique():
        tutor_dropouts = dropout_details[dropout_details['tutor'] == tutor]
        tutor_info = tutor_stats[tutor_stats['tutor'] == tutor].iloc[0]
        
        dropout_names = tutor_dropouts['student'].tolist()
        dropout_str = ", ".join(dropout_names) if len(dropout_names) > 0 else "None"
        
        dropout_summary.append({
            'Tutor': tutor,
            'Total Students': int(tutor_info['n_students']),
            'Dropped Count': len(tutor_dropouts),
            'Drop Rate': f"{tutor_info['share_quick_drop']:.1%}",
            'Retention Score': f"{tutor_info['retention_score']:.2f}",
            'Dropped Students': dropout_str
        })
    
    dropout_summary_df = pd.DataFrame(dropout_summary)
    dropout_summary_df = dropout_summary_df.sort_values('Dropped Count', ascending=False)
    
    # Display with scrollable HTML table
    st.markdown(
        f'<div style="max-height: 400px; overflow-y: auto;">{dropout_summary_df.to_html(index=False, escape=False)}</div>',
        unsafe_allow_html=True
    )
    
    # Download button for the summary
    csv = dropout_summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dropout Summary as CSV",
        data=csv,
        file_name="tutor_dropout_summary.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    st.markdown("### Tutor Retention Scatter (quality vs volume)")
    fig = px.scatter(
        filtered_tutors,
        x="share_quick_drop",
        y="avg_months_per_student",
        size="n_students",
        color="retention_score",
        hover_name="tutor",
        hover_data={
            "share_quick_drop": ":.1%",
            "avg_months_per_student": ":.2f",
            "avg_lessons_per_student": ":.2f",
            "retention_score": ":.2f",
            "n_students": True
        },
        labels={
            "share_quick_drop": "Share of One-Lesson Students (worse →)",
            "avg_months_per_student": "Avg Months per Student (better ↑)",
            "retention_score": "Retention Score"
        },
        color_continuous_scale="viridis"
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ----- Student Retention -----
with tab_students:
    st.markdown("### 🚨 Student Engagement Alerts")
    
    # Calculate alerts
    student_alerts = calculate_student_alerts(df)
    
    if len(student_alerts) > 0:
        # Split into red and amber alerts
        red_alerts = student_alerts[student_alerts['Alert Type'] == '🔴 Red']
        amber_alerts = student_alerts[student_alerts['Alert Type'] == '🟠 Amber']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Red Alerts", len(red_alerts), help="Students with no lesson in 30+ days")
        with col2:
            st.metric("🟠 Amber Alerts", len(amber_alerts), help="Students with decreased lesson frequency")
        
        # Red Alerts
        if len(red_alerts) > 0:
            st.markdown("#### 🔴 Red Alerts - Follow Up Immediately")
            st.markdown("*Students who haven't had a lesson in the past 30 days*")
            
            red_display = red_alerts.copy()
            red_display['Last Lesson'] = red_display['Last Lesson'].dt.strftime('%Y-%m-%d')
            red_display['Avg Gap (days)'] = red_display['Avg Gap (days)'].apply(lambda x: f"{x:.0f}")
            red_display = red_display.sort_values('Days Since Last', ascending=False)
            
            st.markdown(
                f'<div class="alert-red">{red_display.to_html(index=False, escape=False)}</div>',
                unsafe_allow_html=True
            )
        
        # Amber Alerts
        if len(amber_alerts) > 0:
            st.markdown("#### 🟠 Amber Alerts - Check on Engagement")
            st.markdown("*Students whose lesson frequency has significantly decreased*")
            
            amber_display = amber_alerts.copy()
            amber_display['Last Lesson'] = amber_display['Last Lesson'].dt.strftime('%Y-%m-%d')
            amber_display['Avg Gap (days)'] = amber_display['Avg Gap (days)'].apply(lambda x: f"{x:.0f}")
            amber_display = amber_display.sort_values('Days Since Last', ascending=False)
            
            st.markdown(
                f'<div class="alert-amber">{amber_display.to_html(index=False, escape=False)}</div>',
                unsafe_allow_html=True
            )
        
        # Download alerts
        alerts_csv = student_alerts.copy()
        alerts_csv['Last Lesson'] = alerts_csv['Last Lesson'].dt.strftime('%Y-%m-%d')
        st.download_button(
            label="📥 Download Student Alerts as CSV",
            data=alerts_csv.to_csv(index=False),
            file_name="student_alerts.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No student alerts! All students are actively engaged.")
    
    st.markdown("---")
    
    st.markdown("### Student Retention Over Time")

    sr_clean = student_retention.dropna(subset=["retention_rate"])
    fig = px.line(
        sr_clean,
        x="month",
        y="retention_rate",
        markers=True,
        labels={"month": "Month", "retention_rate": "Retention Rate"},
        hover_data=["active_count", "churn_rate"]
    )
    fig.update_layout(height=280, showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Raw monthly student retention / churn")
    sr_display = student_retention.copy()
    sr_display["month"] = sr_display["month"].astype(str)
    sr_display["active_count"] = sr_display["active_count"].apply(lambda x: f"{x:,.0f}")
    sr_display["retained_from_prev"] = sr_display["retained_from_prev"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    sr_display["retention_rate"] = sr_display["retention_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    sr_display["churn_rate"] = sr_display["churn_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    st.markdown(sr_display.to_html(index=False, escape=False), unsafe_allow_html=True)

# ----- MoM Churn (Tutors & Students) -----
with tab_churn:
    st.markdown("### Month-over-Month Churn")

    sr = student_retention.copy()
    tr = tutor_retention.copy()

    sr["type"] = "Student"
    tr["type"] = "Tutor"
    combined = pd.concat([sr, tr], ignore_index=True)

    combined_clean = combined.dropna(subset=["churn_rate"])
    fig = px.line(
        combined_clean,
        x="month",
        y="churn_rate",
        color="type",
        markers=True,
        labels={"month": "Month", "churn_rate": "Churn Rate", "type": "Type"}
    )
    fig.update_layout(height=280, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Raw tables")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Students")
        sr_display2 = sr.copy()
        sr_display2["month"] = sr_display2["month"].astype(str)
        sr_display2["active_count"] = sr_display2["active_count"].apply(lambda x: f"{x:,.0f}")
        sr_display2["retained_from_prev"] = sr_display2["retained_from_prev"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        sr_display2["retention_rate"] = sr_display2["retention_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        sr_display2["churn_rate"] = sr_display2["churn_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        sr_display2 = sr_display2.drop(columns=["type"])
        st.markdown(sr_display2.to_html(index=False, escape=False), unsafe_allow_html=True)
    with c2:
        st.markdown("#### Tutors")
        tr_display = tr.copy()
        tr_display["month"] = tr_display["month"].astype(str)
        tr_display["active_count"] = tr_display["active_count"].apply(lambda x: f"{x:,.0f}")
        tr_display["retained_from_prev"] = tr_display["retained_from_prev"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        tr_display["retention_rate"] = tr_display["retention_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        tr_display["churn_rate"] = tr_display["churn_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        tr_display = tr_display.drop(columns=["type"])
        st.markdown(tr_display.to_html(index=False, escape=False), unsafe_allow_html=True)
