# 📊 Tutor Retention Dashboard

A Streamlit dashboard for analyzing tutor performance, student retention, and identifying red-flag tutors who frequently drop students.

## Features

### 📈 Core Analytics
- **Tutor Retention Quality** - Identify which tutors retain students vs. drop them quickly
- **MoM Churn & Retention** - Track month-over-month retention for both students and tutors
- **Revenue & Activity Trends** - Monitor business growth and engagement
- **Red Flag Detection** - Automatically flag tutors with high student drop rates

### 🎯 Key Metrics
- **Retention Score** - Composite metric ranking tutor performance
- **Quick Drop Rate** - % of students who have only 1 lesson with a tutor
- **Average Months per Student** - How long students stay with each tutor
- **Student/Tutor MoM Retention** - Month-over-month retention rates

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the dashboard:**
```bash
streamlit run app.py
```

3. **Open your browser** to `http://localhost:8501`

## CSV Format

Your CSV file should contain these columns:

| Column | Format | Example | Description |
|--------|--------|---------|-------------|
| Date | DD/MM/YYYY | 02/03/2025 | Lesson date |
| Student First Name | string | John | Student's first name |
| Student Last Name | string | Tan | Student's last name |
| Tutor First Name | string | Abel | Tutor's first name |
| Tutor Last Name | string | Chen | Tutor's last name |
| Service | string | IB Physics HL | Lesson type/subject |
| Status | string | Attended | Lesson status (Attended, Cancelled, etc.) |
| Hours | float | 1.0 | Lesson duration in hours |
| Cost | string | S$130.00 | Lesson cost |

**Note:** Only lessons with Status = "Attended" are included in the analysis.

### Example CSV:
```csv
Date,Title,Tutor First Name,Tutor Last Name,Student First Name,Student Last Name,Service,Status,,Hours,Cost
02/03/2025,Zi Han .,Byron,Chan,Zi Han,.,WP Grade 10/9 Online Tutoring,Attended,,2.0,S$130.00
19/02/2025,Zi Han .,Byron,Chan,Zi Han,.,WP Grade 10/9 Online Tutoring,Attended,,2.0,S$130.00
```

## Dashboard Sections

### 1. Business KPIs
- Monthly revenue trends
- Lesson volume over time
- Active students and tutors tracking

### 2. Tutor Retention
- **Best Tutors** - High retention, low drop rates
- **Red Flag Tutors** - High student drop rates, short relationships
- **Retention Scatter Plot** - Visualize quality vs. volume
- **Adjustable Filters** - Minimum student count threshold

### 3. Student Retention
- Month-over-month student retention trends
- Detailed retention/churn breakdown

### 4. MoM Churn
- Side-by-side student and tutor churn comparison
- Historical churn trends

## Key Insights

### 🟢 Good Tutor Indicators
- Low "quick drop" rate (<20% of students with only 1 lesson)
- High average months per student (>3 months)
- High retention score (>0.7)

### 🔴 Red Flag Indicators
- High quick drop rate (>50% of students with only 1 lesson)
- Low average months per student (<2 months)
- Low retention score (<0.4)

### Retention Score Formula
```
retention_score = (1 - share_quick_drop) * 0.6 + (normalized_avg_months) * 0.4
```

Where:
- `share_quick_drop` = % of students who had only 1 lesson
- `normalized_avg_months` = tutor's avg months / max avg months across all tutors

## Customization

You can adjust the retention score formula in the `tutor_retention_metrics()` function to match your business priorities:

```python
tutor_stats["retention_score"] = (
    (1 - tutor_stats["share_quick_drop"]).fillna(0) * 0.6  # Weight for retention
    + (tutor_stats["avg_months_per_student"] / tutor_stats["avg_months_per_student"].max()).fillna(0) * 0.4  # Weight for longevity
)
```

## Use Cases

1. **Performance Reviews** - Identify tutors who need coaching
2. **Hiring Decisions** - Track new tutor onboarding success
3. **Student Assignment** - Match students with high-performing tutors
4. **Intervention Planning** - Proactively address retention issues
5. **Operational Planning** - Understand seasonal trends

## Technical Details

- **Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Altair
- **Caching:** Uses `@st.cache_data` for performance

## Support

For questions or custom modifications, refer to the inline code comments in `app.py`.

## License

MIT License - Free to use and modify for your business needs.
