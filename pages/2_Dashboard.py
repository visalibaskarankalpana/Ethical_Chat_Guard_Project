import streamlit as st
import pandas as pd
import altair as alt
from services.storage import get_logs

st.set_page_config(page_title="Ethical Pattern Dashboard", layout="wide")

st.title("Ethical Pattern Dashboard")
st.caption("Tracking unethical reasoning patterns across sessions.")

# Load data
logs = get_logs()

if not logs:
    st.info("No audit logs available yet. Start chatting in the main app to generate data.")
    st.stop()

df = pd.DataFrame(logs)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------------------------------------------------------
# High-Level Metrics
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Audits", len(df))
with col2:
    high_risk = df[df['label'] == 'RED'].shape[0]
    st.metric("High Risk (Red)", high_risk)
with col3:
    avg_score = df['score'].mean()
    st.metric("Average Risk Score", f"{avg_score:.1f}")

# ---------------------------------------------------------
# Taxonomy Breakdown
# ---------------------------------------------------------
st.subheader("Detected Ethical Patterns")

# Flatten the categories dictionary
# Each row in df has a 'categories' dict. We want to sum them up.
# Initialize counters
pattern_counts = {}

for record in logs:
    cats = record.get("categories", {})
    for cat, count in cats.items():
        if cat not in pattern_counts:
            pattern_counts[cat] = 0
        pattern_counts[cat] += count

# Convert to DataFrame for plotting
patterns_df = pd.DataFrame(list(pattern_counts.items()), columns=["Pattern", "Count"])
patterns_df = patterns_df.sort_values(by="Count", ascending=False)

if patterns_df.empty or patterns_df['Count'].sum() == 0:
    st.write("No specific patterns detected yet.")
else:
    # Bar Chart
    chart = alt.Chart(patterns_df).mark_bar().encode(
        x=alt.X('Count', title='Occurrences'),
        y=alt.Y('Pattern', sort='-x', title='Ethical Pattern'),
        color=alt.Color('Pattern', legend=None),
        tooltip=['Pattern', 'Count']
    ).properties(
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------------------------------------------
# Timeline
# ---------------------------------------------------------
st.subheader("Risk Score Timeline")

timeline_chart = alt.Chart(df).mark_line(point=True).encode(
    x=alt.X('timestamp', title='Time'),
    y=alt.Y('score', title='Risk Score'),
    tooltip=['timestamp', 'score', 'label']
).properties(
    height=300
)
st.altair_chart(timeline_chart, use_container_width=True)

# ---------------------------------------------------------
# Recent Logs
# ---------------------------------------------------------
st.subheader("Recent Logs")
st.dataframe(
    df[['timestamp', 'score', 'label', 'explanation', 'translated_reply']].sort_values(by='timestamp', ascending=False),
    use_container_width=True
)
