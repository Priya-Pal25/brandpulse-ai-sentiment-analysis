import streamlit as st
import pandas as pd
import joblib
from utils import clean

st.set_page_config(page_title="BrandPulse AI", layout="wide")
st.title("BrandPulse AI: Live Sentiment Monitor")
st.caption("Classifying tweets as Positive, Negative or Neutral")

@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf.pkl")
    lr = joblib.load("lr_model.pkl")
    return tfidf, lr

tfidf, lr = load_models()

def predict(text):
    return lr.predict(tfidf.transform([clean(text)]))[0]

st.subheader("Try your own tweet")
user_text = st.text_area("Type or paste a tweet:", "The service was amazing!")

if st.button("Analyze"):
    st.success(f"Predicted sentiment: **{predict(user_text)}**")


# ---------------- Live stream simulation ----------------
import time
import numpy as np
import plotly.express as px

st.divider()
st.subheader("Live tweet stream (simulated)")

@st.cache_data
def load_tweets():
    return pd.read_csv("Tweets.csv")[["airline", "text"]]

tweets = load_tweets()
n = st.slider("Number of tweets in the feed", 20, 200, 100)
delay = st.slider("Seconds between tweets", 0.0, 1.0, 0.1)

colors = {"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"}

if st.button("Start live stream"):
    batch = tweets.sample(n).reset_index(drop=True)
    batch["sentiment"] = lr.predict(tfidf.transform(batch["text"].apply(clean)))

    # Simulate arrival times spread over the last 24 hours (oldest first)
    now = pd.Timestamp.now()
    offsets = np.sort(np.random.uniform(0, 24, n))[::-1]
    batch["time"] = now - pd.to_timedelta(offsets, unit="h")

    feed_box = st.empty()
    left, right = st.columns(2)
    pie_box = left.empty()
    trend_box = right.empty()

    for i in range(1, n + 1):
        shown = batch.iloc[:i]

        feed_box.dataframe(
            shown.tail(5)[["airline", "text", "sentiment"]].iloc[::-1],
            hide_index=True)

        pie = px.pie(shown, names="sentiment", color="sentiment",
                     color_discrete_map=colors,
                     title=f"Sentiment distribution ({i} tweets)")
        pie_box.plotly_chart(pie, key=f"pie_{i}")

        grouped = (shown.assign(slot=shown["time"].dt.floor("3h"))
                        .groupby(["slot", "sentiment"]).size()
                        .unstack(fill_value=0)
                        .reindex(columns=["negative", "neutral", "positive"], fill_value=0))
        slots = pd.date_range(now.floor("3h") - pd.Timedelta(hours=24),
                              now.floor("3h"), freq="3h")
        grouped = grouped.reindex(slots, fill_value=0).rename_axis("slot").reset_index()
        hourly = grouped.melt(id_vars="slot", var_name="sentiment", value_name="tweets")

        trend = px.line(hourly, x="slot", y="tweets", color="sentiment",
                        color_discrete_map=colors, markers=True,
                        title="Sentiment trend (last 24 hours, 3-hour intervals)")
        trend_box.plotly_chart(trend, key=f"trend_{i}")

        time.sleep(delay)

        
# ---------------- Model comparison ----------------
st.divider()
st.subheader("Model comparison")

results = pd.read_csv("all_results.csv")
st.dataframe(results, hide_index=True)

melted = results.melt(id_vars="Model", value_vars=["Accuracy", "F1 (macro)"],
                      var_name="Metric", value_name="Score")
bar = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
             range_y=[0, 1], title="Accuracy and F1 by model")
st.plotly_chart(bar, key="comparison_bar")
