import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

st.title("Sentiment Analyser")
st.write("Paste a review or any text below to classify its sentiment.")

user_input = st.text_area("Your text", height=150, placeholder="e.g. The movie was absolutely fantastic!")

if st.button("Analyse") and user_input.strip():
    with st.spinner("Analysing..."):
        result = classifier(user_input[:512])[0]

    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        st.success(f"Positive — {score:.1%} confidence")
    else:
        st.error(f"Negative — {score:.1%} confidence")

    all_results = classifier(user_input[:512], top_k=None)
    st.subheader("Score breakdown")
    for r in all_results:
        st.progress(r["score"], text=f"{r['label']}: {r['score']:.1%}")