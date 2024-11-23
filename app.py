import streamlit as st
from transformers import pipeline
import nltk

nltk.download('punkt')

# Load the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit App
def main():
    # App Title
    st.title("Text Summarization with BERT")
    st.write("This app uses a BERT-based model to summarize text.")

    # Input Text
    text = st.text_area("Enter your text here:", height=200)

    # Parameters for summarization
    max_length = st.slider("Maximum Summary Length", 50, 500, 130)
    min_length = st.slider("Minimum Summary Length", 10, 100, 30)

    # Summarize Button
    if st.button("Summarize"):
        if len(text.strip()) == 0:
            st.error("Please enter some text to summarize.")
        else:
            st.info("Generating summary...")
            try:
                summary = summarizer(
                    text, max_length=max_length, min_length=min_length, do_sample=False
                )
                st.success("Summary generated successfully!")
                st.subheader("Summary:")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
