import streamlit as st
from summarization import summarize_arabic_text
from PyPDF2 import PdfReader

# Page configuration
st.set_page_config(page_title="Arabic Text Summarization", layout="centered")

st.title("Arabic Text Summarization")

# Allow user to upload PDF or enter text
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
text_input = st.text_area("Or enter Arabic text to summarize", height=300)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

if st.button("Summarize"):
    # Determine source of text
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        if text is None:
            # error already shown
            st.stop()
        if not text.strip():
            st.warning("No extractable text found in the PDF.")
            st.stop()
    else:
        text = text_input

    # Validate input text
    if not text or text.strip() == "":
        st.warning("Please upload a PDF or enter some Arabic text.")
        st.stop()

    # Summarize with spinner
    with st.spinner("Summarizing..."):
        summary = summarize_arabic_text(text)

    # Display result
    st.subheader("Summary")
    st.success(summary)

# Note:
# Requires installation: pip install PyPDF2
