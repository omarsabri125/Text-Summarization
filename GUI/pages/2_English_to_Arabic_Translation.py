import streamlit as st
from translation_inference import translate_en_to_ar

st.set_page_config(page_title="English to Arabic Translation", layout="centered")

st.title("English to Arabic Translation")

text_input = st.text_area("Enter English text to translate", height=100)

if st.button("Translate"):
    if text_input.strip() == "":
        st.warning("Please enter some English text.")
    else:
        with st.spinner("Translating..."):
            translation = translate_en_to_ar(text_input)
            st.subheader("Translation")
            st.success(translation)
