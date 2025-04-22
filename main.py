import tempfile
from collections import Counter
from typing import cast

import spacy
import streamlit as st
import torch
import whisper

# https://github.com/streamlit/streamlit/issues/10992
torch.classes.__path__ = []

TOP_N = 3

uploaded_file = st.file_uploader("Upload audio file")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_file.getvalue())
        temp_audio_file.flush()
        with st.spinner("Transcribing..."):
            text = cast(str, (whisper.load_model("tiny.en").transcribe(temp_audio_file.name, fp16=False)["text"]))

    col1, col2 = st.columns(2)
    with col1:
        st.write(text)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    with col2:
        for tag, name in {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective"}.items():
            tokens = [token.lemma_ for token in doc if token.pos_ == tag]
            st.write(f"The most frequent {TOP_N} {name}s:")
            st.markdown("\n".join([f"* {noun[0]} ({noun[1]})" for noun in Counter(tokens).most_common(TOP_N)]))
