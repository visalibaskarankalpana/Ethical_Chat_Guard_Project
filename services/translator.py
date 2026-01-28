import streamlit as st
from deep_translator import GoogleTranslator

def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from source_lang to English using GoogleTranslator (deep-translator).
    """
    if source_lang == "en":
        return text

    try:
        # source_lang from langdetect (e.g., 'fr', 'es') is compatible with deep-translator
        translator = GoogleTranslator(source=source_lang, target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text
