import streamlit as st
from services.llm_openai import generate_reply
from services.detector import assess
from utils.helpers import render_highlighted
from services.sbert_lr import predict_proba
from utils.config import load_threshold
from services.storage import log_assessment

st.set_page_config(page_title="Ethical Chat Guard", layout="wide")

CHAT_HEIGHT = 760
PANEL_HEIGHT = 740

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

if "audits" not in st.session_state:
    st.session_state.audits = []

if "mode" not in st.session_state:
    st.session_state.mode = "Balanced"

# -------------------------------
# Header
# -------------------------------
header_left, header_right = st.columns([2.4, 1], gap="large")

with header_left:
    st.title("Ethical Chat Guard")
    st.caption("Chat with an assistant and audit responses for coercive language in real time.")

with header_right:
    st.session_state.mode = st.selectbox(
        "Risk sensitivity",
        ["Conservative", "Balanced", "Aggressive"],
        index=["Conservative", "Balanced", "Aggressive"].index(st.session_state.mode),
    )
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.session_state.audits = []
        st.rerun()

left, right = st.columns([2, 1], gap="large")

# -------------------------------
# Risk Panel
# -------------------------------
with right:
    st.subheader("Risk Panel")
    panel = st.container(height=PANEL_HEIGHT, border=True)

    with panel:
        if not st.session_state.audits:
            st.write("Send a message to see risk analysis.")
        else:
            last = st.session_state.audits[-1]
            label = last.label.upper()

            st.metric("Risk score", f"{last.score}/100")
            st.success(label) if label == "GREEN" else st.warning(label) if label == "YELLOW" else st.error(label)

            st.markdown("**Why this label**")
            st.write(last.explanation)

# -------------------------------
# Chat box
# -------------------------------
with left:
    st.subheader("Conversation")
    chat_box = st.container(height=CHAT_HEIGHT, border=True)

    with chat_box:
        for m in st.session_state.messages:
            if m["role"] not in ("user", "assistant"):
                continue

            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    idx = m.get("audit_idx")
                    if idx is not None:
                        a = st.session_state.audits[idx]
                        html = render_highlighted(m["content"], a.spans)
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.markdown(m["content"])
                else:
                    st.markdown(m["content"])

# -------------------------------
# Input
# -------------------------------
user_msg = st.chat_input("Type your message")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})


    reply = user_msg
    p = predict_proba(reply)
    th = load_threshold()

    # 🔹 Compute model_threshold FIRST
    if st.session_state.mode == "Conservative":
        model_threshold = min(0.95, th + 0.10)
    elif st.session_state.mode == "Aggressive":
        model_threshold = max(0.01, th - 0.10)
    else:
         model_threshold = th

# 🔹 Now assess safely
    a = assess(
     user_msg,
        reply,
        model_proba=p,
        model_threshold=model_threshold,
    )

# 🔹 Choose display text (translated if needed)
    display_text = a.translated_reply if a.translated_reply else reply

    # 🔹 LOG THE ASSESSMENT
    log_assessment(a)

    audit_idx = len(st.session_state.audits)
    st.session_state.audits.append(a)

    # st.session_state.messages.append(
    #     {"role": "assistant", "content": reply, "audit_idx": audit_idx}
    # )

    st.session_state.messages.append({
     "role": "assistant",
     "content": display_text,
    "audit_idx": audit_idx
    })

    # 🔹 SHOW TRANSLATION (AFTER a EXISTS)
    if a.translated_reply:
        st.info("🌍 Message translated to English before analysis:")
        st.code(a.translated_reply)

    st.rerun()
