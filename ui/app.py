
import os
 
import requests
import streamlit as st
 
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 30  # seconds
 
# Used if the API can't be reached (must match schemas.py)
FALLBACK_QUEUES = [
    "Technical Support", "Product Support", "Customer Service", "IT Support",
    "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
    "Sales and Pre-Sales", "Human Resources", "General Inquiry",
]
 
# TEST_MACRO_F1 is filled in after the final evaluation
VAL_MACRO_F1 = 0.665
TEST_MACRO_F1 = 0.674
TEST_MACRO_F1_CI = (0.656, 0.690)  # 95% bootstrap interval
 
PRIORITY_LABEL = {"high": "🔴 HIGH", "medium": "🟠 MEDIUM", "low": "🟢 LOW"}
 
 
@st.cache_data(ttl=300, show_spinner=False)
def load_queues(api_url: str) -> list[str]:
    """Reads the queue list from the API's OpenAPI schema (keeps the UI and API in sync)."""
    try:
        schema = requests.get(f"{api_url}/openapi.json", timeout=5).json()
        return schema["components"]["schemas"]["TicketRequest"]["properties"]["queue"]["enum"]
    except Exception:
        return FALLBACK_QUEUES
 
 
def api_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/status", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False
 
 
def call_predict(payload: dict):
    """Returns (ok, data_or_error_message)."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to the API ({API_URL}). Is the API running?"
    except requests.exceptions.Timeout:
        return False, "The API did not respond in time."
    if r.status_code == 200:
        return True, r.json()
    if r.status_code == 429:
        return False, f"Rate limit reached. Try again in {r.headers.get('Retry-After', '?')} seconds."
    if r.status_code == 422:
        return False, "Invalid input (empty field or text over the length limit)."
    if r.status_code == 503:
        return False, "Model files are missing (on the API side)."
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return False, f"API error ({r.status_code}): {detail}"
 
 
def show_result(title: str, ok: bool, data):
    st.markdown(f"**{title}**")
    if not ok:
        st.error(data)
        return
    st.metric("Priority", PRIORITY_LABEL.get(data["priority"], data["priority"]))
    note = " · from cache" if data.get("cached") else ""
    st.caption(f"Response time: {data['latency_ms']} ms{note}")
 
 
st.title("Ticket Triage")
st.caption("Predicts the urgency (priority) of a customer support ticket.")
 
with st.sidebar:
    st.subheader("API")
    st.write(API_URL)
    if api_is_up():
        st.success("Running")
    else:
        st.error("Unreachable")
 
with st.form("ticket_form"):
    subject = st.text_input("Subject", max_chars=500, key="subject")
    body = st.text_area("Body", height=160, max_chars=5000, key="body")
    queue = st.selectbox("Queue (category)", load_queues(API_URL), key="queue")
    model = st.radio(
        "Model", ["svm", "claude"], horizontal=True, key="model",
        format_func=lambda m: "SVM (fast)" if m == "svm" else "Claude (zero-shot, slow)",
    )
    col1, col2 = st.columns(2)
    single = col1.form_submit_button("Predict")
    compare = col2.form_submit_button("Compare SVM and Claude")
 
if single or compare:
    if not subject.strip() or not body.strip():
        st.warning("Subject and body cannot be empty.")
    else:
        base = {"subject": subject.strip(), "body": body.strip(), "queue": queue}
        if single:
            ok, data = call_predict({**base, "model": model})
            show_result(f"Result ({model})", ok, data)
        else:
            left, right = st.columns(2)
            with left:
                svm_ok, svm = call_predict({**base, "model": "svm"})
                show_result("SVM", svm_ok, svm)
            with right:
                with st.spinner("Claude is responding..."):
                    cl_ok, cl = call_predict({**base, "model": "claude"})
                show_result("Claude", cl_ok, cl)
            if svm_ok and cl_ok:
                if svm["priority"] == cl["priority"]:
                    st.info("The models gave the same answer.")
                else:
                    st.info("The models gave different answers. Without a label, we can't tell which one is right.")
 
with st.expander("Model limitations"):
    test_note = (
        f" Test set: {TEST_MACRO_F1} (95% interval {TEST_MACRO_F1_CI[0]}–{TEST_MACRO_F1_CI[1]})."
        if TEST_MACRO_F1
        else " The test result will be added after the final evaluation."
    )
    st.markdown(
        "- SVM is sensitive to the `queue` category and urgency words (e.g. *urgent*), not to the meaning of the text.\n"
        "- The models were trained on synthetic (template-style) English data; results on real tickets may differ.\n"
        "- Dataset labels don't always match the tone of the text. Claude looks at meaning, so it may answer differently from SVM.\n"
        f"- SVM macro F1 on the validation (val) set ≈ {VAL_MACRO_F1}." + test_note
    )

