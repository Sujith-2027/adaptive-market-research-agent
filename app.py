import json
import pandas as pd
import streamlit as st
from pipeline import load, profile, detect_type, run_pipeline
from playbooks import get_playbook, save_playbook
from llm import USAGE

st.set_page_config(page_title="Adaptive Market Research Agent", layout="wide")
st.title("Adaptive Market Research Agent")
st.caption("Upload your data. The agent detects your business type, cleans the data, adds web context when "
           "needed, and writes a cited, plain-language analysis.")

with st.sidebar:
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    allow_web = st.toggle("Fetch external data from the web", value=True)
    market = st.text_input("Market or country (optional)", "", help="Used to pick relevant web facts. Leave blank if unsure.")
    st.caption("Only the column profile (names, types, 3 sample values), summary statistics and generic "
               "search queries are sent to the AI model. Full rows are not.")
    st.caption(f"LLM calls this session: {USAGE['calls']} (cached: {USAGE['cached']})")

if not up:
    st.info("Upload a file in the sidebar to begin. Try sample_orders.csv from sample_data.py.")
    st.stop()

if st.session_state.get("fname") != up.name:
    with st.spinner("Reading file and detecting business type..."):
        df = load(up)
        prof = profile(df)
        det = detect_type(prof)
    st.session_state.update(fname=up.name, df=df, det=det, res=None)
df, det = st.session_state.df, st.session_state.det

c1, c2 = st.columns([2, 1])
btype = c1.text_input("Detected business type (edit if wrong)", det["type"]).strip().lower().replace(" ", "_")
c2.metric("Detection confidence", f"{det['confidence']:.0%}")
c1.caption(det["reasoning"])
if det["confidence"] < 0.7:
    st.warning("Low confidence. Please confirm or correct the business type above.")
with st.expander("Raw data preview"):
    st.dataframe(df.head(20))

with st.spinner("Loading playbook..."):
    playbook = get_playbook(btype)
pb_text = st.text_area("Playbook: what the agent will check (editable, saved for next time)",
                       json.dumps(playbook, indent=2), height=240)

if st.button("Run analysis", type="primary"):
    try:
        pb = json.loads(pb_text)
    except json.JSONDecodeError:
        st.error("The playbook is not valid JSON. Please fix it and try again.")
        st.stop()
    save_playbook(btype, pb)
    trace = []
    with st.status("Agents working...", expanded=True) as status:
        def log(m):
            trace.append(m)
            st.write(m)
        try:
            res = run_pipeline(df, btype, pb, allow_web, log, market.strip())
            res["trace"] = trace
            st.session_state.res = res
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Run failed: {e}")

res = st.session_state.get("res")
if res:
    t1, t2, t3, t4, t5 = st.tabs(["Report", "Charts", "Clean data", "External data", "Agent trace"])
    with t1:
        st.markdown(res["report"])
        st.download_button("Download report (.md)", res["report"], "report.md")
    with t2:
        if not res["charts"]:
            st.write("Not enough structure in this data to draw charts.")
        for name, s in res["charts"].items():
            st.subheader(name)
            (st.line_chart if name.startswith("Monthly") else st.bar_chart)(s)
    with t3:
        st.dataframe(res["df"])
        st.download_button("Download clean data (.csv)", res["df"].to_csv(index=False), "clean_data.csv")
    with t4:
        if res["facts"]:
            st.dataframe(pd.DataFrame(res["facts"]), column_config={"source": st.column_config.LinkColumn()})
        else:
            st.write("No external facts were used.")
        if res["unfetched"]:
            st.write("Could not find online:", ", ".join(res["unfetched"]))
    with t5:
        for m in res["trace"]:
            st.write(m)
