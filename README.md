# Adaptive Market Research Agent

Upload a CSV or Excel file. The agent detects the business type, cleans and normalizes the data,
fetches external web data only for gaps, computes statistics with pandas, writes a plain-language
report, and checks its own claims before finishing.

## Setup (free, email only)
1. Create a Groq account at console.groq.com and make an API key.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and paste your key.
4. `python sample_data.py` then `streamlit run app.py`

## Flow
Ingest -> Detect type -> Normalize -> Gap check -> Plan and fetch web data -> Analyze (pandas) -> Write -> Critic -> Report

## Files
- `llm.py` provider routing (Groq, optional Gemini), disk cache, JSON retries
- `tools.py` web search and page reader (cached)
- `playbooks.py` per-industry playbooks; unknown types get a proposed one, saved to `playbooks.json`
- `pipeline.py` all agent steps
- `app.py` Streamlit interface

## Privacy
Only the column profile, summary statistics and generic search queries go to the AI model, never full rows.

## Trial and error tips
- Everything is cached in `.cache/`, so reruns are free. Delete the folder to start fresh.
- If you see rate limit errors, wait a minute or set `GROQ_SMART2_MODEL` for a second strong model.
- Fix wrong detections by editing the playbook in the UI instead of changing code.
