"""Streamlit view for the dashboard and the edit chat."""
import hashlib
import streamlit as st
import dashboard as D


def _kpi_grid(df, spec):
    ctx = D.theme_ctx(spec)
    for i, k in enumerate(spec["kpis"]):
        if i % 3 == 0:
            cols = st.columns(3)
        r = D.compute_kpi(df, k, spec)
        with cols[i % 3]:
            st.metric(r["label"], r["value"], r["delta"], delta_color=r["delta_color"], help=r["note"] or None)
            if r.get("status"):
                color = ctx["positive"] if r.get("status_positive") else ctx["negative"]
                arrow = "&#9650;" if r.get("status_positive") else "&#9660;"
                st.markdown(f'<span style="color:{color};font-weight:600">{arrow} {r["status"]}</span>', unsafe_allow_html=True)


def _chart_grid(df, spec):
    ctx = D.theme_ctx(spec)
    for i, c in enumerate(spec["charts"]):
        if i % 2 == 0:
            cols = st.columns(2)
        chart, notes = D.build_chart(df, c, ctx)
        with cols[i % 2]:
            if chart is None:
                st.warning(f"'{c.get('title') or D.auto_title(c)}' has no data to show. " + " ".join(notes))
                continue
            st.altair_chart(chart)
            for n in notes:
                st.caption(n)


def _push_undo(S):
    S.undo.append({"spec": S.spec, "report": S.report})
    S.undo = S.undo[-15:]


def render(res, btype):
    S, df = st.session_state, res["df"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        if S.spec["kpis"]:
            _kpi_grid(df, S.spec)
        if S.spec["charts"]:
            _chart_grid(df, S.spec)
        else:
            st.info("No charts yet. Ask the chat to add one, or pick a suggestion.")
        st.caption(f"Source: your uploaded data, {len(df):,} rows after cleaning. Web facts are listed separately under External data.")
        with st.expander("Integrity checks"):
            for w in D.ethics_review(df, S.spec):
                st.write("- " + w)
        st.download_button("Download dashboard (.html)", D.export_html(df, S.spec, f"{btype.replace('_', ' ').title()} dashboard"),
                           "dashboard.html", mime="text/html")
    with right:
        st.subheader("Edit with chat")
        st.caption("Ask for charts, KPIs, colors, a currency, or changes to the written report.")
        box = st.container(height=360)
        for m in S.chat:
            box.chat_message(m["role"]).markdown(m["content"])
        msg = st.chat_input("For example: add revenue by category, show profit in green and loss in red", key="dash_chat")
        if msg:
            S.chat.append({"role": "user", "content": msg})
            with st.spinner("Working on it..."):
                out = D.handle_message(msg, S.spec, S.report, df, res, S.chat[:-1])
            if out["changed"]:
                _push_undo(S)
                S.spec, S.report = out["spec"], out["report"]
            S.chat.append({"role": "assistant", "content": out["reply"]})
            st.rerun()
        b1, b2 = st.columns(2)
        if b1.button("Undo last change", disabled=not S.undo):
            snap = S.undo.pop()
            S.spec, S.report = snap["spec"], snap["report"]
            st.rerun()
        if b2.button("Reset dashboard"):
            _push_undo(S)
            S.spec = D.default_spec(df, res["stats"], btype)
            st.rerun()
        sugg = D.suggest(df, S.spec, res["stats"])
        if sugg:
            with st.expander("Suggested additions", expanded=True):
                picks = []
                for s in sugg:
                    key = "sug_" + hashlib.md5(s["label"].encode()).hexdigest()[:8]
                    if st.checkbox(s["label"], key=key):
                        picks.append(s["op"])
                if st.button("Add selected", disabled=not picks):
                    new, done, refused, _ = D.apply_ops(S.spec, picks, df)
                    if done:
                        _push_undo(S)
                        S.spec = new
                    S.chat.append({"role": "assistant", "content": "**Changes made**\n" + "\n".join(f"- {d}" for d in done) if done else "Nothing was added."})
                    st.rerun()
