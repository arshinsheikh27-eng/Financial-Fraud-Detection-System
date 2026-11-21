from pathlib import Path
import sys
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.service import predict_transaction, get_history, init_db, MODEL_PATH, THRESHOLD

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Dashboard styling inspired by the supplied dark analytics reference.
st.markdown(r"""
<style>
:root { --bg:#0b0d17; --panel:#141827; --panel2:#191f31; --line:#293149; --muted:#929bb2; --cyan:#39b9e8; --purple:#9860e8; }
.stApp { background: radial-gradient(ellipse at 82% 0%, #171b35 0%, #0b0d17 48%); color:#eef2ff; }
[data-testid="stHeader"] { background:rgba(8,10,18,.92); }
.block-container { padding-top:1.15rem; padding-bottom:2.5rem; max-width:1550px; }
section[data-testid="stSidebar"] { background:#101525; border-right:1px solid #242c42; }
section[data-testid="stSidebar"] > div { padding-top:1.5rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#cbd3e6; }
.stRadio [role="radiogroup"] { gap: .38rem; }
.stRadio [role="radio"] { background:transparent; border:1px solid transparent; padding:11px 12px; border-radius:9px; min-height:42px; }
.stRadio [role="radio"]:hover { background:#192238; border-color:#465273; }
.stRadio [role="radio"]:has(input:checked) { background:#202b46; border-color:#34496f; }
.stRadio [role="radio"] p { font-size:14px; font-weight:550; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#191f32,#141a2a); border:1px solid #2b3550; padding:17px 19px; border-radius:13px; min-height:112px; }
[data-testid="stMetricLabel"] { color:#aab4ca !important; }
[data-testid="stMetricValue"] { color:#f5f7ff !important; }
[data-testid="stMetricDelta"] { color:#60d4b0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div { background:linear-gradient(145deg,rgba(24,30,48,.96),rgba(17,22,36,.96)); border:1px solid #29334c; border-radius:15px; }
div[data-testid="stPlotlyChart"] { margin-top:-3px; }
.stButton button, .stDownloadButton button, button[kind="primary"] { border-radius:9px; }
h1,h2,h3 { letter-spacing:-.025em; }
.small-muted { color:#8f99b2; font-size:12px; }
.topline { display:flex; justify-content:space-between; align-items:center; padding:3px 0 18px; border-bottom:1px solid #252d42; margin-bottom:20px; }
.brand { color:#e9efff; font-weight:700; font-size:14px; letter-spacing:.02em; }
.status-pill { display:inline-block; color:#70e0bc; background:#14352f; border:1px solid #246451; border-radius:20px; padding:5px 11px; font-size:11px; }
.page-subtitle { color:#929bb2; margin-top:-8px; font-size:13px; }
.card-title { color:#f1f4ff; font-size:16px; font-weight:650; margin-bottom:2px; }
.card-subtitle { color:#8994ac; font-size:11px; margin-bottom:12px; }
.integration-row { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #283149; padding:12px 2px; color:#dce3f5; font-size:13px; }
.integration-row:last-child { border-bottom:0; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#42d6a4; margin-right:9px; }
@media (max-width: 900px) { .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
""", unsafe_allow_html=True)

# Data sources supplied in the project archive.
hour_file = ROOT / "reports" / "feature_analysis" / "fraud_rate_by_hour.csv"
intl_file = ROOT / "reports" / "feature_analysis" / "fraud_rate_by_international.csv"
importance_file = ROOT / "reports" / "feature_importance.csv"
comparison_file = ROOT / "reports" / "model_comparison.csv"

hourly = pd.read_csv(hour_file) if hour_file.exists() else pd.DataFrame()
intl = pd.read_csv(intl_file) if intl_file.exists() else pd.DataFrame()
importance = pd.read_csv(importance_file) if importance_file.exists() else pd.DataFrame()
comparison = pd.read_csv(comparison_file) if comparison_file.exists() else pd.DataFrame()
init_db()

with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin:0 0 5px 4px'><span style='font-size:27px'>🛡️</span><span style='font-size:19px;font-weight:750;color:#f0f3ff'>Fraud Detection</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#7f8ba7;letter-spacing:.16em;font-weight:700;margin:0 0 27px 43px'>FINANCIAL SECURITY</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#8290aa;font-size:11px;font-weight:700;letter-spacing:.12em;margin:0 0 10px 8px'>NAVIGATION</div>", unsafe_allow_html=True)
    page_labels = {
        "Overview": "▦   Dashboard",
        "Predict Transaction": "◈   Fraud Detection",
        "Batch Prediction": "▤   Transactions",
        "Investigation History": "⌕   Investigation Reports",
        "Model & Feature Analysis": "◉   Analytics & Model Performance",
    }
    page = st.radio(
        "Navigation",
        options=list(page_labels.keys()),
        format_func=lambda value: page_labels[value],
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:24px'></div><div style='border-top:1px solid #273149;padding-top:15px;color:#8290aa;font-size:11px;font-weight:700;letter-spacing:.12em'>MODEL STATUS</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding-top:11px;font-size:12px;color:#d8e0f1'><span class='dot'></span>Random Forest pipeline</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding-top:8px;font-size:11px;color:#8591aa'>Decision threshold: {THRESHOLD:.0%}</div>", unsafe_allow_html=True)

if page == "Overview":
    st.markdown("<div class='topline'><div class='brand'>◈ &nbsp; SENTINEL / FRAUD INTELLIGENCE</div><div class='status-pill'>● &nbsp; DASHBOARD ONLINE</div></div>", unsafe_allow_html=True)
    st.markdown("# Financial Fraud Detection")
    st.markdown("<div class='page-subtitle'>Transaction risk monitoring · Machine-learning insights · Analyst review workflow</div>", unsafe_allow_html=True)
    st.write("")

    history = get_history(10000)
    live_total = len(history)
    live_flagged = int((history["prediction"] == "Fraudulent").sum()) if live_total else 0
    live_genuine = live_total - live_flagged
    if not intl.empty:
        sample_total = int(intl["Total_Transactions"].sum())
        sample_fraud = int(intl["Fraud_Transactions"].sum())
        sample_genuine = sample_total - sample_fraud
        sample_rate = (sample_fraud / sample_total * 100) if sample_total else 0
    else:
        sample_total = sample_fraud = sample_genuine = 0
        sample_rate = 0

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Transactions analyzed (source sample)", f"{sample_total:,}", help="Aggregated feature-analysis CSV supplied in the project archive.")
    k2.metric("Fraud transactions (source sample)", f"{sample_fraud:,}", f"{sample_rate:.2f}% of sample", help="Historical aggregate analysis, not live confirmed fraud.")
    k3.metric("Predictions made in this app", f"{live_total:,}", f"{live_flagged:,} flagged · {live_genuine:,} genuine")
    k4.metric("Model decision threshold", f"{THRESHOLD:.0%}", "Risk-score cutoff")

    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Fraud detection overview</div><div class='card-subtitle'>Distribution from the supplied domestic / international feature analysis</div>", unsafe_allow_html=True)
            if sample_total:
                donut = go.Figure(data=[go.Pie(labels=["Genuine", "Fraudulent"], values=[sample_genuine, sample_fraud], hole=.70,
                    marker=dict(colors=["#35b8e6", "#965be7"], line=dict(color="#171c2c", width=3)),
                    textinfo="none", hovertemplate="%{label}: %{value:,}<extra></extra>")])
                donut.update_layout(template="plotly_dark", height=255, margin=dict(l=5,r=5,t=5,b=5), showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(text=f"{sample_rate:.1f}%<br><span style='font-size:10px;color:#9aa5bc'>FRAUD RATE</span>", x=.5,y=.5,showarrow=False,font=dict(size=23,color="#f2f4ff"))])
                st.plotly_chart(donut, use_container_width=True, config={"displayModeBar":False})
                a,b,c = st.columns(3)
                a.markdown(f"<div style='color:#9aa6bd;font-size:11px'>● Genuine</div><div style='font-size:18px;font-weight:700'>{sample_genuine:,}</div>", unsafe_allow_html=True)
                b.markdown(f"<div style='color:#9aa6bd;font-size:11px'>● Fraudulent</div><div style='font-size:18px;font-weight:700'>{sample_fraud:,}</div>", unsafe_allow_html=True)
                c.markdown(f"<div style='color:#9aa6bd;font-size:11px'>● Total</div><div style='font-size:18px;font-weight:700'>{sample_total:,}</div>", unsafe_allow_html=True)
            else:
                st.info("Feature-analysis CSV not found.")
    with right:
        with st.container(border=True):
            st.markdown("<div class='card-title'>System integrations</div><div class='card-subtitle'>Application components and local service readiness</div>", unsafe_allow_html=True)
            model_exists = MODEL_PATH.exists()
            db_exists = (ROOT / "fraud_predictions.db").exists()
            rows = [
                ("Machine-learning model", "Random Forest tuned candidate", model_exists),
                ("Prediction service", "Python inference pipeline", True),
                ("SQLite investigation store", "Local prediction history", True),
                ("FastAPI integration", "Available · start separately on port 8000", (ROOT/"backend"/"main.py").exists()),
                ("Feature analysis data", "Hourly + international CSV reports", hour_file.exists() and intl_file.exists()),
            ]
            for name, detail, ok in rows:
                state = "READY" if ok else "MISSING"
                color = "#42d6a4" if ok else "#ff927d"
                st.markdown(f"<div class='integration-row'><div><span class='dot' style='background:{color}'></span><b>{name}</b><div style='color:#8793ac;font-size:10px;margin:3px 0 0 18px'>{detail}</div></div><span style='color:{color};font-size:10px;font-weight:700'>{state}</span></div>", unsafe_allow_html=True)
            st.markdown("<div style='color:#8591aa;font-size:10px;margin-top:12px'>API is included in the project. To start it, run: <code>python -m uvicorn backend.main:app --reload</code></div>", unsafe_allow_html=True)

    st.write("")
    bottom_left, bottom_right = st.columns(2, gap="large")
    with bottom_left:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Fraud rate by transaction hour</div><div class='card-subtitle'>Hourly fraud percentage and transaction volume in the supplied analysis</div>", unsafe_allow_html=True)
            if not hourly.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hourly["Transaction_Hour"], y=hourly["Fraud_Rate_Percent"], mode="lines+markers", name="Fraud rate (%)", line=dict(color="#a56af1",width=3), marker=dict(size=6)))
                fig.add_trace(go.Bar(x=hourly["Transaction_Hour"], y=hourly["Total_Transactions"], name="Transactions", marker_color="rgba(58,174,226,.24)", yaxis="y2"))
                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=8,r=8,t=12,b=8), paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Hour of day",dtick=2,gridcolor="rgba(120,130,160,.10)"), yaxis=dict(title="Fraud rate (%)",gridcolor="rgba(120,130,160,.16)"),
                    yaxis2=dict(title="Transactions",overlaying="y",side="right",showgrid=False),legend=dict(orientation="h",y=1.12,x=0))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            else: st.info("Hourly analysis data not found.")
    with bottom_right:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Model performance comparison</div><div class='card-subtitle'>Accuracy, precision and recall from the supplied model comparison report</div>", unsafe_allow_html=True)
            if not comparison.empty:
                plot_df = comparison.melt(id_vars="Model", value_vars=[c for c in ["Accuracy","Precision","Recall"] if c in comparison.columns], var_name="Metric", value_name="Score")
                fig = px.bar(plot_df,x="Model",y="Score",color="Metric",barmode="group",template="plotly_dark",color_discrete_sequence=["#4b9be8","#9b63e8","#39c4b0"])
                fig.update_layout(height=300,margin=dict(l=8,r=8,t=12,b=8),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",yaxis=dict(tickformat=".0%",range=[0,1],gridcolor="rgba(120,130,160,.16)"),xaxis_title="",yaxis_title="Score",legend=dict(orientation="h",y=1.12,x=0))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            else: st.info("Model comparison report not found.")
    st.caption("Source note: charts use the supplied aggregate feature-analysis and model-comparison CSVs. Live KPI counters reflect predictions made in this application. A model flag is a risk signal for human review, not confirmation of fraud.")

elif page == "Predict Transaction":
    st.markdown("# Predict Transaction")
    st.caption("Enter transaction details to estimate fraud risk. A flagged result should be reviewed by an analyst.")
    with st.container(border=True):
        with st.form("predict_form"):
            a,b,c=st.columns(3)
            date=a.text_input("Transaction date and time","2024-02-20 14:30:00",help="Format: YYYY-MM-DD HH:MM:SS")
            amount=b.number_input("Transaction amount",min_value=0.01,value=50.0)
            avg=c.number_input("Average spend",min_value=0.01,value=100.0)
            merchant=a.selectbox("Merchant category",["Grocery","Electronics","Travel","Food","Shopping","Fuel","Healthcare","Other"])
            payment=b.selectbox("Payment method",["UPI","Credit Card","Debit Card","Net Banking","Wallet","Cash"])
            device=c.selectbox("Device type",["Mobile","Desktop","Tablet","POS"])
            location=a.text_input("Location","Delhi")
            international=b.selectbox("International transaction",[0,1],format_func=lambda x:"Yes" if x else "No")
            previous=c.number_input("Previous transactions",min_value=0,value=150,step=1)
            age=a.number_input("Account age (days)",min_value=0,value=1500,step=1)
            submitted=st.form_submit_button("Analyze transaction",type="primary",use_container_width=True)
    if submitted:
        tx={"Transaction_Date":date,"Transaction_Amount":amount,"Merchant_Category":merchant,"Payment_Method":payment,"Device_Type":device,"Location":location,"Is_International":international,"Previous_Transactions":previous,"Average_Spend":avg,"Account_Age_Days":age}
        try:
            result=predict_transaction(tx)
            probability=float(result["Fraud_Probability"])
            st.metric("Model-estimated fraud risk",f"{probability:.2f}%")
            if result["Prediction"]=="Fraudulent": st.error("Flagged for analyst review — elevated model-estimated risk.")
            else: st.success("Below the configured model threshold. Continue to follow normal controls.")
            st.json(result)
        except Exception as e: st.exception(e)

elif page == "Batch Prediction":
    st.markdown("# Batch Prediction")
    st.caption("Upload a transaction-level CSV to score multiple transactions.")
    required=["Transaction_Date","Transaction_Amount","Merchant_Category","Payment_Method","Device_Type","Location","Is_International","Previous_Transactions","Average_Spend","Account_Age_Days"]
    st.info("Required columns: " + ", ".join(required))
    uploaded=st.file_uploader("Choose transaction CSV",type=["csv"])
    if uploaded:
        df=pd.read_csv(uploaded); st.dataframe(df.head(),use_container_width=True)
        if st.button("Run batch predictions",type="primary"):
            missing=[c for c in required if c not in df.columns]
            if missing: st.error("Missing columns: "+", ".join(missing))
            elif df.empty: st.warning("The uploaded CSV is empty.")
            else:
                results=[]; bar=st.progress(0)
                for i,(_,row) in enumerate(df.iterrows()):
                    try: results.append(predict_transaction(row[required].to_dict()))
                    except Exception as e: results.append({"Fraud_Probability":None,"Prediction":"Error","Suggested_Action":str(e)})
                    bar.progress((i+1)/len(df))
                out=pd.concat([df.reset_index(drop=True),pd.DataFrame(results)],axis=1)
                st.dataframe(out,use_container_width=True)
                st.download_button("Download prediction results",out.to_csv(index=False).encode(),file_name="fraud_predictions.csv",mime="text/csv")

elif page == "Investigation History":
    st.markdown("# Investigation History")
    hist=get_history(2000)
    if hist.empty: st.info("No predictions yet. Use Predict Transaction or Batch Prediction first.")
    else:
        status=st.multiselect("Filter prediction",["Fraudulent","Genuine"],default=["Fraudulent","Genuine"])
        view=hist[hist["prediction"].isin(status)]
        st.dataframe(view,use_container_width=True,hide_index=True)
        st.download_button("Export history CSV",view.to_csv(index=False).encode(),file_name="fraud_investigation_history.csv",mime="text/csv")

else:
    st.markdown("# Model & Feature Analysis")
    st.caption("Evaluation reports and feature-level insights supplied with the project.")
    for fname in ["corrected_threshold_evaluation.csv","model_comparison.csv","tuned_model_evaluation.csv"]:
        path=ROOT/"reports"/fname
        if path.exists():
            with st.expander(fname,expanded=(fname=="model_comparison.csv")):
                st.dataframe(pd.read_csv(path),use_container_width=True,hide_index=True)
    if not importance.empty:
        with st.container(border=True):
            st.subheader("Feature importance")
            imp=importance.sort_values("Importance_Mean",ascending=True)
            fig=px.bar(imp,x="Importance_Mean",y="Feature",orientation="h",template="plotly_dark",color="Importance_Mean",color_continuous_scale=["#3aaedb","#a05de8"])
            fig.update_layout(height=450,margin=dict(l=8,r=8,t=10,b=8),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False,xaxis_title="Mean importance",yaxis_title="")
            st.plotly_chart(fig,use_container_width=True)
    st.warning("The supplied reports contain multiple evaluation runs. Use the report that matches the selected model and threshold when presenting model metrics.")
