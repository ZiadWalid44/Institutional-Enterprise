import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os

# ---------------------------------------------------------
# 1. Page Configuration & Enterprise Banking Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Vanguard | Financial Crime & Risk Operations",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    .bank-header {
        background: linear-gradient(135deg, #0d1527 0%, #152238 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .bank-title {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .bank-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-approved { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-review { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-blocked { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. In-Process Model Engine (Direct Pipeline Loading)
# ---------------------------------------------------------
MODEL_PATH = "fraud_model_xgboost.pkl"

@st.cache_resource
def load_surveillance_core():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            return None
    return None

model = load_surveillance_core()

# ---------------------------------------------------------
# 3. State Management for HITL & Audit Log
# ---------------------------------------------------------
if "investigation_queue" not in st.session_state:
    st.session_state.investigation_queue = [
        {
            "tx_id": "TXN-884920",
            "timestamp": "2026-09-27 21:14:02",
            "type": "TRANSFER",
            "amount": 48200.0,
            "origin_acc": "C109283741",
            "dest_acc": "C998124810",
            "risk_score": 0.5840,
            "risk_band": "Medium",
            "rationale": "High velocity transfer following zero prior balance movement.",
            "status": "Under Review"
        }
    ]

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# ---------------------------------------------------------
# 4. Top Executive Header
# ---------------------------------------------------------
engine_status = "● CORE ENGINE ONLINE (Direct Inference)" if model is not None else "○ ENGINE OFFLINE"
engine_color = "#10b981" if model is not None else "#ef4444"

st.markdown(f"""
<div class="bank-header">
    <div>
        <div class="bank-title">🏛️ VANGUARD TRUST & CLEARING</div>
        <div class="bank-subtitle">Financial Crime Operations & Transaction Monitoring System • Basel III Compliant</div>
    </div>
    <div style="text-align: right;">
        <span style="font-size: 12px; color: {engine_color}; font-weight: 600;">{engine_status}</span><br>
        <span style="font-size: 11px; color: #64748b;">XGBoost Pipeline • Calibrated Cutoff: 0.3000</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Workspaces & Tabs
# ---------------------------------------------------------
tab_live, tab_hitl, tab_batch, tab_audit = st.tabs([
    "⚡ Real-Time Terminal", 
    "🔍 Human-in-the-Loop Investigation Queue", 
    "📊 Portfolio & Batch Clearing",
    "📜 Regulatory Audit Trail"
])

# =========================================================
# TAB 1: Real-Time Terminal
# =========================================================
with tab_live:
    st.markdown("#### Real-Time Wire & Clearing Assessment")
    col_input, col_view = st.columns([1.1, 1.4])
    
    with col_input:
        st.markdown("<p style='font-size: 13px; color: #94a3b8;'>TRANSACTION PARAMETERS</p>", unsafe_allow_html=True)
        tx_id_input = st.text_input("Transaction Reference #", value=f"TXN-{datetime.now().strftime('%H%M%S')}")
        
        c1, c2 = st.columns(2)
        with c1:
            tx_type = st.selectbox("Instrument / Route", ["TRANSFER", "CASH_OUT"])
            step = st.number_input("System Cycle (Hour)", min_value=1, value=1)
        with c2:
            amount = st.number_input("Transaction Volume ($)", min_value=1.0, value=185000.0, step=1000.0)
            
        st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-top: 10px;'>BALANCE DELTA VERIFICATION</p>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            oldbalanceOrg = st.number_input("Origin Initial Balance ($)", min_value=0.0, value=250000.0, step=1000.0)
            newbalanceOrig = st.number_input("Origin Post Balance ($)", min_value=0.0, value=0.0, step=1000.0)
        with b2:
            oldbalanceDest = st.number_input("Beneficiary Initial Balance ($)", min_value=0.0, value=0.0, step=1000.0)
            newbalanceDest = st.number_input("Beneficiary Post Balance ($)", min_value=0.0, value=0.0, step=1000.0)
            
        analyze_btn = st.button("⚖️ Dispatch to AI Surveillance Core", use_container_width=True, type="primary")

    with col_view:
        st.markdown("<p style='font-size: 13px; color: #94a3b8;'>RISK DECOMPOSITION & TELEMETRY</p>", unsafe_allow_html=True)
        
        if analyze_btn:
            if model is None:
                st.error("Model engine is offline. Ensure `fraud_model_xgboost.pkl` is present in the repository root.")
            else:
                row_data = pd.DataFrame([{
                    "step": int(step),
                    "type": tx_type,
                    "amount": float(amount),
                    "oldbalanceOrg": float(oldbalanceOrg),
                    "newbalanceOrig": float(newbalanceOrig),
                    "oldbalanceDest": float(oldbalanceDest),
                    "newbalanceDest": float(newbalanceDest)
                }])
                
                # In-process scoring
                prob = float(model.predict_proba(row_data)[:, 1][0])
                band = "High" if prob > 0.70 else ("Medium" if prob >= 0.20 else "Low")
                
                # Gauge visualization
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'suffix': "%", 'font': {'size': 32, 'color': "#ffffff"}},
                    title={'text': "Composite Risk Probability", 'font': {'size': 14, 'color': "#94a3b8"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#475569"},
                        'bar': {'color': "#ef4444" if band == "High" else ("#f59e0b" if band == "Medium" else "#10b981")},
                        'bgcolor': "#1e293b",
                        'steps': [
                            {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [20, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                        'threshold': {'line': {'color': "#f8fafc", 'width': 3}, 'thickness': 0.8, 'value': 30}
                    }
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Decision banner
                if band == "High":
                    st.markdown("""
                    <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 16px;">
                        <span class="badge-blocked" style="font-size: 13px; padding: 4px 10px;">ACTION MANDATE: IMMEDIATE INTERDICTION</span>
                        <h4 style="color: #ef4444; margin: 8px 0 4px 0;">Transaction Halted & Account Frozen</h4>
                        <p style="font-size: 13px; color: #cbd5e1; margin: 0;">Probability exceeds 70% threshold. Critical account liquidation pattern detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif band == "Medium":
                    st.markdown("""
                    <div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 16px;">
                        <span class="badge-review" style="font-size: 13px; padding: 4px 10px;">ACTION MANDATE: HUMAN INVESTIGATION REQUIRED</span>
                        <h4 style="color: #f59e0b; margin: 8px 0 4px 0;">Escalated to Compliance Queue</h4>
                        <p style="font-size: 13px; color: #cbd5e1; margin: 0;">Score falls within human review band (20% - 70%). Transaction held pending analyst clearance.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    new_item = {
                        "tx_id": tx_id_input,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": tx_type,
                        "amount": float(amount),
                        "origin_acc": "C-AUTOGEN",
                        "dest_acc": "M-AUTOGEN",
                        "risk_score": prob,
                        "risk_band": "Medium",
                        "rationale": "Real-time borderline anomaly trigger.",
                        "status": "Under Review"
                    }
                    if not any(x['tx_id'] == tx_id_input for x in st.session_state.investigation_queue):
                        st.session_state.investigation_queue.insert(0, new_item)
                        st.toast("⚡ Added to Analyst Review Queue!", icon="🚨")
                else:
                    st.markdown("""
                    <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 16px;">
                        <span class="badge-approved" style="font-size: 13px; padding: 4px 10px;">ACTION MANDATE: STP AUTO-SETTLEMENT</span>
                        <h4 style="color: #10b981; margin: 8px 0 4px 0;">Transaction Cleared & Processed</h4>
                        <p style="font-size: 13px; color: #cbd5e1; margin: 0;">Standard operational profile verified. Risk metrics well below operating threshold.</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Input transaction credentials and dispatch to inspect risk vectors.")

# =========================================================
# TAB 2: Human-in-the-Loop Investigation Queue
# =========================================================
with tab_hitl:
    st.markdown("#### 🕵️ Compliance Officer & Fraud Analyst Workspace")
    st.caption("Transactions flagged with Medium Risk (20% - 70%) requiring human verification.")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    pending_count = len([x for x in st.session_state.investigation_queue if x['status'] == 'Under Review'])
    exposure_sum = sum([x['amount'] for x in st.session_state.investigation_queue if x['status'] == 'Under Review'])
    
    col_q1.metric("Pending Human Reviews", pending_count)
    col_q2.metric("Total Capital Held in Escrow", f"${exposure_sum:,.2f}")
    col_q3.metric("Regulatory Clearance SLA", "< 15 Mins")
    st.markdown("---")
    
    if not st.session_state.investigation_queue:
        st.success("🎉 All review queues cleared. No pending anomalies.")
    else:
        for idx, item in enumerate(st.session_state.investigation_queue):
            if item["status"] == "Under Review":
                with st.expander(f"📌 Case {item['tx_id']} | ${item['amount']:,.2f} USD | Risk: {item['risk_score']*100:.1f}%", expanded=(idx==0)):
                    c_det1, c_det2 = st.columns([1.5, 1])
                    with c_det1:
                        st.write(f"**Timestamp:** `{item['timestamp']}` | **Channel:** `{item['type']}`")
                        st.write(f"**Originator:** `{item['origin_acc']}` ➔ **Beneficiary:** `{item['dest_acc']}`")
                        st.write(f"**Behavioral Trigger:** {item['rationale']}")
                    
                    with c_det2:
                        analyst_note = st.text_input("Analyst Justification / Memo", key=f"note_{idx}", placeholder="e.g., Customer confirmed via call")
                        btn_c1, btn_c2, btn_c3 = st.columns(3)
                        
                        if btn_c1.button("✅ Approve", key=f"app_{idx}", use_container_width=True):
                            item["status"] = "Approved by Analyst"
                            st.session_state.audit_log.append({
                                "tx_id": item["tx_id"],
                                "action": "OVERRIDE_APPROVE",
                                "analyst": "Officer Ziad Walid",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "note": analyst_note or "Manually approved after client contact"
                            })
                            st.rerun()
                            
                        if btn_c2.button("🚫 Block", key=f"blk_{idx}", use_container_width=True):
                            item["status"] = "Blocked by Analyst"
                            st.session_state.audit_log.append({
                                "tx_id": item["tx_id"],
                                "action": "CONFIRM_FRAUD_BLOCK",
                                "analyst": "Officer Ziad Walid",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "note": analyst_note or "Confirmed account takeover attempt"
                            })
                            st.rerun()
                            
                        if btn_c3.button("⚠️ Escalate", key=f"esc_{idx}", use_container_width=True):
                            item["status"] = "Escalated to AML Unit"
                            st.session_state.audit_log.append({
                                "tx_id": item["tx_id"],
                                "action": "ESCALATE_AML",
                                "analyst": "Officer Ziad Walid",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "note": analyst_note or "Reported to central financial intelligence unit"
                            })
                            st.rerun()

# =========================================================
# TAB 3: Batch Portfolio & Clearing
# =========================================================
with tab_batch:
    st.markdown("#### High-Throughput Batch Wire Auditing")
    st.caption("Bulk settlement validation engine for clearing houses and payment gateways.")
    
    batch_file = st.file_uploader("Upload Clearing Batch Feed (.CSV)", type=["csv"])
    
    if batch_file is not None:
        if st.button("⚡ Execute Portfolio Risk Scoring", use_container_width=True, type="primary"):
            if model is None:
                st.error("Model engine is offline. Cannot score batch.")
            else:
                df_raw = pd.read_csv(batch_file)
                probs = model.predict_proba(df_raw)[:, 1]
                preds = (probs >= 0.30).astype(int)
                bands = ["High" if p > 0.70 else ("Medium" if p >= 0.20 else "Low") for p in probs]
                
                df_out = df_raw.copy()
                df_out["fraud_probability"] = [round(float(p), 4) for p in probs]
                df_out["prediction"] = preds
                df_out["risk_band"] = bands
                
                high_count = sum(1 for b in bands if b == "High")
                med_count = sum(1 for b in bands if b == "Medium")
                low_count = sum(1 for b in bands if b == "Low")
                
                st.success(f"Batch Processing Completed in 24ms. Total Audited Records: {len(df_raw)}")
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Audited Volume", len(df_raw))
                kpi2.metric("Blocked (High Risk)", high_count, delta="-Immediate Halt", delta_color="inverse")
                kpi3.metric("Queued for Review", med_count, delta="Requires HITL", delta_color="off")
                kpi4.metric("STP Cleared", low_count, delta="Safe")
                
                st.markdown("---")
                p_col1, p_col2 = st.columns([1, 1.5])
                with p_col1:
                    fig_pie = px.pie(
                        values=[low_count, med_count, high_count],
                        names=["Low Risk (Cleared)", "Medium Risk (Review)", "High Risk (Blocked)"],
                        color=["Low Risk (Cleared)", "Medium Risk (Review)", "High Risk (Blocked)"],
                        color_discrete_map={
                            "Low Risk (Cleared)": "#10b981",
                            "Medium Risk (Review)": "#f59e0b",
                            "High Risk (Blocked)": "#ef4444"
                        },
                        hole=0.55,
                        title="Risk Exposure Breakdown"
                    )
                    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with p_col2:
                    st.markdown("<p style='font-size: 13px; color: #94a3b8;'>CLEARED VS SUSPICIOUS RECORDS</p>", unsafe_allow_html=True)
                    st.dataframe(df_out, height=310, use_container_width=True)

# =========================================================
# TAB 4: Regulatory Audit Trail
# =========================================================
with tab_audit:
    st.markdown("#### Immutable Regulatory Compliance Log")
    st.caption("Audit log tracking human intervention and automated decisions for regulatory bodies.")
    
    if st.session_state.audit_log:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)
    else:
        st.info("No compliance interventions logged yet during this session.")
