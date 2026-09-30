# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from langchain_core.messages import HumanMessage, AIMessage
from agents.data_agent import data_agent
from utils.observability import get_active_callbacks

st.set_page_config(
    page_title="AI Data Agent | Multi-Agent Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-sql {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-etl {
        background-color: #ECFDF5;
        color: #047857;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .stMetric {
        background: #F8FAFC;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/database.png", width=60)
    st.markdown("### 🤖 Architecture Specs")
    st.markdown("""
    - **Orchestration:** LangGraph (Hierarchical StateGraph)
    - **Safety:** AST-level Parser (`sqlglot`) + LLM Judge
    - **Self-Healing:** 3-Retry Auto-Repair Loop
    - **Sandbox:** AST Python Execution Guard
    - **Storage:** PostgreSQL + CSV/Parquet
    """)
    st.divider()
    st.markdown("### 💡 Quick Prompt Suggestions")
    sample_prompts = [
        "Show the average rating per vehicle type",
        "What are the top 5 payment methods by total amount?",
        "Extract Pokemon API data and save as parquet",
        "Show list of inactive users in Halifax"
    ]
    for sp in sample_prompts:
        if st.button(sp, key=f"btn_{sp}"):
            st.session_state["preset_input"] = sp

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-title">⚡ Autonomous AI Data Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Natural Language SQL Analytics & Autonomous ETL Pipelines powered by LangGraph</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            badge_class = "badge-sql" if msg["route"] == "sql" else "badge-etl"
            st.markdown(f'<span class="{badge_class}">Active Agent: {msg["route"].upper()}</span>', unsafe_allow_html=True)
        
        st.markdown(msg["content"])
        
        # Display dataframes if stored in history
        if "data" in msg and msg["data"]:
            df_history = pd.DataFrame(msg["data"])
            st.dataframe(df_history, width='stretch')
            
            # Show chart if applicable
            if "chart" in msg and msg["chart"]:
                st.plotly_chart(msg["chart"], width='stretch')
                
            csv_bytes = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name="query_results.csv",
                mime="text/csv",
                key=f"dl_{msg.get('id', id(msg))}"
            )

# Handle input (either from chat_input or sidebar preset)
preset = st.session_state.pop("preset_input", None)
user_prompt = st.chat_input("Ask a data question or request an ETL job...") or preset

if user_prompt:
    # 1. Render User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Execute Agent Workflow
    with st.chat_message("assistant"):
        status_box = st.status("🧠 Multi-agent workflow executing...", expanded=True)
        try:
            status_box.write("🔄 **Router Node:** Classifying user intent...")
            
            callbacks = get_active_callbacks()
            invoke_config = {"callbacks": callbacks} if callbacks else {}
            response = data_agent.invoke({
                "messages": [HumanMessage(content=user_prompt)],
                "route_response": ""
            }, config=invoke_config)
            
            route = response.get("route_response", "sql")
            status_box.write(f"✅ Routed to: **{route.upper()} Analyst Agent**")
            
            if route == "sql":
                status_box.write("🛡️ Validating SQL safety via AST parser & LLM Judge...")
                status_box.write("⚡ Executing query on PostgreSQL database...")
            else:
                status_box.write("📦 Running sandboxed ETL execution...")

            status_box.update(label="✨ Analysis Complete!", state="complete", expanded=False)

            # Extract response details
            messages = response.get("messages", [])
            last_msg = messages[-1] if messages else None
            output_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            structured_data = response.get("structured_data")

            badge_class = "badge-sql" if route == "sql" else "badge-etl"
            st.markdown(f'<span class="{badge_class}">Active Agent: {route.upper()}</span>', unsafe_allow_html=True)
            st.markdown(output_text)

            chart_obj = None
            # Render structured table & auto-visualizations if data is present
            if structured_data and len(structured_data) > 0:
                df = pd.DataFrame(structured_data)
                st.markdown("#### 📊 Query Result Preview")
                st.dataframe(df, width='stretch')

                # Auto-generate chart if numerical & categorical columns exist
                num_cols = df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

                # If numeric values exist with categories, render Plotly chart
                if len(num_cols) >= 1 and len(cat_cols) >= 1 and len(df) <= 25:
                    st.markdown("#### 📈 Interactive Visualization")
                    fig = px.bar(
                        df,
                        x=cat_cols[0],
                        y=num_cols[0],
                        title=f"{num_cols[0].replace('_', ' ').title()} by {cat_cols[0].replace('_', ' ').title()}",
                        color=cat_cols[0],
                        template="plotly_white"
                    )
                    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, width='stretch')
                    chart_obj = fig

                # 1-Click CSV Download Button
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv_bytes,
                    file_name="data_agent_export.csv",
                    mime="text/csv"
                )

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": output_text,
                "route": route,
                "data": structured_data,
                "chart": chart_obj,
                "id": len(st.session_state.messages)
            })

        except Exception as e:
            status_box.update(label="❌ Execution Failed", state="error", expanded=True)
            st.error(f"Error during agent execution: {e}")
