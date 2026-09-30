import os
import sys
import json
import plotly.express as px
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm, extract_text
from utils.database import DatabaseUtil
from utils.sql_validator import validate_sql_safety
from Models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


# -------------------------------------- AI Agent Nodes --------------------------------------

def curate_ques(state: AgentSchema) -> AgentSchema: 
    user_question = state.user_question
    llm = pick_llm("low")
    response = extract_text(llm.invoke(f"Curate and clarify the following data question for SQL analysis: {user_question}").content)

    state.curated_ques = response
    state.messages = state.messages + [HumanMessage(content=f"{response}")]
    return state 


def prompt_query_context(state: AgentSchema) -> AgentSchema:
    curated_question = state.curated_ques

    conn_details = {
        "host": os.environ.get('host', 'localhost'),
        "port": os.environ.get('port', 5432),
        "user": os.environ.get('user', 'postgres'),
        "password": os.environ.get('password', 'potgres'),
        "dbname": os.environ.get('database', os.environ.get('dbname', 'postgres'))
    }
    if os.environ.get('sslmode'):
        conn_details['sslmode'] = os.environ.get('sslmode')

    obj = DatabaseUtil(conn_details)
    schema_info = obj.schema_details("public")

    prompt = f"""
    You are an expert PostgreSQL Analyst. Convert the user's question into an efficient, valid PostgreSQL query.
    
    Database Schema Details:
    {schema_info}
    
    User Query: {curated_question}
    
    Rules:
    1. Output ONLY the raw SQL query without markdown code blocks, backticks, or explanation.
    2. Unless the user explicitly requests all rows, limit results to 10 rows.
    3. Generate standard SELECT statements only.
    """

    state.prompt_query_context = prompt
    return state


def generate_sql(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context
    
    # If this is a self-correction retry, append the error context
    if state.error_trace:
        prompt += f"""
        
        CRITICAL: Your previous query attempt failed with the following database error:
        Failed Query: {state.generated_sql_query}
        Error Message: {state.error_trace}
        
        Please inspect the schema carefully, correct any invalid column names, table aliases, or syntax, and provide a fixed PostgreSQL query.
        """

    llm = pick_llm("medium")
    generated_sql = extract_text(llm.invoke(prompt).content).strip()
    
    # Clean possible markdown ticks
    cleaned_sql = generated_sql.strip('`').lstrip('sql').strip()
    state.generated_sql_query = cleaned_sql
    return state


def is_safe_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    # Stage 1: Deterministic AST Validation
    is_ast_safe, ast_reason = validate_sql_safety(sql_query)
    if not is_ast_safe:
        state.is_safe = "No"
        state.comments = f"Deterministic AST check failed: {ast_reason}"
        return state

    # Stage 2: LLM-as-Judge Validation
    llm = pick_llm("medium")  
    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = f"""
    You are an SQL Security Judge. Evaluate if this SQL query is strictly read-only:
    Query: {sql_query}
    
    The query must NOT contain INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, or system commands.
    Respond with 'Yes' if safe, otherwise 'No' and explain.
    """

    try:
        response = llm_judge.invoke(prompt).model_dump()
        state.is_safe = response['answer']
        state.comments = response['comments']
    except Exception as e:
        # Fallback to AST decision if structured judge fails
        state.is_safe = "Yes" if is_ast_safe else "No"
        state.comments = f"AST Validated. Judge error: {e}"

    return state


def canceled_sql(state: AgentSchema) -> AgentSchema:
    comments = state.comments
    state.final_answer = f"The generated SQL query was deemed unsafe to execute. Reason: {comments}. Execution was canceled for safety."
    state.messages = state.messages + [AIMessage(content=state.final_answer)]
    return state


def execute_sql(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    conn_details = {
        "host": os.environ.get('host', 'localhost'),
        "port": os.environ.get('port', 5432),
        "user": os.environ.get('user', 'postgres'),
        "password": os.environ.get('password', 'potgres'),
        "dbname": os.environ.get('database', os.environ.get('dbname', 'postgres'))
    }
    if os.environ.get('sslmode'):
        conn_details['sslmode'] = os.environ.get('sslmode')

    obj = DatabaseUtil(conn_details)
    rows_dict, err = obj.execute_sql(sql_query)

    if err:
        state.error_trace = err
        state.retry_count += 1
        state.sql_query_execution_result = f"Error: {err}"
        state.structured_data = None
    else:
        state.error_trace = ""
        state.structured_data = rows_dict
        state.sql_query_execution_result = json.dumps(rows_dict, default=str)

    return state


def auto_repair_decision(state: AgentSchema) -> str:
    """
    Conditional routing for self-correction:
    If execution failed and retry_count <= 2, loop back to generate_sql.
    Otherwise proceed to represent final answer.
    """
    if state.error_trace and state.retry_count <= 2:
        return "generate_sql"
    return "represent_final_answer"


def represent_final_answer(state: AgentSchema) -> AgentSchema:
    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques
    has_error = bool(state.error_trace)

    llm = pick_llm("low")

    prompt = f"""
    You are an SQL Data Analyst. Present a clear, helpful, and concise answer to the user based on the execution result.
    
    User's Question: {curated_question}
    Execution Result: {execution_result}
    Has Execution Error: {has_error}
    
    If the query returned data, summarize the insights clearly without mentioning raw SQL code.
    If an error occurred after retries, explain what went wrong politely and suggest how the user might rephrase.
    """

    llm_response = extract_text(llm.invoke(prompt).content)
    state.final_answer = llm_response
    state.messages = state.messages + [AIMessage(content=llm_response)]
    return state


# ------------------------------------------- Graph Building -------------------------------------------

sql_agent_graph = StateGraph(AgentSchema)

# Nodes
sql_agent_graph.add_node("curate_ques", curate_ques)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe_sql", is_safe_sql)
sql_agent_graph.add_node("canceled_sql", canceled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_final_answer", represent_final_answer)

# Edges
sql_agent_graph.add_edge(START, "curate_ques")
sql_agent_graph.add_edge("curate_ques", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

def is_safe_sql_edge(state: AgentSchema) -> str:
    if state.is_safe.lower() == "yes":
        return "execute_sql"
    return "canceled_sql"

sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    is_safe_sql_edge,
    {
        "execute_sql": "execute_sql",
        "canceled_sql": "canceled_sql"
    }
)

# Self-Correction Retry Loop from execute_sql
sql_agent_graph.add_conditional_edges(
    "execute_sql",
    auto_repair_decision,
    {
        "generate_sql": "generate_sql",
        "represent_final_answer": "represent_final_answer"
    }
)

sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("represent_final_answer", END)

# Compile Graph
sql_analyst = sql_agent_graph.compile()
