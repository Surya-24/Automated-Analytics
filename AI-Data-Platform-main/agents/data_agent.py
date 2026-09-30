import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.observability import get_active_callbacks
from Models.schema import RouterSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst

llm = pick_llm("medium")
llm_router = llm.with_structured_output(RouterSchema)


# ---------------------------- DATA AGENT GRAPH ---------------------------- #

def router_node(state: DataAgentSchema):
    message = state.messages[-1].content
    callbacks = get_active_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    try:
        route_response_dict = llm_router.invoke(
            f"Classify whether this prompt is an SQL database query or an ETL data extraction/transformation task: {message}",
            config=config
        ).model_dump()
        route_response = route_response_dict['answer']
    except Exception:
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["extract", "transform", "api", "pokeapi", "parquet", "load"]):
            route_response = "etl"
        else:
            route_response = "sql"

    state.route_response = route_response
    return state


def etl_node(state: DataAgentSchema):
    message = state.messages[-1].content
    callbacks = get_active_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    response = etl_analyst.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    
    if "messages" in response and response["messages"]:
        last_msg = response["messages"][-1]
        state.messages = state.messages + [last_msg]
    else:
        state.messages = state.messages + [AIMessage(content=str(response))]

    return state


def sql_node(state: DataAgentSchema):
    message = state.messages[-1].content
    callbacks = get_active_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}

    input_schema = {
        "messages": [],
        "user_question": f"{message}",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
        "retry_count": 0,
        "error_trace": "",
        "structured_data": None,
        "chart_spec": None
    }

    response = sql_analyst.invoke(input_schema, config=config)
    
    if isinstance(response, dict):
        state.structured_data = response.get("structured_data")
        final_answer = response.get("final_answer", "")
        state.messages = state.messages + [AIMessage(content=final_answer)]
    elif hasattr(response, "final_answer"):
        state.structured_data = getattr(response, "structured_data", None)
        state.messages = state.messages + [AIMessage(content=response.final_answer)]
    else:
        state.messages = state.messages + [AIMessage(content=str(response))]

    return state


data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        return "sql_node"

data_agent_graph.add_conditional_edges(
    "router_node",
    route_edge,
    {
        "sql_node": "sql_node",
        "etl_node": "etl_node"
    }
)

data_agent_graph.add_edge("sql_node", END)
data_agent_graph.add_edge("etl_node", END)

data_agent = data_agent_graph.compile()
