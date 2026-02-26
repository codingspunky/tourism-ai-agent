import os
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
from operator import add
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class IntentType(str, Enum):
    itinerary = "itinerary"
    visa = "visa"
    hotel = "hotel"
    comparison = "comparison"
    emergency = "emergency"
    attraction = "attraction"
    general = "general"

class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add]

    # classification
    is_travel_related: bool
    intent: str
    destination: Optional[str]
    days: Optional[int]
    budget_type: Optional[str]
    place1: Optional[str]
    place2: Optional[str]

    # parallel results
    itinerary_text: Optional[str]
    budget_text: Optional[str]
    risk_text: Optional[str]

class QuerySchema(BaseModel):
    is_travel_related: bool
    intent: IntentType
    destination: Optional[str] = None
    days: Optional[int] = None
    budget_type: Optional[str] = None
    place1: Optional[str] = None
    place2: Optional[str] = None


class RiskSchema(BaseModel):
    has_active_risk: bool
    risk_details: Optional[List[str]] = None

def run_search(query: str, max_results: int = 5) -> str:
    try:
        result = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
        )
        return "\n".join(
            r.get("content", "") for r in result.get("results", [])
        )[:4000]
    except Exception:
        return ""
    
def classify_node(state: ChatState):
    structured_llm = llm.with_structured_output(QuerySchema)

    system_prompt = """
    You are a travel query classifier.

    Travel-related includes:
    - trip planning
    - itineraries
    - hotels
    - visas
    - attractions
    - comparisons
    - travel safety
    - emergency contact numbers in a destination
    - medical emergency numbers in a city
    - police or ambulance numbers for travelers

    Determine:
    - is_travel_related (true/false)
    - intent: itinerary, visa, hotel, comparison, emergency, attraction, general
    - Extract destination, days, budget_type, place1, place2

    If the question asks for emergency numbers in a city,
    intent MUST be: emergency.
    """

    result = structured_llm.invoke(
        state["messages"] + [SystemMessage(content=system_prompt)]
    )

    data = result.model_dump()
    data["intent"] = data["intent"].value
    return data
def route_query(state: ChatState):
    if not state.get("is_travel_related"):
        return "non_travel"
    return state.get("intent", "general")

def itinerary_node(state: ChatState):
    destination = state.get("destination")
    days = state.get("days") or 3

    prompt = f"""
    Create a realistic {days}-day travel itinerary for {destination}.
    No prices.
    Clear daily structure.
    """

    response = llm.invoke(
        state["messages"] + [SystemMessage(content=prompt)]
    )

    return {"itinerary_text": response.content}
def budget_node(state: ChatState):
    destination = state.get("destination")
    days = state.get("days") or 3

    search_text = run_search(
        f"Average mid range travel cost per day in {destination}"
    )

    prompt = f"""
    Based on verified information below,
    estimate total budget for {days} days in {destination}.
    Provide breakdown and total.

    {search_text}
    """

    response = llm.invoke(
        state["messages"] + [SystemMessage(content=prompt)]
    )

    return {"budget_text": response.content}

def risk_node(state: ChatState):
    destination = state.get("destination")

    search_text = run_search(
        f"Current official travel advisory for {destination}"
    )

    structured_llm = llm.with_structured_output(RiskSchema)

    prompt = f"""
    Analyze if active travel risks exist for {destination}.
    If yes, list briefly.

    {search_text}
    """

    result = structured_llm.invoke(
        state["messages"] + [SystemMessage(content=prompt)]
    )

    data = result.model_dump()

    if not data["has_active_risk"]:
        return {"risk_text": "No major active travel advisories."}

    risks = "\n".join(f"- {r}" for r in data.get("risk_details", []))
    return {"risk_text": f"Travel Advisory:\n{risks}"}
    
def combine_node(state: ChatState):
    if not state.get("budget_text") or not state.get("risk_text"):
        return {}

    final = f"""
{state.get("itinerary_text", "")}

Budget Estimate:
{state.get("budget_text", "")}

Travel Advisory:
{state.get("risk_text", "")}
"""

    return {"messages": [AIMessage(content=final)]}

def executor_node(state: ChatState):
    last_message = state["messages"][-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    search_text = run_search(question)

    prompt = f"""
    You are a professional travel assistant.
    Answer clearly and concisely.
    Remove website junk.

    Question:
    {question}

    Verified Info:
    {search_text}
    """

    response = llm.invoke(
        state["messages"] + [SystemMessage(content=prompt)]
    )

    return {
        "messages": [AIMessage(content=response.content)]
    }

def non_travel_node(state: ChatState):
    return {
        "messages": [
            AIMessage(content="I specialize only in travel-related queries.")
        ]
    }
builder = StateGraph(ChatState)

builder.add_node("classify", classify_node)
builder.add_node("itinerary", itinerary_node)
builder.add_node("budget", budget_node)
builder.add_node("risk", risk_node)
builder.add_node("combine", combine_node)
builder.add_node("executor", executor_node)
builder.add_node("non_travel", non_travel_node)

builder.set_entry_point("classify")

builder.add_conditional_edges(
    "classify",
    route_query,
    {
        "itinerary": "itinerary",
        "visa": "executor",
        "hotel": "executor",
        "comparison": "executor",
        "emergency": "executor",
        "attraction": "executor",
        "general": "executor",
        "non_travel": "non_travel",
    }
)

# Parallel fan-out
builder.add_edge("itinerary", "budget")
builder.add_edge("itinerary", "risk")

# Merge
builder.add_edge("budget", "combine")
builder.add_edge("risk", "combine")

# End
builder.add_edge("combine", END)
builder.add_edge("executor", END)
builder.add_edge("non_travel", END)

graph = builder.compile()