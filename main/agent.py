import os
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
from operator import add
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# ================= ENV =================

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ================= STATE =================

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
    is_travel_related: bool
    intent: str
    destination: Optional[str]
    days: Optional[int]
    budget_type: Optional[str]
    place1: Optional[str]
    place2: Optional[str]
    itinerary_text: Optional[str]
    budget_text: Optional[str]
    risk_text: Optional[str]
    combined: Optional[bool]


# ================= SCHEMAS =================

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


# ================= HELPERS =================

def extract_text(message):
    if isinstance(message, BaseMessage):
        return message.content
    if isinstance(message, dict):
        return message.get("content", "")
    return str(message)


def search(query: str) -> str:
    try:
        result = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )
        return "\n".join(
            r.get("content", "") for r in result.get("results", [])
        )[:4000]
    except Exception:
        return ""


def call_llm(state: ChatState, prompt: str, schema=None):
    model = llm.with_structured_output(schema) if schema else llm
    return model.invoke(
        state["messages"] + [SystemMessage(content=prompt)]
    )


# ================= NODES =================

def classify_node(state: ChatState):
    result = call_llm(
        state,
        """
        You are a STRICT travel query classifier.

        Always return all required fields.

        If NOT travel related:
        - is_travel_related = false
        - intent = general
        - destination = null
        - days = null
        - budget_type = null
        - place1 = null
        - place2 = null

        Travel-related includes itineraries, hotels, visas,
        attractions, comparisons, and emergency contact numbers.
        Examples of NOT travel related:Science questions ,Biology question

        If user asks for police, ambulance, fire, hospital,
        SOS or emergency numbers → intent MUST be emergency.

        Return structured output only.
        """,
        QuerySchema,
    )

    data = result.model_dump()
    data["intent"] = data["intent"].value
    return data


def route_query(state: ChatState):
    return state.get("intent") if state.get("is_travel_related") else "non_travel"


# ====== ITINERARY FLOW ======

def itinerary_node(state: ChatState):
    destination = state.get("destination")
    days = state.get("days", 3)

    response = call_llm(
        state,
        f"Create a {days}-day travel itinerary for {destination}. No pricing."
    )

    return {"itinerary_text": response.content}


def budget_node(state: ChatState):
    destination = state.get("destination")
    days = state.get("days", 3)

    info = search(f"Average mid range travel cost per day in {destination}")

    response = call_llm(
        state,
        f"Estimate total budget for {days} days in {destination}.\n{info}"
    )

    return {"budget_text": response.content}


def risk_node(state: ChatState):
    destination = state.get("destination")

    info = search(f"Current official travel advisory for {destination}")

    result = call_llm(
        state,
        f"Analyze travel risks for {destination}.\n{info}",
        RiskSchema,
    )

    data = result.model_dump()

    if not data["has_active_risk"]:
        return {"risk_text": "No major active travel advisories."}

    risks = "\n".join(f"- {r}" for r in data.get("risk_details", []))
    return {"risk_text": f"Travel Advisory:\n{risks}"}


def combine_node(state: ChatState):
    if (
        state.get("combined")
        or not state.get("budget_text")
        or not state.get("risk_text")
    ):
        return {}

    final = f"""
🗺 Itinerary
{state.get("itinerary_text","").strip()}

💰 Budget Estimate
{state.get("budget_text","").strip()}

⚠ Travel Advisory
{state.get("risk_text","").strip()}
"""

    return {"messages": [AIMessage(content=final)], "combined": True}


# ====== SINGLE EXECUTOR ======

def executor_node(state: ChatState):
    question = extract_text(state["messages"][-1])
    intent = state.get("intent")

    info = search(question)

    if intent == "emergency":
        prompt = f"""
        Provide ONLY official emergency contact numbers for {state.get("destination")}.

        Include:
        - Police
        - Ambulance
        - Fire
        - General emergency number

        Do NOT describe disasters.
        Do NOT provide history.
        Use verified information:
        {info}
        """
    else:
        prompt = f"""
        You are a professional travel assistant.
        Answer clearly and concisely.

        Question:
        {question}

        Verified Info:
        {info}
        """

    response = call_llm(state, prompt)

    return {"messages": [AIMessage(content=response.content)]}


def non_travel_node(state: ChatState):
    return {
        "messages": [
            AIMessage(content="I specialize in travel-related queries.")
        ]
    }


# ================= GRAPH =================

uncompiled_graph = StateGraph(ChatState)

uncompiled_graph.add_node("classify", classify_node)
uncompiled_graph.add_node("itinerary", itinerary_node)
uncompiled_graph.add_node("budget", budget_node)
uncompiled_graph.add_node("risk", risk_node)
uncompiled_graph.add_node("combine", combine_node)
uncompiled_graph.add_node("executor", executor_node)
uncompiled_graph.add_node("non_travel", non_travel_node)

uncompiled_graph.set_entry_point("classify")

uncompiled_graph.add_conditional_edges(
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
    },
)

uncompiled_graph.add_edge("itinerary", "budget")
uncompiled_graph.add_edge("itinerary", "risk")

uncompiled_graph.add_edge("budget", "combine")
uncompiled_graph.add_edge("risk", "combine")

uncompiled_graph.add_edge("combine", END)
uncompiled_graph.add_edge("executor", END)
uncompiled_graph.add_edge("non_travel", END)

graph = uncompiled_graph.compile()