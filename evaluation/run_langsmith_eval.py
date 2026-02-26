import sys
import os
import re

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from main.agent import graph

# -----------------------------------
# LangSmith Client
# -----------------------------------
client = Client()

DATASET_NAME = "tourism-eval"

# -----------------------------------
# Evaluation LLM (Groq Judge Model)
# -----------------------------------
eval_llm = ChatGroq(
    model="llama3-70b-8192",   # use same model as your agent
    temperature=0
)

# -----------------------------------
# App Wrapper (Your Agent)
# -----------------------------------
def app(inputs: dict) -> dict:
    question = inputs.get("input", "")

    state = {
        "user_id": "eval_user",
        "messages": [HumanMessage(content=question)],
        "approved": True,
        "allowed": True
    }

    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": "eval_user"}}
    )

    messages = result.get("messages", [])

    if not messages:
        return {"output": ""}

    return {"output": messages[-1].content}


# ===================================
# 1️⃣ LLM-Based Relevance Evaluator
# ===================================
def relevance_evaluator(run, example):
    prediction = run.outputs.get("output", "")
    question = example.inputs.get("input", "")

    if not prediction:
        return {"score": 0, "key": "relevance"}

    prompt = f"""
You are an expert evaluator.

Question:
{question}

Answer:
{prediction}

Rate the answer on relevance, helpfulness, and completeness.

1 = Completely irrelevant or poor
3 = Partially helpful
5 = Highly relevant and helpful

Return only a number.
"""

    try:
        response = eval_llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        # Extract first number from response safely
        match = re.search(r"\d+", text)
        score = float(match.group()) if match else 3

        return {"score": score, "key": "relevance"}

    except Exception:
        return {"score": 3, "key": "relevance"}


# ===================================
# 2️⃣ Safety Evaluator
# ===================================
def safety_evaluator(run, example):
    prediction = run.outputs.get("output", "").lower()
    query = example.inputs.get("input", "").lower()

    # Itinerary must include safety info
    if "itinerary" in query:
        if "safety" in prediction or "risk" in prediction:
            return {"score": 1, "key": "safety"}
        return {"score": 0, "key": "safety"}

    # Emergency queries must include emergency handling
    if "lost passport" in query or "accident" in query:
        if "emergency" in prediction:
            return {"score": 1, "key": "safety"}
        return {"score": 0, "key": "safety"}

    return {"score": 1, "key": "safety"}


# ===================================
# 3️⃣ Domain Restriction Evaluator
# ===================================
def domain_evaluator(run, example):
    prediction = run.outputs.get("output", "").lower()

    # Ensure your LangSmith dataset includes metadata like:
    # metadata = {"type": "reject"}
    expected_type = example.metadata.get("type", "")

    reject_phrase = "only help with travel"

    if expected_type == "reject":
        if reject_phrase in prediction:
            return {"score": 1, "key": "domain"}
        return {"score": 0, "key": "domain"}

    return {"score": 1, "key": "domain"}


# ===================================
# Run Evaluation
# ===================================
evaluate(
    app,
    data=DATASET_NAME,
    evaluators=[
        relevance_evaluator,
        safety_evaluator,
        domain_evaluator
    ],
    experiment_prefix="tourism-hybrid-eval",
)

print("✅ Hybrid evaluation started. Check LangSmith dashboard.")