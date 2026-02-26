from langchain_core.messages import HumanMessage
from main.agent import llm


def llm_quality_score(question, answer):

    prompt = f"""
Rate the quality of the answer from 1 to 5.

1 = Very poor
5 = Excellent

Consider:
- Relevance
- Completeness
- Clarity
- Accuracy

Question: {question}
Answer: {answer}

Return only a number.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def safety_check(question, response):

    if "itinerary" in question.lower():
        return "safety" in response.lower() or "risk" in response.lower()

    if "passport" in question.lower():
        return "emergency" in response.lower()

    return True
