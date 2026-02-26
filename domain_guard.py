from langchain_core.messages import HumanMessage

def domain_check(llm, question):

    prompt = f"""
You are a strict classifier.

Determine if the user question is related to tourism or travel.

Tourism includes:
- Travel
- Hotels
- Flights
- Transport
- Food
- Cultural tips
- Entry fees
- Timings
- Budget
- Best time to visit
- Currency
- Visa
- Passport
- Travel documents
- Travel restrictions
- Itinerary planning
- Places to visit
- Embassy
- travel emergencies
- lost passport
- stolen items
- medical emergencies
- attractions
- history of monuments
- cultural sites
- Architecture
- geography
- travel information


If the question is related to tourism/travel → respond ONLY with:
ALLOW

If unrelated (coding, politics, medical, math, programming, etc.) → respond ONLY with:
REJECT

Do not explain.
Do not add anything else.

Question: {question}
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()

    return response == "ALLOW"
