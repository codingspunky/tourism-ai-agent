import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from langchain_core.messages import HumanMessage
from main.agent import graph


# Load dataset
with open("evaluation/dataset.json", "r") as f:
    dataset = json.load(f)

results = []

for example in dataset:
    print(f"\n🔍 Testing: {example['input']}")

    state = {
        "user_id": "eval_user",
        "messages": [HumanMessage(content=example["input"])],
        "approved": True,
        "allowed": True
    }

    output = graph.invoke(
        state,
        config={"configurable": {"thread_id": "eval_user"}}
    )

    response = output["messages"][-1].content.lower()

    passed = any(keyword.lower() in response for keyword in example["expected_contains"])

    results.append({
        "input": example["input"],
        "passed": passed
    })

    print("✅ PASS" if passed else "❌ FAIL")

# Summary
total = len(results)
passed = sum(1 for r in results if r["passed"])

print("\n======================")
print(f"Total: {total}")
print(f"Passed: {passed}")
print(f"Accuracy: {(passed/total)*100:.2f}%")
