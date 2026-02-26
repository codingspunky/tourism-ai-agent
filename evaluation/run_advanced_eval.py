import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from langchain_core.messages import HumanMessage
from main.agent import graph
from advanced_eval import llm_quality_score
from metrics import update_confusion_matrix

# =============================
# LOAD DATASET
# =============================

with open("evaluation/dataset.json") as f:
    dataset = json.load(f)

# =============================
# METRIC COUNTERS
# =============================

tp = tn = fp = fn = 0
quality_scores = []
safety_failures = 0
hallucination_warnings = 0

# =============================
# EVALUATION LOOP
# =============================

for example in dataset:

    print(f"\n🔍 Testing: {example['input']}")

    state = {
        "user_id": "eval_user",
        "messages": [HumanMessage(content=example["input"])],
        "approved": True,
        "allowed": True
    }

    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": "eval_user"}}
    )

    response = result["messages"][-1].content.lower()

    # =============================
    # STRONG SAFETY REGRESSION
    # =============================

    if example["type"] == "itinerary":
        if "risk" not in response and "safety" not in response:
            safety_failures += 1
            print("⚠ Safety missing in itinerary response")

    if example["type"] == "emergency":
        if "emergency" not in response:
            safety_failures += 1
            print("⚠ Emergency handling failed")

    # =============================
    # SIMPLE HALLUCINATION CHECK
    # =============================

    if "source used: local knowledge base" in response:
        if not any(word in response for word in ["jaipur", "goa", "manali", "amber"]):
            hallucination_warnings += 1
            print("⚠ Possible hallucination detected")

    # =============================
    # CONFUSION MATRIX
    # =============================

    cm = update_confusion_matrix(example["type"], response)

    if cm == "TP": tp += 1
    elif cm == "TN": tn += 1
    elif cm == "FP": fp += 1
    elif cm == "FN": fn += 1

    # =============================
    # LLM QUALITY SCORE
    # =============================

    score = llm_quality_score(example["input"], response)

    try:
        quality_scores.append(int(score))
    except:
        quality_scores.append(3)

    print("✅ Completed")

# =============================
# FINAL METRICS
# =============================

print("\n===== ADVANCED EVALUATION =====")

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

total = tp + tn + fp + fn

accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Accuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1 Score:", round(f1, 2))

avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

print("Average LLM Quality Score:", round(avg_quality, 2))
print("Safety Failures:", safety_failures)
print("Hallucination Warnings:", hallucination_warnings)
