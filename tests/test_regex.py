from main.agent import extract_numeric_budget

def test_extract_numeric_budget():
    result = extract_numeric_budget("My budget is 50000 INR")
    assert result == 50000

def test_no_budget_returns_none():
    result = extract_numeric_budget("Plan a trip")
    assert result is None