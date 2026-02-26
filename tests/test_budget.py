import pytest
from main.agent import estimate_budget

def test_budget_returns_positive_total():
    output, total = estimate_budget("Goa", 3, "standard", "INR")
    assert total > 0

def test_low_budget_less_than_luxury():
    _, low_total = estimate_budget("Goa", 3, "low", "INR")
    _, luxury_total = estimate_budget("Goa", 3, "luxury", "INR")
    assert low_total < luxury_total

def test_budget_output_contains_total():
    output, _ = estimate_budget("Jaipur", 2, "standard", "INR")
    assert "Total Estimated Cost" in output