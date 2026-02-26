from main.agent import extract_comparison_places

def test_compare_and_format():
    p1, p2 = extract_comparison_places("Compare Jaipur and Udaipur")
    assert p1 == "Jaipur"
    assert p2 == "Udaipur"

def test_vs_format():
    p1, p2 = extract_comparison_places("Goa vs Manali")
    assert p1 == "Goa"
    assert p2 == "Manali"