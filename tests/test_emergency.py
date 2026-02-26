from main.agent import is_emergency_query

def test_lost_detection():
    assert is_emergency_query("I lost my passport") == True

def test_accident_detection():
    assert is_emergency_query("I had an accident in London") == True

def test_normal_query_not_emergency():
    assert is_emergency_query("Best time to visit Goa") == False