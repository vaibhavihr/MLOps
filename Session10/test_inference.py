from inference import predict


def test_positive():
    assert predict(0.8) == "Positive"


def test_negative():
    assert predict(0.2) == "Negative"