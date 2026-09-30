from preprocessing import normalize


def test_normalize():
    data = [10, 20, 30]
    result = normalize(data)
    expected = [0.0, 0.5, 1.0]

    assert result == expected