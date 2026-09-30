def normalize(data):
    minimum = min(data)
    maximum = max(data)
    return [(x - minimum) / (maximum - minimum) for x in data]