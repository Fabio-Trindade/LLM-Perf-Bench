def flatten_list(data_list: list):
    for value in data_list:
        if isinstance(value, list):
            yield from flatten_list(value)
        else:
            yield value