def string_to_list(string, delimiter=',', type=int):
    return list(map(type, string.split(delimiter)))


def string_to_tuple(string, delimiter=',', type=int):
    return tuple(string_to_list(string, delimiter, type))