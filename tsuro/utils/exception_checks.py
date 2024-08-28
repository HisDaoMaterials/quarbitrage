from tsuro.utils.exceptions import UnequalLengthError


def check_equal_length(list_dicts: dict[list | dict]) -> Exception:
    """
    Raise exception if length of lists/dictionaries are not equal
    """
    first_key = list(list_dicts.keys())[0]
    first_listdict = list_dicts[first_key]

    list_dicts.pop(first_key)

    for other_key in list_dicts:
        first_len = len(first_listdict)
        other_len = len(list_dicts[other_key])

        if first_len != other_len:
            raise UnequalLengthError(first_len, other_len, first_key, other_key)
