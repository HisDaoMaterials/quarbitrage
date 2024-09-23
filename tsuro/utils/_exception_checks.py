"""
Module for functions that check if specified conditions are satisfied, else an exception is thrown
"""

from tsuro.utils._exceptions import UnequalLengthError, MissingColumnError


def check_if_equal_length(list_dicts: dict[list | dict]) -> Exception:
    """
    Raise exception if length of lists/dictionaries are not equal
    """
    first_key = list(list_dicts.keys())[0]
    first_listdict = list_dicts[first_key]

    list_dicts.pop(first_key)

    for other_key in list_dicts.keys():
        first_len = len(first_listdict)
        other_len = len(list_dicts[other_key])

        if first_len != other_len:
            raise UnequalLengthError(first_len, other_len, first_key, other_key)


def check_if_columns_in_list(
    column_list: list[str], reference_list: list[str]
) -> Exception:
    """
    Check if all columns in 'column_list' are found in 'reference_list'
    """
    column_list = [column.upper() for column in column_list]
    for column in column_list:
        if not column in reference_list:
            raise MissingColumnError(column, reference_list)
