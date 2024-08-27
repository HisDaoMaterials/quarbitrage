

def check_equal_size(
        array_objs: dict[list | dict]
    ) -> Exception:
    """
    Raise exception if size of lists are not equal
    """
    first_key = list(array_objs.keys())[0]
    first_obj = array_objs[first_key]

    array_objs.pop(first_key)
    
    for other_key in array_objs:
        other_list = array_objs[other_key]
        if len(first_obj) != len(other_list):
            raise ValueError(f"Length('{first_key}')={len(first_obj)} != Length('{other_key}')={len(other_list)}. Please ensure same size for both arguments.")