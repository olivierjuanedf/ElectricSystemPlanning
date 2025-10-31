import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import math
from dataclasses import fields, MISSING

CLIM_YEARS_SUFFIX = 'clim-years'


def str_sanitizer(raw_str: Optional[str], replace_empty_char: bool = True,
                  ad_hoc_replacements: dict = None) -> Optional[str]:
    # sanitize only if str
    if not isinstance(raw_str, str):
        return raw_str

    sanitized_str = raw_str
    sanitized_str = sanitized_str.strip()
    if replace_empty_char:
        sanitized_str = sanitized_str.replace(' ', '_')
    sanitized_str = sanitized_str.lower()

    # if specific replacements to be applied
    if ad_hoc_replacements is not None:
        for old_char, new_char in ad_hoc_replacements.items():
            sanitized_str = sanitized_str.replace(old_char, new_char)
    return sanitized_str


def rm_elts_with_none_val(my_dict: dict) -> dict:
    return {key: val for key, val in my_dict.items() if val is not None}


def get_key_of_val(val, my_dict: dict, dict_name: str = None):
    corresp_keys = []
    for key in my_dict:
        if val in my_dict[key]:
            corresp_keys.append(key)
    if dict_name is None:
        dict_name = ''
    else:
        dict_name = f' {dict_name}'
    if len(corresp_keys) == 0:
        logging.warning(f'No corresponding key found in {dict_name} dict. for value {val} -> None returned')
        return None
    if len(corresp_keys) > 1:
        logging.warning(f'Multiple corresponding keys found in{dict_name} dict. for value {val} '
                        f'-> only first one returned')
    return corresp_keys[0]


def is_str_bool(bool_str: Optional[str]) -> bool:
    if not isinstance(bool_str, str):
        return False
    return bool_str.lower() in ['true', 'false']


def cast_str_bool(bool_str: str) -> Union[str, bool]:
    if is_str_bool(bool_str=bool_str):
        return bool(bool_str)
    else:
        return bool_str


def are_lists_eq(list_of_lists: List[list]) -> bool:
    first_list = list_of_lists[0]
    len_first_list = len(first_list)
    set_first_list = set(first_list)
    n_lists = len(list_of_lists)
    for i_list in range(1, n_lists):
        current_list = list_of_lists[i_list]
        if not (len(current_list) == len_first_list and set(current_list) == set_first_list):
            return False
    return True


def lexico_compar_str(string1: str, string2: str, return_tuple: bool = False) -> Union[str, Tuple[str, str]]:
    i = 0
    while i < len(string1) and i < len(string2):
        if string1[i] < string2[i]:
            return (string1, string2) if return_tuple else string1
        elif string1[i] > string2[i]:
            return (string2, string1) if return_tuple else string2
        i += 1
    # one of the strings starts with the other
    if len(string2) > len(string1):
        return (string1, string2) if return_tuple else string1
    else:
        return (string2, string1) if return_tuple else string2


def flatten_list_of_lists(list_of_lists: List[list]) -> list:
    return np.concatenate(list_of_lists).tolist()


def get_intersection_of_lists(list1: list, list2: list) -> list:
    return list(set(list1) & set(list2))


def set_years_suffix(years: List[int], is_climatic_year: bool = False) -> str:
    n_years = len(years)
    if n_years == 0:
        return ''
    if n_years == 1:
        return f'{years[0]}'
    if n_years == 2:
        min_date = f'{min(years)}'
        max_date = f'{max(years)}'
        if min_date[:2] == max_date[:2]:
            return f'{min_date}-{max_date[2:]}'
        else:
            return f'{min_date}-{max_date}'
    suffix = CLIM_YEARS_SUFFIX if is_climatic_year else 'years'
    return f'{n_years}-{suffix}'


def lowest_common_multiple(a, b):
    return abs(a * b) // math.gcd(a, b)


def print_non_default(obj, msg_if_all_defaults: bool = True, obj_name: str = None):
    non_default_msg = ''
    sep = '\n- '
    for f in fields(obj):
        default = f.default if f.default is not MISSING else None
        value = getattr(obj, f.name)
        if value != default:
            non_default_msg += f"{sep}{f.name} = {value}"
    if len(non_default_msg) > 0:
        obj_name_suffix = f' for object {obj_name}' if obj_name is not None else ''
        logging.info(f'Non-default attrs used{obj_name_suffix}:{non_default_msg}')
    elif msg_if_all_defaults:
        logging.info('All default values used')
