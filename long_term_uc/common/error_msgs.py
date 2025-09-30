import sys
import logging
from typing import List


def print_errors_list(error_name: str, errors_list: List[str]):
    error_msg = f'There are error(s) {error_name}:'
    for elt_error in errors_list:
        error_msg += f'\n- {elt_error}'
    error_msg += '\n-> STOP'
    logging.error(error_msg)
    sys.exit(1)
    

def uncoherent_param_stop(param_errors: List[str]):
    print_errors_list(error_name='in JSON params to be modif. file', errors_list=param_errors)
