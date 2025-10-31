import logging

from dataclasses import dataclass
from typing import Dict, List, Union, Optional

from common.constants.datadims import DataDimensions


def to_int_keys_dict(dict_with_level_two_str_keys: Dict[str, Dict[str, str]]) -> Optional[Dict[str, Dict[int, str]]]:
    if dict_with_level_two_str_keys is None:
        return None
    return {name: {int(key): value for key, value in dict_with_str_keys.items()}
            for name, dict_with_str_keys in dict_with_level_two_str_keys.items()}


OWN_PALETTE = 'own'
PLOT_DIMS_ORDER = [DataDimensions.zone, DataDimensions.year, DataDimensions.climatic_year, DataDimensions.agg_prod_type]
TYPE_PARAMS_DEF = Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]]
TYPE_PER_CASE_PARAMS = Union[Dict[str, str], Dict[int, str]]
N_LETTERS_ZONE = 3


def set_per_case_dict(params_def: TYPE_PARAMS_DEF, param_choice: str,
                      param_name: str) -> Optional[TYPE_PER_CASE_PARAMS]:
    if params_def is None:
        return None
    if param_choice in params_def:
        return params_def[param_choice]
    else:
        logging.warning(
            f'{param_name.capitalize()} choice {param_choice} not in {param_name} def. {params_def} '
            f'-> cannot be accounted for in PlotParams')
        return None


@dataclass
class PlotParams:
    dimension: str = None
    # first parameters to set up a choice of parameters for a given execution
    palette_choice: str = OWN_PALETTE
    linestyle_choice: str = OWN_PALETTE
    marker_choice: str = OWN_PALETTE
    order: Union[List[str], List[int]] = None
    # remaining ones for the def of the different possible values for the parameters
    # {name of the palette: {zone name/agg. prod. type/year/climatic year: color}}.
    # N.B. (climatic) year in str when parsing json; in int after processing
    palettes_def: TYPE_PARAMS_DEF = None
    # Idem palettes_def
    linestyles_def: TYPE_PARAMS_DEF = None
    # Idem palettes_def
    markers_def: TYPE_PARAMS_DEF = None
    # simpler dicts, obtained based on choice + def.
    per_case_color: TYPE_PER_CASE_PARAMS = None
    per_case_linestyle: TYPE_PER_CASE_PARAMS = None
    per_case_marker: TYPE_PER_CASE_PARAMS = None

    def process(self):
        # convert str to int keys
        num_plot_dims = [DataDimensions.year, DataDimensions.climatic_year]
        if self.dimension in num_plot_dims:
            self.palettes_def = to_int_keys_dict(dict_with_level_two_str_keys=self.palettes_def)
            self.linestyles_def = to_int_keys_dict(dict_with_level_two_str_keys=self.linestyles_def)
            self.markers_def = to_int_keys_dict(dict_with_level_two_str_keys=self.markers_def)

        # from choice and def. to simpler dicts
        self.per_case_color = set_per_case_dict(params_def=self.palettes_def, param_choice=self.palette_choice,
                                                param_name='palette')
        self.per_case_linestyle = set_per_case_dict(params_def=self.linestyles_def, param_choice=self.linestyle_choice,
                                                    param_name='linestyle')
        self.per_case_marker = set_per_case_dict(params_def=self.markers_def, param_choice=self.marker_choice,
                                                 param_name='marker')

        # set order for numeric idx, if not already provided
        if self.dimension in num_plot_dims and self.order is None:
            self.order = list(self.per_case_color)
            self.order.sort()

    def check(self):
        # TODO: check TB coded
        logging.warning('Not coded for now')
