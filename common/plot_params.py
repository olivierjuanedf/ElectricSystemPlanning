import logging

from dataclasses import dataclass
from typing import Dict, List, Union, Optional


def to_int_keys_dict(dict_with_level_two_str_keys: Dict[str, Dict[str, str]]) -> Optional[Dict[str, Dict[int, str]]]:
    if dict_with_level_two_str_keys is None:
        return None
    return {name: {int(key): value for key, value in dict_with_str_keys.items()}
            for name, dict_with_str_keys in dict_with_level_two_str_keys.items()}


OWN_PALETTE = 'own'
PLOT_DIMS = ['agg_prod_type', 'year', 'climatic_year', 'zone']
TYPE_PARAMS_DEF = Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]]
TYPE_PER_CASE_PARAMS = Union[Dict[str, str], Dict[int, str]]


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
        if self.dimension in ['year', 'climatic_year']:
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

    def check(self):
        # TODO: check TB coded
        logging.warning('Not coded for now')


@dataclass
class PlotParamsFull:
    # first parameters to set up a choice of parameters for a given execution
    zone_palette_choice: str = OWN_PALETTE
    agg_prod_type_palette_choice: str = OWN_PALETTE
    agg_prod_type_marker_choice: str = OWN_PALETTE
    year_palette_choice: str = OWN_PALETTE
    year_linestyle_choice: str = OWN_PALETTE
    climatic_year_palette_choice: str = OWN_PALETTE
    climatic_year_linestyle_choice: str = OWN_PALETTE
    climatic_year_marker_choice: str = OWN_PALETTE
    # remaining ones for the def of the different possible values for the parameters
    zone_order: List[str] = None  # order in which zone curves will be plotted (for those available)
    agg_prod_type_order: List[str] = None  # idem for aggreg. production types
    # {name of the palette: {zone name: color}}
    zone_palettes_def: Dict[str, Dict[str, str]] = None
    # {name of the palette: {aggreg. prod. type name: color}}
    agg_prod_type_palettes_def: Dict[str, Dict[str, str]] = None
    # {name of the palette: {aggreg. prod. type name: marker}}
    agg_prod_type_markers_def: Dict[str, Dict[str, str]] = None
    # {name of the palette: {year: color}}. N.B. year in str when parsing json; in int after processing
    year_palettes_def: Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]] = None
    # {name of the palette: {year: linestyle}}. N.B. Idem; and linestyle used to distinguish different curves when
    # color - common - will have been set by - common - zone
    year_linestyles_def: Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]] = None
    # {name of the palette: {climatic year: color}}. N.B. Idem year
    climatic_year_palettes_def: Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]] = None
    # {name of the palette: {climatic year: linestyle}}. N.B. Idem year
    climatic_year_linestyles_def: Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]] = None
    # {name of the palette: {climatic year: marker}}. N.B. Idem year for typing; markers used when both colors
    # and linestyles will have been commonly set between two curves based on common (zone, year)
    climatic_year_markers_def: Union[Dict[str, Dict[str, str]], Dict[str, Dict[int, str]]] = None
    # simpler dicts, obtained based on choice + def.
    per_zone_color: Dict[str, str] = None
    per_agg_prod_type_color: Dict[str, str] = None
    per_agg_prod_type_marker: Dict[str, str] = None
    per_year_color: Dict[int, str] = None
    per_year_linestyle: Dict[int, str] = None
    per_climatic_year_color: Dict[int, str] = None
    per_climatic_year_linestyle: Dict[int, str] = None
    per_climatic_year_marker: Dict[int, str] = None
