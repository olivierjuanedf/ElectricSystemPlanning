import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Union

from utils.read import check_and_load_json_file
from common.long_term_uc_io import INPUT_FUNC_PARAMS_SUBFOLDER


@dataclass
class PlotParams:
    # first parameters to set up a choice of parameters for a given execution
    zone_palette_choice: str = None
    agg_prod_type_palette_choice: str = None
    year_palette_choice: str = None
    year_linestyle_choice: str = None
    climatic_year_palette_choice: str = None
    climatic_year_linestyle_choice: str = None
    climatic_year_marker_choice: str = None
    # remaining ones for the def of the different possible values for the parameters
    zone_order: List[str] = None  # order in which zone curves will be plotted (for those available)
    agg_prod_type_order: List[str] = None  # idem for aggreg. production types
    # {name of the palette: {zone name: color}}
    zone_palettes_def: Dict[str, Dict[str, str]] = None
    # {name of the palette: {aggreg. prod. type name: color}}
    agg_prod_type_palettes_def: Dict[str, Dict[str, str]] = None
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
    per_zone_color: Dict[str, str] = None
    per_agg_prod_type_color: Dict[str, str] = None

    def process(self):
        # convert str to int keys
        for elt in [self.year_palettes_def, self.year_linestyles_def, self.climatic_year_palettes_def,
                    self.climatic_year_linestyles_def, self.climatic_year_markers_def]:
            if elt is not None:
                elt = {name: {int(key): value for key, value in dict_with_str_keys.items()}
                       for name, dict_with_str_keys in elt.items()}

        self.per_zone_color = self.zone_palettes_def[self.zone_palette_choice]
        self.per_agg_prod_type_color = self.agg_prod_type_palettes_def[self.agg_prod_type_palette_choice]

    def check(self):
        # TODO: check TB coded
        logging.warning('Not coded for now')
