import os
from dataclasses import dataclass
from typing import Dict, List

from long_term_uc.utils.read import check_and_load_json_file
from long_term_uc.common.long_term_uc_io import INPUT_FUNC_PARAMS_SUBFOLDER


@dataclass
class PlotParams:
    json_file: str = os.path.join(INPUT_FUNC_PARAMS_SUBFOLDER, 'plot_params.json')
    zone_palette_choice: str = None
    agg_prod_type_palette_choice: str = None
    zone_order: List[str] = None  # order in which zone curves will be plotted (for those available)
    agg_prod_type_order: List[str] = None  # idem for aggreg. production types
    # {name of the palette: {zone name: color}}
    zone_palettes_def: Dict[str, Dict[str, str]] = None
    # {name of the palette: {aggreg. prod. type name: color}}
    agg_prod_type_palettes_def: Dict[str, Dict[str, str]] = None
    per_zone_color: Dict[str, str] = None
    per_agg_prod_type_color: Dict[str, str] = None

    def read_and_check(self):
        json_plot_params = check_and_load_json_file(json_file=self.json_file, file_descr='JSON plot params')
        # TODO[Q2OJ]: better way to 'unpack'?
        self.zone_palette_choice = json_plot_params['zone_palette_choice']
        self.agg_prod_type_palette_choice = json_plot_params['agg_prod_type_palette_choice']
        self.zone_order = json_plot_params['zone_order']
        self.agg_prod_type_order = json_plot_params['agg_prod_type_order']
        self.zone_palettes_def = json_plot_params['zone_palettes_def']
        self.agg_prod_type_palettes_def = json_plot_params['agg_prod_type_palettes_def']
        # TODO: check TB coded
        self.per_zone_color = self.zone_palettes_def[self.zone_palette_choice]
        self.per_agg_prod_type_color = self.agg_prod_type_palettes_def[self.agg_prod_type_palette_choice]
    