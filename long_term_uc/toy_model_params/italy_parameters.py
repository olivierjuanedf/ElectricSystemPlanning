from typing import Dict, List, Union

import pandas as pd

from long_term_uc.common.fuel_sources import FuelSources, FuelNames, DummyFuelNames
from long_term_uc.include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
gps_coords = (12.5674, 41.8719)


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSources], wind_on_shore_data: pd.DataFrame,
                   wind_off_shore_data: pd.DataFrame, solar_pv_data: pd.DataFrame) -> List[GENERATOR_DICT_TYPE]:
    """
    Get list of generators to be set on a given node of a PyPSA model
    :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
    :param fuel_sources
    :param wind_on_shore_data
    :param wind_off_shore_data
    :param solar_pv_data
    N.B. Better in this function to use CONSTANT names of the different fuel sources to avoid trouble
    in the code (i.e. FuelNames and DummyFuelNames dataclasses = sort of dict.)
    """
    generators = [
        {
            'name': f'{country_trigram}_hard-coal', 'carrier': FuelNames.coal,
            'p_nom': 2362, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources[FuelNames.coal].primary_cost * 0.37,
            'efficiency': 0.37, 'committable': False
        },
        {
            'name': f'{country_trigram}_gas', 'carrier': FuelNames.gas,
            'p_nom': 43672, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources[FuelNames.gas].primary_cost * 0.5,
            'efficiency': 0.5, 'committable': False
        },
        {
            'name': f'{country_trigram}_oil', 'carrier': FuelNames.oil,
            'p_nom': 866, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources[FuelNames.oil].primary_cost * 0.4,
            'efficiency': 0.4, 'committable': False
        },
        {
            'name': f'{country_trigram}_other-non-renewables', 'carrier': FuelNames.other_renewables,
            'p_nom': 8239, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources[FuelNames.other_renewables].primary_cost * 0.4,
            'efficiency': 0.4, 'committable': False
        },
        {
            'name': f'{country_trigram}_wind-on-shore', 'carrier': FuelNames.wind,
            'p_nom': 14512, 'p_min_pu': 0, 'p_max_pu': wind_on_shore_data['value'].values,
            'marginal_cost': fuel_sources[FuelNames.wind].primary_cost, 'efficiency': 1,
            'committable': False
        },
        {
            'name': f'{country_trigram}_wind-off-shore', 'carrier': FuelNames.wind,
            'p_nom': 791, 'p_min_pu': 0, 'p_max_pu': wind_off_shore_data['value'].values,
            'marginal_cost': fuel_sources[FuelNames.wind].primary_cost, 'efficiency': 1,
            'committable': False
        },
        {
            'name': f'{country_trigram}_solar-pv', 'carrier': FuelNames.solar,
            'p_nom': 39954, 'p_min_pu': 0, 'p_max_pu': solar_pv_data['value'].values,
            'marginal_cost': fuel_sources[FuelNames.solar].primary_cost, 'efficiency': 1,
            'committable': False
        },
        {
            'name': f'{country_trigram}_other-renewables', 'carrier': FuelNames.other_renewables,
            'p_nom': 4466, 'p_min_pu': 0, 'p_max_pu': 1, 'marginal_cost': 0, 'efficiency': 1, 'committable': False
        },
        # QUESTION: what is this - very necessary - last fictive asset?
        {
            'name': f'{country_trigram}_failure', 'carrier': DummyFuelNames.failure,
            'p_nom': 1e10, 'p_min_pu': 0, 'p_max_pu': 1, 'marginal_cost': 1e5, 'efficiency': 1, 'committable': False
        }
    ]
    return generators


def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cas as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]
