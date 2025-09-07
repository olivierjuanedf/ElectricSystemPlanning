from typing import Dict, List, Union

from long_term_uc.common.fuel_sources import FuelSources
from long_term_uc.include.dataset_builder import GenerationUnitData


GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
gps_coords = (12.5674, 41.8719)


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSources],
                   wind_on_shore_data, wind_off_shore_data, solar_pv_data) -> List[GENERATOR_DICT_TYPE]:
    """
    Get list of generators to be set on a given node of a PyPSA model
    :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
    :param fuel_sources
    """
    generators = [
        {
            'name': f'{country_trigram}_hard-coal', 'carrier': 'coal',
            'p_nom': 2362, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources['Coal'].primary_cost * 0.37,
            'efficiency': 0.37, 'committable': False
        },
        {
            'name': f'{country_trigram}_gas', 'carrier': 'gas',
            'p_nom': 43672, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources['Gas'].primary_cost * 0.5,
            'efficiency': 0.5, 'committable': False
        },
        {
            'name': f'{country_trigram}_oil', 'carrier': 'oil',
            'p_nom': 866, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources['Gas'].primary_cost * 0.4,
            'efficiency': 0.4, 'committable': False
        },
        {
            'name': f'{country_trigram}_other-non-renewables', 'carrier': 'other-non-renewables',
            'p_nom': 8239, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': fuel_sources['Gas'].primary_cost * 0.4,
            'efficiency': 0.4, 'committable': False
        },
        {
            'name': f'{country_trigram}_wind-on-shore', 'carrier': 'wind',
            'p_nom': 14512, 'p_min_pu': 0, 'p_max_pu': wind_on_shore_data['value'].values,
            'marginal_cost': fuel_sources['Wind'].primary_cost, 'efficiency': 1,
            'committable': False
        },
        {
            'name': f'{country_trigram}_wind-off-shore', 'carrier': 'wind',
            'p_nom': 791, 'p_min_pu': 0, 'p_max_pu': wind_off_shore_data['value'].values,
            'marginal_cost': fuel_sources['Wind'].primary_cost, 'efficiency': 1,
            'committable': False
        },
        {
            'name': f'{country_trigram}_solar-pv', 'carrier': 'solar',
            'p_nom': 39954, 'p_min_pu': 0, 'p_max_pu': solar_pv_data['value'].values,
            'marginal_cost': fuel_sources['Solar'].primary_cost, 'efficiency': 1,
            'committable': False
        },
        {
            'name': f'{country_trigram}_other-renewables', 'carrier': 'other-renewables',
            'p_nom': 4466, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': 0,
            'efficiency': 1, 'committable': False
        },
        # QUESTION: what is this - very necessary - last fictive asset?
        {
            'name': f'{country_trigram}_failure', 'carrier': 'failure',
            'p_nom': 1e10, 'p_min_pu': 0, 'p_max_pu': 1,
            'marginal_cost': 1e5,
            'efficiency': 1, 'committable': False
        }
    ]
    return generators


def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cas as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]
