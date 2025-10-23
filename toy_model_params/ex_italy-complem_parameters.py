from typing import Dict, List

from common.fuel_sources import FuelSource


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource],
                   wind_on_shore_data, wind_off_shore_data, solar_pv_data) -> List[dict]:
    """
    Get list of generators to be set on a given node of a PyPSA model
    :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
    :param fuel_sources
    """
    generators = [
        {'name': f'Battery_{country_trigram}', 'carrier': 'Flexibility', 'p_nom': 4000,
        'p_min_pu': -1, 'p_max_pu': 1
        }
    ]
    return generators
