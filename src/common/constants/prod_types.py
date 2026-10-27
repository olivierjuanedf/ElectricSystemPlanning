from dataclasses import dataclass
from typing import Optional

from src.common.constants.countries import set_country_trigram


@dataclass
class ProdTypeNames:  # by alpha. order
    batteries: str = 'batteries'
    biofuel: str = 'biofuel'
    coal: str = 'coal'
    csp_no_storage: str = 'csp_nostorage'
    demand_side_response: str = 'demand_side_response_capacity'  # TODO: why capa in this name?
    failure: str = 'failure'
    gas: str = 'gas'
    hard_coal: str = 'hard_coal'
    hydro_pondage: str = 'hydro_pondage'
    hydro_pump_storage_closed: str = 'hydro_pump_storage_closed_loop'
    hydro_pump_storage_open: str = 'hydro_pump_storage_open_loop'
    hydro_reservoir: str = 'hydro_reservoir'
    hydro_ror: str = 'hydro_run_of_river'
    lignite: str = 'lignite'
    nuclear: str = 'nuclear'
    oil: str = 'oil'
    others_non_renewable: str = 'others_non-renewable'  # TODO: with '_' instead of '-'? (TB tested)
    others_renewable: str = 'others_renewable'
    solar_pv: str = 'solar_pv'
    solar_thermal: str = 'solar_thermal'
    wind_offshore: str = 'wind_offshore'
    wind_onshore: str = 'wind_onshore'


PT_ORDER_IN_DATA_OUTPUT = [
    # 1st renewables
    ProdTypeNames.solar_pv, ProdTypeNames.solar_thermal, ProdTypeNames.wind_onshore,
    ProdTypeNames.wind_offshore, ProdTypeNames.csp_no_storage, ProdTypeNames.others_renewable,
    ProdTypeNames.hydro_ror, ProdTypeNames.hydro_pondage, ProdTypeNames.hydro_reservoir,
    ProdTypeNames.hydro_pump_storage_open, ProdTypeNames.hydro_pump_storage_closed,  # 2nd hydro
    ProdTypeNames.nuclear,  # thermal - nuclear
    ProdTypeNames.biofuel, ProdTypeNames.coal, ProdTypeNames.hard_coal, ProdTypeNames.lignite,
    ProdTypeNames.oil, ProdTypeNames.gas,  # thermal - others
    "others_fatal",  # TODO: include it in dataclass above?
    ProdTypeNames.others_non_renewable,  # others
    ProdTypeNames.batteries, ProdTypeNames.demand_side_response,  # other storage-like assets
    ProdTypeNames.failure  # last failure (fictive prod.)
]
STOCK_LIKE_PROD_TYPES = [ProdTypeNames.hydro_reservoir, ProdTypeNames.hydro_pump_storage_closed,
                         ProdTypeNames.hydro_pump_storage_open, ProdTypeNames.batteries,
                         ProdTypeNames.demand_side_response]
STORAGE_COL_SUFFIX = {'prod': 'prod', 'cons': 'cons', 'soc': 'soc'}
UNIT_NAME_SEP = '_'


def set_gen_unit_name(country: str, agg_prod_type: str) -> str:
    country_trigram = set_country_trigram(country=country)
    return f'{country_trigram}{UNIT_NAME_SEP}{agg_prod_type}'


def get_country_from_unit_name(prod_unit_name: str) -> str:
    return prod_unit_name.split(UNIT_NAME_SEP)[0]


def get_storage_suffix(prod_unit_name: str) -> Optional[str]:
    storage_suffix = None
    for elt in STORAGE_COL_SUFFIX.values():
        if prod_unit_name.endswith(f"_{elt}"):
            storage_suffix = elt
    return storage_suffix


def get_prod_type_from_unit_name(prod_unit_name: str) -> str:
    """
    ATTENTION (i) Tricky cases if '_' in prod type... the rule is to take the str after country name + separator;
    (ii) Remove storage suffix when present in prod unit name
    Args:
        prod_unit_name:

    Returns:
    """
    country = get_country_from_unit_name(prod_unit_name=prod_unit_name)
    storage_suffix = get_storage_suffix(prod_unit_name=prod_unit_name)
    suffix_len = 0 if storage_suffix is None else len(storage_suffix) + len(UNIT_NAME_SEP)
    pu_name_len = len(prod_unit_name)
    return prod_unit_name[len(country) + len(UNIT_NAME_SEP):pu_name_len-suffix_len]


def add_suffix_to_storage_unit_col(col: str, col_type: str) -> str:
    """
    Suffix added when both prod. and cons. for storage are put in a same df -> to distinguish the 2 columns
    for a same prod unit
    Args:
        col:
        col_type: either prod/cons

    Returns:

    """
    if col_type not in STORAGE_COL_SUFFIX:
        raise Exception(f'Suffix cannot be added to storage column {col}; unknown type {col_type} '
                        f'(it must be in {list(STORAGE_COL_SUFFIX)})')
    return col + UNIT_NAME_SEP + STORAGE_COL_SUFFIX[col_type]
