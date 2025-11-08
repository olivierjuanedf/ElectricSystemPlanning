import logging
from random import random
from typing import List, Dict

from common.constants.extract_eraa_data import ERAADatasetDescr
from common.constants.prod_types import ProdTypeNames


COLD_CLIMATIC_YEARS = [1985, 2001]  # TODO: check with experts
HOT_CLIMATIC_YEARS = [2003]  # TODO: idem
STRESS_TEST_TAG = '[STRESSSSS TEST]'


def set_countries_selection(eraa_data_descr: ERAADatasetDescr, countries_out: List[str] = None) -> List[str]:
    all_countries = eraa_data_descr.available_countries
    if countries_out is not None:
        logging.info(f'{STRESS_TEST_TAG} The following countries are in blackout... '
                     f'they will not be part of current UC run: {countries_out}')
        selected_countries = list(set(all_countries) - set(countries_out))
    else:
        selected_countries = all_countries
    return selected_countries


def set_prod_types_selection(eraa_data_descr: ERAADatasetDescr, selected_countries: List[str],
                             selected_target_year: int, prod_types_out: Dict[str, List[ProdTypeNames]] = None) \
        -> Dict[str, List[ProdTypeNames]]:
    if prod_types_out is not None:
        logging.info(f'{STRESS_TEST_TAG} Technical issue with the following country (key), prod. types (values) '
                     f'that will not be part of current UC run: {prod_types_out}')
        selected_prod_types = {}
        for country in selected_countries:
            all_prod_types = eraa_data_descr.available_aggreg_prod_types[country][selected_target_year]
            if country in prod_types_out:
                selected_prod_types[country] = list(set(all_prod_types) - set(prod_types_out[country]))
            else:
                selected_prod_types[country] = all_prod_types
    else:
        selected_prod_types = {country: eraa_data_descr.available_aggreg_prod_types[country][selected_target_year]
                               for country in selected_countries}

    return selected_prod_types


def set_climatic_years_selection(climatic_year_selec_rule: str = None, climatic_year_vals: List[int] = None,
                                 eraa_data_descr: ERAADatasetDescr = None,
                                 selec_rule_extra_params: dict = None) -> List[int]:
    if climatic_year_selec_rule is None:
        if climatic_year_vals is None:
            raise Exception('Both climatic year selection rule and value are None -> STOP runner')
        else:  # directly applies selection provided in arg.
            return climatic_year_vals
    # climatic year to be set with a "rule", e.g. random draw
    if climatic_year_vals is not None:
        logging.warning(f'In runner, climatic year value(s) {climatic_year_vals} will not be applied; "rule" '
                        f'{climatic_year_selec_rule} will be applied instead to select a value for this parameter')
    all_clim_years = eraa_data_descr.available_climatic_years
    logging.info(f'Apply rule {climatic_year_selec_rule} to select climatic year(s) '
                 f'among available values: {all_clim_years}')
    if climatic_year_selec_rule == 'weather_is_so_unpredictable':
        return random.sample(all_clim_years, selec_rule_extra_params['n_cy'])
    if climatic_year_selec_rule == 'winter_is_coming':
        return random.sample(COLD_CLIMATIC_YEARS, selec_rule_extra_params['n_cy'])
    if climatic_year_selec_rule == 'some_like_summer_hot':
        return random.sample(HOT_CLIMATIC_YEARS, selec_rule_extra_params['n_cy'])
