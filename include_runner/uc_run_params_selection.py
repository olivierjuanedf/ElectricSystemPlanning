import logging
from dataclasses import dataclass
from random import sample
from typing import List, Dict

from common.constants.extract_eraa_data import ERAADatasetDescr
from common.constants.prod_types import ProdTypeNames


@dataclass
class ClimYearsSelecRules:
    random: str = 'weather_is_so_unpredictable'
    cold: str = 'winter_is_coming'
    hot: str = 'some_like_summer_hot'


class UCRunParamsSelector:
    COLD_CLIMATIC_YEARS = [1985, 2001]  # TODO: check with experts
    HOT_CLIMATIC_YEARS = [2003]  # TODO: idem
    STRESS_TEST_TAG = '[STRESSSSS TEST]'

    def __init__(self, eraa_data_descr: ERAADatasetDescr, selected_target_year: int):
        self.available_countries = eraa_data_descr.available_countries
        self.available_clim_years = eraa_data_descr.available_climatic_years
        self.available_aggreg_prod_types = \
            {country: eraa_data_descr.available_aggreg_prod_types[country][selected_target_year]
             for country in eraa_data_descr.available_aggreg_prod_types}
        self.selected_target_year = selected_target_year
        self.selected_countries: List[str] = None
        self.selected_climatic_years: List[int] = None
        self.selected_prod_types: Dict[str, List[ProdTypeNames]] = None

    def set_countries_selection(self, countries_out: List[str] = None):
        if countries_out is not None:
            logging.info(f'{self.STRESS_TEST_TAG} The following countries are in blackout... '
                         f'they will not be part of current UC run: {countries_out}')
            self.selected_countries = list(set(self.available_countries) - set(countries_out))
        else:
            self.selected_countries = self.available_countries

    def set_prod_types_selection(self, prod_types_out: Dict[str, List[ProdTypeNames]] = None):
        if prod_types_out is not None:
            logging.info(f'{self.STRESS_TEST_TAG} Technical issue with the following country (key), '
                         f'prod. types (values) that will not be part of current UC run: {prod_types_out}')
            self.selected_prod_types = {}
            for country in self.selected_countries:
                all_prod_types = self.available_aggreg_prod_types[country]
                if country in prod_types_out:
                    self.selected_prod_types[country] = list(set(all_prod_types) - set(prod_types_out[country]))
                else:
                    self.selected_prod_types[country] = all_prod_types
        else:
            self.selected_prod_types = self.available_aggreg_prod_types

    def set_climatic_years_selection(self, climatic_year_selec_rule: str = None, climatic_year_vals: List[int] = None,
                                     selec_rule_extra_params: dict = None) -> List[int]:
        if climatic_year_selec_rule is None:
            if climatic_year_vals is None:
                raise Exception('Both climatic year selection rule and value are None -> STOP runner')
            else:  # directly applies selection provided in arg.
                logging.info(f'Climatic year(s) selection directly provided in arg: {climatic_year_vals}')
                self.selected_climatic_years = climatic_year_vals
        # climatic year to be set with a "rule", e.g. random draw
        if climatic_year_vals is not None:
            logging.warning(f'In runner, climatic year value(s) {climatic_year_vals} will not be applied; "rule" '
                            f'{climatic_year_selec_rule} will be applied instead to select a value for this parameter')
        logging.info(f'Apply rule {climatic_year_selec_rule} to select climatic year(s) '
                     f'among available values: {self.available_clim_years}')
        if climatic_year_selec_rule == ClimYearsSelecRules.random:
            self.selected_climatic_years = sample(self.available_clim_years, selec_rule_extra_params['n_cy'])
        if climatic_year_selec_rule == ClimYearsSelecRules.cold:
            self.selected_climatic_years = sample(self.COLD_CLIMATIC_YEARS, selec_rule_extra_params['n_cy'])
        if climatic_year_selec_rule == ClimYearsSelecRules.hot:
            self.selected_climatic_years = sample(self.HOT_CLIMATIC_YEARS, selec_rule_extra_params['n_cy'])
