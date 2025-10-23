from dataclasses import dataclass


@dataclass
class CountryJsonParamNames:
    capacities_tb_overwritten: str = 'capacities_tb_overwritten'
    selected_prod_types: str = 'selected_prod_types'
    team: str = 'team'


@dataclass
class EuropeJsonParamNames:
    failure_penalty: str = 'failure_penalty'
    failure_power_capa: str = 'failure_power_capa'
    interco_capas_tb_overwritten: str = 'interco_capas_tb_overwritten'
    selected_climatic_year: str = 'selected_climatic_year'
    selected_countries: str = 'selected_countries'
    selected_target_year: str = 'selected_target_year'
    uc_period_end: str = 'uc_period_end'
    uc_period_start: str = 'uc_period_start'


ALL_KEYWORD = 'all'
