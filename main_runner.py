import logging
from datetime import datetime
from typing import List

from common.constants.usage_params_json import EnvPhaseNames
from common.uc_run_params import UCRunParams
from include_runner.uc_run_params_selection import set_countries_selection, set_climatic_years_selection, \
    set_prod_types_selection
from my_little_europe_lt_uc import run
from utils.read import read_and_check_uc_run_params


def launch_runner(target_year: int, uc_period_start: datetime, uc_period_end: datetime, runner_output_folder: str,
                  countries_out: List[str] = None, climatic_year_selec_rule: str = None,
                  climatic_year_vals: int = None):
    # get ERAA data description from JSON file
    _, eraa_data_descr, _ = read_and_check_uc_run_params(phase_name=EnvPhaseNames.multizones_uc_model)
    # Apply selection for the main UC run parameters
    countries_selec = set_countries_selection(eraa_data_descr=eraa_data_descr, countries_out=countries_out)
    climatic_years_selec = (
        set_climatic_years_selection(climatic_year_selec_rule=climatic_year_selec_rule,
                                     climatic_year_vals=climatic_year_vals, eraa_data_descr=eraa_data_descr,
                                     selec_rule_extra_params={'n_cy': 2})
    )
    prod_types_selec = set_prod_types_selection(eraa_data_descr=eraa_data_descr, selected_countries=countries_selec,
                                                selected_target_year=target_year)
    # TODO: loop over the list of climatic year values
    for clim_year in climatic_years_selec:
        logging.info(f'Runner for climatic year: {clim_year}')
        fixed_uc_run_params_data = {'selected_countries': countries_selec, 'selected_target_year': target_year,
                                    'selected_climatic_year': clim_year, 'selected_prod_types': prod_types_selec,
                                    'uc_period_start': uc_period_start, 'uc_period_end': uc_period_end}
        run(solver_name='highs', fixed_uc_run_params=UCRunParams(**fixed_uc_run_params_data),
            fixed_run_params_fields=list(fixed_uc_run_params_data))
        # TODO: copy/paste obtained results to somme specified folder


if __name__ == '__main__':
    from include_runner.uc_run_params_selection import ClimYearsSelecRules
    launch_runner(target_year=2025, uc_period_start=datetime(1900, 1, 1),
                  uc_period_end=datetime(1900, 1, 15), countries_out=['italy'],
                  climatic_year_selec_rule=ClimYearsSelecRules.random, runner_output_folder=None)
