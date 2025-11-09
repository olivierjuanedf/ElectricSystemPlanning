import logging
import os
import shutil
from datetime import datetime
from typing import List, Dict

from common.constants.prod_types import ProdTypeNames
from common.constants.usage_params_json import EnvPhaseNames
from common.logger import deactivate_verbose_warnings, init_logger, stop_logger
from common.long_term_uc_io import set_full_lt_uc_output_folder
from common.uc_run_params import UCRunParams
from include_runner.uc_run_params_selection import UCRunParamsSelector
from my_little_europe_lt_uc import run
from utils.dir_utils import make_dir, delete_files
from utils.read import read_and_check_uc_run_params


def init(runner_output_folder: str, log_level: str = 'info'):
    # deactivate some annoying and useless warnings in pypsa/pandas
    deactivate_verbose_warnings()

    logger = init_logger(logger_dir=runner_output_folder, logger_name='eraa_lt_uc_runner.log',
                         log_level=log_level)


def copy_results_to_runner_folder(uc_output_folder: str, runner_output_folder: str, climatic_year: int):
    current_cy_output_folder = os.path.join(runner_output_folder, f'cy{climatic_year}')
    shutil.copytree(uc_output_folder, current_cy_output_folder)
    # then suppress files wo cy{climatic_year} suffix
    delete_files(directory=current_cy_output_folder, str_in_file=f'_cy{climatic_year}')


def launch_runner(target_year: int, uc_period_start: datetime, uc_period_end: datetime, runner_output_folder: str,
                  run_idx: int, eur_team_name: str, countries_out: List[str] = None,
                  climatic_year_selec_rule: str = None, climatic_year_vals: int = None,
                  prod_types_out: Dict[str, List[ProdTypeNames]] = None):
    runner_output_folder = os.path.join(runner_output_folder, f'run_{run_idx}', eur_team_name)
    eur_uc_output_folder = set_full_lt_uc_output_folder()

    make_dir(full_path=runner_output_folder)
    init(runner_output_folder=runner_output_folder)
    logging.info(f'Start ERAA-PyPSA long-term European Unit Commitment (UC) runner')

    # get ERAA data description from JSON file
    _, eraa_data_descr, _ = (
        read_and_check_uc_run_params(phase_name=EnvPhaseNames.multizones_uc_model, get_only_eraa_data_descr=True)
    )
    # Apply selection for the main UC run parameters
    uc_run_params_selector = UCRunParamsSelector(eraa_data_descr=eraa_data_descr, selected_target_year=target_year)
    uc_run_params_selector.set_countries_selection(countries_out=countries_out)
    uc_run_params_selector.set_climatic_years_selection(climatic_year_selec_rule=climatic_year_selec_rule,
                                                        climatic_year_vals=climatic_year_vals,
                                                        selec_rule_extra_params={'n_cy': 2})
    uc_run_params_selector.set_prod_types_selection(prod_types_out=prod_types_out)
    # loop over the list of climatic year values
    for clim_year in uc_run_params_selector.selected_climatic_years:
        logging.info(f'Runner for climatic year: {clim_year}')
        fixed_uc_run_params_data = {'selected_countries': uc_run_params_selector.selected_countries,
                                    'selected_target_year': target_year, 'selected_climatic_year': clim_year,
                                    'selected_prod_types': uc_run_params_selector.selected_prod_types,
                                    'uc_period_start': uc_period_start, 'uc_period_end': uc_period_end}
        run(fixed_uc_run_params=UCRunParams(**fixed_uc_run_params_data),
            fixed_run_params_fields=list(fixed_uc_run_params_data))
        # copy/paste obtained results - for current climatic year - to somme specified folder
        copy_results_to_runner_folder(uc_output_folder=eur_uc_output_folder, runner_output_folder=runner_output_folder,
                                      climatic_year=clim_year)

    logging.info('THE END of ERAA-PyPSA long-term UC runner!')
    stop_logger()


if __name__ == '__main__':
    from include_runner.uc_run_params_selection import ClimYearsSelecRules
    output_folder = ('C:\\Users\\B57876\\Documents\\Perso\\Refuge des idees\\2025_EELISA\\ecole\\env_code'
                     '\\tests_ob\\runner')
    launch_runner(target_year=2025, uc_period_start=datetime(1900, 1, 1),
                  uc_period_end=datetime(1900, 1, 15), countries_out=['italy'],
                  climatic_year_selec_rule=ClimYearsSelecRules.random, runner_output_folder=output_folder, run_idx=1,
                  eur_team_name='eu_1')
    # TODO: coherent logs in this case (redundant/confusing with the ones of my_little_lt_uc.py)