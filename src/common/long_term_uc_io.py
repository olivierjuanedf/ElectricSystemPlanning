import logging
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterator, List, Optional
import re

from src.common.constants.countries import set_country_trigram
from src.common.constants.datatypes import DATATYPE_NAMES
from src.common.constants.env_dirs import DT_SUBFOLDERS, DEMAND_ADDER_FILE_PREFIX, \
    OutputFolderNames, INPUT_FUNC_PARAMS_SUBFOLDER, INPUT_DATA_ANALYSIS_SUBFOLDER, INPUT_ENV_PARAMS_SUBFOLDER, \
    INPUT_LT_UC_COUNTRY_SUBFOLDER, INPUT_FUEL_SOURCES_FOLDER, INPUT_LT_UC_SUBFOLDER, INPUT_LT_UC_OPTIONAL_SUBFOLDER, \
    OUTPUT_FOLDER_TOY_MODEL, OUTPUT_FOLDER_LT
from src.common.constants.temporal import Timescale
from src.common.logger import format_relative_path_for_log
from src.utils.dir_utils import make_dir, uniformize_path_os, get_files_in_dir, get_folders_in_dir


# TODO: move part of the following constants in src/common/constants/env_dirs
@dataclass
class ColumnNames:
    date: str = 'date'
    day: str = 'day'
    week: str = 'week'
    target_year: str = 'year'
    climatic_year: str = 'climatic_year'
    production_type: str = 'production_type'
    value: str = 'value'
    min_value: str = 'min_value'
    max_value: str = 'max_value'
    zone: str = 'zone'
    zone_origin: str = 'zone_origin'
    zone_destination: str = 'zone_destination'


@dataclass
class ComplemDataSources:
    from_json_tb_modif: str = 'from_json_tb_modif'
    from_eraa_data: str = 'from_eraa_data'


COLUMN_NAMES = ColumnNames()
COMPLEM_DATA_SOURCES = ComplemDataSources()
DATE_FORMAT_FILE = '%Y-%m-%d'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT_PRINT = '%Y/%m/%d'
GEN_CAPA_SUBDT_COLS = ['power_capacity', 'power_capacity_turbine', 'power_capacity_pumping',
                       'power_capacity_injection', 'power_capacity_offtake', 'energy_capacity']
# N.B. min/max hydro (gen.) levels in a unique file -> share same constants below
HYDRO_FILES = {DATATYPE_NAMES.hydro_ror: 'PECD-hydro-daily-ror-generation.csv',
               DATATYPE_NAMES.hydro_inflows: 'PECD-hydro-weekly-inflows.csv',
               DATATYPE_NAMES.hydro_levels_min: 'PECD-hydro-weekly-reservoir-min-max-levels.csv',
               DATATYPE_NAMES.hydro_gen_min: 'PECD-hydro-weekly-reservoir-min-max-generation.csv'}
HYDRO_FILES[DATATYPE_NAMES.hydro_levels_max] = HYDRO_FILES[DATATYPE_NAMES.hydro_levels_min]
HYDRO_FILES[DATATYPE_NAMES.hydro_gen_max] = HYDRO_FILES[DATATYPE_NAMES.hydro_gen_min]
# hydro datatypes for which data can be optionally obtained from input folder, in which students can do some modifs
HYDRO_DTS_OPT_FROM_INPUT_FOLDER = [DATATYPE_NAMES.hydro_gen_min, DATATYPE_NAMES.hydro_gen_max,
                                   DATATYPE_NAMES.hydro_levels_min, DATATYPE_NAMES.hydro_levels_max]
HYDRO_FILES_PREFIX_IN_INPUT_FOLDER = {DATATYPE_NAMES.hydro_ror: 'hydro-daily-ror-generation',
                                      DATATYPE_NAMES.hydro_inflows: 'hydro-weekly-inflows',
                                      DATATYPE_NAMES.hydro_levels_min: 'hydro-weekly-reservoir-min-max-levels',
                                      DATATYPE_NAMES.hydro_gen_min: 'hydro-weekly-reservoir-min-max-generation'}
HYDRO_FILES_PREFIX_IN_INPUT_FOLDER[DATATYPE_NAMES.hydro_levels_max] = (
    HYDRO_FILES_PREFIX_IN_INPUT_FOLDER)[DATATYPE_NAMES.hydro_levels_min]
HYDRO_FILES_PREFIX_IN_INPUT_FOLDER[DATATYPE_NAMES.hydro_gen_max] = (
    HYDRO_FILES_PREFIX_IN_INPUT_FOLDER)[DATATYPE_NAMES.hydro_gen_min]
HYDRO_KEY_COLUMNS = {DATATYPE_NAMES.hydro_ror:
                         [COLUMN_NAMES.zone, COLUMN_NAMES.day, COLUMN_NAMES.week, COLUMN_NAMES.climatic_year],
                     DATATYPE_NAMES.hydro_inflows: [COLUMN_NAMES.zone, COLUMN_NAMES.week, COLUMN_NAMES.climatic_year],
                     DATATYPE_NAMES.hydro_levels_min: [COLUMN_NAMES.zone, COLUMN_NAMES.week],
                     DATATYPE_NAMES.hydro_gen_min: [COLUMN_NAMES.zone, COLUMN_NAMES.week, COLUMN_NAMES.climatic_year]
                     }
HYDRO_KEY_COLUMNS[DATATYPE_NAMES.hydro_levels_max] = HYDRO_KEY_COLUMNS[DATATYPE_NAMES.hydro_levels_min]
HYDRO_KEY_COLUMNS[DATATYPE_NAMES.hydro_gen_max] = HYDRO_KEY_COLUMNS[DATATYPE_NAMES.hydro_gen_min]
HYDRO_VALUE_COLUMNS = {DATATYPE_NAMES.hydro_ror: [COLUMN_NAMES.value],
                       DATATYPE_NAMES.hydro_inflows:
                           ['cum_inflow_into_reservoirs', 'cum_nat_inflow_into_pump-storage_reservoirs'],
                       DATATYPE_NAMES.hydro_levels_min: [COLUMN_NAMES.min_value, COLUMN_NAMES.max_value],
                       DATATYPE_NAMES.hydro_gen_min: [COLUMN_NAMES.min_value, COLUMN_NAMES.max_value]}
HYDRO_VALUE_COLUMNS[DATATYPE_NAMES.hydro_levels_max] = HYDRO_VALUE_COLUMNS[DATATYPE_NAMES.hydro_levels_min]
HYDRO_VALUE_COLUMNS[DATATYPE_NAMES.hydro_gen_max] = HYDRO_VALUE_COLUMNS[DATATYPE_NAMES.hydro_gen_min]
HYDRO_TS_GRANULARITY = {DATATYPE_NAMES.hydro_ror: Timescale.day,
                        DATATYPE_NAMES.hydro_inflows: Timescale.week,
                        DATATYPE_NAMES.hydro_levels_min: Timescale.week,
                        DATATYPE_NAMES.hydro_levels_max: Timescale.week,
                        DATATYPE_NAMES.hydro_gen_min: Timescale.week,
                        DATATYPE_NAMES.hydro_gen_max: Timescale.week}
HYDRO_DEFAULT_VALUES = {DATATYPE_NAMES.hydro_ror: {COLUMN_NAMES.value: 0},
                        DATATYPE_NAMES.hydro_inflows:
                            {'cum_inflow_into_reservoirs': 0, 'cum_nat_inflow_into_pump-storage_reservoirs': 0},
                        # extreme values found in ERAA2023.2 data(over all countries)
                        DATATYPE_NAMES.hydro_levels_min: {COLUMN_NAMES.min_value: 0, COLUMN_NAMES.max_value: 5},
                        # max value is very large; will be bounded by power capacity a bit later
                        DATATYPE_NAMES.hydro_gen_min: {COLUMN_NAMES.min_value: 0, COLUMN_NAMES.max_value: 1e8}
                        }


# method used when resampling from week/day granularity to hourly one -> (uniform) distribution,
# or all at first hourly time-slot of the week/day - and 0 for the rest (typically for constraints
# on min/max reservoir levels)


@dataclass
class ResampleMethods:
    uniform_distrib: str = 'uniform_distrib'
    all_at_first_ts: str = 'all_at_first_ts'


HYDRO_DATA_RESAMPLE_METHODS = {DATATYPE_NAMES.hydro_ror: ResampleMethods.uniform_distrib,
                               DATATYPE_NAMES.hydro_inflows: ResampleMethods.uniform_distrib,
                               DATATYPE_NAMES.hydro_levels_min: ResampleMethods.all_at_first_ts,
                               DATATYPE_NAMES.hydro_levels_max: ResampleMethods.all_at_first_ts,
                               DATATYPE_NAMES.hydro_gen_min: ResampleMethods.all_at_first_ts,  # TODO: check if ok
                               DATATYPE_NAMES.hydro_gen_max: ResampleMethods.all_at_first_ts}  # TODO: idem
# in input data min/max levels are in p.u., relative to stock capacity
# (ATTENTION in PyPSA, absolute values will have to be used)
HYDRO_LEVELS_RESAMPLE_FILLNA_VALS = {COLUMN_NAMES.min_value: 0, COLUMN_NAMES.max_value: 1}
INTERCO_STR_SEP = '2'
INPUT_CY_STRESS_TEST_SUBFOLDER = 'cy_stress-test'


def get_input_lt_uc_optional_files(optional_subfolder: str, all_years: List[int],
                                   all_countries: List[str]) -> List[str]:
    """
    Get list of allowed optional files for a given subfolder, e.g. "hydro"
    :param optional_subfolder
    """
    # hydro allowed files over all (country, year) cases
    if optional_subfolder == DT_SUBFOLDERS.hydro:
        hydro_datatypes = [DATATYPE_NAMES.hydro_ror, DATATYPE_NAMES.hydro_inflows, DATATYPE_NAMES.hydro_levels_min,
                           DATATYPE_NAMES.hydro_gen_min]
        available_cases = product(hydro_datatypes, all_years, all_countries)
        return [f"{HYDRO_FILES_PREFIX_IN_INPUT_FOLDER[hydro_dt]}_{year}_{country}.csv"
                for (hydro_dt, year, country) in available_cases]
    # demand-adder allowed files over all (country, year) cases
    if optional_subfolder == DT_SUBFOLDERS.demand:
        available_cases = product(all_years, all_countries)
        return [f"{DEMAND_ADDER_FILE_PREFIX}_{year}_{country}.csv" for (year, country) in available_cases]


def check_uc_input_folder_content(all_countries: List[str], all_years: List[int] = None):
    uc_countries_folder = uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_SUBFOLDER, 'countries'))
    files = get_files_in_dir(my_dir=uc_countries_folder)
    gitignore_file = '.gitignore'
    if gitignore_file in files:
        files.remove(gitignore_file)
    allowed_files = set([f'{country}.json' for country in all_countries])
    unknown_files = list(set(files) - allowed_files)
    per_fold_unknown_files = {}
    if len(unknown_files) > 0:
        per_fold_unknown_files["root"] = unknown_files
    # optional sub-folder may be present in this folder
    if INPUT_LT_UC_OPTIONAL_SUBFOLDER in os.listdir(uc_countries_folder):
        if all_years is None:
            raise Exception(f"Available years must be provided to check the content of input folder when "
                            f"{INPUT_LT_UC_OPTIONAL_SUBFOLDER} is present")
        optional_folder = os.path.join(uc_countries_folder, INPUT_LT_UC_OPTIONAL_SUBFOLDER)
        # check coherence of optional subfolder names
        allowed_optional_subfolders = {DT_SUBFOLDERS.hydro, DT_SUBFOLDERS.demand}
        optional_subfolders = get_folders_in_dir(my_dir=optional_folder)
        unknown_subfolders = list(set(optional_subfolders) - allowed_optional_subfolders)
        if len(unknown_subfolders) > 0:
            per_fold_unknown_files["optional"] = unknown_subfolders
        # then names of file in each of the optional (allowed) subfolders
        for subfold in allowed_optional_subfolders:
            if subfold in optional_subfolders:
                optional_folder_path = os.path.join(optional_folder, subfold)
                files = get_files_in_dir(my_dir=optional_folder_path)
                allowed_files = set(get_input_lt_uc_optional_files(optional_subfolder=subfold,
                                                                   all_years=all_years, all_countries=all_countries)
                                    )
                unknown_files = list(set(files) - allowed_files)
                if len(unknown_files) > 0:
                    per_fold_unknown_files[f"optional/{subfold}"] = unknown_files
    if len(per_fold_unknown_files) > 0:
        raise Exception(f'Unknown files in UC input (sub-)folder(s) {uc_countries_folder}: {per_fold_unknown_files}. '
                        f'Remove then and re-run')


def get_json_usage_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_FUNC_PARAMS_SUBFOLDER, 'usage_params.json'))


def get_json_fixed_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_ENV_PARAMS_SUBFOLDER, 'elec-europe_params_fixed.json'))


def get_json_eraa_avail_values_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_SUBFOLDER, 'elec-europe_eraa-available-values.json'))


def get_json_params_tb_modif_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_SUBFOLDER, 'elec-europe_params_to-be-modif.json'))


def get_json_solver_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_ENV_PARAMS_SUBFOLDER, 'solver_params.json'))


def get_json_fuel_sources_tb_modif_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_SUBFOLDER, 'fuel_sources_to-be_modif.json'))


def get_json_params_modif_country_files() -> Iterator[str]:
    return map(
        lambda x: uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_COUNTRY_SUBFOLDER, x)),
        filter(lambda x: x.endswith('.json'),
               os.listdir(INPUT_LT_UC_COUNTRY_SUBFOLDER)))


def get_json_pypsa_static_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_ENV_PARAMS_SUBFOLDER, 'pypsa_static_params.json'))


def get_json_data_analysis_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_DATA_ANALYSIS_SUBFOLDER,
                                                    'data-analysis_params_to-be-modif.json'))


def get_json_plot_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_FUNC_PARAMS_SUBFOLDER, 'plot_params.json'))


def get_json_fuel_sources_raw_params_file() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_FUEL_SOURCES_FOLDER, 'raw_params.json'))


def get_demand_adder_optional_folder() -> str:
    return uniformize_path_os(path_str=os.path.join(INPUT_LT_UC_SUBFOLDER,
                                                    INPUT_LT_UC_OPTIONAL_SUBFOLDER, DT_SUBFOLDERS.demand)
                              )


def get_demand_adder_files(target_year: int, countries: list[str]) -> list[str]:
    """
    Get demand-adder files from input/optional/demand folder
    """
    demand_adder_optional_folder = get_demand_adder_optional_folder()
    if not os.path.exists(demand_adder_optional_folder):
        return []
    all_demand_adder_files = os.listdir(demand_adder_optional_folder)
    # keep only the ones corresponding to target year, and selected countries
    selec_demand_adder_files = []
    countries_with_demand_adder = []
    for file in all_demand_adder_files:
        year, country = get_year_and_country_from_data_file(filename=file, prefix=DEMAND_ADDER_FILE_PREFIX)
        if year is not None and year == target_year and country in countries:
            selec_demand_adder_files.append(file)
            countries_with_demand_adder.append(country)
    # info about countries with demand-adder file, that will be used to modify demand data
    demand_adder_folder_relpath = format_relative_path_for_log(get_demand_adder_optional_folder())
    if len(countries_with_demand_adder) > 0:
        logging.info(f"The following countries have demand-adder files in folder {demand_adder_folder_relpath}: "
                     f"{countries_with_demand_adder}\n -> their demand will be modified using these data")
    # warning about non-used demand-adder files
    useless_files = list(set(all_demand_adder_files) - set(selec_demand_adder_files))
    if len(useless_files) > 0:
        logging.warning(f"There are demand-adder files which will not be used in folder {demand_adder_folder_relpath}: "
                        f"{useless_files} \n(do not correspond to selected target year {target_year} "
                        f"and/or countries {countries})")
    return selec_demand_adder_files


def get_network_figure(toy_model_output: bool = False, country: str = None, create_subdir: bool = True,
                       n_bus: int = None, output_tag: str = None) -> str:
    output_folder = (
        set_full_lt_uc_output_folder(folder_type='figures', country=country, toy_model_output=toy_model_output,
                                     output_tag=output_tag)
    )
    if create_subdir:
        make_dir(full_path=output_folder)

    n_bus_suffix = f'_{n_bus}-bus' if n_bus is not None else ''

    return f'{output_folder}/network{n_bus_suffix}.png'


def get_output_file_suffix(country: str, year: int, climatic_year: int = None, start_horizon: datetime = None) -> str:
    cy_suffix = f'_cy{climatic_year}' if climatic_year is not None else ''
    date_suffix = f'_{start_horizon.strftime(DATE_FORMAT_FILE)}' if start_horizon is not None else ''
    return f'{country}_{year}{cy_suffix}{date_suffix}'


def get_output_file_named(name: str, extension: str, output_dir: str, country: str, year: int, climatic_year: int,
                          start_horizon: datetime = None) -> str:
    file_suffix = get_output_file_suffix(country=country, year=year, climatic_year=climatic_year,
                                         start_horizon=start_horizon)
    return f'{output_dir}/{name}_{file_suffix}.{extension}'


def get_figure_file_named(name: str, country: str, year: int, climatic_year: int = None, start_horizon: datetime = None,
                          toy_model_output: bool = False, output_tag: str = None) -> str:
    output_folder = set_full_lt_uc_output_folder(folder_type='figures', country=country,
                                                 toy_model_output=toy_model_output, output_tag=output_tag)
    return get_output_file_named(name=name, extension='png', output_dir=output_folder, country=country, year=year,
                                 climatic_year=climatic_year, start_horizon=start_horizon)


@dataclass
class FigNamesPrefix:
    capacity: str = 'capa'
    link_capa_saturation: str = 'link_capa_sat'
    production: str = 'prod'
    prices: str = 'prices'
    stock_soc: str = 'stock_soc'


def get_output_figure(fig_name: str, country: str, year: int, climatic_year: int = None, start_horizon: datetime = None,
                      toy_model_output: bool = False, create_subdir: bool = True, output_tag: str = None) -> str:
    output_fig_filepath = get_figure_file_named(name=fig_name, country=country, year=year, climatic_year=climatic_year,
                                                start_horizon=start_horizon, toy_model_output=toy_model_output,
                                                output_tag=output_tag)
    if create_subdir:
        subdir = os.path.split(output_fig_filepath)[0]
        make_dir(full_path=subdir)
    return output_fig_filepath


def add_tag_to_output_folder(output_folder: str, tag: str) -> str:
    return f"{output_folder}_{tag}"


def set_full_lt_uc_output_folder(folder_type: str = None, country: str = None, toy_model_output: bool = False,
                                 output_tag: str = None) -> str:
    subfolder = f'{OUTPUT_FOLDER_TOY_MODEL}_{set_country_trigram(country=country)}' if toy_model_output else f'{OUTPUT_FOLDER_LT}'
    folders_tb_join = [subfolder]

    if folder_type is not None:
        # data/fig subfolder, possibly with output tag to distinguish it from previous runs
        if output_tag is not None:
            per_type_subfolder = add_tag_to_output_folder(output_folder=folder_type, tag=output_tag)
        folders_tb_join.append(per_type_subfolder)
    return str(Path(*folders_tb_join))


def get_per_tag_subfolder_idx(subfolder_name: str) -> Optional[int]:
    idx_suffix = subfolder_name.split("_")[-1]
    if idx_suffix.isdigit():
        return int(idx_suffix)
    return None


def get_current_run_idx_in_output_folder(output_folder: str, output_tag: str) -> int:
    data_subfolder_prefix = add_tag_to_output_folder(output_folder=OutputFolderNames.data, tag=output_tag)
    data_subfolder_selec = [elt for elt in os.listdir(output_folder) if elt.startswith(data_subfolder_prefix)]
    past_run_idx = [get_per_tag_subfolder_idx(subfolder_name=elt) for elt in data_subfolder_selec]
    # first subfolder with idx
    if all([idx is None for idx in past_run_idx]):
        return 1
    max_past_idx = max([idx for idx in past_run_idx if idx is not None])
    return max_past_idx + 1


# TODO: merge 2 following functions
def get_csv_file_named(name: str, year: int, climatic_year: int, start_horizon: datetime, country: str = None,
                       toy_model_output: bool = False, create_subdir: bool = True, output_tag: str = None) -> str:
    output_folder = set_full_lt_uc_output_folder(folder_type=OutputFolderNames.data, country=country,
                                                 toy_model_output=toy_model_output, output_tag=output_tag)
    if create_subdir:
        make_dir(full_path=output_folder)

    return get_output_file_named(name, 'csv', output_folder, country, year, climatic_year, start_horizon)


def get_json_file_named(name: str, country: str, year: int, climatic_year: int, start_horizon: datetime,
                        toy_model_output: bool = False, create_subdir: bool = True, output_tag: str = None) -> str:
    output_folder = set_full_lt_uc_output_folder(folder_type=OutputFolderNames.data, country=country,
                                                 toy_model_output=toy_model_output, output_tag=output_tag)
    if create_subdir:
        make_dir(full_path=output_folder)

    return get_output_file_named(name, 'json', output_folder, country, year, climatic_year, start_horizon)


def get_opt_power_file(country: str, year: int, climatic_year: int, start_horizon: datetime,
                       toy_model_output: bool = False, output_tag: str = None) -> str:
    return get_csv_file_named(name='opt_power', country=country, year=year, climatic_year=climatic_year,
                              start_horizon=start_horizon, toy_model_output=toy_model_output, output_tag=output_tag)


def get_storage_opt_dec_file(country: str, year: int, climatic_year: int, start_horizon: datetime,
                             toy_model_output: bool = False, output_tag: str = None) -> str:
    return get_csv_file_named(name='storage_opt_decisions', country=country, year=year, climatic_year=climatic_year,
                              start_horizon=start_horizon, toy_model_output=toy_model_output, output_tag=output_tag)


def get_link_flow_opt_dec_file(country: str, year: int, climatic_year: int, start_horizon: datetime,
                               toy_model_output: bool = False, output_tag: str = None) -> str:
    return get_csv_file_named(name='link-flow_opt_decisions', country=country, year=year, climatic_year=climatic_year,
                              start_horizon=start_horizon, toy_model_output=toy_model_output, output_tag=output_tag)


def get_marginal_prices_file(country: str, year: int, climatic_year: int, start_horizon: datetime,
                             toy_model_output: bool = False, output_tag: str = None) -> str:
    return get_csv_file_named(name='marginal-prices', country=country, year=year, climatic_year=climatic_year,
                              start_horizon=start_horizon, toy_model_output=toy_model_output, output_tag=output_tag)


def get_link_capa_dual_vars_file(year: int, climatic_year: int, start_horizon: datetime, country: str = 'europe',
                                 output_tag: str = None) -> str:
    return get_csv_file_named(name='link-capa-duals', country=country, year=year, climatic_year=climatic_year,
                              start_horizon=start_horizon, output_tag=output_tag)


def get_uc_summary_file(country: str, year: int, climatic_year: int, start_horizon: datetime,
                        toy_model_output: bool = False, output_tag: str = None) -> str:
    return get_json_file_named(name='uc-summary', country=country, year=year, climatic_year=climatic_year,
                               start_horizon=start_horizon, toy_model_output=toy_model_output, output_tag=output_tag)


def get_country_name_from_json(json_path: str | Path) -> str:
    return Path(json_path).stem


def get_year_and_country_from_data_file(filename: str, prefix: str) -> Optional[tuple[int, str]]:
    pattern = rf"^{re.escape(prefix)}_(\d{{4}})_([^\.]+)\.csv$"

    match = re.match(pattern, Path(filename).name)
    if not match:
        logging.warning(f"Filename '{filename}' does not match '{prefix}_<year>_<country>.csv' -> None returned")
        return None, None

    year = int(match.group(1))
    country = match.group(2)

    return year, country
