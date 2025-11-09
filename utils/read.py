import json
from typing import List, Dict, Optional
import logging

from common.long_term_uc_io import get_json_usage_params_file, get_json_fixed_params_file, \
    get_json_eraa_avail_values_file, get_json_params_tb_modif_file, get_json_pypsa_static_params_file, \
    get_json_params_modif_country_files, get_json_fuel_sources_tb_modif_file, \
    get_json_data_analysis_params_file, get_json_plot_params_file
from common.constants.extract_eraa_data import ERAADatasetDescr, \
    PypsaStaticParams, UsageParameters
from common.constants.uc_json_inputs import CountryJsonParamNames, EuropeJsonParamNames, ALL_KEYWORD
from common.constants.usage_params_json import USAGE_PARAMS_SHORT_NAMES, EnvPhaseNames
from common.uc_run_params import UCRunParams
from include.dataset_analyzer import DataAnalysis
from common.plot_params import PlotParams, DEFAULT_PLOT_DIMS_ORDER
from utils.dir_utils import check_file_existence
from utils.plot import FigureStyle


def check_and_load_json_file(json_file: str, file_descr: str = None) -> dict:
    check_file_existence(file=json_file, file_descr=file_descr)

    f = open(json_file, mode='r', encoding='utf-8')

    # rk: when reading null values in a JSON file they are converted to None
    json_data = json.loads(f.read())

    return json_data


def apply_per_country_json_file_params(countries_data: dict, available_countries: List[str],
                                       mode_name: str, team_name: str) -> dict:
    for file in get_json_params_modif_country_files():
        json_country = check_and_load_json_file(
            json_file=file,
            file_descr='JSON country capacities'
        )
        country = json_country[CountryJsonParamNames.team]
        # TODO[CR]: solo check from global constant/Mode defined in extract_eraa_data.py
        if mode_name == 'solo' and team_name != country:
            continue
        if country not in available_countries:
            logging.error(f'Incorrect country found in file {file}: {country} is not available in dataset')
            exit(1)
        for k, _ in countries_data.items():
            # TODO[CR]: solo check from global constant/Mode defined in extract_eraa_data.py
            if mode_name == 'solo':
                for c, _ in json_country[k].items():
                    logging.info(f'Updating {k} for country {c} from file {file}')
                    countries_data[k][c] = json_country[k][c]
            else:
                logging.info(f'Updating {k} for {country} from file {file}')
                countries_data[k][country] = json_country[k][country]
                for c, _ in json_country[k].items():
                    if c != country:
                        logging.warning(f'Ignoring {k} for {country} from file {file}')
    return countries_data


def update_country_json_params(countries_data: dict, json_params_tb_modif: dict) -> (dict, dict):
    """
    Update parameter values obtained from collective file input/long_term_uc/elec-europe_params_to-be-modif.json
    based on the ones of files input/long_term_uc/countries/{country.json}
    """
    selected_pt_param_name = CountryJsonParamNames.selected_prod_types
    if len(countries_data[selected_pt_param_name]) > 0:
        for elt_country, new_prod_types in countries_data[selected_pt_param_name].items():
            logging.info(f'Selected production type overwritten (not all the ones from ERAA) for {elt_country}')
            json_params_tb_modif[selected_pt_param_name][elt_country] = new_prod_types
    # suppress pt selection key in countries data dict, not to have multiple values for same attr.
    # when creating UCRunParams object hereafter
    del countries_data[selected_pt_param_name]
    return countries_data, json_params_tb_modif


def set_json_usage_params_data() -> dict:
    # read
    json_usage_params_data = check_and_load_json_file(json_file=get_json_usage_params_file(),
                                                      file_descr='JSON usage params')
    # replace long key names by short names (attribute names of following object created)
    json_usage_params_data = {USAGE_PARAMS_SHORT_NAMES[key]: val for key, val in json_usage_params_data.items()}
    return json_usage_params_data


def set_json_eraa_avail_values() -> dict:
    # read
    json_eraa_avail_values = check_and_load_json_file(json_file=get_json_eraa_avail_values_file(),
                                                      file_descr='JSON ERAA available values')
    # add 'available_' to the different keys of JSON available values to make them more explicit in the following
    json_eraa_avail_values = {f'available_{key}': val for key, val in json_eraa_avail_values.items()}
    return json_eraa_avail_values


def set_json_params_fixed() -> dict:
    json_params_fixed = check_and_load_json_file(json_file=get_json_fixed_params_file(), file_descr='JSON fixed params')
    json_eraa_avail_values = set_json_eraa_avail_values()
    # put this dictionary of available values into the 'fixed values' one
    json_params_fixed |= json_eraa_avail_values
    return json_params_fixed


def set_json_params_tb_modif() -> dict:
    return check_and_load_json_file(json_file=get_json_params_tb_modif_file(), file_descr='JSON params to be modif.')


def set_json_fuel_sources_tb_modif() -> dict:
    return check_and_load_json_file(json_file=get_json_fuel_sources_tb_modif_file(),
                                    file_descr='JSON fuel sources params to be modif.')


def set_usage_params(json_usage_params_data: dict) -> UsageParameters:
    usage_params = UsageParameters(**json_usage_params_data)
    usage_params.process()
    return usage_params


def set_eraa_data_descr(json_params_fixed: dict) -> ERAADatasetDescr:
    eraa_data_descr = ERAADatasetDescr(**json_params_fixed)
    eraa_data_descr.check_types()
    eraa_data_descr.process()
    return eraa_data_descr


def set_uc_run_params(json_params_tb_modif: dict, countries_data: dict, json_fuel_sources_tb_modif: dict,
                      eraa_data_descr: ERAADatasetDescr) -> UCRunParams:
    uc_run_params = UCRunParams(**json_params_tb_modif, **countries_data,
                                updated_fuel_sources_params=json_fuel_sources_tb_modif)
    uc_run_params.process(available_countries=eraa_data_descr.available_countries)
    uc_run_params.set_is_stress_test(avail_cy_stress_test=eraa_data_descr.available_climatic_years_stress_test)
    uc_run_params.coherence_check(eraa_data_descr=eraa_data_descr)
    return uc_run_params


def set_countries_data(usage_params: UsageParameters, phase_name: str, available_countries: List[str],
                       json_params_tb_modif: dict) -> (dict, dict):
    selected_pt_param_name = CountryJsonParamNames.selected_prod_types
    countries_data = {
        CountryJsonParamNames.capacities_tb_overwritten: {},
        selected_pt_param_name: {}
    }
    # apply data selection - based on the different JSON files to be modified, and the chosen mode (solo/team)
    apply_per_country_params = usage_params.apply_per_country_json_file_params[phase_name]
    if apply_per_country_params:
        countries_data = (
            apply_per_country_json_file_params(countries_data=countries_data, available_countries=available_countries,
                                               mode_name=usage_params.mode, team_name=usage_params.team)
        )

    # init. selected prod. types, with 'all' value for all selected countries
    json_params_tb_modif[selected_pt_param_name] = \
        {c: [ALL_KEYWORD] for c in json_params_tb_modif[EuropeJsonParamNames.selected_countries]}
    countries_data, json_params_tb_modif = (
        update_country_json_params(countries_data=countries_data, json_params_tb_modif=json_params_tb_modif)
    )

    return countries_data, json_params_tb_modif


def read_and_check_uc_run_params(phase_name: str, get_only_eraa_data_descr: bool = False) \
        -> tuple[Optional[UsageParameters], ERAADatasetDescr, Optional[UCRunParams]]:
    """
    Read and check parameters for UC run - based on different JSON files (with only part of them that can be modified
    by the users/students in this environment)
    :param phase_name: name of the phase for which this function is run (data analysis, 1-zone UC toy model, X-zones
    Eur. UC model)
    :param get_only_eraa_data_descr: used for the runner
    """
    if get_only_eraa_data_descr:
        info_msg = 'Get ERAA data description parameters'
    else:
        json_params_tb_modif_file = get_json_params_tb_modif_file()
        info_msg = f'Read and check long-term UC parameters; the ones modified in file {json_params_tb_modif_file}'
    logging.info(info_msg)

    # check that modifications in JSON in which it is allowed are allowed/coherent
    logging.info('... and check that modifications done are coherent with available ERAA data')
    if not get_only_eraa_data_descr:
        # Get JSON usage params, the ones to control the behaviour of the UC run -> UsageParameters object
        usage_params = set_usage_params(json_usage_params_data=set_json_usage_params_data())
    else:
        usage_params = None

    # get JSON fixed parameters -> ERAA data description object - including in particular the set of
    # available values for countries, (climatic) years, etc.
    eraa_data_descr = set_eraa_data_descr(json_params_fixed=set_json_params_fixed())

    # Set countries data, applying data selection/overwriting based on JSON file with values to be modified
    if not get_only_eraa_data_descr:
        json_params_tb_modif = set_json_params_tb_modif()
        countries_data, json_params_tb_modif = (
            set_countries_data(usage_params=usage_params, phase_name=phase_name,
                               available_countries=eraa_data_descr.available_countries,
                               json_params_tb_modif=json_params_tb_modif)
        )

    # init. UC run params object
    if not get_only_eraa_data_descr:
        # fuel sources values modif.
        json_fuel_sources_tb_modif = set_json_fuel_sources_tb_modif()
        uc_run_params = set_uc_run_params(json_params_tb_modif=json_params_tb_modif, countries_data=countries_data,
                                          json_fuel_sources_tb_modif=json_fuel_sources_tb_modif,
                                          eraa_data_descr=eraa_data_descr)
    else:
        uc_run_params = None

    return usage_params, eraa_data_descr, uc_run_params


def read_and_check_pypsa_static_params() -> PypsaStaticParams:
    json_pypsa_static_params_file = get_json_pypsa_static_params_file()
    logging.debug(f'Read and check PyPSA static parameters file {json_pypsa_static_params_file}')
    json_pypsa_static_params = check_and_load_json_file(json_file=json_pypsa_static_params_file,
                                                        file_descr='JSON PyPSA static params')
    pypsa_static_params = PypsaStaticParams(**json_pypsa_static_params)
    pypsa_static_params.check_types()
    pypsa_static_params.process()
    return pypsa_static_params


def read_and_check_data_analysis_params(eraa_data_descr: ERAADatasetDescr) -> List[DataAnalysis]:
    json_data_analysis_params_file = get_json_data_analysis_params_file()
    logging.info(f'Read and check data analysis parameters file; '
                 f'the ones modified in file {json_data_analysis_params_file}')

    json_data_analysis_params = check_and_load_json_file(json_file=json_data_analysis_params_file,
                                                         file_descr='JSON data analysis params')
    data_analysis_params = json_data_analysis_params['data_analysis_list']
    data_analyses = [DataAnalysis(**param_vals) for param_vals in data_analysis_params]
    # check types
    for elt_analysis in data_analyses:
        elt_analysis.check_types()
        elt_analysis.process(eraa_data_descr=eraa_data_descr)
        elt_analysis.coherence_check(eraa_data_descr=eraa_data_descr)
    return data_analyses


def read_given_phase_plot_params(phase_name: str) -> FigureStyle:
    json_plot_params_file = get_json_plot_params_file()
    logging.debug(f'Read and check {phase_name} plot parameters file: {json_plot_params_file}')
    json_data_analysis_plot_params = check_and_load_json_file(json_file=json_plot_params_file,
                                                              file_descr=f'JSON {phase_name} plot params')
    return FigureStyle(**json_data_analysis_plot_params[f'fig_style_{phase_name}'])


def read_plot_params() -> Dict[str, PlotParams]:
    json_plot_params_file = get_json_plot_params_file()
    logging.info(f'Read and check plot parameters file: {json_plot_params_file}')

    json_plot_params = check_and_load_json_file(json_file=json_plot_params_file, file_descr='JSON plot params')
    # remove elt used only for FigureStyle of data analysis
    for phase_name in [EnvPhaseNames.data_analysis, EnvPhaseNames.monozone_toy_uc_model,
                       EnvPhaseNames.multizones_uc_model]:
        current_key = f'fig_style_{phase_name}'
        if current_key in json_plot_params:
            del json_plot_params[current_key]

    per_dim_plot_params = {}
    for plot_dim in DEFAULT_PLOT_DIMS_ORDER:
        dict_params = json_plot_params[plot_dim]
        dict_params['dimension'] = plot_dim
        current_plot_params = PlotParams(**dict_params)
        current_plot_params.process()
        # TODO: add checker
        per_dim_plot_params[plot_dim] = current_plot_params

    return per_dim_plot_params
