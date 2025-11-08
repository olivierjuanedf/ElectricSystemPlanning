"""
Read JSON parametrization files... and check coherence of them
"""
import warnings

from common.constants.usage_params_json import EnvPhaseNames
from utils.basic_utils import print_non_default

# deactivate some annoying and useless warnings in pypsa/pandas
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import logging

from common.fuel_sources import set_fuel_sources_from_json, DUMMY_FUEL_SOURCES, DummyFuelNames
fuel_sources = set_fuel_sources_from_json()
from common.logger import init_logger, stop_logger
from common.long_term_uc_io import OUTPUT_FOLDER_LT, set_full_lt_uc_output_folder
from include.dataset_builder import PypsaModel
from utils.dates import get_period_str
from include.dataset import Dataset
from common.constants.optimisation import OPTIM_RESOL_STATUS
from utils.read import read_and_check_uc_run_params, read_and_check_pypsa_static_params, read_given_phase_plot_params

# name of current "phase" (of the course), the one associated to this script: a multi-zone (Eur.) Unit Commitment model
from common.constants.usage_params_json import EnvPhaseNames
phase_name = EnvPhaseNames.multizones_uc_model

usage_params, eraa_data_descr, uc_run_params = read_and_check_uc_run_params(phase_name=EnvPhaseNames.multizones_uc_model)

output_folder = set_full_lt_uc_output_folder()
logger = init_logger(logger_dir=output_folder, logger_name='eraa_lt_uc_pb.log',
                     log_level=usage_params.log_level)

logging.info(f'Start ERAA-PyPSA long-term European Unit Commitment (UC) simulation')

"""
Get needed data (demand, RES Capa. Factors, installed generation capacities)
"""
uc_period_msg = get_period_str(period_start=uc_run_params.uc_period_start,
                               period_end=uc_run_params.uc_period_end)

logging.info(f'Read needed ERAA ({eraa_data_descr.eraa_edition}) data for period {uc_period_msg}')
# initialize dataset object
eraa_dataset = Dataset(source=f'eraa_{eraa_data_descr.eraa_edition}',
                       agg_prod_types_with_cf_data=eraa_data_descr.agg_prod_types_with_cf_data,
                       is_stress_test=uc_run_params.is_stress_test)

eraa_dataset.get_countries_data(uc_run_params=uc_run_params,
                                aggreg_prod_types_def=eraa_data_descr.aggreg_prod_types_def)

logging.info('Get generation units data, from both ERAA data - read just before - and JSON parameter file')
eraa_dataset.get_generation_units_data(uc_run_params=uc_run_params,
                                       pypsa_unit_params_per_agg_pt=eraa_data_descr.pypsa_unit_params_per_agg_pt,
                                       units_complem_params_per_agg_pt=eraa_data_descr.units_complem_params_per_agg_pt)
eraa_dataset.set_committable_param()
# TODO[perpi]: connect this properly
# if len(uc_run_params.updated_fuel_sources_params) > 0:
#   generation_units_data = overwrite_gen_units_fuel_src_params(generation_units_data=generation_units_data, 
#                                                               updated_fuel_sources_params=uc_run_params.updated_fuel_sources_params)

logging.info(
    'Check that "minimal" PyPSA parameters for unit creation have been provided (in JSON files)/read (from ERAA data)')
pypsa_static_params = read_and_check_pypsa_static_params()
eraa_dataset.control_min_pypsa_params_per_gen_units(
    pypsa_min_unit_params_per_agg_pt=pypsa_static_params.min_unit_params_per_agg_pt)

# create PyPSA network
pypsa_model = PypsaModel(name='my little europe')
date_idx = eraa_dataset.demand[uc_run_params.selected_countries[0]].index
import pandas as pd

horizon = pd.date_range(
    start=uc_run_params.uc_period_start.replace(year=uc_run_params.selected_target_year),
    end=uc_run_params.uc_period_end.replace(year=uc_run_params.selected_target_year),
    freq='h'
)
pypsa_model.init_pypsa_network(date_idx=date_idx, date_range=horizon)
# add GPS coordinates
selec_countries_gps_coords = \
    {country: gps_coords for country, gps_coords in eraa_data_descr.gps_coordinates.items()
     if country in uc_run_params.selected_countries}
pypsa_model.add_gps_coordinates(countries_gps_coords=selec_countries_gps_coords)
fuel_sources |= DUMMY_FUEL_SOURCES
pypsa_model.add_energy_carriers(fuel_sources=fuel_sources)
pypsa_model.add_generators(generators_data=eraa_dataset.generation_units_data)
pypsa_model.add_loads(demand=eraa_dataset.demand, carrier_name=DummyFuelNames.load)
pypsa_model.add_interco_links(countries=uc_run_params.selected_countries, interco_capas=eraa_dataset.interco_capas)
logging.info(f'PyPSA network main properties: {pypsa_model.network}')
# plot network
from utils.read import read_plot_params
per_dim_plot_params = read_plot_params()
fig_style = read_given_phase_plot_params(phase_name=phase_name)
print_non_default(obj=fig_style, obj_name=f'FigureStyle - for phase {phase_name}')

pypsa_model.plot_network(toy_model_output=False)
# use alternatively set_optim_solver(name='gurobi', licence_file='gurobi.lic') to use Gurobi,
# with gurobi.lic file provided at root of this project (see readme.md on procedure to obtain such a lic file)
pypsa_model.set_optim_solver()
result = pypsa_model.optimize_network(year=uc_run_params.selected_target_year,
                                      n_countries=len(uc_run_params.selected_countries),
                                      period_start=uc_run_params.uc_period_start)

pypsa_opt_resol_status = OPTIM_RESOL_STATUS.optimal
# if optimal resolution status, save output data and plot associated figures
if result[1] == pypsa_opt_resol_status:
    # get objective value, and associated optimal decisions / dual variables
    objective_value = pypsa_model.get_opt_value(pypsa_resol_status=pypsa_opt_resol_status)
    pypsa_model.get_prod_var_opt()
    pypsa_model.get_storage_vars_opt()
    pypsa_model.get_sde_dual_var_opt()
    # get plot parameters associated to aggreg. production types
    from common.constants.datadims import DataDimensions
    plot_params_agg_pt = per_dim_plot_params[DataDimensions.agg_prod_type]
    plot_params_zone = per_dim_plot_params[DataDimensions.zone]

    # plot - per country - opt prod profiles 'stacked'
    for country in uc_run_params.selected_countries:
        pypsa_model.plot_opt_prod_var(plot_params_agg_pt=plot_params_agg_pt, country=country,
                                      year=uc_run_params.selected_target_year,
                                      climatic_year=uc_run_params.selected_climatic_year,
                                      start_horizon=uc_run_params.uc_period_start)
    # plot 'marginal price' figure
    pypsa_model.plot_marginal_price(plot_params_zone=plot_params_zone, year=uc_run_params.selected_target_year,
                                    climatic_year=uc_run_params.selected_climatic_year,
                                    start_horizon=uc_run_params.uc_period_start)

    # save optimal prod. decision to an output file
    pypsa_model.save_opt_decisions_to_csv(year=uc_run_params.selected_target_year,
                                          climatic_year=uc_run_params.selected_climatic_year,
                                          start_horizon=uc_run_params.uc_period_start)

    # save marginal prices to an output file
    pypsa_model.save_marginal_prices_to_csv(year=uc_run_params.selected_target_year,
                                            climatic_year=uc_run_params.selected_climatic_year,
                                            start_horizon=uc_run_params.uc_period_start)
else:
    logging.info(
        f'Optimisation resolution status is not {pypsa_opt_resol_status} -> output data (resp. figures) cannot be saved (resp. plotted)')

logging.info('THE END of ERAA-PyPSA long-term UC simulation!')
stop_logger()
