# -*- coding: utf-8 -*-
"""
First very simple toy Unit Commitment model of Italy zone - alone -> with PyPSA and ERAA data
"""
import warnings
# deactivate some annoying and useless warnings in pypsa/pandas
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# use global constant names of different prod. types to be sure of extracting data wo any pb  
from long_term_uc.common.constants.prod_types import ProdTypeNames
AGG_PROD_TYPES_DEF = {
    'res_capa-factors': {
        'solar_pv': [ProdTypeNames.solar_pv_in_cf_data],
        'solar_thermal': [ProdTypeNames.csp_no_storage],
        'wind_offshore': [ProdTypeNames.wind_offshore],
        'wind_onshore': [ProdTypeNames.wind_onshore]
    },
    'generation_capas': {
        'batteries': [ProdTypeNames.batteries],
        'biofuel': [ProdTypeNames.biofuel],
        'coal': [ProdTypeNames.coal, ProdTypeNames.hard_coal, ProdTypeNames.lignite],
        'dsr': [ProdTypeNames.demand_side_response],
        'gas': [ProdTypeNames.gas],
        'hydro_pondage': [ProdTypeNames.hydro_pondage],
        'hydro_pump_storage_closed_loop': [ProdTypeNames.hydro_pump_storage_closed],
        'hydro_pump_storage_open_loop': [ProdTypeNames.hydro_pump_storage_open],
        'hydro_reservoir': [ProdTypeNames.hydro_reservoir],
        'hydro_run_of_river': [ProdTypeNames.hydro_ror],
        'nuclear': [ProdTypeNames.nuclear], 
        'oil': [ProdTypeNames.oil],
        'others_fatal': [ProdTypeNames.others_non_renewable, ProdTypeNames.others_renewable],
        'solar_pv': [ProdTypeNames.solar_pv],
        'solar_thermal': [ProdTypeNames.solar_thermal],
        'wind_offshore': [ProdTypeNames.wind_offshore], 
        'wind_onshore': [ProdTypeNames.wind_onshore]
    }
}

"""
# I) Set global parameters for the case simulated
"""
# unique country modeled in this example -> some comments are provided 
# below to explain how PyPSA model could be extended to a multiple countries case
# -> look at "[Multiple-countries ext.]" tags
country = 'italy'
# select first ERAA year available, as an example 
# -> see values in input/long_term_uc/elec-europe_eraa-available-values.json
year = 2025  # Can be changed -> available values [2025, 2033]
# and a given "climatic year" (to possibly test different climatic*weather conditions)
# -> idem
# N.B. Ask Boutheina OUESLATI on Wednesday to get an idea of the 'climatic' versus 'weather' conditions 
climatic_year = 1989  # Can be changed -> available values [1982, 1989, 1996, 2003, 2010, 2016]
# TODO: used?
agg_prod_types_selec = [ProdTypeNames.wind_offshore, ProdTypeNames.wind_offshore, ProdTypeNames.solar_pv]

"""
II) Initialize a UCRunParams object 
"""
# N.B. UC = Unit Commitment, i.e. supply-demand equilibrium problem 
# - for given electricity generation capacities
from datetime import datetime, timedelta
from long_term_uc.common.uc_run_params import UCRunParams
# Set start and end date corresponding to the period to be simulated
# ATTENTION: uc_period_end not included -> here period {1900/1/1 00:00, 1900/1/1 01:00, ..., 1900/1/13 23:00}
# N.B. Calendar of year 1900 used here, to make explicit the fact that ERAA data are 'projected' 
# on a fictive calendar; as made of 52 full weeks
uc_period_start = datetime(year=1900, month=1, day=1)
uc_period_end = uc_period_start + timedelta(days=14)
selected_countries = [country]  # [Multiple-countries ext.] List with multiple country names
uc_run_params = UCRunParams(selected_countries=selected_countries, selected_target_year=year, 
                            selected_climatic_year=climatic_year, 
                            selected_prod_types={'italy': agg_prod_types_selec},
                            uc_period_start=uc_period_start,
                            uc_period_end=uc_period_end)

import pandas as pd
horizon = pd.date_range(
    start = uc_period_start.replace(year=uc_run_params.selected_target_year),
    end = uc_period_end.replace(year=uc_run_params.selected_target_year),
    freq = 'h'
)

# initialize dataset object
from long_term_uc.common.constants.extract_eraa_data import ERAADatasetDescr
from long_term_uc.include.dataset import Dataset
from long_term_uc.utils.read import check_and_load_json_file
from long_term_uc.common.long_term_uc_io import get_json_fixed_params_file  

json_fixed_params_file = get_json_fixed_params_file()
json_params_fixed = check_and_load_json_file(json_file=json_fixed_params_file,
                                             file_descr='JSON fixed params')
json_available_values_dummy = {'available_climatic_years': None, 
                               'available_countries': None, 
                               'available_aggreg_prod_types': None,
                               'available_intercos': None,
                               'available_target_years': None}
json_params_fixed |= json_available_values_dummy

eraa_data_descr = ERAADatasetDescr(**json_params_fixed)

# initialize dataset object
eraa_dataset = Dataset(source=f'eraa_{eraa_data_descr.eraa_edition}',
                       agg_prod_types_with_cf_data=eraa_data_descr.agg_prod_types_with_cf_data,
                       is_stress_test=uc_run_params.is_stress_test)
    
"""
III) Get needed data - from ERAA csv files in data\\ERAA_2023-2
"""

# III.1) Get data for Italy... just for test -> data used when writing PyPSA model will be re-obtained afterwards
eraa_dataset.get_countries_data(uc_run_params=uc_run_params,
                                aggreg_prod_types_def=eraa_data_descr.aggreg_prod_types_def)

# III.2) In this case, decompose aggreg. CF data into three sub-dicts (for following ex. to be more explicit)
from long_term_uc.utils.df_utils import selec_in_df_based_on_list
solar_pv = {
    country: selec_in_df_based_on_list(df=eraa_dataset.agg_cf_data[country], selec_col='production_type_agg',
                                       selec_vals=[ProdTypeNames.solar_pv], rm_selec_col=True)
}
wind_on_shore = {
    country: selec_in_df_based_on_list(df=eraa_dataset.agg_cf_data[country], selec_col='production_type_agg',
                                       selec_vals=[ProdTypeNames.wind_onshore], rm_selec_col=True)
}
wind_off_shore = {
    country: selec_in_df_based_on_list(df=eraa_dataset.agg_cf_data[country], selec_col='production_type_agg',
                                       selec_vals=[ProdTypeNames.wind_offshore], rm_selec_col=True)
}

"""
IV) Build PyPSA model - with unique country (Italy here)
"""
# IV.1) Initialize PyPSA Network (basis of all your simulations this week!). 
import pypsa
print('Initialize PyPSA network')
# Here snapshots is used to defined the temporal period associated to considered UC model
# -> for ex. as a list of indices (other formats; like data ranges can be used instead) 
from long_term_uc.include.dataset_builder import PypsaModel
pypsa_model = PypsaModel(name='my little europe')
date_idx = eraa_dataset.demand[uc_run_params.selected_countries[0]].index
# set a date horizon, to have more explicit axis labels hereafter
import pandas as pd
horizon = pd.date_range(
    start = uc_run_params.uc_period_start.replace(year=uc_run_params.selected_target_year),
    end = uc_run_params.uc_period_end.replace(year=uc_run_params.selected_target_year),
    freq = 'h'
)
pypsa_model.init_pypsa_network(date_idx=date_idx, date_range=horizon)

# And print it to check that for now it is... empty
print(pypsa_model.network)

#################################################
# KEY POINT: main parameters needed for Italy description in PyPSA are set in script
# long_term_uc.toy_model_params.italy_parameters.py
# To get the meaning and format of main PyPSA objects/attributes look 
# at file doc/toy-model_tutorial.md
#################################################

# IV.2) Add bus for considered country
# N.B. Italy coordinates set randomly! (not useful in the calculation that will be done this week)
from long_term_uc.toy_model_params.italy_parameters import gps_coords
unique_country = 'italy' 
coordinates = {unique_country: gps_coords}
# IV.2.1) For brevity, set country trigram as the 'id' of this zone in following model definition (and observed outputs)
from long_term_uc.include.dataset_builder import set_country_trigram
country_trigram = set_country_trigram(country=country)
# N.B. Multiple bus would be added if multiple countries were considered
pypsa_model.add_gps_coordinates(countries_gps_coords=coordinates)
# [Multiple-count. ext., start] Loop over the different countries to add an associated bus
# for country in modeled_countries:
#    network.add('Bus', name=country_trigram, x=coordinates[country][0], y=coordinates[country][1])
# [Multiple-count. ext., end]

# IV.4) [VERY KEY STAGE] Generators definition, beginning with only simple parameters. 
# Almost 'real Italy'... excepting hydraulic storage and Demand-Side Response capacity 
# (we will come back on this later)
# Thanks to Tim WALTER - student of last year ATHENS course, detailed parameter values associated 
# to different fuel sources are available in following dictionary. You can use it or search/define 
# fictive alternative values instead -> plenty infos on Internet on this... sometimes of 'varying' quality! 
# (keeping format of dataclass - sort of enriched dictionary -, just change values in 
# file long_term_uc/common/fuel_sources.py)
from long_term_uc.common.fuel_sources import FUEL_SOURCES
from long_term_uc.toy_model_params.italy_parameters import get_generators, set_gen_as_list_of_gen_units_data
# IV.4.1) get generators to be set on the unique considered bus here 
# -> from long_term_uc.toy_model_params.italy_parameters.py script
generators = get_generators(country_trigram=country_trigram, fuel_sources=FUEL_SOURCES, 
                            wind_on_shore_data=wind_on_shore[country], wind_off_shore_data=wind_off_shore[country],
                            solar_pv_data=solar_pv[country])
# set generation units data from this list
from long_term_uc.include.dataset_builder import GenerationUnitData
generation_units_data = set_gen_as_list_of_gen_units_data(generators=generators)
eraa_dataset.set_generation_units_data(gen_units_data={unique_country: generation_units_data})

# IV.4.2) Loop over previous list of dictionaries to add each of the generators to PyPSA network
# [Coding trick] ** used to 'unpack' the dictionary as named parameters
pypsa_model.add_energy_carrier(fuel_sources=FUEL_SOURCES)
pypsa_model.add_generators(generators_data=eraa_dataset.generation_units_data)

# [Multiple-count. ext., start] Idem but adding the different generators to the bus (country) they are connected to
# -> a correspondence (for ex. with a dictionary) between bus names and list of associated 
# generators is then needed
# [Multiple-count. ext., end]

# IV.5) Add load
# N.B. 'carrier' associated to demand here just to explicit that an AC current network is considered
# IV.5.1) Setting attribute values in a dictionary
# loads = [
#    {
#        'name': f'{country_trigram}-load', 'bus': country_trigram,
#        'carrier': 'AC', 'p_set': demand[country]['value'].values
#    }
#]
# IV.5.2) Then adding Load objects to PyPSA model
# for load in loads:
#    network.add('Load', **load)

# [Coding trick] f'{my_var} is associated to {my_country}' is a f-string or formatted-string (https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals)
# [Multiple-count. ext., start] Multiple dictionaries in previous list, 
# each of them corresponding to a given bus (country)
# [Multiple-count. ext., end]
# Here attribute eraa_dataset.demand has a unique key -> 'italy'
pypsa_model.add_loads(demand=eraa_dataset.demand)

# IV.6) A few prints to check/observe that created PyPSA model be coherent 
# IV.6.1) Print the network after having completed it
print(f'PyPSA network main properties: {pypsa_model.network}')
# IV.6.2) And plot it. Surely better when having multiple buses (countries)!!
# plot network
from long_term_uc.include.plotter import PlotParams
plot_params = PlotParams()
plot_params.read_and_check()
pypsa_model.plot_network()
# IV.6.3) Print out list of generators
print(pypsa_model.network.generators)

# IV.7) 'Optimize network' i.e., solve the associated Unit-Commitment problem
# IV.7.1) Solve and print result. N.B. Default solver used is highs, that is 'sufficient' for a
# 1-zone model as the one solved here
result = pypsa_model.optimize_network(year=uc_run_params.selected_target_year,
                                      n_countries=len(uc_run_params.selected_countries),
                                      period_start=uc_run_params.uc_period_start)
# IV.7.2) [Optional] For those who want to get a standard .lp file containing 
# the equations associated to the solved problem
# -> will be saved in output folder output/long_term_uc/data
# you can observe if you find the equations corresponding to the UC problem modeled

print(result)  # Note 2nd component of result, the resolution status (optimal?)
# get objective value, and associated optimal decisions / dual variables
from long_term_uc.utils.pypsa_utils import OPTIM_RESOL_STATUS
pypsa_opt_resol_status = OPTIM_RESOL_STATUS.optimal
# if optimal resolution status, save output data and plot associated figures
if result[1] == pypsa_opt_resol_status:
    objective_value = pypsa_model.get_opt_value(pypsa_resol_status=pypsa_opt_resol_status)
    print(f'Total cost at optimum: {objective_value:.2f}')
    pypsa_model.get_prod_var_opt()
    pypsa_model.get_storage_vars_opt()
    pypsa_model.get_sde_dual_var_opt()

    # IV.8) Plot a few info/results
    print('Plot installed capas (parameters), generation and prices (optim. outputs) figures')
    # IV.8.1) Plot installed capacities
    pypsa_model.plot_installed_capas(country=unique_country, year=uc_run_params.selected_target_year)
    # IV.8.2) Plot 'stack' of optimized production profiles -> key graph to interpret UC solution -> will be 
    # saved in file output/long_term_uc/figures/prod_italy_{year}_{period start, under format %Y-%m-%d}.png
    pypsa_model.plot_opt_prod_var(plot_params=plot_params, country=unique_country,
                                year=uc_run_params.selected_target_year,
                                climatic_year=uc_run_params.selected_climatic_year,
                                start_horizon=uc_run_params.uc_period_start)
    # IV.8.2bis) Specific prod. profile: the one of fictive failure asset
    pypsa_model.plot_failure_at_opt(country=unique_country, year=uc_run_params.selected_target_year,
                                    climatic_year=uc_run_params.selected_climatic_year,
                                    start_horizon=uc_run_params.uc_period_start)
    # IV.8.3) Finally, 'marginal prices' -> QUESTION: meaning? 
    # -> saved in file output/long_term_uc/figures/prices_italy_{year}_{period start, under format %Y-%m-%d}.png
    # QUESTION: how can you interpret the very constant value plotted? 
    pypsa_model.plot_marginal_price(plot_params=plot_params, year=uc_run_params.selected_target_year,
                                    climatic_year=uc_run_params.selected_climatic_year,
                                    start_horizon=uc_run_params.uc_period_start) 


    # IV.9) Save optimal decisions to output csv files
    print('Save optimal dispatch decisions to .csv file')
    # save optimal prod. decision to an output file
    pypsa_model.save_opt_decisions_to_csv(year=uc_run_params.selected_target_year,
                                        climatic_year=uc_run_params.selected_climatic_year,
                                        start_horizon=uc_run_params.uc_period_start)

    # save marginal prices to an output file
    pypsa_model.save_marginal_prices_to_csv(year=uc_run_params.selected_target_year,
                                            climatic_year=uc_run_params.selected_climatic_year,
                                            start_horizon=uc_run_params.uc_period_start)
else:
    print(f'Optimisation resolution status is not {pypsa_opt_resol_status} -> output data (resp. figures) cannot be saved (resp. plotted), excepting installed capas one')
    pypsa_model.plot_installed_capas(country=unique_country, year=uc_run_params.selected_target_year)

print(f'THE END of ERAA-PyPSA long-term UC toy model of country {unique_country} simulation!')
