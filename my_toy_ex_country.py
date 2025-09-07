# -*- coding: utf-8 -*-
"""
First very simple toy Unit Commitment model of... YOUR COUNTRY zone - alone -> with PyPSA and ERAA data
-> 
(i) copy/paste/rename this file according to your country name
(ii) copy/paste pieces of code from my_toy_ex_italy.py to this created file
(iii) Also, set the parameters of your country electricity generators in file {your country}_parameters.py,
following/adapting model of long_term_uc/toy_model_params/italy_parameters.py
"""
import warnings
#deactivate some annoying and useless warnings in pypsa/pandas
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# use global constant names of different prod. types to be sure of extracting data wo any pb  
from long_term_uc.common.constants.prod_types import ProdTypeNames
AGG_PROD_TYPES_DEF = {
    'res_capa-factors': {
        'solar_pv': [ProdTypeNames.solar_pv_in_cf_data],
        'solar_thermal': [ProdTypeNames.solar_thermal],
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

