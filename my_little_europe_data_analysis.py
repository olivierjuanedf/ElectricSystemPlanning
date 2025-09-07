import numpy as np
import logging

from long_term_uc.common.constants.datatypes import DATATYPE_NAMES, UNITS_PER_DT
from long_term_uc.common.logger import init_logger, stop_logger
from long_term_uc.common.long_term_uc_io import OUTPUT_DATA_ANALYSIS_FOLDER, OUTPUT_FOLDER_LT
from long_term_uc.utils.basic_utils import get_period_str
from long_term_uc.include.dataset import Dataset
from long_term_uc.include.dataset_analyzer import ANALYSIS_TYPES
from long_term_uc.include.uc_timeseries import UCTimeseries, set_uc_ts_name
from long_term_uc.utils.read import read_and_check_data_analysis_params, read_and_check_uc_run_params


usage_params, eraa_data_descr, uc_run_params = read_and_check_uc_run_params()
data_analyses = read_and_check_data_analysis_params(eraa_data_descr=eraa_data_descr)

logger = init_logger(logger_dir=OUTPUT_FOLDER_LT, logger_name='eraa_input_data_analysis',
                     log_level=usage_params.log_level)
logging.info('START ERAA (input) data analysis')

uc_period_msg = get_period_str(period_start=uc_run_params.uc_period_start, 
                               period_end=uc_run_params.uc_period_end)

date_col = 'date'
value_col = 'value'

# loop over the different cases to be analysed
for elt_analysis in data_analyses:
    logging.info(elt_analysis)
    # set UC run params to the ones corresponding to this analysis
    uc_run_params.set_countries(countries=[elt_analysis.country])
    uc_run_params.set_target_year(year=elt_analysis.year)
    uc_run_params.set_climatic_year(climatic_year=elt_analysis.climatic_year)
    # Attention check at each time if stress test based on the set year
    uc_run_params.set_is_stress_test(avail_cy_stress_test=eraa_data_descr.available_climatic_years_stress_test)
    # And if coherent climatic year, i.e. in list of available data
    uc_run_params.coherence_check_ty_and_cy(eraa_data_descr=eraa_data_descr, stop_if_error=True)

    logging.info(f'Read needed ERAA ({eraa_data_descr.eraa_edition}) data for period {uc_period_msg}')
    # initialize dataset object
    eraa_dataset = Dataset(source=f'eraa_{eraa_data_descr.eraa_edition}', 
                        agg_prod_types_with_cf_data=eraa_data_descr.agg_prod_types_with_cf_data, 
                        is_stress_test=uc_run_params.is_stress_test)

    if elt_analysis.data_subtype is not None:
        subdt_selec = [elt_analysis.data_subtype]
    else:
        subdt_selec = None
    eraa_dataset.get_countries_data(uc_run_params=uc_run_params,
                                    aggreg_prod_types_def=eraa_data_descr.aggreg_prod_types_def,
                                    datatypes_selec=elt_analysis.data_type, subdt_selec=subdt_selec)
    # create Unit Commitment Timeseries object from data read
    if elt_analysis.data_type == DATATYPE_NAMES.demand:
        current_df = eraa_dataset.demand[elt_analysis.country]
    elif elt_analysis.data_type == DATATYPE_NAMES.capa_factor:
        current_df = eraa_dataset.agg_cf_data[elt_analysis.country]
    try:
        dates = list(current_df[date_col])
    except:
        logging.error(f'No dates obtained from data -> move directly to next data analysis')
        continue
    # if data available continue analysis (and plot)
    dates = [elt_date.replace(year=elt_analysis.year) for elt_date in dates]
    values = np.array(current_df[value_col])

    current_full_dt = elt_analysis.get_full_datatype()
    uc_ts_name = set_uc_ts_name(full_data_type=current_full_dt, country=elt_analysis.country, 
                                year=elt_analysis.year, climatic_year=elt_analysis.climatic_year)
    uc_timeseries = UCTimeseries(name=uc_ts_name, data_type=current_full_dt, dates=dates, 
                                 values=values, unit=UNITS_PER_DT[elt_analysis.data_type])
    # And apply calc./plot... and other operations
    if elt_analysis.analysis_type == ANALYSIS_TYPES.plot:
        uc_timeseries.plot(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER)
    elif elt_analysis.analysis_type == ANALYSIS_TYPES.plot_duration_curve:
        uc_timeseries.plot_duration_curve(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER)
    elif elt_analysis.analysis_type in [ANALYSIS_TYPES.extract, ANALYSIS_TYPES.extract_to_mat]:
        to_matrix = True if elt_analysis == ANALYSIS_TYPES.extract_to_mat else False
        uc_timeseries.to_csv(to_matrix_format=to_matrix)

logging.info('THE END of ERAA (input) data analysis!')
stop_logger()
