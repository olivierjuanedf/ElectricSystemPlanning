To get some insights on the data used in this code environment, from European Resource Adequacy Assessment (ERAA), you can plot very easily some quantities you would like to observe for different countries, years, climatic years. This is done running script *my_little_europe_data_analysis.py*, as explained below.

# How to run data analysis

**Update the JSON input file** dedicated to data analysis: *input/long_term_uc/data_analysis/data-analysis_params_to-be-modif.json*. It contains a list of the quantities you would like to get plotted/saved in csv files; each element of this list being a dictionary with fields to specify the analysis/plot to be done:

  - **analysis_type** (str): "plot" - to get some curves plotted e.g., demand of a given (country, year, climatic year); "plot_duration_curve" - idem for duration curve of a given quantity, typically (net) demand; "extract" to get some ERAA data extracted to a .csv file. 
  - **data_type** (str): datatype to analyze/plot; its value must be in the list of available values given in file *input/long_term_uc/functional_available-values.json* (e.g., "demand", "net_demand", "res_capa-factors", "generation_capas", etc.).
  - **country** (str or list of str): it must be in the list of values given in file *input/long_term_uc/elec-europe_eraa-available-values.json* (field "countries").
    N.B. If list of countries: if plots are displayed, multiple curves will be obtained (one for each country - on the same graph); if csv is written, data of the different countries will be concatenated.
  - **year** (int or list of int): year(s) to be considered for the data analysis; its value must be in the list of values given in file *input/long_term_uc/elec-europe_eraa-available-values.json* (field “target_years”).
    N.B. If list of years: if plots are displayed, multiple curves will be obtained (one for each year - on the same graph); if csv is written, data of the different years will be concatenated.
  - **climatic_year** (int or list of int): the (past) year from which weather conditions will be "extracted" and applied to current year; it must be in list given in file *input/long_term_uc/elec-europe_eraa-available-values.json* (field "climatic_years")  
    N.B. If list of climatic years: if plots are displayed, multiple curves will be obtained (one for each climatic year - on the same graph); if csv is written, data of the different climatic years will be concatenated.
  - **period_start** (str, with date format yyyy/mm/dd): start date of the period to be considered
  - **period_end** (idem): idem, end date
  - **extra_params** (dict or list of dict): to specify extra-parameters that can be used to analyse data, e.g. fixed capacities for RES sources for net demand calculation.
    This dictionary has the following fields:
    - (optional) **label**: name of the case associated to this extra-params choice, that will be used for plot/csv saving
    - **values**: a dictionary with {param name: param value}. Only available param is currently **capas_aggreg_pt_with_cf**, for which 
    the following values can be used for ex. {"wind_onshore": 10000, "wind_offshore": 500, "solar_pv": 10000} to set capacity values of Wind on-/off-shore and Solar PV
    to 10GW, 500MW and 10GW respectively.

N.B. If list are provided for countries, years, climatic years, and extra-params: if plots are displayed, a curve will be obtained for each case in the product of requested lists; if csv is written, concatenation will be done over the product of cases.
For plots a maximal number of 6 cases is allowed, so that obtained graph be readable.

**Run script *my_little_europe_data_analysis.py***

**Outputs**

They will be obtained in folder *output/data_analysis*; **either .png files** (if "plot" chosen for "analysis_type"; cf. description above) or **.csv ones** (if XXX tb completed xxx)
