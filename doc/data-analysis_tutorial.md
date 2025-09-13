To get some insights on the data used in this code environment, from European Resource Adequacy Assessment (ERAA), you can plot very easily some quantities you would like to observe for different countries, years, climatic years. This is done running script *my_little_europe_data_analysis.py*, as explained below.

# How to run data analysis

**Update the JSON input file** dedicated to data analysis: *input/long_term_uc/data_analysis/data-analysis_params_to-be-modif.json*. It contains a list of the quantities you would like to get plotted; each element of this list being a dictionary with fields to specify the analysis/plot to be done:

    - **analysis_type** (str): "plot" XXX available values/update code to enrich them... (only plot currently) XXX
    - **data_type** (str): datatype to analyze/plot; its value must be in the list of available values given in file *input/long_term_uc/functional_available-values.json* (e.g., "demand", "res_capa-factors", "generation_capas", etc.)
    - **country** (str): it must be in the list of values given in file *input/long_term_uc/elec-europe_eraa-available-values.json* (field "countries")
    - **year** (int): idem (field "target_years")
    - **climatic_year** (int): the (past) year from which weather conditions will be "extracted" and applied to current year; it must be in list given in file *input/long_term_uc/elec-europe_eraa-available-values.json* (field "climatic_years")  
    - **period_start** (str, with date format yyyy/mm/dd): start date of the period to be considered
    - **period_end** (idem): idem, end date

**Run script *my_little_europe_data_analysis.py***

**Outputs**

They will be obtained in folder *output/data_analysis*; **either .png files** (if "plot" chosen for "analysis_type"; cf. description above) or **.csv ones** (if XXX tb completed xxx)
