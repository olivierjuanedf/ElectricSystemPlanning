# Description of the main elements parametrizing this code environment, in teacher role

## data

Mainly containing ERAA data used to parametrize the UC problems simulated in this environment

## doc

Including all infos for students - distinguished per script of this environment, 
and a specific folder for teachers [doc/teachers](../../doc/teachers)

## input

Containing parameters to be changed by students to play with the different scripts of this environment

Following are additional info vs. *European-system_tutorial.md* doc, not put in students doc for simplicity:

- [elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json):  
    - (optional) "<span style="color:#32B032; font-weight:bold">add_date_tag_to_output</span>": to **add a tag indicating the run date** to the obtained data/figure output folders. Default: no such tag, in this case a previous run results may be overwritten.
    N.B. This tag is applied prioritarily versus the run index one (activated with parameter **add_run_idx_tag_to_output**)
- On "Sumprod" constraints -> env. allow a more complex usage than the one actually described to the students 
  - Some constraints expressed based on a weighted sum of production can be very helpful to enrich the description of UC 
problems; 2 typical cases are (i) An upper bound on CO2 emissions; (ii) A maximal UC cost. N.B. In current version of 
student environment only CO2 emissions bound available
  - [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json): 
   - in "extra_params" dict.
   - with following fields
     - **max_co2_emis_constraints** (resp. **xxx**) for an upper bound on CO2 emissions (resp. UC cost)
       - **temporal_granularity**: "day", "week" or "whole_period". Default: "whole_period", the one used by students 
       in current version of the environment. N.B. When weekly periods are considered they are starting on Mondays
       - **upper_bound**: can be either a float, or list of values. If float either it corresponds to a case with 
         temporal_granularity=whole_period, or in which the same bound will be applied to each period. N.B. (i) It can 
         be replaced by **lower_bound** resp **xxx**) if a lower bound (resp. equality) constraint is to be applied. 
         (ii) If a vector is provided its size must correspond to the number of (daily/weekly) periods in considered
         optimization period. If not the case, or the number of time-slots in considered model does not correspond 
         the code will raise an error/warning; (iii) When a weekly temporal granularity is used and some weeks are only 
         partial (with strictly less than 7 days) a proportional factor is applied in the code to lower the value written
         in this JSON (which must be interpreted as corresponding to a whole week, to be comparable with other ones in
         provided vector)

```json
{
  "max_co2_emis_constraints": {
    "temporal_granularity": "week",
    "cases": [
      {
        "countries": [
          "france",
          "germany"
        ],
        "upper_bound": [
          1,
          2,
          3,
          4
        ]
      }
    ]
  }
}
```

## output

Data obtaining after having run one of the scripts of this environement. Two main subfolders are distinguished:
- **data_analysis**: for... data analysis script
- **long_term_uc**: for the 1-country/European UC simulation. In this second subfolder 1-country/European output 
folders will be obtained, in which **data** (.csv files, and .lp with optim model solved) and **figures** (.png) 
outputs will be separated 

## src\functional_params

- [plot_params.json](../../src/functional_params/plot_params.json): a few parameters for (color) palette 
definition - possibly using a standard one (e.g., the one of Eco2mix), linestyles, markers, orders of curves. 
It can be done on different dimensions (production types, zones, years, etc.)
- [usage_params.json](../../src/functional_params/usage_params.json): idem, to allow (or not)/parametrize some 
operations for the students. Per parameter:
  - **allow_adding_interco_capas**: possible to add some interconnection (pair of zones) w.r.t. the list in ERAA data?
  - **allow_overwriting_eraa_interco_capa_vals**: possible to overwrite interco. capa values of ERAA? 
  (specifying values in [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json))
  - **allow_manually_adding_demand**: possible to add supplementary demand in addition to ERAA one? 
  N.B. Not functional currently; could be used to integrate "exogeneous" Demand-Side-Response 
  (e.g. based on Time-of-Use tariff, set independently of the UC solution)
  - **allow_manually_adding_generators**: ??
  N.B. Idem
  - **apply_cf_techno_breakthrough**: apply some "breakthrough" on a technology associated to CF data. In this case, 
  CF values will be applied an increasing transformation to represent this improvement
  - **apply_per_country_json_file_params**: dictionary with following values, indicating for each of the scripts if run
  parameters for a given country must be obtained rom individual country/global european json input file 
    - **data_analysis**: false 
    - **monozone_toy_uc_model**: true
    - **multizones_uc_model**: true
  Q: still useful? (in multizones_uc_model case, behaviour dependently on the chosen mode solo/europe)
  - **get_optional_const_from_data_folder**: get data for optional constraints from data folder (available only to the 
  teachers) or input one (in which students can do some modifications)? Can be either a boolean, or a dictionary to 
  specifiy this property per type of constraint; in this latter case available keys are "hydro", "max_co2_emissions", 
  and "max_prod_cost"
  - **res_cf_stress_test_folder**: folder in which RES CF of the "stress test" will be found (different 
  folder from the one in which original CF ERAA data are found)
  - **res_cf_stress_test_cy**: CY to be used for RES CF stress test; for which associated data must be found in 
  the files provided in **res_cf_stress_test_folder** 
  - **use_first_year_capas_as_default**: use first year available (in ERAA data) to set generation capas by default, 
  i.e. before applying values set in input JSON per-country files
  - **log_level**
