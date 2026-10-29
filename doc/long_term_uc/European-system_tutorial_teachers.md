Following same strcuture as in European-system_tutorial.md doc, following are additional parameters that are not put in students doc for simplicity:

- [elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json):  
    - (optional) "<span style="color:#32B032; font-weight:bold">add_date_tag_to_output</span>": to **add a tag indicating the run date** to the obtained data/figure output folders. Default: no such tag, in this case a previous run results may be overwritten.
    N.B. This tag is prioritary versus the run index one (activated with parameter **add_run_idx_tag_to_output**)

### "Sumprod" constraints

Some constraints expressed based on a weighted sum of production can be very helpful to enrich the description of UC 
problems; 2 typical cases are (i) An upper bound on CO2 emissions; (ii) A maximal UC cost. N.B. In current version of 
student environment only CO2 emissions bound available

[input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json): 
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