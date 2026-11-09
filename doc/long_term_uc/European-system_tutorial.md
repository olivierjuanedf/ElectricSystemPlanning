# To start with: some general documentation

See [doc/useful_references.md](doc/useful_references.md) file to get 
- infos about ERAA (data) and PyPSA (framework/modeler)
- also some complementary sources of infos regarding the different countries you will be analysing

See [doc/glossary.md](doc/glossary.md) to get the **definition of some concepts or abbreviations** you may not be familiar 
with: starting with CY (Climatic Year) and TY (Target Year) that will be used repeatedly in this class.

# Tutorial - Long-Term Unit Commitment (UC) part

## Input files

With the provided code environment you will be able to **run a European Unit Commitment model by simply modifying the values in the 2 following files**:

1) [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json) -> contain **some default values and global parameters** (e.g., temporal ones - with the UC period to be simulated). **See dedicated appendix below** for a detailed description of the different fields in this file

2) [input/long_term_uc/countries/](../../input/long_term_uc/countries/)*{**country**}.json* with "**country**" the name of your considered country -> the **values used in this file will overwrite the ones of preceding file**. This is to make your own country choice. **N.B.** In this file not only your own country parameters can be defined, but also the ones of the other countries - typically neighboring ones. This may seem surprising, but is related to the "solo" mode of this code environment described justafter.

## Two modes : "Solo" and "Europe"
**You can choose between two different code behaviors: "solo" mode or "europe" one** - as defined in file [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json), field "mode":
- if mode is set to **"solo", all country parameters (for your own country, but also for the rest of them) will be read from your own file *{country}.json***. 
Example: if in [germany.json](../../input/long_term_uc/countries/germany.json) dictionary associated to key "capacities_tb_overwritten" contains "france": {"nuclear": 0}, the French nuclear capacity will be set to 0 MW for the UC simulated
- if mode is **"europe", parameters of each country will be extracted from file *{country}.json*; the rest of the values in this file being not accounted for**. Example: specifying "france": {"nuclear": 0} 
in [germany.json](../../input/long_term_uc/countries/germany.json) will be useless in this mode, as nuclear capacity value will be obtained from [france.json](../../input/long_term_uc/countries/france.json)

## Python script

- **Open and run [my_little_europe_lt_uc.py](../../my_little_europe_lt_uc.py)**
- You should **get a log "THE END of ERAA-PyPSA long-term UC simulation!"** in the terminal window. If not, the "checkers" should have indicated you some aspects to be corrected in your - modified - parametrization (e.g., using some unavailable values for country or production types). 

N.B. Even if normally not the case with parametrizations used in this class... it can be that no optimal solution 
be found (e.g., when the problem is "infeasible"): in this case (i) Run stops correctly - with an explicit error message 
indicating this infeasibility status in the logs but; (ii) No output data (neither figures) are obtained in the folders 
described just below.

## Output files
**Obtained data (respectively plotted figures) results** are given in [output/long_term_uc/data](../../output/long_term_uc/data) (resp. [output/long_term_uc/figures](../../output/long_term_uc/figures)) folders.
Note that, **as explained with the description of** [elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json) input file in Appendix section, 
**a "tag" - indicating name of the run**, and optionally a counter index - **can be added to data/figures subfolder to identify them more simply**.

For all files obtained, suffixes indicate the (i) Country corresponding to the content ("europe" if for whole continent);
(ii) TY and CY; (iii) Start date of the UC period simulated.

In detail, and **except if the resolution of PyPSA optimization model was not successful**, it will give you: 
* (*data* subfolder) **optimal production of all generators** considered in Europe, in *opt_power_europe_{TY and CY}_{start date}.csv* file. 
* (*data*) **"prices" for all countries** considered in Europe, in *marginal-prices_europe_{TY and CY}_{start date}.csv* file. 
**N.B.** Specifically, these prices are the optimal values of dual variables associated to supply-demand equilibrium (for those who are familiar with optimization; otherwise it will be explained!). 
* (*data*) **optimal flows in the interconnections** in *link-flow_opt_decisions_europe_{TY and CY}_{start date}.csv*.
* (*data*) **optimal decisions for the storages** - production (turbining, discharging; depending on the type of asset), 
consumption (pumping, charging) and "State-of-Charge" (water/battery level) in *storage_opt_decisions_europe_{TY and CY}_{start date}.csv*.
* (*data*) summary of (per-country) main results in *uc-summary_europe_{TY and CY}_{start date}.json*. In detail, all metrics being calculated over the whole UC period:
  - **per_country_ens**: Energy Not Served, i.e. the yearly volume of production of (fictive) failure asset (MWh).
  - **per_country_n_failure_hours**: number of failure hours, i.e. with a strictly positive level of failure production (no unit).
  - **total_cost** (resp. **per_country_total_cost**): of the UC problem simulated over Europe (resp. for a given country), i.e. "value" of the optimization problem (M€).
  - **total_operational_cost** (resp. **per_country_total_operational_cost**): previous value over Europe (resp. for a given country), from which failure cost (penalty) has been subtracted (M€).
  - **total_co2_emissions** (resp. **per_country_co2_emissions**): CO2 emissions over Europe (resp. for a given country) (XXX).
* (*data*) An LP file containing the description of UC optimization problem solved, in *model_7-countries_{start date}.lp*.
N.B. The suffix of this file contain the trigrams of the considered countries if they are very few; otherwise their number 
(7 if you are simulating the whole perimeter available for this class).
* (*figures* subfolder) A - very schematic - description of the "network" corresponding to the UC model simulated, in *network_7-bus.png*.
N.B. Idem regarding the suffix.
* (*figures*) A **"cumulated (stacked) vision" of the production**, in *prod_{country}_{TY and CY}_{start date}.png* files.
* (*figures*) **Price curves**, for the different countries in *prices_europe_{TY and CY}_{start date}.png* file.

## Start preparing the "design" of your country/Europe system by playing with this UC tool

**Based on the numeric results obtained for each of the simulated configurations you can start "designing" (i.e. sizing the production capacities) your own country** (if in "solo" mode)/the European system ("Europe" mode). Consider different:
* **seasons** -> by changing **uc_period_start** in file [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json). **Question**: how would you select a few typical, or extreme, weeks to be considered to size your system? Is it possible to do it *ex-ante*, i.e. only looking at input data (e.g., demand, RES CF, installed generation capacities, etc.) or do you need some iterative process with UC runs to do that?
* **(target) years** -> using 2025 or 2033. **Question**: would your capacity design be similarly "efficient" at both horizons?
* **climatic years** -> how sensitive are your results to the choice of this parameter? in combination with the ones of the season (associated period)? **Question** how would you choose one/a few scenarios used for your investment planning decision-making?
* **interconnection capacities** -> how are your results sensitive to the limit on the flows that can be exchanged between your 7 countries? **N.B.** Playing with parameter **interco_capas_tb_overwritten** in file [input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json), you can get some preliminar insights on this.
* (in solo mode) **What if... my neighbouring countries..." -> how are your individual country results sensitive to the decisions made by your neighbours? **N.B.** In solo mode you can exactly simulate the desired cases to try answering this question, by testing different configurations for your neighbours - imposing their decisions in your own *{country}.json* file

# Appendices

## Input data description

**Preliminary remarks**: 

On the **format of JSON files**:

<span style="margin-left: 40px;">(i) Used to store (multi-levels) dict-like infos.  
<span style="margin-left: 40px;">(ii) Must start "directly" with "{" and end with "}".  
<span style="margin-left: 40px;">(iii) "null" is used for None.  
<span style="margin-left: 40px;">(iv) 'my nice text' (single quotes) not allowed; use "my nice text" (double quotes) instead.  
<span style="margin-left: 40px;">(v) Tuples ("france", "italy") not allowed; use rather lists ["france", "italy"].  

The ones in folder [input/long_term_uc](../../input/long_term_uc/); **file by file description** - starting with the two only ones to be modified:

- [elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json):  
    - "<span style="color:#32B032; font-weight:bold">selected_climatic_year</span>": to **choose climatic year** considered for UC model (unique deterministic scenario, <span style="color:#257cbd; font-weight:bold">int</span> value; that must be in set of allowed ones -> see [elec-europe_eraa-available-values.json](../../input/long_term_uc/elec-europe_eraa-available-values.json)).
    - "<span style="color:#32B032; font-weight:bold">selected_countries</span>": to **choose countries** that you would like to be part of your European - copper-plate - long-term UC model (<span style="color:#257cbd; font-weight:bold">list of string</span>; that must be in the set of considered countries for this class, see same file). 
    
      N.B. Except if values are overwritten based on *{country}.json* files described hereafter, all production types available in ERAA data will be considered built for the countries in this list.
    - "<span style="color:#32B032; font-weight:bold">selected_target_year</span>": to choose the ERAA (target) **year** to be simulated (<span style="color:#257cbd; font-weight:bold">int</span>, either 2025 or 2033).
    - "<span style="color:#32B032; font-weight:bold">uc_period_start</span>": **date from which UC optimization period starts; under format "1900/%M/%d"** (<span style="color:#257cbd; font-weight:bold">string</span>). Ex.: "1900/1/1" to start from beginning of the year. 
  
      N.B. "1900" to clearly indicate that a "fictive calendar" (modelling one) be used in ERAA data, with 364 days (to get 52 full weeks... an important granularity for some unit optimization, as discussed in class).
    - (optional) "uc_period_end": idem, **end of period; same format** (<span style="color:#257cbd; font-weight:bold">string</span>). 
  
      Default: period of 9 days starting from "uc_period_start".
    - "<span style="color:#32B032; font-weight:bold">failure_power_capa</span>": **capacity of the failure - fictive - asset** considered (<span style="color:#257cbd; font-weight:bold">non-negative float</span>). 
  
      N.B. Common to all countries.
    - "<span style="color:#32B032; font-weight:bold">failure_penalty</span>": **failure asset variable cost**, or more standardly "penalty" (<span style="color:#257cbd; font-weight:bold">non-negative float</span>). 
  
      N.B. Typically set to a "very big" value, so that this asset be used as a last recourse - i.e. after having used all other production units at maximal power available.
    - "<span style="color:#32B032; font-weight:bold">interco_capas_tb_overwritten</span>": **values for the interconnection capacities**; to be used to overwrite - or complete - ERAA data (<span style="color:#257cbd; font-weight:bold">dictionary</span> with <span style="color:#257cbd; font-weight:bold">str</span> keys and <span style="color:#257cbd; font-weight:bold">non-negative float</span> values, with keys under format {origin country}2{destination country}). 
  
      Ex:  
      ```json
      {"france2poland": 10, "france2scandinavia": 0, "italy2iberian-peninsula": 0} 
      ```

      will set interconnection capacity from France to Poland to 10GW (very fictive!) and from France to both Scandinavia et Iberian-Peninsula to 0GW. Note that regarding the France to Scandinavia link this value will be useless as there is no ERAA data associated to this link; in turn our code already used 0GW as the value.
    - (optional) "<span style="color:#32B032; font-weight:bold">output_tag</span>": to **add a tag to the obtained data/figure** output folders. 
      
      Default: no tag, in this case a previous run results may be overwritten.
      
      Ex: if output_tag="preliminar", data (resp. figure) results will be obtained in "data_preliminar" (resp. "figures_preliminar") subfolders
    - (optional) "<span style="color:#32B032; font-weight:bold">add_run_idx_tag_to_output</span>": to **add a tag indicating the run index** to the obtained data (resp. figure) output folders. 
   
      Default: no such index, in this case a previous run results may be overwritten.
    
- [countries/](../../input/long_term_uc/countries/){**country**}.json: 
    - "<span style="color:#32B032; font-weight:bold">team</span>": **name of your team**, i.e. name of the country you are "responsible for" (<span style="color:#257cbd; font-weight:bold">string</span>, that must be in the set {"benelux", "germany", "iberian-peninsula", "italy", "poland", "scandinavia"} - use lower letters).
    - "<span style="color:#32B032; font-weight:bold">selected_prod_types</span>": **per country production types to be selected - using aggregate classes** defined in [input/long_term_uc/elec-europe_eraa-available-values.json](../../input/long_term_uc/elec-europe_eraa-available-values.json) file/field "aggreg_prod_types" (<span style="color:#257cbd; font-weight:bold">dictionary</span> {country name: list of aggreg. production types to be selected for run}). 
  
      Ex: 
      ```json
      {"france": ["nuclear", "failure"], "germany": ["coal", "wind_onshore", "wind_offshore"]} 
      ```
      will lead to a run with only nuclear and failure (resp. coal and wind on-/off-shore) units in France (resp. Germany). 
    - "<span style="color:#32B032; font-weight:bold">capacities_tb_overwritten</span>": **aggreg. production units for which you want to update the capacities - versus the ones in ERAA data**, and for the considered (target) year (2-level <span style="color:#257cbd; font-weight:bold">dictionary</span> {country: {aggreg. prod. type: updated capacity}}). 
  
      Ex: 
      ```json
      {
        "france": {
          "nuclear": 100000,
          "failure": 100000
        }
      }
      ```
      will update French nuclear (aggreg.) capacity to 100GW and set a big failure asset of the same capacity. 
      
      In this exemple, capacities for Germany will be taken as given in ERAA data. 

      Ex2: 
      ```json
      {
        "france": {
          "nuclear": 100000,
          "failure": 100000
        },
        "germany": {
          "oil": 500
        }
      }
      ```
      In this 2nd example, the modification on French capacities is identical to the 1st one and, if in solo mode, oil capacity in Germany will be modified to (only!) 500MW.

- (if too slow UC resolution time...) [solver_params.json](../../input/_env_params/solver_params.json):
    - "<span style="color:#32B032; font-weight:bold">name</span>": **name of the solver to be used** (<span style="color:#257cbd; font-weight:bold">string</span>, that must be in the set {"highs", "gurobi"} - use lower letters).
    - (optional) "<span style="color:#32B032; font-weight:bold">license_file</span>": **name of the solver license file** (only for Gurobi; to be obtained from https://portal.gurobi.com/iam/register with your student email).
    
      N.B. This file has to be provided at the root of this project.

- **[NOT TO BE MODIFIED during this practical class]** [elec-europe_eraa-available-values.json](../../input/long_term_uc/elec-europe_eraa-available-values.json): containing values available in the ERAA extract provided in folder [data/](../../data/): 
    - "<span style="color:#32B032; font-weight:bold">climatic_years</span>": **past historical years weather conditions** that are 'projected' on ERAA "target year" (<span style="color:#257cbd; font-weight:bold">list of int</span> values).
    - "<span style="color:#32B032; font-weight:bold">countries</span>": your seven **(meta-)countries**, the only ones for which ERAA data are made available in this code environment (<span style="color:#257cbd; font-weight:bold">list of str</span>).
    - "<span style="color:#32B032; font-weight:bold">aggreg_prod_types</span>": **per country and year aggregated production types** (<span style="color:#257cbd; font-weight:bold">two-level dictionary</span> in format {country name: {(target) year: list of aggregated production types available in the extract of ERAA data}}). 
  
      N.B. As "aggregated production types" are only used here to simplify the considered model (diminishing its size), availability of such a type means that at least one of the corresponding - more detailed - ERAA production types is available in data.
    - "<span style="color:#32B032; font-weight:bold">target_years</span>": list of **years available** here - 2025 or 2033 here (<span style="color:#257cbd; font-weight:bold">list of int</span>).
    - "<span style="color:#32B032; font-weight:bold">intercos</span>": list of **interconnection with available data** (<span style="color:#257cbd; font-weight:bold">list of str</span>, with str under format {origin country}2{destination country}). 
  
      N.B. Obtained by simple aggregation of ERAA data when multiple sub-zones are present in our (meta-)countries.

**ATTENTION** (for next class): actually marginal cost values for Eur. UC model are extracted from following file... put it in the list of the ones to be modified by students?

## Optional functionalities

### Hydraulic assets: a more realistic modelling - already integrated in provided environement

In order to get a dispatch of hydraulic assets closer to the one observed in reality - especially for (meta-)countries 
in which it represents a significant part of the capacities (e.g., Scandinavia!), it may be useful to specify:
- **min/max generation levels**: adding and completing a file XXX.
  
  **Format** (columns, ";" separator): week;climatic_year;min_value;max_value with
    - "week": index in 1, ..., 53
    - "climatic_year": at least the value used for simulation
    - "min_value" (resp. "max_value"): min (resp. max) WEEKLY generation level (MWh). In turn, the sum of production decisions in the week will be lower- (resp. upper-) bounded by this value
- **min/max reservoir levels** ("State-of-(Water) Charge" of the reservoir): idem with a file [PECD-hydro-weekly-reservoir-min-max-levels.csv](../../data/ERAA_2023-2/hydro/PECD-hydro-weekly-reservoir-min-max-levels.csv).

  **Format** (columns, ";" separator): week;climatic_year;min_value;max_value with
    - "week": index in 1, ..., 53
    - "min_value" (resp. "max_value"): min (resp. max) END-OF-WEEK reservoir charging level (%). In turn, the end-of-week SoC, i.e. the start-of-week SoC + sum of production - consumption decisions in the week  will be lower- 
      (resp. upper-) bounded by this value
  
      N.B. On the contrary to the data in generation level, here the provided parameters are the same whatever the CY.

### Bound on CO2 emissions - to activate on your side

To integrate this additional functionality, it consists in adding/filling files in
[optional input folder](../../input/long_term_uc/optional). 


These constraints can be added in the proposed environment, adding a dictionary in file 
[input/long_term_uc/elec-europe_params_to-be-modif.json](../../input/long_term_uc/elec-europe_params_to-be-modif.json): 
   - in "extra_params" dict.
   - with following fields
     - **max_co2_emis_constraints**
       - **cases**: on which this sumprod constraint is to be applied, each of them consisting in a dict. with fields:
         - **countries**: list of countries on which the sum must be calculated - in addition to the temporal sum 
         done on whole temporal period. "europe" can also be used to apply this constraint on whole continent - i.e.
         on all selected countries for the UC simulated.
         - **upper_bound**: a float, the maximal amount of CO2 emissions (XXX unit XXX).

Ex: to add a (unique) constraint of this type on France-Germany
```json
{
  "max_co2_emis_constraints": {
    "cases": [
      {
        "countries": [
          "france",
          "germany"
        ],
        "upper_bound": 3.14
      }
    ]
  }
}
```