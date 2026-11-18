# Investment Runner User Guide

This document describes how the **Investment Runner** (`main_runner.py`) works. It is designed to collect investment choices from each student group (for each country), inject these capacities into the corresponding country files (`input/long_term_uc/countries/{country}.json`), and execute a unified simulation of the interconnected European grid (**Exclusive Europe Mode**).

> **Note on Solo Mode**: The runner operates **exclusively in Europe mode**. Solo mode is not used for the runner; if a student wishes to explore their individual country and evaluate sensitivity to neighboring countries, they can run `my_little_europe_lt_uc.py` directly by configuring their JSON file as described in the [European system tutorial](../long_term_uc/European-system_tutorial.md).

---

## 1. Operating Principle

The runner can operate with two data sources:
- **TXT mode (default)**: collects investment text files from student groups (e.g., `input/runner/france_europe1.txt`, `input/runner/france_europe2.txt`...), parses them, and automatically updates the corresponding `input/long_term_uc/countries/{country}.json` files.
- **JSON mode**: uses country JSON files (e.g., `input/runner/france_europe1.json`, `input/runner/france_europe2.json`, or in `input/long_term_uc/countries/`).

### Dual Europe Support (Europe 1 & Europe 2)
The runner supports two European configurations or competing scenario mixes:
- Filename pattern: `{country}_europe1.{txt|json}` and `{country}_europe2.{txt|json}`.
- If both `europe1` and `europe2` are present, the runner:
  1. Configures and simulates **Europe 1**.
  2. Configures and simulates **Europe 2**.
  3. Displays a comprehensive **Comparative Dashboard** in the terminal (System Cost, OPEX, CAPEX, €/MWh, CO2, ENS, Failure Hours, with Δ and % differences).
  4. Automatically saves the comparative results to `output/runner/`:
     - `output/runner/comparison_europe1_vs_europe2_YYYYMMDD_HHMMSS.txt` (timestamped text table)
     - `output/runner/comparison_europe1_vs_europe2_YYYYMMDD_HHMMSS.json` (timestamped metrics)
     - `output/runner/comparison_europe1_vs_europe2_latest.txt` (latest text table)
     - `output/runner/comparison_europe1_vs_europe2_latest.json` (latest metrics)
  5. Automatically computes, displays, and saves the **simulation evolution history** across successive runs for Europe 1 (and for each country) to:
     - `output/runner/history_evolutions_europe1.txt`
     - `output/runner/history_evolutions_europe1.json`
- If only one Europe is present (or without suffix), the runner executes that single Europe and displays the standard dashboard.

In all cases, the `main_runner.py` script:
   - Discovers and validates files for each country (non-negative capacities in MW).
   - In TXT mode: updates the `input/long_term_uc/countries/{country}.json` files.
   - Automatically configures `elec-europe_params_to-be-modif.json`:
     - Enforces **`mode: "europe"`**: each country takes the capacities defined by its JSON file and cross-border interconnected exchanges are simulated.
     - Enforces the **full-year simulation horizon** (`"uc_period_start": "1900/1/1"`, `"uc_period_end": "1901/1/1"`, 8,760 hours) by default, or a fast **2-week test period** (`"1900/1/1"` to `"1900/1/15"`) when `--test` is specified, ensuring both OPEX and CAPEX are expressed on a consistent basis.
   - Executes the European unit commitment optimization simulation (`my_little_europe_lt_uc.py`).
   - Displays a comparative summary table for every European country and for Europe as a whole (ENS, failure hours, total cost, CO2 emissions).

---

## 2. Investment File Format

Each line specifies a production asset and its installed capacity in **MW**:

```text
# Country name (optional if the filename is already the country name, e.g., france.txt)
country: france

# Low-carbon & Renewable Generation
nuclear: 61000
solar_pv: 35000
wind_onshore: 25000
wind_offshore: 10000

# Hydro
hydro_reservoir: 8500
hydro_run_of_river: 4500
hydro_pump_storage_closed_loop: 5000
hydro_pump_storage_open_loop: 1000

# Storage & Flexibility
batteries: 3000
dsr: 2500

# Peaking / Thermal
gas: 4000
```

### Syntax Rules:
- **Separators**: colon `:`, equals sign `=`, or whitespace (e.g., `nuclear: 60000`, `nuclear = 60000`, or `nuclear 60000`).
- **Comments**: any line starting with `#` or `//` is ignored.
- **Units**: capacities are in **MW** (the same unit as in `country.json`). The suffix `MW` is optional (e.g., `60000 MW`).
- **Asset Name Tolerance**: common aliases and French names are supported (`wind_on_shore`, `eolien_terrestre`, `solar`, `pv`, `solaire`, `gaz`, `step_closed`, etc.).

### Missing Countries & Empty Files Behavior:

| Situation | In the Simulation Grid | Capacities Applied |
| :--- | :--- | :--- |
| **No file provided for a country** | The country is **excluded** from the simulation (`selected_countries`). | **Not simulated** (0 cost/gen in the dashboard). |
| **Empty file (or only `country: <name>`)** | The country is **included** in the European simulation. | **0 MW for all assets**. Only the fictive `failure` generator is active (100% loss of load / ENS). |
| **Asset omitted from the file** | The asset is not included in `selected_prod_types`. | **0 MW** (there is **no automatic fallback to ERAA** capacities). |

### Using Official ERAA Reference Mixes:
You can instruct a country to adopt official ENTSO-E ERAA reference capacities directly in its `.txt` file:

```text
# Option 1: 100% ERAA roadmap (target year, e.g. 2033)
country: italy
eraa

# Option 2: ERAA baseline from a specific year (2025 or 2033)
country: poland
eraa: 2025

# Option 3: ERAA base with custom overrides for specific technologies
country: france
base: eraa
nuclear: 50000
solar_pv: 60000
```
In Option 3, all official ERAA assets are loaded, and only `nuclear` and `solar_pv` are replaced by your custom choices.

### Automatically Completing Missing Countries with ERAA:
When only a few student teams have submitted files, you can simulate the full 7-country European interconnected grid by passing the `--fill-missing-with-eraa` CLI option (or shortcut `make runner-fill-eraa`):
```bash
python main_runner.py --fill-missing-with-eraa
python main_runner.py --fill-missing-with-eraa --dry-run
```
Any European country without a submitted investment file is automatically populated with its official ERAA reference capacities.

---

## 3. Handling of Cross-Border Interconnections

For cross-border electricity exchanges between European countries, the model takes into account **both native ERAA data and user-defined overwrites**:
1. **Base ERAA Data**: Native interconnections and capacities defined in `data/ERAA_2023-2/interco_capas/interco-capas_{year}.csv` are loaded by default.
2. **User Overwrites (`interco_capas_tb_overwritten`)**: Any capacity specified in `input/long_term_uc/elec-europe_params_to-be-modif.json` under `interco_capas_tb_overwritten` overrides the ERAA value (e.g., `"germany2italy": 0` or `"france2germany": 5000`).
3. **Unspecified Links**: All cross-border links not explicitly overridden in `interco_capas_tb_overwritten` automatically retain their native ERAA values.
4. **New Links**: Additional links between countries not originally in ERAA can also be introduced (allowed by `"allow_adding_interco_capas": true` in `usage_params.json`).

---

## 4. Main Commands

### Mode TXT (Default - Student Submissions)
Place student `.txt` files into `input/runner/`, then execute:

```bash
# Run the simulation for all collected countries in Europe mode
python main_runner.py
# or explicitly:
python main_runner.py txt

# Preview changes without solving (dry-run, updates country.json files only)
python main_runner.py --dry-run
# or:
python main_runner.py txt --dry-run
```

### Mode JSON (Direct Pre-configured Countries)
Uses the existing JSON files in `input/long_term_uc/countries/` directly without rewriting them:

```bash
# Run full European collective simulation using existing JSON files
python main_runner.py json
# or using the --source option:
python main_runner.py --source json

# Preview configuration without solving
python main_runner.py json --dry-run
```

### Fast Debug Mode (`--test`)
For rapid testing and debugging, you can simulate a short **2-week period** (14 days / 336 hours, from `1900/1/1` to `1900/1/15`) instead of the entire year. This executes the full optimization problem in just a few seconds:
```bash
# Fast test simulation (2 weeks)
python main_runner.py --test

# Fast test simulation in dry-run mode
python main_runner.py --test --dry-run

# Or via Makefile shortcut
make runner-test
```

### Clearing Previous Runner Outputs & History
To prevent historical runs from blurring or polluting subsequent incremental simulation results when presenting iterative updates:
```bash
# Clear all output files in output/runner/ via Python CLI
python main_runner.py --clean
# or using positional keyword:
python main_runner.py clean

# Or via Makefile shortcut
make runner-clean
```
You can also combine `--clean` with simulation flags (e.g., `python main_runner.py --clean --test`) to clean previous outputs before starting a fresh run.

### Comparison with Official ENTSO-E ERAA Baselines (2025 & 2033)
The runner automatically compares student investment mixes (Europe 1 and Europe 2) against the official **ERAA 2025** (starting mix) and **ERAA 2033** (target roadmap) baseline simulations:
- **Baseline Caching**: pure ERAA simulations for the simulated test year and period are automatically computed and cached in `output/runner/baselines/` on their first execution. Subsequent runs reload the cache instantaneously (0 additional solver time).
- **Disable Baseline Comparison**: use `--no-compare-eraa` if you only wish to compare Europe 1 vs Europe 2 without running or loading ERAA baselines:
  ```bash
  python main_runner.py --no-compare-eraa
  ```
- **Force Baseline Recalculation**: use `--force-eraa-run` to regenerate the baseline cache (e.g. after updating input files or solver parameters):
  ```bash
  python main_runner.py --force-eraa-run
  ```

### Custom Files or Folders
You can also specify a custom collection directory or specific files:
```bash
python main_runner.py my_collection_folder/
```

Or using the Makefile shortcut targets:
```bash
# Preview update of country JSONs (TXT mode)
make runner-dry-run

# Run fast test simulation (2 weeks)
make runner-test

# Clear previous simulation outputs and history
make runner-clean

# Run full European collective simulation (full year)
make runner
```

---

## 5. Capacity Visibility and ERAA Comparisons

### 5.1. Installed Capacities Table 
Right upon parsing investment choices, the runner displays an aligned capacity table in **GW** for each country and for Europe as a whole, accessible immediately even in `--dry-run`:
- Per-technology columns: Nuclear, Solar PV, Wind Onshore, Wind Offshore, Hydro Reservoir, Hydro RoR, Hydro PHS (Closed/Open), Batteries, DSR, Gas, and Other.
- Summary columns:
  - **Total Capa (GW)**
  - **Low-Carbon / Decarbonized Share (%)**: proportion of non-fossil generation, storage and flexibility.
  - **Overnight CAPEX (B€)**: total initial capital expenditure.
  - **Annualized CAPEX (M€/year)**: annual capital recovery annuity based on WACC and asset lifetime.

### 5.2. Capacity Comparison with ERAA References (2025 & 2033)
The runner evaluates student choices against official ENTSO-E PEMMDB data:
- **$\Delta$ vs ERAA 2025 (GW / %)**: net capacity additions since the 2025 European starting point.
- **$\Delta$ vs ERAA 2033 (GW / %)**: difference relative to the official 2033 ENTSO-E trajectory.

### 5.3. Incremental Capacity Trajectory across ERAA References and Historical Runs
To clearly track the progression of choices across rounds and compare them to the official references, the runner generates a unified multi-column capacity trajectory table (`output/runner/history_capacities_trajectory.txt`):
```text
============================================================================================================================================
              INCREMENTAL CAPACITY TRAJECTORY: ERAA REFERENCES vs SUCCESSIVE RUNS (GW)
============================================================================================================================================
Country / Technology           ERAA 2025   ERAA 2033    Run 1 (E1)  Run 1 (E2)    Run 2 (E1)  Run 2 (E2)     Δ Net (E1)     Δ Net (E2)
--------------------------------------------------------------------------------------------------------------------------------------------
[FRANCE]
  nuclear                           61.8        61.8          50.0        50.0          55.0        50.0           +5.0            0.0
  solar_pv                          18.2        54.3          45.0        40.0          48.0        42.0           +3.0           +2.0
  wind_onshore                      24.1        36.1          35.0        30.0          36.0        32.0           +1.0           +2.0
  wind_offshore                      2.0        11.6          15.0        12.0          15.0        14.0            0.0           +2.0
  hydro_reservoir                    9.5         9.8           8.5         8.5           8.5         8.5            0.0            0.0
  hydro_run_of_river                13.6        13.6           4.5         4.5           4.5         4.5            0.0            0.0
  hydro_pump_storage_closed_loop     2.0         2.1           5.0         4.0           5.0         5.0            0.0           +1.0
  hydro_pump_storage_open_loop       1.8         2.0           1.0         1.0           1.0         1.0            0.0            0.0
  batteries                          0.5         0.5           6.0         4.0           8.0         5.0           +2.0           +1.0
  dsr                                4.2         5.6           3.5         3.0           4.0         3.5           +0.5           +0.5
  gas                                7.2         7.2           3.0         5.0           2.0         4.0           -1.0           -1.0
  Total france                     154.3       210.4         176.5       162.0         187.0       169.0          +10.5           +7.0
--------------------------------------------------------------------------------------------------------------------------------------------
...
[TOTAL EUROPE]
  ...
============================================================================================================================================
```
This table gives teachers and students an immediate, intuitive view of:
1. **Starting Point**: ERAA 2025.
2. **Roadmap Objective**: ERAA 2033.
3. **Round-by-Round Progression**: Run 1 (E1 & E2) $\to$ Run 2 (E1 & E2) $\to$ Latest Run.
4. **Net Adjustments**: Total additions/retirements made between the first and last run ($\Delta_{\text{net}}$).

Additionally, across iterative runs, individual simulation history files (`history_evolutions_europe1.txt`) track capacity deltas under section `[5] CAPACITY EVOLUTION ACROSS SUCCESSIVE RUNS (GW)`.

### 5.4. Unified 4-Way Comparative Dashboard
When both Europe 1, Europe 2, and the ERAA baselines (2025 and 2033) are simulated:
- **Global Table**: compares Europe 1 | Europe 2 | ERAA 2025 | ERAA 2033 | $\Delta$(E1 - ERAA33) | $\Delta$(E2 - ERAA33).
- **Per-Country Breakdown**: side-by-side comparison of costs, CO2 emissions, and loss-of-load hours across all 4 configurations.
- **Persisted Outputs**: saved in `output/runner/comparison_europe1_vs_europe2_YYYYMMDD_HHMMSS.txt` and `.json`.

---

## 6. Simulation Results and Interpretation

At the end of a collective simulation, the runner outputs the European comparative dashboard including both **operational costs (OPEX)** and **annualized investment costs (CAPEX)**:

```text
==============================================================================================================
                                RESULTS OF THE EUROPEAN INTERCONNECTED SIMULATION
==============================================================================================================
COUNTRY              |  ENS (MWh) | FAIL.(h) |  OPEX(M€/yr) |  CAPEX(M€/yr) |  TOTAL(M€/yr) |  COST(€/MWh) | CO2 (Mt)
--------------------------------------------------------------------------------------------------------------
benelux *            |        0.0 |      0.0 |        611.0 |       7,450.2 |       8,061.2 |         68.5 |     5.69
france *             |        0.0 |      0.0 |        319.0 |      20,207.5 |      20,526.5 |         51.3 |     0.00
germany *            |   17,652.0 |    456.0 |        310.0 |      14,830.0 |      15,140.0 |         42.1 |     3.70
iberian-peninsula *  |        0.0 |      0.0 |        118.0 |      11,250.0 |      11,368.0 |         45.2 |     0.44
italy *              |        0.0 |      0.0 |        176.0 |       9,120.0 |       9,296.0 |         48.9 |     0.84
poland *             |        0.0 |      0.0 |        443.0 |       6,840.0 |       7,283.0 |         56.2 |     6.52
scandinavia *        |        0.0 |      0.0 |         17.0 |       8,900.0 |       8,917.0 |         32.4 |     0.09
--------------------------------------------------------------------------------------------------------------
TOTAL EUROPE         |   17,652.0 |    456.0 |      1,994.0 |      78,604.7 |      80,598.7 |         49.8 |    17.28
==============================================================================================================
  Total operational cost (OPEX) : 1,994.00 M€/year
  Total annualized CAPEX        : 78,604.70 M€/year (Total overnight: ~1,250.4 B€)
  Total system cost (OPEX+CAPEX): 80,598.70 M€/year
  Total generation              : 1,618.40 TWh (1,618,400 GWh)
  Average electricity cost      : 49.80 €/MWh (OPEX: 1.23 €/MWh, CAPEX: 48.57 €/MWh)
  (*) Countries updated by the runner during this session: benelux, france, germany, ...

[!] LOSS OF LOAD / FAILURE DETECTED in: germany
    These countries must reinforce dispatchable capacities, renewables, or flexibility/interconnections.
```

### Metrics Explained:
- **ENS (MWh)**: Energy Not Served (unmet demand volume).
- **FAIL. (h)**: Number of hours where supply shortfall / loss of load occurred.
- **OPEX (M€/yr)**: Annual operational cost calculated by the PyPSA dispatch model over the simulation period (fuel costs + carbon tax + variable O&M).
- **CAPEX (M€/yr)**: Annualized capital investment cost calculated from student capacities and techno-economic parameters (`data/fuel_sources/technology.json`) using the capital recovery annuity:
  $$\text{Annuity} = \text{CAPEX (€/MW)} \times \frac{r (1 + r)^n}{(1 + r)^n - 1} \quad (r = \text{WACC},\; n = \text{Lifetime})$$
- **TOTAL (M€/yr)**: Total annual system cost ($\text{OPEX} + \text{CAPEX annualisé}$).
- **COST (€/MWh)**: Average cost per unit of electricity generated (LCOE), computed as:
  $$\text{Cost (€/MWh)} = \frac{\text{TOTAL (M€/yr)} \times 10^6}{\text{Annual Generation (MWh)}} = \frac{\text{TOTAL (M€/yr)} \times 1000}{\text{Annual Generation (GWh)}}$$
- **CO2 (Mt)**: Total carbon emissions over the simulated period.
- **Economic and Environmental balance**: the distribution of operational costs and CO2 emissions across Europe.
