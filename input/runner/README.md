# Student Investment Choices

This directory allows you to define your generation mix choices in a simple text format.

## Investment File Format

Each line represents an asset and its installed capacity in **MW** (megawatts):

```text
# Specify your country (or name the file with the country name, e.g., france.txt)
country: france

# List of assets and capacities (in MW)
nuclear: 61000
solar_pv: 35000
wind_onshore: 25000
wind_offshore: 10000
hydro_reservoir: 8500
hydro_run_of_river: 4500
gas: 4000
batteries: 3000
```

### Syntax Rules:
- **Accepted separators**: colon `:`, equals sign `=`, or space (e.g., `nuclear: 60000` or `nuclear 60000`).
- **Comments**: lines starting with `#` or `//` are ignored.
- **Units**: capacities are in **MW**. You can optionally append `MW` (e.g., `60000 MW`).
- **Recognized asset names**:
  - `nuclear` (aliases: `nucleaire`, `nuc`)
  - `wind_onshore` (aliases: `wind_on_shore`, `eolien_terrestre`)
  - `wind_offshore` (aliases: `wind_off_shore`, `eolien_en_mer`)
  - `solar_pv` (aliases: `solar`, `pv`, `solaire`)
  - `gas` (aliases: `gaz`, `ccgt`, `ocgt`)
  - `coal` (aliases: `charbon`)
  - `lignite`
  - `oil` (aliases: `fioul`, `fuel`)
  - `biofuel` (aliases: `biomasse`)
  - `batteries` (aliases: `battery`)
  - `dsr` (aliases: `effacement`)
  - `hydro_reservoir` (aliases: `reservoir`, `barrage`)
  - `hydro_run_of_river` (aliases: `run_of_river`, `ror`, `fil_de_leau`)
  - `hydro_pondage` (aliases: `pondage`, `eclusee`)
  - `hydro_pump_storage_closed_loop` (aliases: `step_closed`, `step_fermee`)
  - `hydro_pump_storage_open_loop` (aliases: `step_open`, `step_ouverte`)

## How to Run the Collective Runner?

From the root of the project, you can choose the investment source to use:

### TXT Mode (Default)
Uses the `.txt` investment files from this directory (`input/runner/`) and updates the corresponding `country.json` files:

```bash
# Preview changes without solving (updates country JSON files)
python main_runner.py --dry-run
# or explicitly:
python main_runner.py txt --dry-run

# Full European collective simulation run
python main_runner.py
# or:
python main_runner.py txt
```

### JSON Mode
Directly uses the existing `.json` files from `input/long_term_uc/countries/` without overwriting them or looking for text files:

```bash
# Preview configuration in JSON mode without solving
python main_runner.py json --dry-run
# or with the --source option:
python main_runner.py --source json --dry-run

# Full European collective simulation run in JSON mode
python main_runner.py json
```

### Execution Workflow:
1. Discovers country files according to the requested source (`.txt` in `input/runner/` or `.json` in `input/long_term_uc/countries/`).
2. In TXT mode: updates each corresponding file in `input/long_term_uc/countries/{country}.json` (in JSON mode, existing files are used directly without modification).
3. Automatically sets `"europe"` mode and the annual horizon in `input/long_term_uc/elec-europe_params_to-be-modif.json`.
4. Executes the interconnected European unit commitment optimization simulation (`my_little_europe_lt_uc.py`).
5. Displays a comparative dashboard by country (Energy Not Served, loss of load hours, OPEX, CAPEX, total cost, cost per MWh, and CO2 emissions).
