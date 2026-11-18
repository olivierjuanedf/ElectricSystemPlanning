# Student Investment Choices

This directory allows you to define your generation mix choices in a simple text format (`.txt`) or JSON format (`.json`).

## Dual Europe Configurations (Europe 1 & Europe 2)

The runner supports simulating and comparing two European scenarios or team configurations (**Europe 1** and **Europe 2**).

File naming convention:
- **TXT format**: `{country}_europe1.txt` and `{country}_europe2.txt` (e.g. `france_europe1.txt`, `france_europe2.txt`, `germany_europe1.txt`, `germany_europe2.txt`).
- **JSON format**: `{country}_europe1.json` and `{country}_europe2.json` (e.g. `france_europe1.json`, `france_europe2.json`).
- If no Europe suffix is provided (e.g. `france.txt`), the file is attributed by default to `europe1`.

When files for both Europe 1 and Europe 2 are present, the runner:
1. Runs the European simulation for **Europe 1**.
2. Runs the European simulation for **Europe 2**.
3. Displays a comprehensive **Comparative Dashboard** (Europe 1 vs Europe 2 with Δ and % differences) in the terminal.
4. Saves the comparative results into `output/runner/`:
   - `output/runner/comparison_europe1_vs_europe2_YYYYMMDD_HHMMSS.txt` (timestamped text report)
   - `output/runner/comparison_europe1_vs_europe2_YYYYMMDD_HHMMSS.json` (timestamped structured metrics)
   - `output/runner/comparison_europe1_vs_europe2_latest.txt` (latest text report)
   - `output/runner/comparison_europe1_vs_europe2_latest.json` (latest structured metrics)
5. Automatically tracks and displays the **Simulation Evolution History** across successive runs for Europe 1 and for each country (saved in `output/runner/history_evolutions_europe1.txt` and `.json`).

---

## Investment File Format

Each line represents an asset and its installed capacity in **MW** (megawatts):

```text
# Specify your country (optional if the country name is in the filename)
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

### Missing Countries & Empty Files Behavior:

| Situation | In the Simulation Grid | Capacities Applied |
| :--- | :--- | :--- |
| **No file provided for a country** | The country is **excluded** from the simulation (`selected_countries`). | **Not simulated** (0 cost/gen in the dashboard). |
| **Empty file (or only `country: <name>`)** | The country is **included** in the European simulation. | **0 MW for all assets**. Only the fictive `failure` generator is active (100% loss of load / ENS). |
| **Asset omitted from the file** | The asset is not included in `selected_prod_types`. | **0 MW** (there is **no automatic fallback to ERAA** capacities). |

> **Note on ERAA References**: The runner does not inject default ERAA capacities into incomplete or missing student mixes. However, it independently simulates and displays pure **ERAA 2025** and **ERAA 2033** baselines in the comparative dashboard and trajectory table as reference benchmarks.

## How to Run the Collective Runner?

From the root of the project, you can choose the investment source to use:

### TXT Mode (Default)
Uses the `.txt` investment files from this directory (`input/runner/`):

```bash
# Preview changes without solving (updates country JSON files for all detected Europes)
python main_runner.py --dry-run
# or explicitly:
python main_runner.py txt --dry-run

# Full simulation of both Europes + Comparative Dashboard
python main_runner.py
# or:
python main_runner.py txt

# Simulate only Europe 1 or Europe 2:
python main_runner.py --europe europe1
python main_runner.py --europe europe2
```

### JSON Mode
Uses `.json` investment files from `input/runner/`:

```bash
# Preview configuration in JSON mode without solving
python main_runner.py json --dry-run
# or with the --source option:
python main_runner.py --source json --dry-run

# Full European collective simulation run in JSON mode
python main_runner.py json
```

### Execution Workflow:
1. Discovers country files according to the requested source (`.txt` or `.json` in `input/runner/`).
2. Groups files by Europe configuration (`europe1`, `europe2`).
3. Sequentially updates `input/long_term_uc/countries/{country}.json` and executes the optimization simulation (`my_little_europe_lt_uc.py`) for each Europe.
4. Displays the **Comparative Dashboard** comparing Europe 1 and Europe 2 (ENS, loss of load hours, OPEX, CAPEX, total cost, cost per MWh, and CO2 emissions).
5. Automatically saves `output/long_term_uc/comparison_europe1_vs_europe2.txt` and `.json`.
