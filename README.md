# Electric System Planning

This code will help you simulate your own European electric system. It is written in Python, using the [PyPSA](https://pypsa.org/) library to build Unit Commitment (UC) models, and it relies on the ERAA (European Resource Adequacy Assessment) dataset.

The repository supports three complementary use cases:
1. **European Unit Commitment (European mode)**: Simulate multi-country or single-country dispatch and optimal power flow across Europe (available on all branches).
2. **Data Crunch**: Analyze ERAA raw data, capacity factors, residual demand, and run exploratory notebooks (`main` branch).
3. **Toy Models**: Discover PyPSA modeling with simple single-zone pedagogical examples (`main-full` branch).

## Requirements 

* A **GitHub account**: If not the case, go to [https://github.com/](https://github.com/) and sign up.


## Setup

### Preferred: Run remotely with GitHub Codespaces (an online VS Code running in GitHub infrastructure)

1. Select the **Codespaces** tab on the repository page.
2. Click **Create codespace on main** (or on `main-full`).
3. Your Codespace is configuring... Please wait:  
   ![step 3, configuring](./.assets/img/building_codespace.png)
4. Your Codespace is ready! You can open the Python example script and run it:  
   ![step 4, ready](./.assets/img/ready.png)

> **Note for Jupyter Notebooks in Codespaces (e.g. `data_crunch.ipynb`)**:  
> The codespace automatically installs dependencies and sets up a dedicated Jupyter kernel named **`project`** (Python 3.11). When opening a notebook, click **Select Kernel** $\rightarrow$ **Python Environments...** $\rightarrow$ choose the **`project`** kernel (or the recommended Python 3.11 environment).


### Alternative: Run locally

<details>
  <summary>Help me, I really really want to run it locally</summary>

[blue pill or red pill ?](https://en.wikipedia.org/wiki/Red_pill_and_blue_pill)
![blue pill or red pill](https://upload.wikimedia.org/wikipedia/commons/5/52/Red_and_blue_pill.jpg)

Are you sure you want to choose the red pill over the blue pill?

<details>
  <summary>Yes, I'm sure</summary>

You have to install Python (version 3.11) and VS Code.

#### 1. Python

Install a compatible Python version 3.11:

<details>
  <summary>On Windows</summary>

Either via `winget`:
```powershell
winget install Python.Python.3.11
```
or download and install a stable release from [Python.org Downloads for Windows](https://www.python.org/downloads/windows/).
</details>

<details>
  <summary>On macOS</summary>

If you have Homebrew installed:
```bash
brew install python@3.11
```
or download and install from [Python.org Downloads for macOS](https://www.python.org/downloads/macos/).
</details>

<details>
  <summary>On Linux</summary>

Install Python 3.11 and the venv module using your package manager (e.g. on Ubuntu/Debian):
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv
```
</details>


#### 2. VS Code

Download and install [Visual Studio Code](https://code.visualstudio.com/download).


#### 3. Environment & Dependencies

Open the repository folder in VS Code. In the VS Code terminal:

**Create a virtual environment:**
```bash
python3.11 -m venv uc-env
# (or 'python -m venv uc-env' depending on your installation)
```

**Activate the virtual environment:**
- **Linux / macOS**:
  ```bash
  source uc-env/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\uc-env\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  uc-env\Scripts\activate.bat
  ```
- **Windows (Git Bash)**:
  ```bash
  source uc-env/Scripts/activate
  ```

**Install requirements:**
```bash
pip install -r requirements.txt
python -m ipykernel install --user --name uc_env --display-name "Python (uc_env)"
```

**Run the simulation:**
```bash
python my_little_europe_lt_uc.py
```

</details>

</details>


## Performance Improvement for students and Academics: install Gurobi

By default, the code uses the open-source solver **HiGHS**, which works out of the box without any license. For faster solving times on large European instances, academic users can install Gurobi:

<details>
  <summary>How to configure Gurobi academic license</summary>

* Connect to your university Wi-Fi or eduroam Wi-Fi.
* **Register** with your student/academic email: [https://portal.gurobi.com/iam/register](https://portal.gurobi.com/iam/register)  
  ![registration page 1/3](./.assets/img/gurobi_registration.png)
* Register **as a student / academic**  
  ![registration page 2/3](./.assets/img/gurobi_registration2.png)
* Set your password and activate your account.
* **Request a 'WLS license'** on this page: [https://portal.gurobi.com/iam/licenses/request](https://portal.gurobi.com/iam/licenses/request)  
  ![license 1/3](./.assets/img/gurobi_license1.png)
* **Download your license file** (`gurobi.lic`): [https://license.gurobi.com/manager/licenses](https://license.gurobi.com/manager/licenses)  
  ![license 2/3](./.assets/img/gurobi_license2.png)
* **Place your `gurobi.lic` file at the root of your project directory** (drag and drop in VS Code):  
  ![license 3/3](./.assets/img/gurobi_license3.png)

Finally, edit [`input/long_term_uc/solver_params.json`](input/long_term_uc/solver_params.json) to set your solver to `"gurobi"` and specify your license file:
```json
{
  "name": "gurobi",
  "license_file": "gurobi.lic"
}
```

</details>


## Structure of the repository

* `src/` : Core Python source code (modules: `common`, `include`, `include_runner`, `utils`).
* `data/` : Input data from ERAA 2023-2 (hourly demand, generation capacities, hydro parameters, interconnection capacities, RES capacity factors) and fuel/technology data.
* `input/` : User configuration parameters:
  * `input/long_term_uc/` : Parameters for the European long-term UC model (global configuration and per-country overrides in `countries/`).
  * `input/data_analysis/` : Configuration for raw data extraction and plots (`main-full` branch).
  * `input/toy_model_params/` : Parameters for single-zone toy models (`main-full` branch).
* `output/` : Simulation outputs (generated CSV data tables, network LP formulation, summary JSON, and plotted figures).
* `doc/` : Detailed documentation and tutorials:
  * `doc/long_term_uc/` : European UC tutorial (`European-system_tutorial.md`).
  * `doc/data-analysis/` : Data analysis tutorials and guide.
  * `doc/toy_model/` : Toy model tutorial and task descriptions.
  * `doc/teachers/` : Pedagogical plans, developer notes, and TODOs.
* `tests/` : Automated test suite (`pytest`).
* **Main entry point scripts** (at the root of the repository):
  * `my_little_europe_lt_uc.py` : Main script for the European Unit Commitment simulation.
  * `my_little_europe_data_analysis.py` : Data extraction and plotting script (`main-full`).
  * `data_crunch.ipynb` : Interactive data exploration notebook (`main-full`).
  * `my_toy_ex_italy.py` / `my_toy_ex_country.py` : Simplified pedagogical single-country examples (`main-full`).


## Quick Start

### 1. European Mode (All branches / Student course)

* **Main entry script**: `my_little_europe_lt_uc.py`
  ```bash
  python my_little_europe_lt_uc.py
  ```
* **Configuration files to customize**:
  * Global UC parameters (simulated period, mode, target year, climatic year...):  
    [`input/long_term_uc/elec-europe_params_to-be-modif.json`](input/long_term_uc/elec-europe_params_to-be-modif.json)
  * Per-country overrides:  
    `input/long_term_uc/countries/{country}.json`  
    *(Example files [`france.json`](input/long_term_uc/countries/france.json) and [`germany.json`](input/long_term_uc/countries/germany.json) are provided. Available ERAA country nodes are: `benelux`, `france`, `germany`, `iberian-peninsula`, `italy`, `poland`, `scandinavia`)*.
* **Detailed tutorial**: see [`doc/long_term_uc/European-system_tutorial.md`](doc/long_term_uc/European-system_tutorial.md).

---

### 2. Data Crunch Mode (`main` branch)


* **Interactive notebook**: `data_crunch.ipynb`


---

### 3. Toy Model Mode (`main-full` branch)

* **Pedagogical single-zone scripts**:
  * `my_toy_ex_italy.py` : Toy example based on Italy.
  * `my_toy_ex_country.py` : Generic country toy model.
* **Configuration files**: `input/toy_model_params/`
* **Detailed tutorial**: see [`doc/toy_model/toy-model_tutorial.md`](doc/toy_model/toy-model_tutorial.md).

---

## Git Branches & Teacher Information

The repository uses two primary branches to serve different pedagogical and development objectives:

* **Branch `main`** : Light version designed for students.
  * Focuses on the data crunch notebook and the European UC simulation mode.
  * Simplifies the repository view by omitting exploratory data analysis and toy models.
* **Branch `main-full`** : Full version with all features.
  * Contains all simulation modes (European UC, Data Analysis, Toy Models).
  * Includes teacher notes, backlog, and development plans in [`doc/teachers/`](doc/teachers/).
  * **Use this branch** for new developments, contributions, and instructor reference.
