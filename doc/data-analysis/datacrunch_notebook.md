# Data Crunch Notebook — Electric System Planning

## Goals

This notebook is the entry point of a hands-on power system sizing exercise. It serves as a **sandbox** to:

- Discover the electricity profile of the studied country (or zone);
- Understand the key indicators that shape a power system (load duration curves, controllable/fatal generation, base/peak load, gross/net demand, the Boiteux stacking method);
- Characterize the main power generation technologies (technical, environmental and economic parameters);
- Simulate various power mixes, from the simplest (100% thermal) to the most complete (renewables + storage + constraints);
- Arrive at a well-argued power mix choice for the studied country.

## Overall approach and methodology

The notebook follows a step-by-step pedagogical progression: each section introduces a new layer of complexity relative to the previous one, rather than starting directly with a fully complete model.

1. **Understand demand** before looking at supply — load curves, seasonal/daily variability, the notion of risk and unserved energy.
2. **Size a 100% thermal system** using the Boiteux graphical stacking method (stacking technologies on the load duration curve based on their fixed/variable costs), to build economic intuition around the merit order before adding complexity.
3. **Introduce variable renewable energy sources (vRES)** one at a time (solar, onshore/offshore wind, run-of-river hydro), observing their effect on the net demand curve.
4. **Introduce storage** (batteries, pumped hydro) as a tool to smooth this net demand.
5. **Move from a graphical/heuristic approach to mathematical optimization** (linear programming) to size an optimal mix under economic, environmental and technical constraints (including more realistic constraints on nuclear).
6. **Compare with the country's actual power mix** (ERAA data) and settle on a target mix for 2033, which serves as input for the detailed PyPSA modeling later in the week.

The data used comes from the **ERAA 2023.2** dataset (ENTSO-E): hourly demand by climatic year, renewable capacity factors, existing generation capacities.

## Expected final output

- A well-argued power mix choice for the studied country, for the 2033 horizon, accounting for the economic, environmental and technical constraints explored throughout the notebook.
- This mix, together with the technologies not modeled here (pumped hydro, hydro reservoirs, DSR, etc.), serves as the starting point (input) for the PyPSA modeling that follows in the rest of the practical work.
- A short presentation of the studied country, its demand profile, the theoretically ideal mix (built from scratch), the actual current mix (2025), and its proposed evolution up to 2033.

## Table of contents and section content

### 1 — Imports
Loading the libraries used (pandas, numpy, plotly, seaborn/matplotlib, json).

### 2 — Choice of study case
Selection of the studied country/zone (among: benelux, france, germany, iberian-peninsula, italy, poland, scandinavia) and of the ERAA projection years considered (2025, 2033).

### 3 — Power Demand
**3.1 — Download of gross demand data**: import of hourly demand data by ERAA climatic year, formatting and renaming of climatic scenarios (WS1, WS2...), construction of an average weather scenario (with a warning on the non-equiprobability of climatic years).

**3.2 — Data analysis (average weather scenario)**: plotting gross demand, computing profile indicators (total demand, peak, seasonal/daily/weekly variation), building the load duration curve.

**3.3 — Data analysis (including climatic years)**: same analyses as 3.2, extended to all available climatic years, to visualize inter-annual dispersion (quantile envelopes).

**3.4 — Risk management and unsupplied energy**: introduction of the deficit/load-shedding concept as the system's balancing variable.

### 4 — Generation
**4.1 — Notion of controllable and variable generation**: conceptual distinction between commitable technologies (thermal, nuclear...) and fatal (variable renewable) generation.

**4.2 — Optimal power mix with controllable generation**: presentation of controllable technologies and their economic characteristics (fixed/variable costs) and environmental characteristics (lifecycle GHG emissions, IPCC AR5 data, and impacts beyond carbon); application of the Boiteux graphical stacking method to stack technologies on the load duration curve and obtain a first 100% thermal mix, followed by a critical assessment of this mix.

**4.3 — Introduction of variable Renewable Energy Sources (vRES)**: production profiles (solar PV, onshore/offshore wind, run-of-river hydro) across all climatic years; effect of each technology on the net demand curve (marginal analysis, system value); economic and environmental characteristics of vRES.

### 5 — Storage
**5.1 — Batteries**: simplified modeling (intraday load shifting with efficiency losses), impact on net demand with and without vRES, economic characteristics (cost review sourced from IRENA/NREL/BNEF/IEA) and environmental considerations.

**5.2 — Pumped Hydro**: treatment of this existing storage technology.

### 6 — Exploring optimal power mix design
**6.1 — Optimal power mix with vRES deployment target**: optimization under a target renewable capacity constraint.

**6.2 — Optimal power mix (without vRES target)**: move to a full linear programming model (controllable capacities, curtailable vRES, battery with state of charge), including:
- a first resolution with no additional constraint (and the observation that unconstrained nuclear can cannibalize the value of renewables);
- a CO2 emissions reduction constraint (with an analysis of the shadow price associated with the constraint);
- a discussion of political mix choices (min/max capacity bounds per technology);
- (optional) realistic dynamic constraints on nuclear (minimum stable level, ramp rate).

**6.3 — Choice of a mix considering various climatic years**: solving the optimization model across all available climatic years, and synthesizing the distribution of resulting mixes, costs and emissions.

### 7 — Choice of a power mix
**7.1 — Comparison with the ERAA power mix**: confronting the optimized mix with the actual installed capacities in the country per ERAA, guided by a set of questions to analyze the gaps.

**7.3 — Capacity expansion starting from the current power mix**: moving from a "from scratch" optimization to a capacity-expansion optimization, where already-installed capacity is free (no CAPEX), with only new capacity (or decommissioning) carrying a cost — including a methodological note on technologies not modeled in this notebook (pumped hydro, hydro reservoirs, DSR, other renewable/non-renewable) and simplified modeling suggestions for each.

**7.4 — Choice of power mix for PyPSA**: final synthesis of the retained 2033 mix (capacities by technology), to be documented and justified, to be used as input data for the PyPSA modeling in the rest of the practical work.
