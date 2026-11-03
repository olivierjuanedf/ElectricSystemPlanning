# Data Crunch Notebook — Electric System Planning

## Goals

This notebook is the entry point of a hands-on power system sizing exercise. It serves as a **sandbox** to:

- Discover the electricity profile of the studied country (or zone);
- Understand the key indicators that shape a power system (load duration curves, controllable/fatal generation, base/peak load, gross/net demand, the Boiteux stacking method);
- Characterize the main power generation technologies (technical, environmental and economic parameters);
- Understand how to choose the power generation capacities to be installed to satisfy the demand using the Boiteux's stacking method;
- Arrive at a well-argued power mix choice for the studied country.

## Overall approach and methodology

The notebook follows a step-by-step pedagogical progression:

1. **Focus on the demand** before looking at supply — load curves, seasonal/daily variability, the notion of risk and unserved energy.
2. **Focus on thermal dispatchable technologies** to understand their economic, technical and environmental characteristics
3. **Understand how to size a 100% thermal system** using the Boiteux graphical stacking method (stacking technologies on the load duration curve based on their fixed/variable costs)
4. **Introduce variable renewable energy sources (vRES)** one at a time (solar, onshore/offshore wind, run-of-river hydro), observing their effect on the net demand curve.
5. **Understand how to size a system with both thermal and renewable energy sources** using the Boiteux graphical stacking method
6. **Size the country's power mix and compare with the country's actual power mix** (ERAA data). After discussing the differences, settle on a target mix for 2033, which serves as input for the detailed PyPSA modeling later in the week.

The data used comes from the **ERAA 2023.2** dataset (ENTSO-E): hourly demand by climatic year, renewable capacity factors, existing generation capacities.

## Expected final output

- A well-argued power mix choice for the studied country, for the 2033 horizon, accounting for the economic, environmental and technical constraints explored throughout the notebook.
- This mix, together with the technologies not modeled here (pumped hydro, hydro reservoirs, DSR, etc.), serves as the starting point (input) for the PyPSA modeling that follows in the rest of the practical work.
- A short presentation of the studied country, its demand profile, the theoretically ideal mix (built from scratch), the actual current mix (2025), and its proposed evolution up to 2033.

## Table of contents and section content

### Imports
Loading the libraries used (pandas, numpy, plotly, seaborn/matplotlib, json).

### Choice of study case
Selection of the studied country/meta-country (among: benelux, france, germany, iberian-peninsula, italy, poland, scandinavia) and of the ERAA projection years considered (2025, 2033).

### 1 — Power Demand
- Import of hourly demand data from ERAA
- Data analysis plotting gross demand, computing profile indicators (total demand, peak, seasonal variation), building the load duration curve.
- Risk management and unsupplied energy: introduction of the deficit/load-shedding concept as the system's balancing variable.

### 2 — Thermal Controllable Generation
- Conceptual distinction between commitable technologies (thermal, nuclear...) and fatal (variable renewable) generation.
- Presentation of controllable technologies and their economic characteristics (fixed/variable costs) and environmental characteristics (lifecycle GHG emissions, IPCC AR5 data, and impacts beyond carbon)

### 3 - Boiteux stacking method (1/2) - only with controllable generation technologies
- Application of the Boiteux graphical stacking method to stack technologies on the load duration curve and obtain a first 100% thermal mix, followed by a critical assessment of this mix.

### 4 - Introduction of variable Renewable Energy Sources (vRES)
- Production profiles (solar PV, onshore/offshore wind, run-of-river hydro) 
- Effect of each technology on the net demand curve (marginal analysis, system value)

### 5 - Boiteux stacking method (2/2) - with controllable and renewable generation technologies
- Application of the Boiteux graphical stacking method to stack technologies on the load duration curve when considering both renewable and controllable technologie, followed by a critical assessment of this mix.

### 6 — Exploring optimal power mix design
- Use of the Boiteux stacking method to obtain a power generation mix for the selected country or meta-country
- Comparison with the ERAA power mix: confronting the optimized mix with the actual installed capacities in the country per ERAA, guided by a set of questions to analyze the gaps.
- Choice of power mix to start the serious game: final synthesis of the retained 2033 mix (capacities by technology), to be documented and justified, to be used as input data for the PyPSA modeling in the rest of the practical work.
