import glob
import json
import logging
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Union

from src.common.constants.countries import set_country_trigram
from src.common.constants.prod_types import ProdTypeNames

AVAILABLE_COUNTRIES: List[str] = [
    "benelux",
    "france",
    "germany",
    "iberian-peninsula",
    "italy",
    "poland",
    "scandinavia",
]

COUNTRY_ALIASES: Dict[str, str] = {
    "france": "france",
    "fr": "france",
    "germany": "germany",
    "de": "germany",
    "allemagne": "germany",
    "deutschland": "germany",
    "italy": "italy",
    "it": "italy",
    "italie": "italy",
    "italia": "italy",
    "poland": "poland",
    "pl": "poland",
    "pologne": "poland",
    "polska": "poland",
    "benelux": "benelux",
    "be": "benelux",
    "belgique": "benelux",
    "netherlands": "benelux",
    "pays-bas": "benelux",
    "nl": "benelux",
    "scandinavia": "scandinavia",
    "scandinavie": "scandinavia",
    "scand": "scandinavia",
    "se": "scandinavia",
    "no": "scandinavia",
    "suede": "scandinavia",
    "iberian-peninsula": "iberian-peninsula",
    "iberian_peninsula": "iberian-peninsula",
    "iberia": "iberian-peninsula",
    "peninsule_iberique": "iberian-peninsula",
    "péninsule ibérique": "iberian-peninsula",
    "es": "iberian-peninsula",
    "espagne": "iberian-peninsula",
    "spain": "iberian-peninsula",
    "portugal": "iberian-peninsula",
    "pt": "iberian-peninsula",
}

ASSET_ALIASES: Dict[str, str] = {
    # Nuclear
    "nuclear": ProdTypeNames.nuclear,
    "nucleaire": ProdTypeNames.nuclear,
    "nucléaire": ProdTypeNames.nuclear,
    "nuc": ProdTypeNames.nuclear,
    # Wind onshore
    "wind_onshore": ProdTypeNames.wind_onshore,
    "wind_on_shore": ProdTypeNames.wind_onshore,
    "wind onshore": ProdTypeNames.wind_onshore,
    "onshore_wind": ProdTypeNames.wind_onshore,
    "onshore": ProdTypeNames.wind_onshore,
    "eolien_terrestre": ProdTypeNames.wind_onshore,
    "éolien_terrestre": ProdTypeNames.wind_onshore,
    "eolien terrestre": ProdTypeNames.wind_onshore,
    "éolien terrestre": ProdTypeNames.wind_onshore,
    "eolien_onshore": ProdTypeNames.wind_onshore,
    # Wind offshore
    "wind_offshore": ProdTypeNames.wind_offshore,
    "wind_off_shore": ProdTypeNames.wind_offshore,
    "wind offshore": ProdTypeNames.wind_offshore,
    "offshore_wind": ProdTypeNames.wind_offshore,
    "offshore": ProdTypeNames.wind_offshore,
    "eolien_en_mer": ProdTypeNames.wind_offshore,
    "éolien_en_mer": ProdTypeNames.wind_offshore,
    "eolien en mer": ProdTypeNames.wind_offshore,
    "éolien en mer": ProdTypeNames.wind_offshore,
    "eolien_offshore": ProdTypeNames.wind_offshore,
    # Solar PV
    "solar_pv": ProdTypeNames.solar_pv,
    "solar": ProdTypeNames.solar_pv,
    "pv": ProdTypeNames.solar_pv,
    "solaire": ProdTypeNames.solar_pv,
    "solaire_pv": ProdTypeNames.solar_pv,
    "solar pv": ProdTypeNames.solar_pv,
    # Thermal & Fossil
    "gas": ProdTypeNames.gas,
    "gaz": ProdTypeNames.gas,
    "ccgt": ProdTypeNames.gas,
    "ocgt": ProdTypeNames.gas,
    "coal": ProdTypeNames.coal,
    "charbon": ProdTypeNames.coal,
    "lignite": ProdTypeNames.lignite,
    "oil": ProdTypeNames.oil,
    "fioul": ProdTypeNames.oil,
    "fuel": ProdTypeNames.oil,
    "petrole": ProdTypeNames.oil,
    "pétrole": ProdTypeNames.oil,
    "biofuel": ProdTypeNames.biofuel,
    "biomasse": ProdTypeNames.biofuel,
    "biomass": ProdTypeNames.biofuel,
    # Storage & Flexibility
    "batteries": ProdTypeNames.batteries,
    "battery": ProdTypeNames.batteries,
    "batterie": ProdTypeNames.batteries,
    "dsr": ProdTypeNames.dsr,
    "demand_side_response": ProdTypeNames.dsr,
    "effacement": ProdTypeNames.dsr,
    # Hydro
    "hydro_reservoir": ProdTypeNames.hydro_reservoir,
    "reservoir": ProdTypeNames.hydro_reservoir,
    "réservoir": ProdTypeNames.hydro_reservoir,
    "barrage": ProdTypeNames.hydro_reservoir,
    "hydro_pondage": ProdTypeNames.hydro_pondage,
    "pondage": ProdTypeNames.hydro_pondage,
    "eclusee": ProdTypeNames.hydro_pondage,
    "éclusée": ProdTypeNames.hydro_pondage,
    "hydro_run_of_river": ProdTypeNames.hydro_ror,
    "run_of_river": ProdTypeNames.hydro_ror,
    "ror": ProdTypeNames.hydro_ror,
    "fil_de_leau": ProdTypeNames.hydro_ror,
    "fil d'eau": ProdTypeNames.hydro_ror,
    "hydro_pump_storage_closed_loop": ProdTypeNames.hydro_pump_storage_closed,
    "pump_storage_closed_loop": ProdTypeNames.hydro_pump_storage_closed,
    "step_closed": ProdTypeNames.hydro_pump_storage_closed,
    "step_fermee": ProdTypeNames.hydro_pump_storage_closed,
    "step fermée": ProdTypeNames.hydro_pump_storage_closed,
    "hydro_pump_storage_open_loop": ProdTypeNames.hydro_pump_storage_open,
    "pump_storage_open_loop": ProdTypeNames.hydro_pump_storage_open,
    "step_open": ProdTypeNames.hydro_pump_storage_open,
    "step_ouverte": ProdTypeNames.hydro_pump_storage_open,
    "step ouverte": ProdTypeNames.hydro_pump_storage_open,
    # Other
    "others_fatal": ProdTypeNames.others_fatal,
    "fatal": ProdTypeNames.others_fatal,
    "failure": ProdTypeNames.failure,
    "defaillance": ProdTypeNames.failure,
    "défaillance": ProdTypeNames.failure,
}


def normalize_country_name(name: str) -> str:
    """Normalize and validate country name."""
    clean = name.strip().lower()
    clean = clean.replace(" ", "-").replace("_", "-")
    if clean in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[clean]
    if clean in AVAILABLE_COUNTRIES:
        return clean
    raise ValueError(
        f"Unknown country '{name}'. Available countries: {', '.join(AVAILABLE_COUNTRIES)}"
    )


def normalize_asset_name(name: str) -> str:
    """Normalize and validate production asset name."""
    clean = name.strip().lower()
    clean_key = clean.replace("-", "_")
    if clean_key in ASSET_ALIASES:
        return ASSET_ALIASES[clean_key]
    if clean in ASSET_ALIASES:
        return ASSET_ALIASES[clean]
    valid_names = [getattr(ProdTypeNames, attr) for attr in dir(ProdTypeNames)
                   if not attr.startswith("_") and isinstance(getattr(ProdTypeNames, attr), str)]
    if clean_key in valid_names:
        return clean_key
    raise ValueError(
        f"Unknown asset type '{name}'. Supported assets: nuclear, wind_onshore, wind_offshore, "
        f"solar_pv, gas, coal, lignite, oil, biofuel, batteries, dsr, hydro_reservoir, "
        f"hydro_pondage, hydro_run_of_river, hydro_pump_storage_closed_loop, hydro_pump_storage_open_loop"
    )


def parse_capacity_value(val_str: str) -> float:
    """Parse capacity value string in MW (e.g. '60000', '60000 MW', '60000.5')."""
    cleaned = val_str.strip().upper()
    if cleaned.endswith("MW"):
        cleaned = cleaned[:-2].strip()
    try:
        val = float(cleaned)
    except ValueError:
        raise ValueError(
            f"Invalid capacity value '{val_str}'. Please provide a number in MW (e.g. 60000)."
        )
    if val < 0:
        raise ValueError(f"Capacity cannot be negative ({val} MW).")
    return val


def parse_investment_file(filepath: str, country: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    Parse a student investment text file.
    Each line can be:
        nuclear: 60000
        wind_onshore = 25000
        solar_pv 35000
        country: france
    Lines starting with '#' or '//' are treated as comments.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Investment file not found: '{filepath}'")

    file_country: Optional[str] = None
    basename = os.path.basename(filepath)
    stem = os.path.splitext(basename)[0].lower()
    for c in AVAILABLE_COUNTRIES:
        if c in stem:
            file_country = c
            break

    capacities: Dict[str, float] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            if ":" in line:
                parts = line.split(":", 1)
            elif "=" in line:
                parts = line.split("=", 1)
            elif "," in line:
                parts = line.split(",", 1)
            else:
                parts = line.split(None, 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Syntax error on line {line_idx} in '{filepath}': '{line}'. "
                    f"Expected format: 'asset: capacity' (e.g., 'nuclear: 60000')."
                )

            key = parts[0].strip().lower()
            val = parts[1].strip()

            if key in ("country", "pays"):
                file_country = normalize_country_name(val)
                continue

            asset = normalize_asset_name(key)
            capa = parse_capacity_value(val)
            capacities[asset] = capa

    resolved_country = country or file_country
    if not resolved_country:
        raise ValueError(
            f"Country not specified for file '{filepath}'. "
            f"Please name your file '{AVAILABLE_COUNTRIES[1]}.txt', add 'country: <country>' "
            f"in the file, or pass '--country <country>' as argument."
        )

    resolved_country = normalize_country_name(resolved_country)
    return resolved_country, capacities


def parse_country_json_file(filepath: str, country: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    Parse an existing country JSON file (e.g. input/long_term_uc/countries/{country}.json).
    Extracts the country name and capacity dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Country JSON file not found: '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file '{filepath}': {exc}")

    file_country: Optional[str] = data.get("team")
    if not file_country:
        basename = os.path.basename(filepath)
        stem = os.path.splitext(basename)[0].lower()
        for c in AVAILABLE_COUNTRIES:
            if c in stem:
                file_country = c
                break

    resolved_country = country or file_country
    if not resolved_country:
        raise ValueError(
            f"Country not specified in JSON file '{filepath}' (missing 'team' key)."
        )

    resolved_country = normalize_country_name(resolved_country)

    raw_capas: dict = {}
    if "capacities_tb_overwritten" in data and isinstance(data["capacities_tb_overwritten"], dict):
        raw_capas = data["capacities_tb_overwritten"].get(resolved_country, {})
        if not raw_capas and len(data["capacities_tb_overwritten"]) == 1:
            raw_capas = next(iter(data["capacities_tb_overwritten"].values()))
    elif "capacities" in data and isinstance(data["capacities"], dict):
        raw_capas = data["capacities"]

    capacities: Dict[str, float] = {}
    for k, v in raw_capas.items():
        try:
            asset = normalize_asset_name(k)
            capacities[asset] = float(v)
        except Exception:
            pass

    return resolved_country, capacities


def load_country_capacities(filepath: str, country: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """Load country name and capacities from either a .txt or a .json file."""
    if filepath.endswith(".json"):
        return parse_country_json_file(filepath=filepath, country=country)
    return parse_investment_file(filepath=filepath, country=country)


def collect_investment_files(target: Union[str, List[str]], extension: str = ".txt") -> List[str]:
    """Find all investment files (.txt or .json) in a given directory or from a list of paths."""
    ext = extension if extension.startswith(".") else f".{extension}"
    pattern = f"*{ext}"

    if isinstance(target, list):
        files = []
        for item in target:
            if os.path.isdir(item):
                files.extend(glob.glob(os.path.join(item, pattern)))
            elif os.path.isfile(item) and item.endswith(ext):
                files.append(item)
        return sorted(list(set(files)))

    if os.path.isdir(target):
        files = glob.glob(os.path.join(target, pattern))
        return sorted(files)
    elif os.path.isfile(target):
        return [target] if target.endswith(ext) else []
    return []


def get_selected_target_year(
    europe_params_path: str = "input/long_term_uc/elec-europe_params_to-be-modif.json",
    default: int = 2033,
) -> int:
    """Read the target year from elec-europe_params_to-be-modif.json."""
    if os.path.isfile(europe_params_path):
        try:
            with open(europe_params_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("selected_target_year", default))
        except Exception:
            pass
    return default


def get_eraa_available_prod_types(
    country: str,
    target_year: int = 2033,
    eraa_values_path: str = "input/long_term_uc/elec-europe_eraa-available-values.json",
) -> Optional[List[str]]:
    """Return the list of aggregate production types available in ERAA for a given country and year."""
    if not os.path.isfile(eraa_values_path):
        return None
    try:
        with open(eraa_values_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        types_by_country = data.get("aggreg_prod_types", {}).get(country, {})
        return types_by_country.get(str(target_year)) or types_by_country.get(target_year)
    except Exception as e:
        logging.warning(f"Could not read {eraa_values_path}: {e}")
        return None


def write_country_json(country: str, capacities: Dict[str, float],
                       countries_dir: str = "input/long_term_uc/countries",
                       target_year: Optional[int] = None) -> str:
    """
    Write or update input/long_term_uc/countries/{country}.json.
    Sets team, selected_prod_types, and capacities_tb_overwritten.
    Filters out assets not present in the ERAA dataset for this country and year.
    """
    os.makedirs(countries_dir, exist_ok=True)
    json_path = os.path.join(countries_dir, f"{country}.json")

    if target_year is None:
        target_year = get_selected_target_year()

    available_eraa = get_eraa_available_prod_types(country=country, target_year=target_year)
    valid_capacities: Dict[str, float] = {}
    for asset, capa in capacities.items():
        if available_eraa is not None and asset not in available_eraa:
            print(f"      [!] Warning: '{asset}' is not available in the ERAA dataset for '{country}' in {target_year}. Asset ignored.")
            continue
        valid_capacities[asset] = capa

    active_assets = [asset for asset, capa in valid_capacities.items() if capa > 0]
    if ProdTypeNames.failure not in active_assets:
        active_assets.append(ProdTypeNames.failure)

    data_to_write = {
        "team": country,
        "selected_prod_types": {country: active_assets},
        "capacities_tb_overwritten": {country: valid_capacities}
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_to_write, f, indent=2, ensure_ascii=False)

    return json_path


def update_europe_params(mode: str = "europe", team: Optional[str] = None,
                         period_start: str = "1900/1/1", period_end: str = "1901/1/1",
                         europe_params_path: str = "input/long_term_uc/elec-europe_params_to-be-modif.json") -> None:
    """
    Enforce and set 'mode' to 'europe', and configure the simulation period to the full year
    ('1900/1/1' to '1901/1/1') in elec-europe_params_to-be-modif.json.
    The runner exclusively operates in European mode ('europe'). Solo mode is not supported for the runner.
    """
    if mode != "europe":
        raise ValueError(
            f"The runner exclusively operates in 'europe' mode (received mode='{mode}'). "
            f"Solo mode is not supported by the runner."
        )

    if not os.path.isfile(europe_params_path):
        return

    try:
        with open(europe_params_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        modified = False
        if data.get("mode") != "europe":
            data["mode"] = "europe"
            modified = True
        if team and data.get("team") != team:
            data["team"] = team
            modified = True
        if period_start and data.get("uc_period_start") != period_start:
            data["uc_period_start"] = period_start
            modified = True
        if period_end and data.get("uc_period_end") != period_end:
            data["uc_period_end"] = period_end
            modified = True

        if modified:
            with open(europe_params_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(
                f"Updated {europe_params_path}: mode='europe' team='{team or data.get('team')}' "
                f"period=[{period_start} - {period_end}]"
            )
    except Exception as e:
        logging.warning(f"Could not update {europe_params_path}: {e}")


def update_team_in_europe_params(country: str,
                                 europe_params_path: str = "input/long_term_uc/elec-europe_params_to-be-modif.json") -> None:
    """Ensure the 'team' parameter matches the country in elec-europe_params_to-be-modif.json."""
    update_europe_params(mode="europe", team=country, europe_params_path=europe_params_path)


def find_latest_summary_file(output_base_dir: str = "output/long_term_uc") -> Optional[str]:
    """Find the most recently generated uc-summary_*.json file."""
    pattern = os.path.join(output_base_dir, "**", "uc-summary_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


# Mapping of model prod types to technology.json keys
TECH_MAPPING: Dict[str, str] = {
    "nuclear": "Nuclear",
    "solar_pv": "Solar PV",
    "wind_onshore": "Wind Onshore",
    "wind_offshore": "Wind Offshore",
    "hydro_run_of_river": "Hydro ROR",
    "gas": "CCGT",
    "oil": "Oil",
    "coal": "Hard Coal",
    "lignite": "Lignite",
    "biofuel": "Biomass",
}

# Reference techno-economic parameters (fallback if technology.json is missing or for storage/hydro assets)
# CAPEX: EUR/MW, Lifetime: years, WACC: fraction
DEFAULT_TECH_PARAMS: Dict[str, Dict[str, float]] = {
    "nuclear": {"CAPEX": 4500e3, "Lifetime": 60, "WACC": 0.04},
    "solar_pv": {"CAPEX": 500e3, "Lifetime": 20, "WACC": 0.05},
    "wind_onshore": {"CAPEX": 1200e3, "Lifetime": 25, "WACC": 0.05},
    "wind_offshore": {"CAPEX": 2200e3, "Lifetime": 25, "WACC": 0.08},
    "hydro_run_of_river": {"CAPEX": 1600e3, "Lifetime": 50, "WACC": 0.04},
    "hydro_reservoir": {"CAPEX": 2000e3, "Lifetime": 50, "WACC": 0.04},
    "hydro_pondage": {"CAPEX": 1600e3, "Lifetime": 50, "WACC": 0.04},
    "hydro_pump_storage_closed_loop": {"CAPEX": 1800e3, "Lifetime": 40, "WACC": 0.05},
    "hydro_pump_storage_open_loop": {"CAPEX": 1500e3, "Lifetime": 40, "WACC": 0.05},
    "batteries": {"CAPEX": 1200e3, "Lifetime": 15, "WACC": 0.06},  # 300 €/kWh * 4h * 1000 kW/MW
    "dsr": {"CAPEX": 50e3, "Lifetime": 10, "WACC": 0.05},
    "gas": {"CAPEX": 1000e3, "Lifetime": 30, "WACC": 0.08},
    "oil": {"CAPEX": 500e3, "Lifetime": 20, "WACC": 0.08},
    "coal": {"CAPEX": 1800e3, "Lifetime": 35, "WACC": 0.08},
    "lignite": {"CAPEX": 1800e3, "Lifetime": 35, "WACC": 0.08},
    "biofuel": {"CAPEX": 1500e3, "Lifetime": 40, "WACC": 0.08},
    "others_fatal": {"CAPEX": 1000e3, "Lifetime": 25, "WACC": 0.05},
    "failure": {"CAPEX": 0.0, "Lifetime": 1, "WACC": 0.0},
}


def calc_crf(wacc: float, lifetime: int) -> float:
    """Capital Recovery Factor (CRF) / annuity factor."""
    if wacc <= 0 or lifetime <= 0:
        return 1.0 / max(lifetime, 1)
    return wacc * ((1.0 + wacc) ** lifetime) / (((1.0 + wacc) ** lifetime) - 1.0)


def calc_annualized_capex_per_mw(capex_eur_per_mw: float, wacc: float, lifetime: int) -> float:
    """Calculate annualized CAPEX in EUR / MW / year."""
    crf = calc_crf(wacc=wacc, lifetime=lifetime)
    return capex_eur_per_mw * crf


def get_country_capacities_from_json(country: str,
                                     countries_dir: str = "input/long_term_uc/countries") -> Dict[str, float]:
    """Load capacities for a given country from its input/long_term_uc/countries/{country}.json file."""
    json_path = os.path.join(countries_dir, f"{country}.json")
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("capacities_tb_overwritten", {}).get(country, {})
    except Exception:
        return {}


def compute_country_annualized_capex(
    capacities: Dict[str, float],
    tech_data_path: str = "data/fuel_sources/technology.json",
) -> Tuple[float, float]:
    """
    Compute total overnight CAPEX (EUR) and total annualized CAPEX (EUR/year)
    for a dictionary of production asset capacities in MW.
    Returns: (total_overnight_capex_eur, total_annualized_capex_eur_per_year)
    """
    tech_data: dict = {}
    if os.path.isfile(tech_data_path):
        try:
            with open(tech_data_path, "r", encoding="utf-8") as f:
                tech_data = json.load(f)
        except Exception:
            pass

    total_overnight = 0.0
    total_annualized = 0.0

    for asset, capa_mw in capacities.items():
        if capa_mw <= 0 or asset == ProdTypeNames.failure:
            continue

        tech_key = TECH_MAPPING.get(asset)
        params = None
        if tech_key and tech_key in tech_data:
            params = tech_data[tech_key]
        elif asset in DEFAULT_TECH_PARAMS:
            params = DEFAULT_TECH_PARAMS[asset]

        if not params:
            continue

        capex_per_mw = float(params.get("CAPEX", 0.0))
        lifetime = int(params.get("Lifetime", 25))
        wacc = float(params.get("WACC", 0.05))

        total_overnight += capa_mw * capex_per_mw
        total_annualized += capa_mw * calc_annualized_capex_per_mw(capex_per_mw, wacc, lifetime)

    return total_overnight, total_annualized


def get_production_from_opt_power_csv(summary_file_path: Optional[str]) -> Dict[str, float]:
    """Compute per-country total generated energy in GWh from opt_power_*.csv if available."""
    if not summary_file_path or not os.path.isfile(summary_file_path):
        return {}
    dir_path = os.path.dirname(summary_file_path)
    csv_files = glob.glob(os.path.join(dir_path, "opt_power_*.csv"))
    if not csv_files:
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        prod_cols = [c for c in df.columns if c != "date" and not c.endswith("_failure")]
        per_country_prod_gwh: Dict[str, float] = {}
        for col in prod_cols:
            tri = col.split("_")[0]
            per_country_prod_gwh[tri] = per_country_prod_gwh.get(tri, 0.0) + float(df[col].sum()) / 1e3
        return per_country_prod_gwh
    except Exception:
        return {}


def display_summary(summary_data: dict, country: str) -> None:
    """
    Display summary for a country within the European simulation.
    Delegates to display_multi_country_summary as the runner runs exclusively in Europe mode.
    """
    display_multi_country_summary(summary_data=summary_data, countries=[country])


def display_multi_country_summary(summary_data: dict,
                                  countries: Optional[List[str]] = None,
                                  updated_countries: Optional[List[str]] = None,
                                  per_country_capacities: Optional[Dict[str, Dict[str, float]]] = None,
                                  summary_file_path: Optional[str] = None) -> None:
    """
    Print a clean comparative multi-country summary table for European mode.
    Displays ENS, Loss of Load hours, OPEX, annualized CAPEX, TOTAL cost, cost per MWh produced, and CO2 emissions.
    Supports both full country names and 3-letter trigrams in summary_data keys.
    """
    total_cost = summary_data.get("total_cost", 0.0)
    total_op_cost = summary_data.get("total_operational_cost", 0.0)
    total_co2 = summary_data.get("total_co2_emissions", 0.0)

    per_country_ens = summary_data.get("per_country_ens", {}) or {}
    per_country_fail_h = summary_data.get("per_country_n_failure_hours", {}) or {}
    per_country_cost = summary_data.get("per_country_total_cost", {}) or {}
    per_country_op_cost = summary_data.get("per_country_total_operational_cost", {}) or {}
    per_country_co2 = summary_data.get("per_country_co2_emissions", {}) or {}

    # Production data (in GWh)
    per_country_prod = summary_data.get("per_country_production", {}) or {}
    if not per_country_prod and summary_file_path:
        per_country_prod = get_production_from_opt_power_csv(summary_file_path=summary_file_path)

    total_ens = sum(per_country_ens.values()) if per_country_ens else 0.0
    max_fail_h = max(per_country_fail_h.values()) if per_country_fail_h else 0.0

    display_countries = countries if countries is not None else AVAILABLE_COUNTRIES
    updated_set = set(updated_countries or [])

    print("\n" + "=" * 110)
    print("                              RESULTS OF THE EUROPEAN INTERCONNECTED SIMULATION")
    print("=" * 110)
    header = f"{'COUNTRY':<20} | {'ENS (MWh)':>10} | {'FAIL.(h)':>8} | {'OPEX(M€/yr)':>11} | {'CAPEX(M€/yr)':>12} | {'TOTAL(M€/yr)':>12} | {'COST(€/MWh)':>11} | {'CO2 (Mt)':>8}"
    print(header)
    print("-" * 110)

    total_europe_capex_m_yr = 0.0
    total_europe_overnight_b = 0.0
    total_europe_opex = 0.0
    total_europe_prod_gwh = 0.0

    for country in sorted(display_countries):
        tri = set_country_trigram(country)
        ens = per_country_ens.get(tri, per_country_ens.get(country, 0.0))
        fail_h = per_country_fail_h.get(tri, per_country_fail_h.get(country, 0.0))

        # OPEX for country (operational cost, fallback to total cost)
        opex = per_country_op_cost.get(tri, per_country_op_cost.get(country, per_country_cost.get(tri, per_country_cost.get(country, 0.0))))
        total_europe_opex += opex

        # CAPEX for country
        country_capas = {}
        if per_country_capacities and country in per_country_capacities:
            country_capas = per_country_capacities[country]
        else:
            country_capas = get_country_capacities_from_json(country=country)

        overnight_eur, annualized_eur = compute_country_annualized_capex(capacities=country_capas)
        capex_m_yr = annualized_eur / 1e6
        overnight_b = overnight_eur / 1e9

        total_europe_capex_m_yr += capex_m_yr
        total_europe_overnight_b += overnight_b

        total_m_yr = opex + capex_m_yr
        co2 = per_country_co2.get(tri, per_country_co2.get(country, 0.0))

        # Cost per MWh produced (€/MWh = (M€ * 1e6) / (GWh * 1e3) = M€ * 1000 / GWh)
        prod_gwh = per_country_prod.get(tri, per_country_prod.get(country, 0.0))
        total_europe_prod_gwh += prod_gwh
        if prod_gwh > 0:
            cost_per_mwh = (total_m_yr * 1000.0) / prod_gwh
            cost_mwh_str = f"{cost_per_mwh:>11,.1f}"
        else:
            cost_mwh_str = f"{'-':>11}"

        opex_str = f"{opex:>11,.1f}" if opex > 0 else f"{'-':>11}"
        capex_str = f"{capex_m_yr:>12,.1f}" if capex_m_yr > 0 else f"{'-':>12}"
        total_str = f"{total_m_yr:>12,.1f}" if total_m_yr > 0 else f"{'-':>12}"
        co2_str = f"{co2:>8,.2f}" if co2 > 0 else f"{'-':>8}"

        tag = " *" if country in updated_set else ""
        c_display = f"{country}{tag}"
        print(f"{c_display:<20} | {ens:>10,.1f} | {fail_h:>8.1f} | {opex_str} | {capex_str} | {total_str} | {cost_mwh_str} | {co2_str}")

    # European total OPEX: prefer global metric if available
    display_europe_opex = total_op_cost if total_op_cost > 0 else total_europe_opex
    total_europe_cost = display_europe_opex + total_europe_capex_m_yr

    # European cost per MWh
    if total_europe_prod_gwh > 0:
        total_europe_cost_mwh = (total_europe_cost * 1000.0) / total_europe_prod_gwh
        total_europe_cost_mwh_str = f"{total_europe_cost_mwh:>11,.1f}"
    else:
        total_europe_cost_mwh_str = f"{'-':>11}"

    print("-" * 110)
    print(f"{'TOTAL EUROPE':<20} | {total_ens:>10,.1f} | {max_fail_h:>8.1f} | {display_europe_opex:>11,.1f} | {total_europe_capex_m_yr:>12,.1f} | {total_europe_cost:>12,.1f} | {total_europe_cost_mwh_str} | {total_co2:>8,.2f}")
    print("=" * 110)
    print(f"  Total operational cost (OPEX) : {display_europe_opex:,.2f} M€/year")
    print(f"  Total annualized CAPEX        : {total_europe_capex_m_yr:,.2f} M€/year (Total overnight: ~{total_europe_overnight_b:,.1f} B€)")
    print(f"  Total system cost (OPEX+CAPEX): {total_europe_cost:,.2f} M€/year")
    if total_europe_prod_gwh > 0:
        total_europe_prod_twh = total_europe_prod_gwh / 1000.0
        avg_opex_mwh = (display_europe_opex * 1000.0) / total_europe_prod_gwh
        avg_capex_mwh = (total_europe_capex_m_yr * 1000.0) / total_europe_prod_gwh
        avg_cost_mwh = (total_europe_cost * 1000.0) / total_europe_prod_gwh
        print(f"  Total generation              : {total_europe_prod_twh:,.2f} TWh ({total_europe_prod_gwh:,.0f} GWh)")
        print(f"  Average electricity cost      : {avg_cost_mwh:,.2f} €/MWh (OPEX: {avg_opex_mwh:,.2f} €/MWh, CAPEX: {avg_capex_mwh:,.2f} €/MWh)")
    if updated_set:
        print(f"  (*) Countries updated by the runner during this session: {', '.join(sorted(updated_set))}")

    # Countries with ENS
    countries_with_fail = [
        c for c in display_countries
        if per_country_ens.get(set_country_trigram(c), per_country_ens.get(c, 0.0)) > 0.1
        or per_country_fail_h.get(set_country_trigram(c), per_country_fail_h.get(c, 0.0)) > 0
    ]
    if countries_with_fail:
        print(f"\n[!] LOSS OF LOAD / FAILURE DETECTED in: {', '.join(countries_with_fail)}")
        print("    These countries must reinforce dispatchable capacities, renewables, or flexibility/interconnections.")
    else:
        print("\n[OK] NO LOSS OF LOAD: The entire European system is balanced!")
    print()


def run_simulation(script_path: str = "my_little_europe_lt_uc.py") -> int:
    """Execute the simulation script via subprocess, streaming stdout/stderr."""
    cmd = [sys.executable, script_path]
    logging.info(f"Running simulation: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode
