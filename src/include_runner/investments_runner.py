import glob
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
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

AVAILABLE_CLIMATIC_YEARS: List[int] = [
    1982,
    1989,
    1996,
    2003,
    2010,
    2016,
]

# Baseline dictionary of non-adjacent European country pairs set to 0 MW.
# PyPSA requires every pair of countries without an ERAA link to have an explicit 0 MW entry.
# Values must be int to satisfy UCRunParams and type checker constraints.
DEFAULT_BASE_INTERCO_OVERWRITTEN: Dict[str, int] = {
    "france2poland": 0,
    "france2scandinavia": 0,
    "italy2iberian-peninsula": 0,
    "italy2poland": 0,
    "italy2benelux": 0,
    "italy2scandinavia": 0,
    "germany2iberian-peninsula": 0,
    "iberian-peninsula2italy": 0,
    "iberian-peninsula2germany": 0,
    "iberian-peninsula2poland": 0,
    "iberian-peninsula2benelux": 0,
    "iberian-peninsula2scandinavia": 0,
    "poland2france": 0,
    "poland2italy": 0,
    "poland2iberian-peninsula": 0,
    "poland2benelux": 0,
    "benelux2italy": 0,
    "benelux2iberian-peninsula": 0,
    "benelux2poland": 0,
    "scandinavia2france": 0,
    "scandinavia2italy": 0,
    "scandinavia2iberian-peninsula": 0,
    "germany2italy": 0,
}

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


def normalize_interco_key(key: str) -> str:
    """
    Normalize an interconnection key (e.g. 'france2germany', 'fr2de', 'france -> germany', 'france-germany')
    into the canonical format '{origin}2{destination}'.
    """
    cleaned = key.strip().lower()

    if "->" in cleaned:
        parts = cleaned.split("->", 1)
    elif " to " in cleaned:
        parts = cleaned.split(" to ", 1)
    elif "_to_" in cleaned:
        parts = cleaned.split("_to_", 1)
    elif "2" in cleaned:
        parts = cleaned.split("2", 1)
    elif "--" in cleaned:
        parts = cleaned.split("--", 1)
    else:
        parts = cleaned.split("-", 1)
        if len(parts) == 2:
            try:
                c1 = normalize_country_name(parts[0])
                c2 = normalize_country_name(parts[1])
                return f"{c1}2{c2}"
            except ValueError:
                pass
        raise ValueError(
            f"Invalid interconnection link format: '{key}'. "
            f"Expected format '<origin>2<destination>' (e.g. 'france2germany' or 'france -> germany')."
        )

    c1 = normalize_country_name(parts[0].strip())
    c2 = normalize_country_name(parts[1].strip())
    if c1 == c2:
        raise ValueError(f"Origin and destination countries cannot be identical: '{key}'.")
    return f"{c1}2{c2}"


def parse_interco_file(filepath: str) -> Dict[str, float]:
    """
    Parse an interconnection configuration JSON file.
    Supports:
      1. Flat dictionary: {"france2germany": 5000, ...}
      2. Nested dictionary: {"interco_capas_tb_overwritten": {"france2germany": 5000, ...}}
    Ignores metadata keys starting with '_' or named 'comment', 'description', etc.
    Returns:
      Dict mapping canonical '{origin}2{destination}' to capacity in MW.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Interconnection file not found: '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in interconnection file '{filepath}': {exc}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object/dictionary in '{filepath}', got {type(data).__name__}")

    raw_items = data.get("interco_capas_tb_overwritten", data)
    if not isinstance(raw_items, dict):
        raise ValueError(
            f"Expected dictionary for 'interco_capas_tb_overwritten' in '{filepath}', got {type(raw_items).__name__}"
        )

    overwrites: Dict[str, float] = {}
    ignored_keys = {"comment", "description", "note", "notes", "europe", "scenario", "team"}

    for raw_k, raw_v in raw_items.items():
        k_str = str(raw_k).strip()
        if k_str.startswith("_") or k_str.lower() in ignored_keys:
            continue
        norm_key = normalize_interco_key(k_str)
        capa = parse_capacity_value(str(raw_v))
        overwrites[norm_key] = capa

    return overwrites


def find_interco_file(europe_id: str, runner_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Search for an interconnection configuration file for a given Europe identifier ('europe1', 'europe2').
    Searches in runner_dirs (default: ['input/runner']).
    """
    dirs = runner_dirs or ["input/runner"]
    eur_clean = europe_id.lower().strip()
    candidates = [
        f"interco_{eur_clean}.json",
        f"interconnections_{eur_clean}.json",
        f"interco-{eur_clean}.json",
        f"{eur_clean}_interco.json",
    ]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for cand in candidates:
            path = os.path.join(d, cand)
            if os.path.isfile(path):
                return path
    return None


def load_europe_interco_overwrites(
    europe_id: str,
    runner_dirs: Optional[List[str]] = None,
    custom_filepath: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load interconnection overwrites for a given Europe scenario.
    Returns empty dict if file does not exist or has no overrides.
    """
    target_file = custom_filepath or find_interco_file(europe_id=europe_id, runner_dirs=runner_dirs)
    if not target_file or not os.path.isfile(target_file):
        return {}
    try:
        return parse_interco_file(target_file)
    except Exception as exc:
        logging.warning(f"Could not load interconnection file '{target_file}': {exc}")
        return {}


def extract_europe_id(filepath_or_name: str, default_europe: str = "europe1") -> str:
    """
    Extract Europe identifier ('europe1' or 'europe2') from a filename or its contents.
    Supported patterns in filename:
        france_europe1.txt -> 'europe1'
        france_europe2.txt -> 'europe2'
        germany_europe_1.json -> 'europe1'
        germany_eur2.json -> 'europe2'
    If not specified in filename or file contents, defaults to default_europe.
    """
    basename = os.path.basename(filepath_or_name)
    stem = os.path.splitext(basename)[0].lower()

    if re.search(r"(?:^|[_\-\s])(?:europe[_\-\s]*2|eur[_\-\s]*2)(?:[_\-\s]|$)", stem):
        return "europe2"
    if re.search(r"(?:^|[_\-\s])(?:europe[_\-\s]*1|eur[_\-\s]*1)(?:[_\-\s]|$)", stem):
        return "europe1"

    # Inspect file content if available
    if os.path.isfile(filepath_or_name):
        try:
            if filepath_or_name.endswith(".json"):
                with open(filepath_or_name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    val = data.get("europe") or data.get("europe_id") or data.get("scenario")
                    if val:
                        val_str = str(val).strip().lower()
                        if "2" in val_str:
                            return "europe2"
                        if "1" in val_str:
                            return "europe1"
            else:
                with open(filepath_or_name, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or line.startswith("//"):
                            continue
                        if ":" in line or "=" in line:
                            sep = ":" if ":" in line else "="
                            parts = line.split(sep, 1)
                            key = parts[0].strip().lower()
                            if key in ("europe", "europe_id", "scenario"):
                                val_str = parts[1].strip().lower()
                                if "2" in val_str:
                                    return "europe2"
                                if "1" in val_str:
                                    return "europe1"
        except Exception:
            pass

    return default_europe


def extract_country_and_europe(filepath: str, default_europe: str = "europe1") -> Tuple[str, str]:
    """Extract country name and Europe identifier for an investment file."""
    country, _ = load_country_capacities(filepath=filepath)
    europe_id = extract_europe_id(filepath_or_name=filepath, default_europe=default_europe)
    return country, europe_id


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

    if not file_country:
        tokens = re.split(r"[_\-\s.]+", stem)
        for token in tokens:
            if token in COUNTRY_ALIASES:
                file_country = COUNTRY_ALIASES[token]
                break

    use_eraa = False
    eraa_year: Optional[int] = None
    capacities: Dict[str, float] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            line_lower = line.lower()
            if line_lower in ("eraa", "use_eraa", "eraa_2025", "eraa_2033", "eraa 2025", "eraa 2033"):
                use_eraa = True
                if "2025" in line_lower:
                    eraa_year = 2025
                elif "2033" in line_lower:
                    eraa_year = 2033
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
                key_candidate = parts[0].strip().lower()
                if key_candidate in ("eraa", "use_eraa"):
                    use_eraa = True
                    continue
                raise ValueError(
                    f"Syntax error on line {line_idx} in '{filepath}': '{line}'. "
                    f"Expected format: 'asset: capacity' (e.g., 'nuclear: 60000') or 'eraa'."
                )

            key = parts[0].strip().lower()
            val = parts[1].strip()

            if key in ("country", "pays"):
                file_country = normalize_country_name(val)
                continue

            if key in ("eraa", "base", "mix", "source"):
                val_lower = val.lower()
                if "eraa" in val_lower or val_lower in ("true", "yes", "1", "2025", "2033"):
                    use_eraa = True
                    if "2025" in val_lower:
                        eraa_year = 2025
                    elif "2033" in val_lower:
                        eraa_year = 2033
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

    if use_eraa:
        if eraa_year is None:
            try:
                eraa_year = get_selected_target_year()
            except Exception:
                eraa_year = 2033
        eraa_caps_all = load_eraa_country_capacities(year=eraa_year)
        country_eraa = eraa_caps_all.get(resolved_country, {})
        final_capacities = dict(country_eraa)
        final_capacities.update(capacities)
        capacities = final_capacities

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
        if not file_country:
            tokens = re.split(r"[_\-\s.]+", stem)
            for token in tokens:
                if token in COUNTRY_ALIASES:
                    file_country = COUNTRY_ALIASES[token]
                    break

    resolved_country = country or file_country
    if not resolved_country:
        raise ValueError(
            f"Country not specified in JSON file '{filepath}' (missing 'team' key)."
        )

    resolved_country = normalize_country_name(resolved_country)

    use_eraa = False
    eraa_year: Optional[int] = None
    if data.get("eraa") or str(data.get("base", "")).lower() == "eraa" or str(data.get("mix", "")).lower() == "eraa":
        use_eraa = True
        val_eraa = str(data.get("eraa", "")).lower()
        if "2025" in val_eraa:
            eraa_year = 2025
        elif "2033" in val_eraa:
            eraa_year = 2033
    elif isinstance(data.get("selected_prod_types"), dict):
        prod_types = data["selected_prod_types"].get(resolved_country, [])
        if "all" in prod_types:
            use_eraa = True

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

    if use_eraa:
        if eraa_year is None:
            try:
                eraa_year = get_selected_target_year()
            except Exception:
                eraa_year = 2033
        eraa_caps_all = load_eraa_country_capacities(year=eraa_year)
        country_eraa = eraa_caps_all.get(resolved_country, {})
        final_capacities = dict(country_eraa)
        final_capacities.update(capacities)
        capacities = final_capacities

    return resolved_country, capacities


def load_country_capacities(filepath: str, country: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """Load country name and capacities from either a .txt or a .json file."""
    if filepath.endswith(".json"):
        return parse_country_json_file(filepath=filepath, country=country)
    return parse_investment_file(filepath=filepath, country=country)


def group_files_by_europe(
    filepaths: List[str],
    default_europe: str = "europe1",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Groups collected investment files by Europe identifier ('europe1', 'europe2', ...).
    Returns:
        {europe_id: {country: {asset: capacity}}}
    """
    grouped_capas, _ = group_files_by_europe_with_sources(filepaths, default_europe=default_europe)
    return grouped_capas


def group_files_by_europe_with_sources(
    filepaths: List[str],
    default_europe: str = "europe1",
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, str]]]:
    """
    Groups collected investment files by Europe identifier and tracks file sources.
    Returns:
        (grouped_capacities, grouped_sources)
        where:
        grouped_capacities: {europe_id: {country: capacities}}
        grouped_sources: {europe_id: {country: filepath}}
    """
    grouped_capas: Dict[str, Dict[str, Dict[str, float]]] = {}
    grouped_srcs: Dict[str, Dict[str, str]] = {}
    for fpath in filepaths:
        country, capacities = load_country_capacities(filepath=fpath)
        europe_id = extract_europe_id(filepath_or_name=fpath, default_europe=default_europe)
        if europe_id not in grouped_capas:
            grouped_capas[europe_id] = {}
            grouped_srcs[europe_id] = {}
        grouped_capas[europe_id][country] = capacities
        grouped_srcs[europe_id][country] = fpath
    return grouped_capas, grouped_srcs


def fill_missing_countries_with_eraa(
    grouped_capas: Dict[str, Dict[str, Dict[str, float]]],
    grouped_srcs: Dict[str, Dict[str, str]],
    target_year: Optional[int] = None,
    available_countries: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, str]]]:
    """
    For any country in available_countries not present in grouped_capas for a given Europe,
    populate it with official ENTSO-E ERAA reference capacities.
    """
    if target_year is None:
        try:
            target_year = get_selected_target_year()
        except Exception:
            target_year = 2033

    target_countries = available_countries if available_countries is not None else AVAILABLE_COUNTRIES
    eraa_ref_capas = load_eraa_country_capacities(year=target_year)

    for eur in list(grouped_capas.keys()):
        for c in target_countries:
            if c not in grouped_capas[eur]:
                grouped_capas[eur][c] = dict(eraa_ref_capas.get(c, {}))
                grouped_srcs[eur][c] = f"ERAA {target_year} reference"

    return grouped_capas, grouped_srcs


def collect_investment_files(target: Union[str, List[str]], extension: str = ".txt") -> List[str]:
    """Find all investment files (.txt or .json) in a given directory or from a list of paths."""
    ext = extension if extension.startswith(".") else f".{extension}"
    pattern = f"*{ext}"

    def _is_interco_file(fname: str) -> bool:
        base = os.path.basename(fname).lower()
        return "interco" in base or "interconnection" in base

    if isinstance(target, list):
        files = []
        for item in target:
            if os.path.isdir(item):
                files.extend([f for f in glob.glob(os.path.join(item, pattern)) if not _is_interco_file(f)])
            elif os.path.isfile(item) and item.endswith(ext) and not _is_interco_file(item):
                files.append(item)
        return sorted(list(set(files)))

    if os.path.isdir(target):
        files = [f for f in glob.glob(os.path.join(target, pattern)) if not _is_interco_file(f)]
        return sorted(files)
    elif os.path.isfile(target):
        return [target] if (target.endswith(ext) and not _is_interco_file(target)) else []
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


def get_selected_climatic_year(
    europe_params_path: str = "input/long_term_uc/elec-europe_params_to-be-modif.json",
    default: int = 1989,
) -> int:
    """Read the climatic year from elec-europe_params_to-be-modif.json."""
    if os.path.isfile(europe_params_path):
        try:
            with open(europe_params_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("selected_climatic_year", default))
        except Exception:
            pass
    return default


def get_period_tag(period_start: str, period_end: str) -> str:
    """Generate a clean filesystem tag for simulation period."""
    if period_start == TEST_PERIOD_START and period_end == TEST_PERIOD_END:
        return "test_2w"
    if period_start == DEFAULT_PERIOD_START and period_end == DEFAULT_PERIOD_END:
        return "full_year"
    clean_s = period_start.replace("/", "").replace("-", "")
    clean_e = period_end.replace("/", "").replace("-", "")
    return f"{clean_s}_{clean_e}"


def get_eraa_baseline_cache_path(
    eraa_year: int,
    period_start: str,
    period_end: str,
    target_year: int = 2033,
    climatic_year: int = 1989,
    countries: Optional[List[str]] = None,
    output_dir: str = "output/runner",
) -> str:
    """Return the cache file path for an ERAA baseline simulation."""
    baselines_dir = os.path.join(output_dir, "baselines")
    period_tag = get_period_tag(period_start, period_end)
    if countries and len(countries) < len(AVAILABLE_COUNTRIES):
        c_tag = "_" + "-".join(sorted(countries))
    else:
        c_tag = ""
    filename = f"eraa_baseline_{eraa_year}_target{target_year}_cy{climatic_year}{c_tag}_{period_tag}.json"
    return os.path.join(baselines_dir, filename)


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
    Only writes capacities_tb_overwritten for active assets to prevent PyPSA unused capacity warnings.
    """
    os.makedirs(countries_dir, exist_ok=True)
    json_path = os.path.join(countries_dir, f"{country}.json")

    if target_year is None:
        target_year = get_selected_target_year()

    available_eraa = get_eraa_available_prod_types(country=country, target_year=target_year)
    valid_capacities: Dict[str, float] = {}
    for asset, capa in capacities.items():
        if available_eraa is not None and asset not in available_eraa:
            continue
        valid_capacities[asset] = capa

    active_assets = [asset for asset, capa in valid_capacities.items() if capa > 0]
    if ProdTypeNames.failure not in active_assets:
        active_assets.append(ProdTypeNames.failure)

    # Only include capacities for active assets in capacities_tb_overwritten (except failure)
    overwritten_caps = {
        asset: capa for asset, capa in valid_capacities.items()
        if asset in active_assets and asset != ProdTypeNames.failure
    }

    data_to_write = {
        "team": country,
        "selected_prod_types": {country: active_assets},
        "capacities_tb_overwritten": {country: overwritten_caps}
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_to_write, f, indent=2, ensure_ascii=False)

    return json_path


def sync_country_json_files(
    active_countries: List[str],
    country_capacities: Dict[str, Dict[str, float]],
    countries_dir: str = "input/long_term_uc/countries",
    target_year: Optional[int] = None,
) -> None:
    """
    Synchronize the input/long_term_uc/countries directory:
    - Writes/updates JSON files for active_countries.
    - Removes any obsolete JSON files for countries NOT in active_countries
      to prevent PyPSA from aborting due to mismatch between selected_countries
      and the files found in input/long_term_uc/countries/.
    """
    os.makedirs(countries_dir, exist_ok=True)
    # 1. Write active country JSONs
    for c in active_countries:
        c_caps = country_capacities.get(c, {})
        write_country_json(country=c, capacities=c_caps, countries_dir=countries_dir, target_year=target_year)

    # 2. Clean up any country file in countries_dir that is not in active_countries
    for fname in os.listdir(countries_dir):
        if fname.endswith(".json"):
            c_name = fname[:-5]
            if c_name not in active_countries and c_name in AVAILABLE_COUNTRIES:
                fpath = os.path.join(countries_dir, fname)
                try:
                    os.remove(fpath)
                    logging.info(f"Cleaned up inactive country file {fpath}")
                except OSError as exc:
                    logging.warning(f"Could not remove {fpath}: {exc}")


DEFAULT_PERIOD_START: str = "1900/1/1"
DEFAULT_PERIOD_END: str = "1901/1/1"
TEST_PERIOD_START: str = "1900/1/1"
TEST_PERIOD_END: str = "1900/1/15"  # 14 days / 2 weeks for fast debugging


def resolve_simulation_period(
    is_test: bool = False,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve the simulation start and end dates based on flags and parameters.

    In test/debug mode (is_test=True):
      - Defaults to a 2-week period (1900/1/1 to 1900/1/15).
      - If a non-default period_start is provided, it is preserved.
      - If a non-default period_end is provided, it is preserved.
    In standard mode (is_test=False):
      - Defaults to the full year (1900/1/1 to 1901/1/1).
    """
    if is_test:
        start = period_start if (period_start and period_start != DEFAULT_PERIOD_START) else TEST_PERIOD_START
        end = period_end if (period_end and period_end != DEFAULT_PERIOD_END) else TEST_PERIOD_END
        return start, end

    return period_start or DEFAULT_PERIOD_START, period_end or DEFAULT_PERIOD_END


def update_europe_params(
    mode: str = "europe",
    team: Optional[str] = None,
    period_start: str = DEFAULT_PERIOD_START,
    period_end: str = DEFAULT_PERIOD_END,
    selected_countries: Optional[List[str]] = None,
    climatic_year: Optional[int] = None,
    interco_capas: Optional[Dict[str, float]] = None,
    europe_params_path: str = "input/long_term_uc/elec-europe_params_to-be-modif.json",
) -> None:
    """
    Enforce and set 'mode' to 'europe', configure the simulation period,
    update 'selected_countries' to strictly match active participating countries,
    and optionally configure 'selected_climatic_year' and 'interco_capas_tb_overwritten'.
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
        if selected_countries is not None:
            sorted_countries = sorted(selected_countries)
            if data.get("selected_countries") != sorted_countries:
                data["selected_countries"] = sorted_countries
                modified = True
        if climatic_year is not None and data.get("selected_climatic_year") != climatic_year:
            data["selected_climatic_year"] = int(climatic_year)
            modified = True

        if interco_capas is not None:
            combined_intercos = dict(DEFAULT_BASE_INTERCO_OVERWRITTEN)
            combined_intercos.update({k: int(round(v)) for k, v in interco_capas.items()})
            if data.get("interco_capas_tb_overwritten") != combined_intercos:
                data["interco_capas_tb_overwritten"] = combined_intercos
                modified = True

        if modified:
            with open(europe_params_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(
                f"Updated {europe_params_path}: mode='europe' team='{team or data.get('team')}' "
                f"period=[{period_start} - {period_end}] countries={data.get('selected_countries')} "
                f"climatic_year={data.get('selected_climatic_year')}"
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


def find_new_summary_file(before_files: set, output_base_dir: str = "output/long_term_uc") -> Optional[str]:
    """Find the summary file created after before_files was captured."""
    pattern = os.path.join(output_base_dir, "**", "uc-summary_*.json")
    current_files = set(glob.glob(pattern, recursive=True))
    new_files = current_files - before_files
    if new_files:
        return sorted(list(new_files), key=os.path.getmtime, reverse=True)[0]
    return find_latest_summary_file(output_base_dir=output_base_dir)


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


# Mapping from raw production types in ERAA generation-capa CSVs to model aggregate technologies
ERAA_MAPPING_CSV_TO_MODEL: Dict[str, str] = {
    "batteries": "batteries",
    "biofuel": "biofuel",
    "demand_side_response_capacity": "dsr",
    "dsr": "dsr",
    "gas": "gas",
    "hard_coal": "coal",
    "lignite": "coal",
    "coal": "coal",
    "hydro_pondage": "hydro_pondage",
    "hydro_pump_storage_closed_loop": "hydro_pump_storage_closed_loop",
    "hydro_pump_storage_open_loop": "hydro_pump_storage_open_loop",
    "hydro_reservoir": "hydro_reservoir",
    "hydro_run_of_river": "hydro_run_of_river",
    "nuclear": "nuclear",
    "oil": "oil",
    "others_non_renewable": "others_fatal",
    "others_renewable": "others_fatal",
    "others_fatal": "others_fatal",
    "solar_pv": "solar_pv",
    "solar_thermal": "solar_thermal",
    "wind_offshore": "wind_offshore",
    "wind_onshore": "wind_onshore",
}

FOSSIL_TECH_KEYS = {"gas", "coal", "lignite", "oil"}

# Compact display names for table rows to prevent column misalignment on long asset names
TECH_DISPLAY_NAMES: Dict[str, str] = {
    "hydro_pump_storage_closed_loop": "hydro_phs_closed",
    "hydro_pump_storage_open_loop": "hydro_phs_open",
    "demand_side_response_capacity": "dsr",
    "others_non_renewable": "others_fatal",
    "others_renewable": "others_fatal",
}


def get_tech_display_name(tech: str) -> str:
    """Return a clean, compact technology display name to preserve ASCII table alignment."""
    return TECH_DISPLAY_NAMES.get(tech, tech)


def load_eraa_country_capacities(
    year: int,
    data_dir: str = "data/ERAA_2023-2/generation_capas",
    countries: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load official ENTSO-E ERAA reference capacities (in MW) for a given target year (e.g. 2025 or 2033).
    Parses generation-capa_{year}_{country}.csv for all available countries.
    Aggregates raw production types to model standard production types.
    Returns:
        {country: {tech_name: capacity_mw}}
    """
    target_countries = countries if countries is not None else AVAILABLE_COUNTRIES
    country_capacities: Dict[str, Dict[str, float]] = {}

    for country in target_countries:
        filename = f"generation-capa_{year}_{country}.csv"
        filepath = os.path.join(data_dir, filename)
        capas: Dict[str, float] = {}

        if not os.path.isfile(filepath):
            country_capacities[country] = capas
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
                if not header_line:
                    country_capacities[country] = capas
                    continue
                headers = [h.strip() for h in header_line.split(";")]
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) < len(headers):
                        continue
                    row = dict(zip(headers, parts))
                    prod_type_raw = row.get("production_type", "").strip().lower()
                    prod_type_clean = prod_type_raw.replace(" - ", "_").replace(" ", "_").replace("-", "_")
                    agg_pt = ERAA_MAPPING_CSV_TO_MODEL.get(prod_type_clean, prod_type_clean)

                    # Extract power capacity: for storage/hydro/turbines, turbine or injection might be used
                    p_cap = float(row.get("power_capacity", 0.0) or 0.0)
                    p_turb = float(row.get("power_capacity_turbine", 0.0) or 0.0)
                    p_inj = float(row.get("power_capacity_injection", 0.0) or 0.0)
                    capa_val = max(p_cap, p_turb, p_inj)

                    capas[agg_pt] = capas.get(agg_pt, 0.0) + capa_val
        except Exception as exc:
            logging.warning(f"Error loading ERAA capacities from {filepath}: {exc}")

        country_capacities[country] = capas

    return country_capacities


def format_installed_capacities_table(
    country_capacities: Dict[str, Dict[str, float]],
    title: str = "INSTALLED CAPACITIES (GW)",
) -> str:
    """
    Format installed capacities across countries into an aligned ASCII table.
    Capacities are presented in GW (1 decimal).
    Includes summary columns: Total Capa (GW), Low-Carbon Share (%),
    Overnight CAPEX (B€), and Annualized CAPEX (M€/year).
    """
    ordered_techs = [
        ("nuclear", "Nuc"),
        ("solar_pv", "Solar"),
        ("wind_onshore", "WindOn"),
        ("wind_offshore", "WindOff"),
        ("hydro_reservoir", "HydroRes"),
        ("hydro_run_of_river", "HydroRoR"),
        ("hydro_pump_storage_closed_loop", "PHS_C"),
        ("hydro_pump_storage_open_loop", "PHS_O"),
        ("batteries", "Battery"),
        ("dsr", "DSR"),
        ("gas", "Gas"),
    ]

    countries = sorted(list(country_capacities.keys()))

    # Check if any other technologies have capacity > 0
    has_other = False
    for c in countries:
        c_caps = country_capacities[c]
        other_val = sum(v for k, v in c_caps.items() if k not in [t[0] for t in ordered_techs] and k != "failure")
        if other_val > 0.01:
            has_other = True
            break

    columns = list(ordered_techs)
    if has_other:
        columns.append(("other", "Other"))

    # Table headers
    hdr_country = f"{'Country':<18}"
    tech_hdrs = "".join(f"{lbl:>9}" for _, lbl in columns)
    summary_hdrs = f"{'Total':>9} {'LowCarb%':>9} {'Overnight':>11} {'Ann.CAPEX':>11}"
    full_hdr = f"{hdr_country}{tech_hdrs}  {summary_hdrs}"
    line_sep = "-" * len(full_hdr)
    title_bar = "=" * len(full_hdr)

    lines = [
        title_bar,
        f"   {title}",
        line_sep,
        full_hdr,
        line_sep,
    ]

    # Totals across all displayed countries
    totals_by_tech: Dict[str, float] = {t[0]: 0.0 for t in columns}
    tot_europe_mw = 0.0
    tot_europe_clean_mw = 0.0
    tot_europe_overnight_eur = 0.0
    tot_europe_annualized_eur = 0.0

    for c in countries:
        caps = country_capacities[c]
        row_str = f"{c:<18}"
        for tech_key, _ in columns:
            if tech_key == "other":
                val_mw = sum(v for k, v in caps.items() if k not in [t[0] for t in ordered_techs] and k != "failure")
            else:
                val_mw = caps.get(tech_key, 0.0)
            totals_by_tech[tech_key] += val_mw
            val_gw = val_mw / 1000.0
            row_str += f"{val_gw:>9.1f}"

        # Summary metrics
        total_mw = sum(v for k, v in caps.items() if k != "failure")
        clean_mw = sum(v for k, v in caps.items() if k not in FOSSIL_TECH_KEYS and k not in ("failure", "others_fatal"))
        low_carb_pct = (clean_mw / total_mw * 100.0) if total_mw > 0 else 0.0
        overnight_eur, annualized_eur = compute_country_annualized_capex(caps)

        tot_europe_mw += total_mw
        tot_europe_clean_mw += clean_mw
        tot_europe_overnight_eur += overnight_eur
        tot_europe_annualized_eur += annualized_eur

        total_gw = total_mw / 1000.0
        overnight_b = overnight_eur / 1e9
        annualized_m = annualized_eur / 1e6

        row_str += f"  {total_gw:>9.1f} {low_carb_pct:>8.1f}% {overnight_b:>9.1f} B€ {annualized_m:>9.0f} M€"
        lines.append(row_str)

    lines.append(line_sep)

    # European total row
    tot_str = f"{'TOTAL EUROPE':<18}"
    for tech_key, _ in columns:
        tot_gw = totals_by_tech[tech_key] / 1000.0
        tot_str += f"{tot_gw:>9.1f}"

    tot_europe_gw = tot_europe_mw / 1000.0
    tot_low_carb_pct = (tot_europe_clean_mw / tot_europe_mw * 100.0) if tot_europe_mw > 0 else 0.0
    tot_overnight_b = tot_europe_overnight_eur / 1e9
    tot_annualized_m = tot_europe_annualized_eur / 1e6

    tot_str += f"  {tot_europe_gw:>9.1f} {tot_low_carb_pct:>8.1f}% {tot_overnight_b:>9.1f} B€ {tot_annualized_m:>9.0f} M€"
    lines.append(tot_str)
    lines.append(title_bar)

    return "\n".join(lines)


def display_installed_capacities(table_str: str) -> None:
    """Print the formatted installed capacities table to console."""
    print(table_str)


def compute_capacity_comparison_with_eraa(
    current_capacities: Dict[str, Dict[str, float]],
    eraa_capacities: Dict[str, Dict[str, float]],
) -> dict:
    """
    Compute deltas between current capacities and ERAA capacities (in MW, GW, and %).
    Returns structured comparison per country and for total Europe.
    """
    comparison: dict = {"per_country": {}, "total_europe": {}}
    countries = sorted(list(current_capacities.keys()))

    europe_cur_by_tech: Dict[str, float] = {}
    europe_eraa_by_tech: Dict[str, float] = {}

    for country in countries:
        cur_caps = current_capacities.get(country, {})
        eraa_caps = eraa_capacities.get(country, {})
        all_techs = sorted(list(set(list(cur_caps.keys()) + list(eraa_caps.keys()))))
        country_comp: dict = {}

        cur_tot = 0.0
        eraa_tot = 0.0

        for tech in all_techs:
            if tech == "failure":
                continue
            c_mw = float(cur_caps.get(tech, 0.0))
            e_mw = float(eraa_caps.get(tech, 0.0))
            diff_mw = c_mw - e_mw
            diff_pct = (diff_mw / e_mw * 100.0) if e_mw > 0 else (100.0 if c_mw > 0 else 0.0)

            cur_tot += c_mw
            eraa_tot += e_mw
            europe_cur_by_tech[tech] = europe_cur_by_tech.get(tech, 0.0) + c_mw
            europe_eraa_by_tech[tech] = europe_eraa_by_tech.get(tech, 0.0) + e_mw

            country_comp[tech] = {
                "current_mw": c_mw,
                "eraa_mw": e_mw,
                "diff_mw": diff_mw,
                "diff_gw": diff_mw / 1000.0,
                "diff_pct": diff_pct,
            }

        tot_diff_mw = cur_tot - eraa_tot
        tot_diff_pct = (tot_diff_mw / eraa_tot * 100.0) if eraa_tot > 0 else (100.0 if cur_tot > 0 else 0.0)
        country_comp["total"] = {
            "current_mw": cur_tot,
            "eraa_mw": eraa_tot,
            "diff_mw": tot_diff_mw,
            "diff_gw": tot_diff_mw / 1000.0,
            "diff_pct": tot_diff_pct,
        }
        comparison["per_country"][country] = country_comp

    # Total Europe across displayed countries
    all_eur_techs = sorted(list(set(list(europe_cur_by_tech.keys()) + list(europe_eraa_by_tech.keys()))))
    eur_cur_tot = sum(europe_cur_by_tech.values())
    eur_eraa_tot = sum(europe_eraa_by_tech.values())
    tot_eur_comp: dict = {}

    for tech in all_eur_techs:
        c_mw = europe_cur_by_tech.get(tech, 0.0)
        e_mw = europe_eraa_by_tech.get(tech, 0.0)
        diff_mw = c_mw - e_mw
        diff_pct = (diff_mw / e_mw * 100.0) if e_mw > 0 else (100.0 if c_mw > 0 else 0.0)
        tot_eur_comp[tech] = {
            "current_mw": c_mw,
            "eraa_mw": e_mw,
            "diff_mw": diff_mw,
            "diff_gw": diff_mw / 1000.0,
            "diff_pct": diff_pct,
        }

    eur_tot_diff_mw = eur_cur_tot - eur_eraa_tot
    eur_tot_diff_pct = (eur_tot_diff_mw / eur_eraa_tot * 100.0) if eur_eraa_tot > 0 else (100.0 if eur_cur_tot > 0 else 0.0)
    tot_eur_comp["total"] = {
        "current_mw": eur_cur_tot,
        "eraa_mw": eur_eraa_tot,
        "diff_mw": eur_tot_diff_mw,
        "diff_gw": eur_tot_diff_mw / 1000.0,
        "diff_pct": eur_tot_diff_pct,
    }
    comparison["total_europe"] = tot_eur_comp

    return comparison


def format_capacity_comparison_table(
    current_capacities: Dict[str, Dict[str, float]],
    eraa_2025_capas: Dict[str, Dict[str, float]],
    eraa_2033_capas: Dict[str, Dict[str, float]],
    europe_name: str = "Europe 1",
) -> str:
    """
    Format a comprehensive comparison table showing current capacities vs ERAA 2025 (starting mix)
    and ERAA 2033 (official roadmap). Displays both absolute deltas (GW) and relative changes (%).
    """
    comp_2025 = compute_capacity_comparison_with_eraa(current_capacities, eraa_2025_capas)
    comp_2033 = compute_capacity_comparison_with_eraa(current_capacities, eraa_2033_capas)

    countries = sorted(list(current_capacities.keys()))

    line_w = 114
    title_bar = "=" * line_w
    sep_bar = "-" * line_w

    lines = [
        title_bar,
        f"   CAPACITY COMPARISON vs ERAA REFERENCES - {europe_name.upper()} (GW)",
        sep_bar,
        f"{'Country / Technology':<30} {'Current':>9} {'ERAA 2025':>11} {'Δ vs 2025 (GW / %)':>24} {'ERAA 2033':>11} {'Δ vs 2033 (GW / %)':>24}",
        sep_bar,
    ]

    def _fmt_diff(diff_gw: float, diff_pct: float) -> str:
        sign = "+" if diff_gw >= 0 else ""
        return f"{sign}{diff_gw:.1f} ({sign}{diff_pct:.1f}%)"

    for c in countries:
        c_25 = comp_2025["per_country"].get(c, {})
        c_33 = comp_2033["per_country"].get(c, {})
        all_techs = sorted([k for k in c_25.keys() if k != "total"])

        lines.append(f"[{c.upper()}]")
        for tech in all_techs:
            tech_disp = get_tech_display_name(tech)
            cur_gw = c_25[tech]["current_mw"] / 1000.0
            e25_gw = c_25[tech]["eraa_mw"] / 1000.0
            d25_str = _fmt_diff(c_25[tech]["diff_gw"], c_25[tech]["diff_pct"])

            e33_gw = c_33.get(tech, {}).get("eraa_mw", 0.0) / 1000.0
            d33_gw = c_33.get(tech, {}).get("diff_gw", 0.0)
            d33_pct = c_33.get(tech, {}).get("diff_pct", 0.0)
            d33_str = _fmt_diff(d33_gw, d33_pct)

            lines.append(f"  {tech_disp[:28]:<28} {cur_gw:>9.1f} {e25_gw:>11.1f} {d25_str:>24} {e33_gw:>11.1f} {d33_str:>24}")

        # Total for country
        tot_cur_gw = c_25.get("total", {}).get("current_mw", 0.0) / 1000.0
        tot_e25_gw = c_25.get("total", {}).get("eraa_mw", 0.0) / 1000.0
        tot_d25_str = _fmt_diff(c_25.get("total", {}).get("diff_gw", 0.0), c_25.get("total", {}).get("diff_pct", 0.0))

        tot_e33_gw = c_33.get("total", {}).get("eraa_mw", 0.0) / 1000.0
        tot_d33_str = _fmt_diff(c_33.get("total", {}).get("diff_gw", 0.0), c_33.get("total", {}).get("diff_pct", 0.0))

        lines.append(f"  {('Total ' + c)[:28]:<28} {tot_cur_gw:>9.1f} {tot_e25_gw:>11.1f} {tot_d25_str:>24} {tot_e33_gw:>11.1f} {tot_d33_str:>24}")
        lines.append(sep_bar)

    # European Total
    tot_eur_25 = comp_2025.get("total_europe", {})
    tot_eur_33 = comp_2033.get("total_europe", {})
    eur_cur_gw = tot_eur_25.get("total", {}).get("current_mw", 0.0) / 1000.0
    eur_e25_gw = tot_eur_25.get("total", {}).get("eraa_mw", 0.0) / 1000.0
    eur_d25_str = _fmt_diff(tot_eur_25.get("total", {}).get("diff_gw", 0.0), tot_eur_25.get("total", {}).get("diff_pct", 0.0))

    eur_e33_gw = tot_eur_33.get("total", {}).get("eraa_mw", 0.0) / 1000.0
    eur_d33_str = _fmt_diff(tot_eur_33.get("total", {}).get("diff_gw", 0.0), tot_eur_33.get("total", {}).get("diff_pct", 0.0))

    lines.append(f"{'TOTAL EUROPE':<30} {eur_cur_gw:>9.1f} {eur_e25_gw:>11.1f} {eur_d25_str:>24} {eur_e33_gw:>11.1f} {eur_d33_str:>24}")
    lines.append(title_bar)

    return "\n".join(lines)


def load_all_simulation_runs(output_dir: str = "output/runner") -> List[dict]:
    """
    Load all historical simulation runs from output_dir sorted chronologically.
    Extracts both Europe 1 and Europe 2 capacities and metrics for each run.
    """
    if not os.path.isdir(output_dir):
        return []

    pattern = os.path.join(output_dir, "*.json")
    all_files = glob.glob(pattern)

    sim_files = [
        f for f in all_files
        if not os.path.basename(f).startswith("history_")
        and not os.path.basename(f).startswith("eraa_baseline_")
        and not os.path.basename(f).endswith("_latest.json")
    ]

    history = []
    for fpath in sim_files:
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            dt_str = data.get("timestamp_str")
            if not dt_str:
                mtime = os.path.getmtime(fpath)
                dt_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            # Extract Europe 1 capacities
            e1_caps = (
                data.get("europe1_capacities")
                or data.get("europe1_metrics", {}).get("per_country_capacities")
                or {}
            )
            if not e1_caps and "per_country" in data:
                e1_caps = {
                    c: cdata.get("capacities", {})
                    for c, cdata in data["per_country"].items()
                    if isinstance(cdata, dict) and cdata.get("capacities")
                }

            # Extract Europe 2 capacities
            e2_caps = (
                data.get("europe2_capacities")
                or data.get("europe2_metrics", {}).get("per_country_capacities")
                or {}
            )

            if e1_caps or e2_caps:
                history.append({
                    "filepath": fpath,
                    "filename": os.path.basename(fpath),
                    "timestamp": dt_str,
                    "europe1_capacities": e1_caps,
                    "europe2_capacities": e2_caps,
                    "europe1_metrics": data.get("europe1_metrics"),
                    "europe2_metrics": data.get("europe2_metrics"),
                })
        except Exception as exc:
            logging.warning(f"Could not load simulation history file {fpath}: {exc}")

    history.sort(key=lambda item: item["timestamp"])
    for idx, item in enumerate(history, start=1):
        item["run_index"] = idx

    return history


def format_incremental_capacities_trajectory_table(
    eraa_2025_capas: Dict[str, Dict[str, float]],
    eraa_2033_capas: Dict[str, Dict[str, float]],
    history_runs: Optional[List[dict]] = None,
    current_europe1_capas: Optional[Dict[str, Dict[str, float]]] = None,
    current_europe2_capas: Optional[Dict[str, Dict[str, float]]] = None,
    displayed_countries: Optional[List[str]] = None,
) -> str:
    """
    Format a unified multi-column table displaying the incremental trajectory of installed capacities,
    grouping successive runs of Europe 1 together, then successive runs of Europe 2 together:
    ERAA 2025 | ERAA 2033 | [--- EUROPE 1 RUNS ---] Δ Net (E1) | [--- EUROPE 2 RUNS ---] Δ Net (E2).
    """
    runs_e1: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    runs_e2: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    past_runs = history_runs or []

    for h in past_runs:
        idx = h["run_index"]
        if h.get("europe1_capacities"):
            runs_e1.append((f"R{idx} (E1)", h["europe1_capacities"]))
        if h.get("europe2_capacities"):
            runs_e2.append((f"R{idx} (E2)", h["europe2_capacities"]))

    last_e1 = runs_e1[-1][1] if runs_e1 else None
    last_e2 = runs_e2[-1][1] if runs_e2 else None

    curr_idx = len(past_runs) + 1 if past_runs else 1
    if current_europe1_capas and current_europe1_capas != last_e1:
        lbl = f"R{curr_idx} (E1)*" if past_runs else "R1 (E1)"
        runs_e1.append((lbl, current_europe1_capas))
    if current_europe2_capas and current_europe2_capas != last_e2:
        lbl = f"R{curr_idx} (E2)*" if past_runs else "R1 (E2)"
        runs_e2.append((lbl, current_europe2_capas))

    if not runs_e1 and not runs_e2:
        return "[i] No capacity runs available to display."

    sel_e1 = runs_e1 if len(runs_e1) <= 4 else (runs_e1[:2] + runs_e1[-2:])
    sel_e2 = runs_e2 if len(runs_e2) <= 4 else (runs_e2[:2] + runs_e2[-2:])

    if displayed_countries is not None:
        c_list = sorted(displayed_countries)
    else:
        found_countries = set()
        for _, c_caps in runs_e1 + runs_e2:
            found_countries.update(c_caps.keys())
        if not found_countries:
            found_countries.update(AVAILABLE_COUNTRIES)
        c_list = sorted(list(found_countries))

    ordered_techs = [
        "nuclear",
        "solar_pv",
        "wind_onshore",
        "wind_offshore",
        "hydro_reservoir",
        "hydro_run_of_river",
        "hydro_pump_storage_closed_loop",
        "hydro_pump_storage_open_loop",
        "batteries",
        "dsr",
        "gas",
    ]

    w_tech = 26
    w_col = 11
    w_delta = 12

    e1_col_headers = "".join(f"{lbl:>{w_col}}" for lbl, _ in sel_e1)
    if sel_e1:
        e1_col_headers += f"{'Δ Net (E1)':>{w_delta}}"

    e2_col_headers = "".join(f"{lbl:>{w_col}}" for lbl, _ in sel_e2)
    if sel_e2:
        e2_col_headers += f"{'Δ Net (E2)':>{w_delta}}"

    header_parts = [f"{'Country / Technology':<{w_tech}}", f"{'ERAA 2025':>{w_col}}", f"{'ERAA 2033':>{w_col}}"]
    if sel_e1:
        header_parts.append(f" |{e1_col_headers}")
    if sel_e2:
        header_parts.append(f" |{e2_col_headers}")

    full_header = " ".join(header_parts)
    line_w = max(len(full_header), 110)
    title_bar = "=" * line_w
    sep_bar = "-" * line_w

    lines = [
        title_bar,
        f"{'INCREMENTAL CAPACITY TRAJECTORY: ERAA REFERENCES vs HISTORICAL RUNS (GW)':^{line_w}}",
        title_bar,
    ]

    def _fmt_val(mw: float) -> str:
        return f"{mw / 1000.0:>{w_col}.1f}"

    def _fmt_diff(diff_mw: float) -> str:
        diff_gw = diff_mw / 1000.0
        sign = "+" if diff_gw > 0 else ""
        return f"{sign}{diff_gw:>{w_delta - 1}.1f} "

    for country in c_list:
        lines.append(f"[{country.upper()}]")
        lines.append(full_header)
        lines.append(sep_bar)
        e25_c = eraa_2025_capas.get(country, {})
        e33_c = eraa_2033_capas.get(country, {})

        c_tot_25 = sum(v for k, v in e25_c.items() if k != "failure")
        c_tot_33 = sum(v for k, v in e33_c.items() if k != "failure")

        for tech in ordered_techs:
            v25 = e25_c.get(tech, 0.0)
            v33 = e33_c.get(tech, 0.0)
            v_e1 = [caps.get(country, {}).get(tech, 0.0) for _, caps in sel_e1]
            v_e2 = [caps.get(country, {}).get(tech, 0.0) for _, caps in sel_e2]

            if v25 == 0 and v33 == 0 and all(v == 0 for v in v_e1 + v_e2):
                continue

            tech_disp = get_tech_display_name(tech)
            row_parts = [f"  {tech_disp[:w_tech - 2]:<{w_tech - 2}}", _fmt_val(v25), _fmt_val(v33)]
            if sel_e1:
                e1_str = " |" + "".join(_fmt_val(vr) for vr in v_e1)
                e1_first = runs_e1[0][1].get(country, {}).get(tech, 0.0)
                e1_last = runs_e1[-1][1].get(country, {}).get(tech, 0.0)
                e1_str += _fmt_diff(e1_last - e1_first)
                row_parts.append(e1_str)
            if sel_e2:
                e2_str = " |" + "".join(_fmt_val(vr) for vr in v_e2)
                e2_first = runs_e2[0][1].get(country, {}).get(tech, 0.0)
                e2_last = runs_e2[-1][1].get(country, {}).get(tech, 0.0)
                e2_str += _fmt_diff(e2_last - e2_first)
                row_parts.append(e2_str)

            lines.append(" ".join(row_parts))

        # Total for Country
        tot_label = f"Total {country}"
        tot_parts = [f"  {tot_label[:w_tech - 2]:<{w_tech - 2}}", _fmt_val(c_tot_25), _fmt_val(c_tot_33)]
        if sel_e1:
            tot_e1_vals = [sum(v for k, v in caps.get(country, {}).items() if k != "failure") for _, caps in sel_e1]
            e1_str = " |" + "".join(_fmt_val(vr) for vr in tot_e1_vals)
            e1_first_tot = sum(v for k, v in runs_e1[0][1].get(country, {}).items() if k != "failure")
            e1_last_tot = sum(v for k, v in runs_e1[-1][1].get(country, {}).items() if k != "failure")
            e1_str += _fmt_diff(e1_last_tot - e1_first_tot)
            tot_parts.append(e1_str)
        if sel_e2:
            tot_e2_vals = [sum(v for k, v in caps.get(country, {}).items() if k != "failure") for _, caps in sel_e2]
            e2_str = " |" + "".join(_fmt_val(vr) for vr in tot_e2_vals)
            e2_first_tot = sum(v for k, v in runs_e2[0][1].get(country, {}).items() if k != "failure")
            e2_last_tot = sum(v for k, v in runs_e2[-1][1].get(country, {}).items() if k != "failure")
            e2_str += _fmt_diff(e2_last_tot - e2_first_tot)
            tot_parts.append(e2_str)

        lines.append(" ".join(tot_parts))
        lines.append(sep_bar)

    # European Total
    lines.append("[TOTAL EUROPE]")
    lines.append(full_header)
    lines.append(sep_bar)
    eur_tot_25 = sum(sum(v for k, v in eraa_2025_capas.get(c, {}).items() if k != "failure") for c in c_list)
    eur_tot_33 = sum(sum(v for k, v in eraa_2033_capas.get(c, {}).items() if k != "failure") for c in c_list)

    for tech in ordered_techs:
        v25 = sum(eraa_2025_capas.get(c, {}).get(tech, 0.0) for c in c_list)
        v33 = sum(eraa_2033_capas.get(c, {}).get(tech, 0.0) for c in c_list)
        v_e1 = [sum(caps.get(c, {}).get(tech, 0.0) for c in c_list) for _, caps in sel_e1]
        v_e2 = [sum(caps.get(c, {}).get(tech, 0.0) for c in c_list) for _, caps in sel_e2]

        if v25 == 0 and v33 == 0 and all(v == 0 for v in v_e1 + v_e2):
            continue

        tech_disp = get_tech_display_name(tech)
        row_parts = [f"  {tech_disp[:w_tech - 2]:<{w_tech - 2}}", _fmt_val(v25), _fmt_val(v33)]
        if sel_e1:
            e1_str = " |" + "".join(_fmt_val(vr) for vr in v_e1)
            e1_first_eur = sum(runs_e1[0][1].get(c, {}).get(tech, 0.0) for c in c_list)
            e1_last_eur = sum(runs_e1[-1][1].get(c, {}).get(tech, 0.0) for c in c_list)
            e1_str += _fmt_diff(e1_last_eur - e1_first_eur)
            row_parts.append(e1_str)
        if sel_e2:
            e2_str = " |" + "".join(_fmt_val(vr) for vr in v_e2)
            e2_first_eur = sum(runs_e2[0][1].get(c, {}).get(tech, 0.0) for c in c_list)
            e2_last_eur = sum(runs_e2[-1][1].get(c, {}).get(tech, 0.0) for c in c_list)
            e2_str += _fmt_diff(e2_last_eur - e2_first_eur)
            row_parts.append(e2_str)

        lines.append(" ".join(row_parts))

    tot_eur_parts = [f"  {'TOTAL EUROPE':<{w_tech - 2}}", _fmt_val(eur_tot_25), _fmt_val(eur_tot_33)]
    if sel_e1:
        tot_e1_eur_vals = [sum(sum(v for k, v in caps.get(c, {}).items() if k != "failure") for c in c_list) for _, caps in sel_e1]
        e1_str = " |" + "".join(_fmt_val(vr) for vr in tot_e1_eur_vals)
        e1_first_eur_tot = sum(sum(v for k, v in runs_e1[0][1].get(c, {}).items() if k != "failure") for c in c_list)
        e1_last_eur_tot = sum(sum(v for k, v in runs_e1[-1][1].get(c, {}).items() if k != "failure") for c in c_list)
        e1_str += _fmt_diff(e1_last_eur_tot - e1_first_eur_tot)
        tot_eur_parts.append(e1_str)
    if sel_e2:
        tot_e2_eur_vals = [sum(sum(v for k, v in caps.get(c, {}).items() if k != "failure") for c in c_list) for _, caps in sel_e2]
        e2_str = " |" + "".join(_fmt_val(vr) for vr in tot_e2_eur_vals)
        e2_first_eur_tot = sum(sum(v for k, v in runs_e2[0][1].get(c, {}).items() if k != "failure") for c in c_list)
        e2_last_eur_tot = sum(sum(v for k, v in runs_e2[-1][1].get(c, {}).items() if k != "failure") for c in c_list)
        e2_str += _fmt_diff(e2_last_eur_tot - e2_first_eur_tot)
        tot_eur_parts.append(e2_str)

    lines.append(" ".join(tot_eur_parts))
    lines.append(title_bar)
    lines.append("(*) Runs marked with * represent current in-memory configurations awaiting solver execution.\n")

    return "\n".join(lines)


def format_incremental_capacities_summary_table(
    eraa_2025_capas: Dict[str, Dict[str, float]],
    eraa_2033_capas: Dict[str, Dict[str, float]],
    history_runs: Optional[List[dict]] = None,
    current_europe1_capas: Optional[Dict[str, Dict[str, float]]] = None,
    current_europe2_capas: Optional[Dict[str, Dict[str, float]]] = None,
    displayed_countries: Optional[List[str]] = None,
) -> str:
    """
    Format a concise summary table showing Total Capacity (GW) and Low-Carbon Share (%)
    for each country and Total Europe across ERAA 2025, ERAA 2033, and successive simulation runs,
    grouping Europe 1 runs together, then Europe 2 runs together.
    """
    runs_e1 = []
    runs_e2 = []
    past_runs = history_runs or []

    for h in past_runs:
        idx = h["run_index"]
        if h.get("europe1_capacities"):
            runs_e1.append((f"R{idx} (E1)", h["europe1_capacities"]))
        if h.get("europe2_capacities"):
            runs_e2.append((f"R{idx} (E2)", h["europe2_capacities"]))

    curr_idx = len(past_runs) + 1 if past_runs else 1
    if current_europe1_capas and (not runs_e1 or current_europe1_capas != runs_e1[-1][1]):
        runs_e1.append((f"R{curr_idx} (E1)*" if past_runs else "R1 (E1)", current_europe1_capas))
    if current_europe2_capas and (not runs_e2 or current_europe2_capas != runs_e2[-1][1]):
        runs_e2.append((f"R{curr_idx} (E2)*" if past_runs else "R1 (E2)", current_europe2_capas))

    if not runs_e1 and not runs_e2:
        return ""

    sel_e1 = runs_e1 if len(runs_e1) <= 4 else (runs_e1[:2] + runs_e1[-2:])
    sel_e2 = runs_e2 if len(runs_e2) <= 4 else (runs_e2[:2] + runs_e2[-2:])

    if displayed_countries is not None:
        c_list = sorted(displayed_countries)
    else:
        found_countries = set()
        for _, c_caps in runs_e1 + runs_e2:
            found_countries.update(c_caps.keys())
        c_list = sorted(list(found_countries)) if found_countries else AVAILABLE_COUNTRIES

    w_c = 18
    w_col = 16

    e1_hdrs = "".join(f"{lbl:>{w_col}}" for lbl, _ in sel_e1)
    e2_hdrs = "".join(f"{lbl:>{w_col}}" for lbl, _ in sel_e2)

    header_parts = [f"{'Country':<{w_c}}", f"{'ERAA 2025':>{w_col}}", f"{'ERAA 2033':>{w_col}}"]
    if sel_e1:
        header_parts.append(f" |{e1_hdrs}")
    if sel_e2:
        header_parts.append(f" |{e2_hdrs}")

    full_header = " ".join(header_parts)
    line_w = max(len(full_header), 90)
    title_bar = "=" * line_w
    sep_bar = "-" * line_w

    lines = [
        title_bar,
        f"{'TOTAL CAPACITY & LOW-CARBON SHARE TRAJECTORY (GW / %)':^{line_w}}",
        title_bar,
        full_header,
        sep_bar,
    ]

    def _fmt_cap_share(caps: Dict[str, float]) -> str:
        tot_mw = sum(v for k, v in caps.items() if k != "failure")
        clean_mw = sum(v for k, v in caps.items() if k not in FOSSIL_TECH_KEYS and k not in ("failure", "others_fatal"))
        tot_gw = tot_mw / 1000.0
        pct = (clean_mw / tot_mw * 100.0) if tot_mw > 0 else 0.0
        return f"{tot_gw:>6.1f} ({pct:>4.1f}%)"

    for c in c_list:
        v25 = _fmt_cap_share(eraa_2025_capas.get(c, {}))
        v33 = _fmt_cap_share(eraa_2033_capas.get(c, {}))
        row_parts = [f"{c:<{w_c}}", f"{v25:>{w_col}}", f"{v33:>{w_col}}"]
        if sel_e1:
            row_parts.append(" |" + "".join(f"{_fmt_cap_share(caps.get(c, {})):>{w_col}}" for _, caps in sel_e1))
        if sel_e2:
            row_parts.append(" |" + "".join(f"{_fmt_cap_share(caps.get(c, {})):>{w_col}}" for _, caps in sel_e2))
        lines.append(" ".join(row_parts))

    lines.append(sep_bar)

    def _fmt_eur_cap_share(all_caps: Dict[str, Dict[str, float]]) -> str:
        tot_mw = sum(sum(v for k, v in all_caps.get(c, {}).items() if k != "failure") for c in c_list)
        clean_mw = sum(sum(v for k, v in all_caps.get(c, {}).items() if k not in FOSSIL_TECH_KEYS and k not in ("failure", "others_fatal")) for c in c_list)
        tot_gw = tot_mw / 1000.0
        pct = (clean_mw / tot_mw * 100.0) if tot_mw > 0 else 0.0
        return f"{tot_gw:>6.1f} ({pct:>4.1f}%)"

    eur_25 = _fmt_eur_cap_share(eraa_2025_capas)
    eur_33 = _fmt_eur_cap_share(eraa_2033_capas)
    eur_parts = [f"{'TOTAL EUROPE':<{w_c}}", f"{eur_25:>{w_col}}", f"{eur_33:>{w_col}}"]
    if sel_e1:
        eur_parts.append(" |" + "".join(f"{_fmt_eur_cap_share(caps):>{w_col}}" for _, caps in sel_e1))
    if sel_e2:
        eur_parts.append(" |" + "".join(f"{_fmt_eur_cap_share(caps):>{w_col}}" for _, caps in sel_e2))
    lines.append(" ".join(eur_parts))
    lines.append(title_bar)

    return "\n".join(lines)


def save_incremental_capacities_trajectory(
    report_text: str,
    output_dir: str = "output/runner",
) -> str:
    """Save the incremental capacity trajectory table to text file."""
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "history_capacities_trajectory.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return txt_path


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


def extract_simulation_metrics(
    summary_data: dict,
    per_country_capacities: Optional[Dict[str, Dict[str, float]]] = None,
    summary_file_path: Optional[str] = None,
    countries: Optional[List[str]] = None,
) -> dict:
    """
    Extract standardized European and per-country metrics from summary_data.
    """
    display_countries = countries if countries is not None else AVAILABLE_COUNTRIES

    total_op_cost = float(summary_data.get("total_operational_cost", 0.0))
    total_co2 = float(summary_data.get("total_co2_emissions", 0.0))

    per_country_ens = summary_data.get("per_country_ens", {}) or {}
    per_country_fail_h = summary_data.get("per_country_n_failure_hours", {}) or {}
    per_country_cost = summary_data.get("per_country_total_cost", {}) or {}
    per_country_op_cost = summary_data.get("per_country_total_operational_cost", {}) or {}
    per_country_co2 = summary_data.get("per_country_co2_emissions", {}) or {}

    per_country_prod = summary_data.get("per_country_production", {}) or {}
    if not per_country_prod and summary_file_path:
        per_country_prod = get_production_from_opt_power_csv(summary_file_path=summary_file_path)

    total_ens = sum(float(v) for v in per_country_ens.values()) if per_country_ens else 0.0
    max_fail_h = max(float(v) for v in per_country_fail_h.values()) if per_country_fail_h else 0.0

    total_europe_capex_m_yr = 0.0
    total_europe_overnight_b = 0.0
    total_europe_opex = 0.0
    total_europe_prod_gwh = 0.0

    per_country_metrics: Dict[str, dict] = {}

    for country in sorted(display_countries):
        tri = set_country_trigram(country)
        ens = float(per_country_ens.get(tri, per_country_ens.get(country, 0.0)))
        fail_h = float(per_country_fail_h.get(tri, per_country_fail_h.get(country, 0.0)))

        opex = float(per_country_op_cost.get(tri, per_country_op_cost.get(country, per_country_cost.get(tri, per_country_cost.get(country, 0.0)))))
        total_europe_opex += opex

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
        raw_co2 = float(per_country_co2.get(tri, per_country_co2.get(country, 0.0)))
        # CO2 emissions in UCSummaryMetrics are in ktCO2eq. Convert to Mt if in kt (> 500)
        co2 = (raw_co2 / 1000.0) if raw_co2 > 500.0 else raw_co2

        prod_gwh = float(per_country_prod.get(tri, per_country_prod.get(country, 0.0)))
        total_europe_prod_gwh += prod_gwh
        cost_per_mwh = (total_m_yr * 1000.0) / prod_gwh if prod_gwh > 0 else 0.0

        per_country_metrics[country] = {
            "ens_mwh": ens,
            "failure_hours": fail_h,
            "opex_m_yr": opex,
            "capex_m_yr": capex_m_yr,
            "total_cost_m_yr": total_m_yr,
            "cost_per_mwh": cost_per_mwh,
            "co2_mt": co2,
            "production_gwh": prod_gwh,
            "capacities": country_capas,
        }

    display_europe_opex = total_op_cost if total_op_cost > 0 else total_europe_opex
    total_europe_cost = display_europe_opex + total_europe_capex_m_yr
    avg_cost_mwh = (total_europe_cost * 1000.0) / total_europe_prod_gwh if total_europe_prod_gwh > 0 else 0.0

    total_co2_raw = float(summary_data.get("total_co2_emissions", 0.0))
    total_co2_mt = (total_co2_raw / 1000.0) if total_co2_raw > 500.0 else total_co2_raw

    return {
        "summary_file": summary_file_path,
        "total_cost_m_yr": total_europe_cost,
        "total_operational_cost_m_yr": display_europe_opex,
        "total_annualized_capex_m_yr": total_europe_capex_m_yr,
        "total_overnight_capex_b": total_europe_overnight_b,
        "avg_cost_per_mwh": avg_cost_mwh,
        "total_co2_emissions_mt": total_co2_mt,
        "total_production_gwh": total_europe_prod_gwh,
        "total_production_twh": total_europe_prod_gwh / 1000.0,
        "total_ens_mwh": total_ens,
        "max_failure_hours": max_fail_h,
        "per_country": per_country_metrics,
        "per_country_capacities": {c: per_country_metrics[c]["capacities"] for c in sorted(display_countries)},
    }


def compute_europe_comparison(metrics_eur1: dict, metrics_eur2: dict) -> dict:
    """
    Compute delta (Eur2 - Eur1) and relative change (%) between Europe 1 and Europe 2.
    """
    def _diff(val1: float, val2: float) -> dict:
        delta = val2 - val1
        pct = ((val2 - val1) / val1 * 100.0) if abs(val1) > 1e-6 else (0.0 if abs(val2) < 1e-6 else None)
        return {"eur1": val1, "eur2": val2, "delta": delta, "pct_change": pct}

    global_keys = [
        "total_cost_m_yr",
        "total_operational_cost_m_yr",
        "total_annualized_capex_m_yr",
        "total_overnight_capex_b",
        "avg_cost_per_mwh",
        "total_co2_emissions_mt",
        "total_production_twh",
        "total_ens_mwh",
        "max_failure_hours",
    ]

    global_diffs = {}
    for k in global_keys:
        v1 = float(metrics_eur1.get(k, 0.0))
        v2 = float(metrics_eur2.get(k, 0.0))
        global_diffs[k] = _diff(v1, v2)

    countries = sorted(list(set(metrics_eur1.get("per_country", {}).keys()) | set(metrics_eur2.get("per_country", {}).keys())))
    country_diffs = {}
    for c in countries:
        c1 = metrics_eur1.get("per_country", {}).get(c, {})
        c2 = metrics_eur2.get("per_country", {}).get(c, {})
        country_diffs[c] = {
            "total_cost_m_yr": _diff(c1.get("total_cost_m_yr", 0.0), c2.get("total_cost_m_yr", 0.0)),
            "opex_m_yr": _diff(c1.get("opex_m_yr", 0.0), c2.get("opex_m_yr", 0.0)),
            "capex_m_yr": _diff(c1.get("capex_m_yr", 0.0), c2.get("capex_m_yr", 0.0)),
            "co2_mt": _diff(c1.get("co2_mt", 0.0), c2.get("co2_mt", 0.0)),
            "ens_mwh": _diff(c1.get("ens_mwh", 0.0), c2.get("ens_mwh", 0.0)),
            "failure_hours": _diff(c1.get("failure_hours", 0.0), c2.get("failure_hours", 0.0)),
            "production_gwh": _diff(c1.get("production_gwh", 0.0), c2.get("production_gwh", 0.0)),
            "cost_per_mwh": _diff(c1.get("cost_per_mwh", 0.0), c2.get("cost_per_mwh", 0.0)),
        }

    return {
        "global": global_diffs,
        "per_country": country_diffs,
        "europe1_summary_file": metrics_eur1.get("summary_file"),
        "europe2_summary_file": metrics_eur2.get("summary_file"),
    }


def format_comparison_table(
    metrics_eur1: dict,
    metrics_eur2: dict,
    comparison: dict,
    eur1_label: str = "Europe 1",
    eur2_label: str = "Europe 2",
) -> str:
    """Format a comprehensive comparative dashboard into a string."""
    lines = []
    width = 133
    sep = "=" * width
    thin_sep = "-" * width

    title = f"COMPARATIVE DASHBOARD: {eur1_label.upper()} vs {eur2_label.upper()}"
    lines.append("")
    lines.append(sep)
    lines.append(f"{title:^{width}}")
    lines.append(sep)

    lines.append("\n[1] GLOBAL EUROPE SYSTEM METRICS:")
    lines.append(thin_sep)
    hdr1 = f"{'METRIC':<48} | {eur1_label:>18} | {eur2_label:>18} | {'DELTA (E2-E1)':>20} | {'CHANGE (%)':>16}"
    lines.append(hdr1)
    lines.append(thin_sep)

    def _fmt_pct(val):
        if val is None:
            return f"{'N/A':>16}"
        sign = "+" if val > 0 else ""
        return f"{sign}{val:>15.1f}%"

    g = comparison.get("global", {})
    metric_rows = [
        ("Total System Cost (OPEX+CAPEX)", "total_cost_m_yr", "M€/yr", "{:>18,.2f}", "{:>+20,.2f}"),
        ("Operational Cost (OPEX)", "total_operational_cost_m_yr", "M€/yr", "{:>18,.2f}", "{:>+20,.2f}"),
        ("Annualized CAPEX", "total_annualized_capex_m_yr", "M€/yr", "{:>18,.2f}", "{:>+20,.2f}"),
        ("Total Overnight CAPEX", "total_overnight_capex_b", "B€", "{:>18,.2f}", "{:>+20,.2f}"),
        ("Average Electricity Cost", "avg_cost_per_mwh", "€/MWh", "{:>18,.2f}", "{:>+20,.2f}"),
        ("CO2 Emissions", "total_co2_emissions_mt", "Mt", "{:>18,.2f}", "{:>+20,.2f}"),
        ("Total Generation", "total_production_twh", "TWh", "{:>18,.2f}", "{:>+20,.2f}"),
        ("Energy Not Served (ENS)", "total_ens_mwh", "MWh", "{:>18,.1f}", "{:>+20,.1f}"),
        ("Loss of Load Max Duration", "max_failure_hours", "h", "{:>18.1f}", "{:>+20.1f}"),
    ]

    for label, key, unit, val_fmt, delta_fmt in metric_rows:
        item = g.get(key, {})
        v1 = item.get("eur1", 0.0)
        v2 = item.get("eur2", 0.0)
        delta = item.get("delta", 0.0)
        pct = item.get("pct_change")
        l_str = f"{label} ({unit})" if unit else label
        v1_str = val_fmt.format(v1)
        v2_str = val_fmt.format(v2)
        d_str = delta_fmt.format(delta)
        pct_str = _fmt_pct(pct)
        lines.append(f"{l_str:<48} | {v1_str} | {v2_str} | {d_str} | {pct_str}")

    lines.append(thin_sep)

    lines.append("\n[2] PER-COUNTRY COMPARISON:")
    lines.append(thin_sep)
    hdr2 = (
        f"{'COUNTRY':<18} | {'TOTAL SYSTEM COST (M€/yr)':^29} | {'CO2 EMISSIONS (Mt)':^24} | "
        f"{'ENERGY NOT SERVED (MWh)':^26} | {'FAILURE HOURS (h)':^24}"
    )
    subhdr2 = (
        f"{'':<18} | {'Eur 1':>8} | {'Eur 2':>8} | {'Delta':>7} | {'Eur 1':>6} | {'Eur 2':>6} | {'Delta':>6} | "
        f"{'Eur 1':>7} | {'Eur 2':>7} | {'Delta':>6} | {'Eur 1':>6} | {'Eur 2':>6} | {'Delta':>6}"
    )
    sub_sep = (
        f"{'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
        f"{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}"
    )
    lines.append(hdr2)
    lines.append(subhdr2)
    lines.append(sub_sep)

    c_diffs = comparison.get("per_country", {})
    for country in sorted(c_diffs.keys()):
        cd = c_diffs[country]
        cost = cd.get("total_cost_m_yr", {})
        co2 = cd.get("co2_mt", {})
        ens = cd.get("ens_mwh", {})
        fail = cd.get("failure_hours", {})

        cost_e1 = f"{cost.get('eur1', 0.0):>8,.0f}"
        cost_e2 = f"{cost.get('eur2', 0.0):>8,.0f}"
        cost_d = f"{cost.get('delta', 0.0):>+7,.0f}"

        # CO2 emissions in Mt: if raw values exceed 500 (e.g. from tests or raw inputs), divide by 1000
        co2_val1 = co2.get('eur1', 0.0)
        co2_val2 = co2.get('eur2', 0.0)
        c1 = co2_val1 / 1000.0 if abs(co2_val1) > 500.0 else co2_val1
        c2 = co2_val2 / 1000.0 if abs(co2_val2) > 500.0 else co2_val2
        c_delta = c2 - c1
        co2_e1 = f"{c1:>6.1f}"
        co2_e2 = f"{c2:>6.1f}"
        co2_d = f"{c_delta:>+6.1f}"

        ens_e1 = f"{ens.get('eur1', 0.0):>7,.0f}"
        ens_e2 = f"{ens.get('eur2', 0.0):>7,.0f}"
        ens_d = f"{ens.get('delta', 0.0):>+6,.0f}"

        fail_e1 = f"{fail.get('eur1', 0.0):>6,.0f}"
        fail_e2 = f"{fail.get('eur2', 0.0):>6,.0f}"
        fail_d = f"{fail.get('delta', 0.0):>+6,.0f}"

        lines.append(
            f"{country:<18} | {cost_e1} | {cost_e2} | {cost_d} | {co2_e1} | {co2_e2} | {co2_d} | "
            f"{ens_e1} | {ens_e2} | {ens_d} | {fail_e1} | {fail_e2} | {fail_d}"
        )

    lines.append(sep)
    return "\n".join(lines) + "\n"


def get_or_run_eraa_baseline(
    eraa_year: int,
    period_start: str,
    period_end: str,
    countries: Optional[List[str]] = None,
    target_year: int = 2033,
    climatic_year: int = 1989,
    output_dir: str = "output/runner",
    dry_run: bool = False,
    force_run: bool = False,
    sim_output_base_dir: str = "output/long_term_uc",
) -> Optional[dict]:
    """
    Get or compute pure ERAA baseline simulation metrics for the active European countries.
    Uses cached result in output/runner/baselines/ if available.
    """
    active_countries = sorted(countries) if countries is not None else AVAILABLE_COUNTRIES

    cache_path = get_eraa_baseline_cache_path(
        eraa_year=eraa_year,
        period_start=period_start,
        period_end=period_end,
        target_year=target_year,
        climatic_year=climatic_year,
        countries=active_countries,
        output_dir=output_dir,
    )

    if os.path.isfile(cache_path) and not force_run:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_metrics = json.load(f)
            print(f"[CACHE HIT] Loaded pure ERAA {eraa_year} baseline from {cache_path}")
            return cached_metrics
        except Exception as exc:
            logging.warning(f"Could not read baseline cache from {cache_path}: {exc}")

    if dry_run:
        print(f"[INFO] Pure ERAA {eraa_year} baseline not in cache. Skipping solver execution (--dry-run).")
        return None

    # Run PyPSA simulation for pure ERAA capacities
    print(f"\n--- [CONFIGURING PURE ERAA {eraa_year} BASELINE] ({len(active_countries)} countries: {', '.join(active_countries)}) ---")
    eraa_capas = load_eraa_country_capacities(year=eraa_year)
    sync_country_json_files(active_countries=active_countries, country_capacities=eraa_capas, target_year=target_year)

    update_europe_params(
        mode="europe",
        team=active_countries[0],
        period_start=period_start,
        period_end=period_end,
        selected_countries=active_countries,
        climatic_year=climatic_year,
        interco_capas={},
    )

    pattern = os.path.join(sim_output_base_dir, "**", "uc-summary_*.json")
    before_files = set(glob.glob(pattern, recursive=True))

    print(f"\n[RUN] Launching simulation for pure ERAA {eraa_year} baseline (my_little_europe_lt_uc.py)...")
    print("-" * 70)
    return_code = run_simulation()
    print("-" * 70)

    if return_code != 0:
        logging.error(f"ERAA {eraa_year} baseline simulation failed with return code {return_code}.")
        return None

    summary_file = find_new_summary_file(before_files=before_files, output_base_dir=sim_output_base_dir)
    if not summary_file or not os.path.isfile(summary_file):
        logging.warning(f"No summary file found for pure ERAA {eraa_year} baseline.")
        return None

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        metrics = extract_simulation_metrics(
            summary_data=summary_data,
            per_country_capacities=eraa_capas,
            summary_file_path=summary_file,
            countries=active_countries,
        )
        metrics["eraa_baseline_year"] = eraa_year
        metrics["target_year"] = target_year
        metrics["climatic_year"] = climatic_year
        metrics["period_start"] = period_start
        metrics["period_end"] = period_end

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[CACHE SAVED] Saved pure ERAA {eraa_year} baseline to {cache_path}")
        return metrics
    except Exception as exc:
        logging.warning(f"Error processing pure ERAA {eraa_year} baseline metrics: {exc}")
        return None


def compute_four_way_comparison(
    metrics_eur1: Optional[dict] = None,
    metrics_eur2: Optional[dict] = None,
    metrics_eraa2025: Optional[dict] = None,
    metrics_eraa2033: Optional[dict] = None,
) -> dict:
    """
    Compute structured 4-way comparison across Europe 1, Europe 2, ERAA 2025 and ERAA 2033 baselines.
    Computes deltas of Europe 1 and Europe 2 against the official ERAA 2033 roadmap.
    """
    global_keys = [
        ("total_cost_m_yr", "Total system cost (M€/yr)"),
        ("total_operational_cost_m_yr", "Operational cost OPEX (M€/yr)"),
        ("total_annualized_capex_m_yr", "Annualized CAPEX (M€/yr)"),
        ("total_overnight_capex_b", "Overnight CAPEX (B€)"),
        ("avg_cost_per_mwh", "Average cost (€/MWh)"),
        ("total_co2_emissions_mt", "CO2 emissions (Mt)"),
        ("total_production_twh", "Total generation (TWh)"),
        ("total_ens_mwh", "Energy not served / ENS (MWh)"),
        ("max_failure_hours", "Max loss of load hours (h)"),
    ]

    def _extract_metric(m: Optional[dict], key: str) -> Optional[float]:
        if not m or not isinstance(m, dict):
            return None
        val = m.get(key)
        return float(val) if val is not None else None

    def _calc_delta(val: Optional[float], ref: Optional[float]) -> dict:
        if val is None or ref is None:
            return {"delta": None, "pct": None}
        diff = val - ref
        pct = (diff / ref * 100.0) if abs(ref) > 1e-6 else (0.0 if abs(val) < 1e-6 else None)
        return {"delta": diff, "pct": pct}

    global_comparison: dict = {}
    for key, label in global_keys:
        v_e1 = _extract_metric(metrics_eur1, key)
        v_e2 = _extract_metric(metrics_eur2, key)
        v_25 = _extract_metric(metrics_eraa2025, key)
        v_33 = _extract_metric(metrics_eraa2033, key)

        d_e1_33 = _calc_delta(v_e1, v_33)
        d_e2_33 = _calc_delta(v_e2, v_33)

        global_comparison[key] = {
            "label": label,
            "eur1": v_e1,
            "eur2": v_e2,
            "eraa2025": v_25,
            "eraa2033": v_33,
            "delta_eur1_vs_33": d_e1_33["delta"],
            "pct_eur1_vs_33": d_e1_33["pct"],
            "delta_eur2_vs_33": d_e2_33["delta"],
            "pct_eur2_vs_33": d_e2_33["pct"],
        }

    # Per-country comparison
    country_comparison: dict = {}
    all_countries = sorted(list(set(
        list(metrics_eur1.get("per_country", {}).keys() if metrics_eur1 else [])
        + list(metrics_eur2.get("per_country", {}).keys() if metrics_eur2 else [])
        + list(metrics_eraa2025.get("per_country", {}).keys() if metrics_eraa2025 else [])
        + list(metrics_eraa2033.get("per_country", {}).keys() if metrics_eraa2033 else [])
    )))

    per_country_keys = [
        ("total_cost_m_yr", "Total Cost (M€/yr)"),
        ("opex_m_yr", "OPEX (M€/yr)"),
        ("capex_m_yr", "CAPEX (M€/yr)"),
        ("co2_mt", "CO2 Emissions (Mt)"),
        ("ens_mwh", "ENS (MWh)"),
        ("failure_hours", "Loss of Load (h)"),
    ]

    for c in all_countries:
        c_comp: dict = {}
        for key, label in per_country_keys:
            v_e1 = _extract_metric(metrics_eur1.get("per_country", {}).get(c) if metrics_eur1 else None, key)
            v_e2 = _extract_metric(metrics_eur2.get("per_country", {}).get(c) if metrics_eur2 else None, key)
            v_25 = _extract_metric(metrics_eraa2025.get("per_country", {}).get(c) if metrics_eraa2025 else None, key)
            v_33 = _extract_metric(metrics_eraa2033.get("per_country", {}).get(c) if metrics_eraa2033 else None, key)

            d_e1_33 = _calc_delta(v_e1, v_33)
            d_e2_33 = _calc_delta(v_e2, v_33)

            c_comp[key] = {
                "label": label,
                "eur1": v_e1,
                "eur2": v_e2,
                "eraa2025": v_25,
                "eraa2033": v_33,
                "delta_eur1_vs_33": d_e1_33["delta"],
                "pct_eur1_vs_33": d_e1_33["pct"],
                "delta_eur2_vs_33": d_e2_33["delta"],
                "pct_eur2_vs_33": d_e2_33["pct"],
            }
        country_comparison[c] = c_comp

    return {
        "global": global_comparison,
        "per_country": country_comparison,
        "has_eraa_baselines": (metrics_eraa2025 is not None or metrics_eraa2033 is not None),
    }


def format_four_way_comparison_table(four_way_data: dict) -> str:
    """
    Format the 4-way comparative dashboard into an aligned ASCII table.
    Compares Europe 1, Europe 2, ERAA 2025, ERAA 2033, and deltas vs ERAA 2033.
    """
    line_w = 126
    sep_bar = "=" * line_w
    thin_bar = "-" * line_w

    lines = [
        sep_bar,
        f"{'4-WAY COMPARISON: STUDENT MIXES vs OFFICIAL ERAA BASELINES':^{line_w}}",
        sep_bar,
        f"{'GLOBAL METRIC':<32} {'EUROPE 1':>13} {'EUROPE 2':>13} {'ERAA 2025':>13} {'ERAA 2033':>13} {'Δ(E1 - ERAA33)':>22} {'Δ(E2 - ERAA33)':>22}",
        thin_bar,
    ]

    def _fmt_val(v: Optional[float], decimals: int = 2) -> str:
        if v is None:
            return f"{'N/A':>13}"
        return f"{v:>13,.{decimals}f}"

    def _fmt_delta(d: Optional[float], pct: Optional[float], decimals: int = 2) -> str:
        if d is None:
            return f"{'-':>22}"
        sign = "+" if d >= 0 else ""
        if pct is not None:
            pct_sign = "+" if pct >= 0 else ""
            res = f"{sign}{d:,.{decimals}f} ({pct_sign}{pct:.1f}%)"
        else:
            res = f"{sign}{d:,.{decimals}f}"
        return f"{res:>22}"

    glob = four_way_data.get("global", {})
    for key, item in glob.items():
        lbl = item["label"]
        dec = 1 if "ENS" in lbl or "hours" in lbl or "Overnight" in lbl or "Average" in lbl else 2
        v_e1 = _fmt_val(item["eur1"], dec)
        v_e2 = _fmt_val(item["eur2"], dec)
        v_25 = _fmt_val(item["eraa2025"], dec)
        v_33 = _fmt_val(item["eraa2033"], dec)
        d_e1 = _fmt_delta(item["delta_eur1_vs_33"], item["pct_eur1_vs_33"], dec)
        d_e2 = _fmt_delta(item["delta_eur2_vs_33"], item["pct_eur2_vs_33"], dec)
        lines.append(f"{lbl:<32} {v_e1} {v_e2} {v_25} {v_33} {d_e1} {d_e2}")

    lines.append(thin_bar)
    lines.append("\nPER-COUNTRY BREAKDOWN:")
    lines.append(thin_bar)
    country_hdr = f"{'COUNTRY / METRIC':<32} {'EUROPE 1':>13} {'EUROPE 2':>13} {'ERAA 2025':>13} {'ERAA 2033':>13} {'Δ(E1 - ERAA33)':>22} {'Δ(E2 - ERAA33)':>22}"

    per_c = four_way_data.get("per_country", {})
    for c in sorted(per_c.keys()):
        lines.append(f"[{c.upper()}]")
        lines.append(country_hdr)
        lines.append(thin_bar)
        for key, item in per_c[c].items():
            lbl = f"  {item['label']}"
            dec = 1 if "ENS" in lbl or "Load" in lbl else 2
            v_e1 = _fmt_val(item["eur1"], dec)
            v_e2 = _fmt_val(item["eur2"], dec)
            v_25 = _fmt_val(item["eraa2025"], dec)
            v_33 = _fmt_val(item["eraa2033"], dec)
            d_e1 = _fmt_delta(item["delta_eur1_vs_33"], item["pct_eur1_vs_33"], dec)
            d_e2 = _fmt_delta(item["delta_eur2_vs_33"], item["pct_eur2_vs_33"], dec)
            lines.append(f"{lbl:<32} {v_e1} {v_e2} {v_25} {v_33} {d_e1} {d_e2}")
        lines.append(thin_bar)

    lines.append(sep_bar)
    return "\n".join(lines)


def display_europe_comparison(formatted_report: str) -> None:
    """Print comparative report to console."""
    print(formatted_report)


def save_comparison_results(
    comparison_data: dict,
    formatted_report: str,
    output_dir: str = "output/runner",
    filename_prefix: str = "comparison_europe1_vs_europe2",
    metrics_eur1: Optional[dict] = None,
    metrics_eur2: Optional[dict] = None,
    timestamp: Optional[datetime] = None,
    metrics_eraa2025: Optional[dict] = None,
    metrics_eraa2033: Optional[dict] = None,
    four_way_comparison: Optional[dict] = None,
    four_way_report: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Save the comparative results to both a timestamped text file and a JSON file
    in output_dir (default: output/runner/), and update latest pointer files.
    """
    os.makedirs(output_dir, exist_ok=True)
    dt = timestamp or datetime.now()
    ts_str = dt.strftime("%Y%m%d_%H%M%S")
    ts_readable = dt.strftime("%Y-%m-%d %H:%M:%S")

    # Enrich comparison data with metadata and raw Europe metrics for history tracking
    enriched_data = dict(comparison_data)
    enriched_data["timestamp"] = dt.isoformat()
    enriched_data["timestamp_str"] = ts_readable
    if metrics_eur1:
        enriched_data["europe1_metrics"] = metrics_eur1
        if "per_country_capacities" in metrics_eur1:
            enriched_data["europe1_capacities"] = metrics_eur1["per_country_capacities"]
    if metrics_eur2:
        enriched_data["europe2_metrics"] = metrics_eur2
        if "per_country_capacities" in metrics_eur2:
            enriched_data["europe2_capacities"] = metrics_eur2["per_country_capacities"]
    if metrics_eraa2025:
        enriched_data["eraa2025_metrics"] = metrics_eraa2025
    if metrics_eraa2033:
        enriched_data["eraa2033_metrics"] = metrics_eraa2033
    if four_way_comparison:
        enriched_data["four_way_comparison"] = four_way_comparison

    full_report = formatted_report
    if four_way_report:
        full_report = f"{formatted_report}\n\n{four_way_report}"

    txt_filename = f"{filename_prefix}_{ts_str}.txt"
    json_filename = f"{filename_prefix}_{ts_str}.json"
    txt_path = os.path.join(output_dir, txt_filename)
    json_path = os.path.join(output_dir, json_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    # Also update latest convenience copies
    latest_txt_path = os.path.join(output_dir, f"{filename_prefix}_latest.txt")
    latest_json_path = os.path.join(output_dir, f"{filename_prefix}_latest.json")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logging.warning(f"Could not update latest pointer files: {exc}")

    return txt_path, json_path


def load_simulation_history(
    output_dir: str = "output/runner",
    target_europe: str = "europe1",
) -> List[dict]:
    """
    Load historical simulation runs from output_dir sorted chronologically.
    Filters out '_latest.json' and 'history_' files to avoid duplicates.
    Extracts metrics for target_europe ('europe1' or 'europe2').
    """
    if not os.path.isdir(output_dir):
        return []

    pattern = os.path.join(output_dir, "*.json")
    all_files = glob.glob(pattern)

    sim_files = [
        f for f in all_files
        if not os.path.basename(f).startswith("history_")
        and not os.path.basename(f).endswith("_latest.json")
    ]

    history = []
    for fpath in sim_files:
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            dt_str = data.get("timestamp_str")
            if not dt_str:
                mtime = os.path.getmtime(fpath)
                dt_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            metrics = None
            if f"{target_europe}_metrics" in data and isinstance(data[f"{target_europe}_metrics"], dict):
                metrics = dict(data[f"{target_europe}_metrics"])
            elif "per_country" in data:
                # Extract from comparison structure
                eur_key = "eur1" if target_europe == "europe1" else "eur2"
                glob_data = data.get("global", {})
                
                def _get_glob(k: str) -> float:
                    sub = glob_data.get(k, {})
                    if isinstance(sub, dict):
                        return float(sub.get(eur_key, sub.get("eur1", 0.0)))
                    return float(data.get(k, 0.0))

                metrics = {
                    "total_cost_m_yr": _get_glob("total_cost_m_yr"),
                    "total_operational_cost_m_yr": _get_glob("total_operational_cost_m_yr"),
                    "total_annualized_capex_m_yr": _get_glob("total_annualized_capex_m_yr"),
                    "total_co2_emissions_mt": _get_glob("total_co2_emissions_mt"),
                    "total_ens_mwh": _get_glob("total_ens_mwh"),
                    "max_failure_hours": _get_glob("max_failure_hours"),
                    "per_country": {},
                }
                for c, cdata in data.get("per_country", {}).items():
                    if isinstance(cdata, dict):
                        def _get_c(k: str) -> float:
                            sub = cdata.get(k, {})
                            if isinstance(sub, dict):
                                return float(sub.get(eur_key, sub.get("eur1", 0.0)))
                            return float(sub) if sub is not None else 0.0

                        metrics["per_country"][c] = {
                            "total_cost_m_yr": _get_c("total_cost_m_yr"),
                            "co2_mt": _get_c("co2_mt"),
                            "ens_mwh": _get_c("ens_mwh"),
                            "failure_hours": _get_c("failure_hours"),
                        }

            # Check for per-country capacities in data or metrics
            if metrics and "per_country" in metrics:
                capas_source = (
                    data.get(f"{target_europe}_capacities")
                    or data.get("per_country_capacities")
                    or metrics.get("per_country_capacities")
                )
                for c, c_info in metrics["per_country"].items():
                    if isinstance(c_info, dict) and "capacities" not in c_info:
                        if capas_source and isinstance(capas_source, dict) and c in capas_source:
                            c_info["capacities"] = capas_source[c]
                        elif isinstance(data.get("per_country", {}).get(c), dict):
                            c_info["capacities"] = data["per_country"][c].get("capacities", {})

            if metrics and metrics.get("per_country"):
                history.append({
                    "filepath": fpath,
                    "filename": os.path.basename(fpath),
                    "timestamp": dt_str,
                    "metrics": metrics,
                })
        except Exception as exc:
            logging.warning(f"Could not load simulation history file {fpath}: {exc}")

    # Sort chronologically by timestamp
    history.sort(key=lambda item: item["timestamp"])
    for idx, item in enumerate(history, start=1):
        item["run_index"] = idx

    return history


def compute_simulation_evolutions(
    history: List[dict],
    eraa_2025_capas: Optional[Dict[str, Dict[str, float]]] = None,
    eraa_2033_capas: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    """
    Compute evolution progression across historical simulation runs.
    Includes ERAA 2025 and ERAA 2033 as reference baselines in capacity evolutions.
    """
    if not history:
        return {"has_history": False, "history_len": 0}

    if eraa_2025_capas is None:
        try:
            eraa_2025_capas = load_eraa_country_capacities(year=2025)
        except Exception:
            eraa_2025_capas = {}
    if eraa_2033_capas is None:
        try:
            eraa_2033_capas = load_eraa_country_capacities(year=2033)
        except Exception:
            eraa_2033_capas = {}

    runs_data = []
    for h in history:
        m = h["metrics"]
        runs_data.append({
            "run_index": h["run_index"],
            "filename": h.get("filename", ""),
            "timestamp": h["timestamp"],
            "total_cost_m_yr": float(m.get("total_cost_m_yr", 0.0)),
            "total_operational_cost_m_yr": float(m.get("total_operational_cost_m_yr", 0.0)),
            "total_annualized_capex_m_yr": float(m.get("total_annualized_capex_m_yr", 0.0)),
            "total_co2_emissions_mt": float(m.get("total_co2_emissions_mt", 0.0)),
            "total_ens_mwh": float(m.get("total_ens_mwh", 0.0)),
            "max_failure_hours": float(m.get("max_failure_hours", 0.0)),
        })

    # Net global evolution from first to last run
    first_run = runs_data[0]
    last_run = runs_data[-1]
    
    def _diff_stat(key: str) -> dict:
        v1 = first_run[key]
        v2 = last_run[key]
        delta = v2 - v1
        pct = (delta / v1 * 100.0) if abs(v1) > 1e-6 else (0.0 if abs(v2) < 1e-6 else None)
        return {"first": v1, "last": v2, "delta": delta, "pct_change": pct}

    global_net_evolution = {
        "total_cost_m_yr": _diff_stat("total_cost_m_yr"),
        "total_operational_cost_m_yr": _diff_stat("total_operational_cost_m_yr"),
        "total_annualized_capex_m_yr": _diff_stat("total_annualized_capex_m_yr"),
        "total_co2_emissions_mt": _diff_stat("total_co2_emissions_mt"),
        "total_ens_mwh": _diff_stat("total_ens_mwh"),
        "max_failure_hours": _diff_stat("max_failure_hours"),
    }

    # Per-country evolution across all runs
    all_countries = sorted(list(set().union(*(h["metrics"].get("per_country", {}).keys() for h in history))))
    country_evolutions = {}

    for c in all_countries:
        c_runs_cost = []
        c_runs_co2 = []
        c_runs_ens = []
        c_runs_fail = []

        for h in history:
            cm = h["metrics"].get("per_country", {}).get(c, {})
            cost_val = float(cm.get("total_cost_m_yr", 0.0))
            raw_co2 = float(cm.get("co2_mt", 0.0))
            co2_val = (raw_co2 / 1000.0) if abs(raw_co2) > 500.0 else raw_co2
            ens_val = float(cm.get("ens_mwh", 0.0))
            fail_val = float(cm.get("failure_hours", 0.0))

            c_runs_cost.append(cost_val)
            c_runs_co2.append(co2_val)
            c_runs_ens.append(ens_val)
            c_runs_fail.append(fail_val)

        cost_delta = c_runs_cost[-1] - c_runs_cost[0]
        cost_pct = (cost_delta / c_runs_cost[0] * 100.0) if abs(c_runs_cost[0]) > 1e-6 else 0.0

        co2_delta = c_runs_co2[-1] - c_runs_co2[0]
        co2_pct = (co2_delta / c_runs_co2[0] * 100.0) if abs(c_runs_co2[0]) > 1e-6 else 0.0

        ens_delta = c_runs_ens[-1] - c_runs_ens[0]
        fail_delta = c_runs_fail[-1] - c_runs_fail[0]

        country_evolutions[c] = {
            "costs": c_runs_cost,
            "cost_delta": cost_delta,
            "cost_pct": cost_pct,
            "co2": c_runs_co2,
            "co2_delta": co2_delta,
            "co2_pct": co2_pct,
            "ens": c_runs_ens,
            "ens_delta": ens_delta,
            "failure": c_runs_fail,
            "failure_delta": fail_delta,
        }

    # 5. Capacity evolution across runs
    capacity_evolutions = {
        "per_country": {},
        "total_europe": {},
        "has_capacity_data": False,
    }

    country_capas_across_runs: Dict[str, Dict[str, List[float]]] = {c: {} for c in all_countries}
    europe_capas_across_runs: Dict[str, List[float]] = {}
    any_capa_found = False

    for h_idx, h in enumerate(history):
        m = h["metrics"]
        p_c = m.get("per_country", {})
        eur_tech_totals: Dict[str, float] = {}

        for c in all_countries:
            c_caps = p_c.get(c, {}).get("capacities", {})
            if c_caps:
                any_capa_found = True
            for tech, capa in c_caps.items():
                if tech == "failure":
                    continue
                if tech not in country_capas_across_runs[c]:
                    country_capas_across_runs[c][tech] = [0.0] * h_idx
                country_capas_across_runs[c][tech].append(float(capa))
                eur_tech_totals[tech] = eur_tech_totals.get(tech, 0.0) + float(capa)

            # For techs not present in this run but in other runs for country c
            for tech in list(country_capas_across_runs[c].keys()):
                if tech not in c_caps:
                    country_capas_across_runs[c][tech].append(0.0)

        for tech, tot in eur_tech_totals.items():
            if tech not in europe_capas_across_runs:
                europe_capas_across_runs[tech] = [0.0] * h_idx
            europe_capas_across_runs[tech].append(tot)
        for tech in list(europe_capas_across_runs.keys()):
            if tech not in eur_tech_totals:
                europe_capas_across_runs[tech].append(0.0)

    if any_capa_found:
        capacity_evolutions["has_capacity_data"] = True
        for c in all_countries:
            c_evol: Dict[str, dict] = {}
            for tech, values in country_capas_across_runs[c].items():
                first_val = values[0] if values else 0.0
                last_val = values[-1] if values else 0.0
                prev_val = values[-2] if len(values) >= 2 else first_val
                delta_prev = last_val - prev_val
                delta_net = last_val - first_val
                pct_net = (delta_net / first_val * 100.0) if abs(first_val) > 1e-6 else (0.0 if abs(last_val) < 1e-6 else None)

                e25_val = eraa_2025_capas.get(c, {}).get(tech, 0.0) if eraa_2025_capas else 0.0
                e33_val = eraa_2033_capas.get(c, {}).get(tech, 0.0) if eraa_2033_capas else 0.0

                c_evol[tech] = {
                    "runs_mw": values,
                    "runs_gw": [v / 1000.0 for v in values],
                    "eraa_2025_gw": e25_val / 1000.0,
                    "eraa_2033_gw": e33_val / 1000.0,
                    "delta_prev_gw": delta_prev / 1000.0,
                    "delta_net_gw": delta_net / 1000.0,
                    "delta_vs_2025_gw": (last_val - e25_val) / 1000.0,
                    "delta_vs_2033_gw": (last_val - e33_val) / 1000.0,
                    "pct_net": pct_net,
                }
            capacity_evolutions["per_country"][c] = c_evol

        tot_eur_evol: Dict[str, dict] = {}
        for tech, values in europe_capas_across_runs.items():
            first_val = values[0] if values else 0.0
            last_val = values[-1] if values else 0.0
            prev_val = values[-2] if len(values) >= 2 else first_val
            delta_prev = last_val - prev_val
            delta_net = last_val - first_val
            pct_net = (delta_net / first_val * 100.0) if abs(first_val) > 1e-6 else (0.0 if abs(last_val) < 1e-6 else None)

            e25_eur = sum(eraa_2025_capas.get(c, {}).get(tech, 0.0) for c in all_countries) if eraa_2025_capas else 0.0
            e33_eur = sum(eraa_2033_capas.get(c, {}).get(tech, 0.0) for c in all_countries) if eraa_2033_capas else 0.0

            tot_eur_evol[tech] = {
                "runs_mw": values,
                "runs_gw": [v / 1000.0 for v in values],
                "eraa_2025_gw": e25_eur / 1000.0,
                "eraa_2033_gw": e33_eur / 1000.0,
                "delta_prev_gw": delta_prev / 1000.0,
                "delta_net_gw": delta_net / 1000.0,
                "delta_vs_2025_gw": (last_val - e25_eur) / 1000.0,
                "delta_vs_2033_gw": (last_val - e33_eur) / 1000.0,
                "pct_net": pct_net,
            }
        capacity_evolutions["total_europe"] = tot_eur_evol

    return {
        "has_history": len(history) >= 2,
        "history_len": len(history),
        "runs": runs_data,
        "global_net_evolution": global_net_evolution,
        "country_evolutions": country_evolutions,
        "capacity_evolutions": capacity_evolutions,
    }


def format_simulation_evolutions_table(evolution_data: dict, target_europe: str = "Europe 1") -> str:
    """
    Format a complete simulation evolution dashboard into an aligned ASCII table.
    """
    if not evolution_data.get("has_history"):
        return f"\n[i] Only 1 simulation run found for {target_europe}. History tracking requires at least 2 runs.\n"

    runs = evolution_data["runs"]
    num_runs = len(runs)
    width = 133
    sep = "=" * width
    thin_sep = "-" * width

    lines = []
    title = f"SIMULATION EVOLUTION HISTORY: {target_europe.upper()} ({num_runs} RUNS RECORDED)"
    lines.append("")
    lines.append(sep)
    lines.append(f"{title:^{width}}")
    lines.append(sep)

    # 1. Global System Evolution
    lines.append(f"\n[1] GLOBAL SYSTEM EVOLUTION ACROSS SIMULATIONS ({target_europe.upper()}):")
    lines.append(thin_sep)
    g_hdr = (
        f"{'RUN':<5} | {'TIMESTAMP':<19} | {'TOTAL COST (M€/yr)':>18} | {'OPEX (M€/yr)':>16} | "
        f"{'CAPEX (M€/yr)':>16} | {'CO2 (Mt)':>12} | {'ENS (MWh)':>14} | {'FAILURE (h)':>13}"
    )
    lines.append(g_hdr)
    lines.append(thin_sep)

    for r in runs:
        idx_str = f"#{r['run_index']}"
        ts_str = r['timestamp']
        cost_str = f"{r['total_cost_m_yr']:>18,.2f}"
        opex_str = f"{r['total_operational_cost_m_yr']:>16,.2f}"
        capex_str = f"{r['total_annualized_capex_m_yr']:>16,.2f}"
        co2_str = f"{r['total_co2_emissions_mt']:>12,.2f}"
        ens_str = f"{r['total_ens_mwh']:>14,.1f}"
        fail_str = f"{r['max_failure_hours']:>13,.1f}"
        lines.append(
            f"{idx_str:<5} | {ts_str:<19} | {cost_str} | {opex_str} | {capex_str} | {co2_str} | {ens_str} | {fail_str}"
        )

    lines.append(thin_sep)
    gnet = evolution_data["global_net_evolution"]
    summary_label = f"TOTAL CHANGE (#{runs[0]['run_index']} -> #{runs[-1]['run_index']})"
    cost_d = f"{gnet['total_cost_m_yr']['delta']:>+18,.2f}"
    opex_d = f"{gnet['total_operational_cost_m_yr']['delta']:>+16,.2f}"
    capex_d = f"{gnet['total_annualized_capex_m_yr']['delta']:>+16,.2f}"
    co2_d = f"{gnet['total_co2_emissions_mt']['delta']:>+12,.2f}"
    ens_d = f"{gnet['total_ens_mwh']['delta']:>+14,.1f}"
    fail_d = f"{gnet['max_failure_hours']['delta']:>+13,.1f}"
    lines.append(
        f"{summary_label:<27} | {cost_d} | {opex_d} | {capex_d} | {co2_d} | {ens_d} | {fail_d}"
    )
    lines.append(thin_sep)

    # Helper for per-country tables: decide which runs to display as columns
    max_run_cols = min(num_runs, 4)
    displayed_run_indices = list(range(max_run_cols)) if num_runs <= 4 else [0, num_runs - 2, num_runs - 1]

    # 2. Per-Country Total Cost Evolution
    lines.append(f"\n[2] PER-COUNTRY TOTAL SYSTEM COST (M€/yr) EVOLUTION:")
    lines.append(thin_sep)

    run_col_headers = "".join(f" | {f'RUN #{runs[i][run_index_key]}':>12}" for i, run_index_key in [(i, 'run_index') for i in displayed_run_indices])
    cost_hdr = f"{'COUNTRY':<18}{run_col_headers} | {'NET CHANGE (Δ)':>15} | {'CHANGE (%)':>12}"
    lines.append(cost_hdr)
    lines.append(thin_sep)

    cev = evolution_data["country_evolutions"]
    for c in sorted(cev.keys()):
        cdata = cev[c]
        run_vals_str = "".join(f" | {cdata['costs'][i]:>12,.0f}" for i in displayed_run_indices)
        d_str = f"{cdata['cost_delta']:>+15,.0f}"
        pct_sign = "+" if cdata['cost_pct'] > 0 else ""
        pct_str = f"{pct_sign}{cdata['cost_pct']:>11.1f}%"
        lines.append(f"{c:<18}{run_vals_str} | {d_str} | {pct_str}")

    lines.append(thin_sep)

    # 3. Per-Country CO2 Emissions Evolution
    lines.append(f"\n[3] PER-COUNTRY CO2 EMISSIONS (Mt) EVOLUTION:")
    lines.append(thin_sep)
    co2_hdr = f"{'COUNTRY':<18}{run_col_headers} | {'NET CHANGE (Δ)':>15} | {'CHANGE (%)':>12}"
    lines.append(co2_hdr)
    lines.append(thin_sep)

    for c in sorted(cev.keys()):
        cdata = cev[c]
        run_vals_str = "".join(f" | {cdata['co2'][i]:>12.1f}" for i in displayed_run_indices)
        d_str = f"{cdata['co2_delta']:>+15.1f}"
        pct_sign = "+" if cdata['co2_pct'] > 0 else ""
        pct_str = f"{pct_sign}{cdata['co2_pct']:>11.1f}%"
        lines.append(f"{c:<18}{run_vals_str} | {d_str} | {pct_str}")

    lines.append(thin_sep)

    # 4. Per-Country Security of Supply Evolution
    lines.append(f"\n[4] PER-COUNTRY SECURITY OF SUPPLY (ENS & LOSS OF LOAD) EVOLUTION:")
    lines.append(thin_sep)

    # 4.1 ENS Evolution per country across all runs
    lines.append("  * ENERGY NOT SERVED - ENS (MWh):")
    ens_hdr = f"    {'COUNTRY':<18}{run_col_headers} | {'NET CHANGE (Δ)':>15}"
    lines.append(ens_hdr)
    lines.append("    " + "-" * (len(ens_hdr) - 4))

    for c in sorted(cev.keys()):
        cdata = cev[c]
        run_vals_str = "".join(f" | {cdata['ens'][i]:>12,.0f}" for i in displayed_run_indices)
        d_str = f"{cdata['ens_delta']:>+15,.0f}"
        lines.append(f"    {c:<18}{run_vals_str} | {d_str}")

    lines.append("    " + "-" * (len(ens_hdr) - 4))

    # 4.2 Loss of Load Hours per country across all runs
    lines.append("\n  * LOSS OF LOAD - FAILURE (h):")
    fail_hdr = f"    {'COUNTRY':<18}{run_col_headers} | {'NET CHANGE (Δ)':>15}"
    lines.append(fail_hdr)
    lines.append("    " + "-" * (len(fail_hdr) - 4))

    for c in sorted(cev.keys()):
        cdata = cev[c]
        run_vals_str = "".join(f" | {cdata['failure'][i]:>12,.1f}" for i in displayed_run_indices)
        d_str = f"{cdata['failure_delta']:>+15,.1f}"
        lines.append(f"    {c:<18}{run_vals_str} | {d_str}")

    lines.append(thin_sep)

    # 5. Capacity evolution across runs vs ERAA references
    cap_data = evolution_data.get("capacity_evolutions", {})
    if cap_data and cap_data.get("has_capacity_data"):
        lines.append(f"\n[5] CAPACITY EVOLUTION ACROSS SUCCESSIVE RUNS vs ERAA REFERENCES (GW):")
        lines.append(thin_sep)

        cap_col_headers = "".join(f" | {f'RUN #{runs[i][run_index_key]}':>9}" for i, run_index_key in [(i, 'run_index') for i in displayed_run_indices])
        cap_hdr = f"{'COUNTRY / TECH':<24} | {'ERAA 2025':>9} | {'ERAA 2033':>9}{cap_col_headers} | {'Δ vs PREV':>11} | {'Δ vs 2025':>11} | {'Δ vs 2033':>11}"
        lines.append(cap_hdr)
        lines.append(thin_sep)

        def _fmt_d(gw: float) -> str:
            sign = "+" if gw > 0 else ""
            return f"{sign}{gw:.1f}"

        per_c_cap = cap_data.get("per_country", {})
        for c in sorted(per_c_cap.keys()):
            techs = sorted(per_c_cap[c].keys())
            if not techs:
                continue
            lines.append(f"[{c.upper()}]")
            lines.append(cap_hdr)
            lines.append(thin_sep)
            for tech in techs:
                t_info = per_c_cap[c][tech]
                tech_disp = get_tech_display_name(tech)
                e25_str = f"{t_info.get('eraa_2025_gw', 0.0):>9.1f}"
                e33_str = f"{t_info.get('eraa_2033_gw', 0.0):>9.1f}"
                run_vals = "".join(f" | {t_info['runs_gw'][i]:>9.1f}" for i in displayed_run_indices)
                d_prev = _fmt_d(t_info["delta_prev_gw"])
                d_25 = _fmt_d(t_info.get("delta_vs_2025_gw", 0.0))
                d_33 = _fmt_d(t_info.get("delta_vs_2033_gw", 0.0))
                lines.append(f"  {tech_disp[:22]:<22} | {e25_str} | {e33_str}{run_vals} | {d_prev:>11} | {d_25:>11} | {d_33:>11}")
            lines.append(thin_sep)

        # Total Europe
        tot_eur = cap_data.get("total_europe", {})
        if tot_eur:
            lines.append("[TOTAL EUROPE]")
            lines.append(cap_hdr)
            lines.append(thin_sep)
            for tech in sorted(tot_eur.keys()):
                t_info = tot_eur[tech]
                tech_disp = get_tech_display_name(tech)
                e25_str = f"{t_info.get('eraa_2025_gw', 0.0):>9.1f}"
                e33_str = f"{t_info.get('eraa_2033_gw', 0.0):>9.1f}"
                run_vals = "".join(f" | {t_info['runs_gw'][i]:>9.1f}" for i in displayed_run_indices)
                d_prev = _fmt_d(t_info["delta_prev_gw"])
                d_25 = _fmt_d(t_info.get("delta_vs_2025_gw", 0.0))
                d_33 = _fmt_d(t_info.get("delta_vs_2033_gw", 0.0))
                lines.append(f"  {tech_disp[:22]:<22} | {e25_str} | {e33_str}{run_vals} | {d_prev:>11} | {d_25:>11} | {d_33:>11}")
            lines.append(thin_sep)

    lines.append(sep)
    return "\n".join(lines) + "\n"


def display_simulation_evolutions(report_text: str) -> None:
    """Print simulation evolution report to console."""
    print(report_text)


def save_simulation_evolutions(
    evolution_data: dict,
    report_text: str,
    output_dir: str = "output/runner",
    target_europe: str = "europe1",
) -> Tuple[str, str]:
    """
    Save simulation evolution data and formatted text report to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"history_evolutions_{target_europe}.txt")
    json_path = os.path.join(output_dir, f"history_evolutions_{target_europe}.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evolution_data, f, indent=2, ensure_ascii=False)

    return txt_path, json_path


def clear_runner_outputs(output_dir: str = "output/runner") -> List[str]:
    """
    Remove all runner output files (comparisons, latest pointers, and history evolutions)
    from output_dir to prevent blurring incremental results.
    Preserves hidden files such as .gitkeep or .gitignore.
    Returns the list of removed file paths.
    """
    if not os.path.isdir(output_dir):
        return []

    removed = []
    for item in sorted(os.listdir(output_dir)):
        if item.startswith("."):
            continue
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            try:
                os.remove(item_path)
                removed.append(item_path)
            except OSError as exc:
                logging.warning(f"Could not remove runner output file {item_path}: {exc}")
    return removed

