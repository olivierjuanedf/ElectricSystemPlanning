#!/usr/bin/env python3
"""
Collective Energy Investment Runner - European Simulation (Exclusive Europe Mode).

Collects investment choice files from student groups (one file per country in input/runner/ as .txt,
or directly from input/long_term_uc/countries/ as .json), configures the model
exclusively in "europe" mode, runs the unified simulation of the interconnected European grid,
and displays a comparative dashboard of results by country.

The runner operates exclusively in "europe" mode (solo mode is not supported by the runner).

Usage:
    # Standard execution: collect all country files in input/runner/ (.txt) and run European simulation:
    python main_runner.py
    python main_runner.py txt

    # JSON execution: use existing country files in input/long_term_uc/countries/ (.json):
    python main_runner.py json
    python main_runner.py --source json

    # Preview mode (updates country .json files without solving in txt mode):
    python main_runner.py --dry-run
    python main_runner.py json --dry-run

    # Specify a custom folder or files:
    python main_runner.py custom_folder/
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

from src.include_runner.investments_runner import (
    AVAILABLE_COUNTRIES,
    parse_investment_file,
    parse_country_json_file,
    load_country_capacities,
    collect_investment_files,
    write_country_json,
    update_europe_params,
    run_simulation,
    find_latest_summary_file,
    display_multi_country_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Collective Runner: collects student investments for all countries and executes the interconnected European simulation (exclusive Europe mode)."
    )
    parser.add_argument(
        "folder_or_files",
        nargs="*",
        default=None,
        help="Directory, list of country files, or source keyword ('txt' or 'json') (default: input/runner/)",
    )
    parser.add_argument(
        "--source",
        choices=["txt", "json"],
        default=None,
        help="Source format to use: 'txt' (input/runner/*.txt) or 'json' (input/long_term_uc/countries/*.json). Default: 'txt'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Update country .json files without executing the PyPSA simulation",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not display the comparative summary table at the end of simulation",
    )
    parser.add_argument(
        "--period-start",
        default="1900/1/1",
        help="Simulation start date (default: '1900/1/1', full year)",
    )
    parser.add_argument(
        "--period-end",
        default="1901/1/1",
        help="Simulation end date (default: '1901/1/1', full year)",
    )

    args = parser.parse_args()

    # Determine source mode and input targets
    raw_targets = list(args.folder_or_files) if args.folder_or_files else []
    source_mode = args.source

    # Check if a positional keyword was used ('txt' or 'json')
    if raw_targets and len(raw_targets) == 1 and raw_targets[0].lower() in ("txt", "json"):
        if source_mode is None:
            source_mode = raw_targets[0].lower()
        raw_targets = []

    # Default source mode is "txt" if not specified
    if source_mode is None:
        if raw_targets and all(f.endswith(".json") for f in raw_targets if os.path.isfile(f)):
            source_mode = "json"
        else:
            source_mode = "txt"

    if not raw_targets:
        if source_mode == "json":
            default_dir = os.path.join("input", "long_term_uc", "countries")
        else:
            runner_dir = os.path.join("input", "runner")
            default_dir = runner_dir

        if os.path.isdir(default_dir):
            raw_targets = [default_dir]
        else:
            parser.print_help()
            print(f"\nError: Directory '{default_dir}' not found.")
            sys.exit(1)

    ext = ".json" if source_mode == "json" else ".txt"
    matched_files = collect_investment_files(raw_targets, extension=ext)

    if not matched_files:
        print(f"[!] No {ext} files found in: {raw_targets}")
        if source_mode == "json":
            print(f"    Please ensure country JSON files exist in input/long_term_uc/countries/ (e.g., france.json...)")
        else:
            print(f"    Please place country files in input/runner/ (e.g., france.txt, germany.txt...)")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"   COLLECTIVE RUNNER FOR STUDENT INVESTMENTS - EUROPE MODE ({source_mode.upper()} SOURCE)")
    print("=" * 70 + "\n")

    # 1. Read and validate all country investment files
    per_country_capacities: Dict[str, Dict[str, float]] = {}
    source_files: Dict[str, str] = {}

    for fpath in matched_files:
        try:
            country, capacities = load_country_capacities(filepath=fpath)
            per_country_capacities[country] = capacities
            source_files[country] = fpath
        except Exception as exc:
            print(f"[!] Error while reading '{fpath}': {exc}", file=sys.stderr)
            sys.exit(1)

    collected_countries = sorted(list(per_country_capacities.keys()))
    print(f"[1/3] Collected country files from {source_mode.upper()} ({len(collected_countries)} countries):")
    for c in collected_countries:
        print(f"      - {c.upper():<20} <- {source_files[c]} ({len(per_country_capacities[c])} assets defined)")

    # 2. Update per-country JSON files (in TXT mode only)
    if source_mode == "txt":
        print(f"\n[2/3] Injecting capacities into input/long_term_uc/countries/:")
        for country in collected_countries:
            json_path = write_country_json(country=country, capacities=per_country_capacities[country])
            print(f"      -> {json_path} updated")
    else:
        print(f"\n[2/3] Using existing country JSON files in input/long_term_uc/countries/ (no rewriting needed):")
        print(f"      -> {len(collected_countries)} country file(s) loaded directly from JSON.")

    # Strictly enforce "europe" mode and full-year horizon for the collective simulation
    update_europe_params(
        mode="europe",
        team=collected_countries[0],
        period_start=args.period_start,
        period_end=args.period_end,
    )
    print(
        f"      -> Global parameters: mode='europe', full year period [{args.period_start} - {args.period_end}] "
        f"activated in elec-europe_params_to-be-modif.json"
    )

    if args.dry_run:
        print(f"\n[INFO] --dry-run mode activated: configuration validated, simulation not started.")
        print(f"       You can review files in input/long_term_uc/countries/.")
        return

    # 3. Launch interconnected European simulation
    print(f"\n[3/3] Launching interconnected European simulation (my_little_europe_lt_uc.py)...")
    print("-" * 70)
    return_code = run_simulation()
    print("-" * 70)

    if return_code != 0:
        print(f"\n[!] The simulation finished with non-zero exit code ({return_code}).")
        sys.exit(return_code)

    # 4. Display comparative European dashboard
    if not args.no_summary:
        summary_file = find_latest_summary_file()
        if summary_file and os.path.isfile(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                updated_countries = collected_countries if source_mode == "txt" else []
                display_multi_country_summary(
                    summary_data=summary_data,
                    countries=AVAILABLE_COUNTRIES,
                    updated_countries=updated_countries,
                    per_country_capacities=per_country_capacities,
                    summary_file_path=summary_file,
                )
            except Exception as e:
                logging.warning(f"Could not read summary file {summary_file}: {e}")
        else:
            print("\n[i] Simulation completed. No summary file found.")


if __name__ == "__main__":
    main()
