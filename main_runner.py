#!/usr/bin/env python3
"""
Collective Energy Investment Runner - European Simulation (Exclusive Europe Mode).

Collects investment choice files from student groups (e.g. france_europe1.txt, france_europe2.txt in input/runner/,
or corresponding JSON files), configures the model exclusively in "europe" mode, runs the unified simulation
for Europe 1 and Europe 2 sequentially, displays a comparative dashboard, and saves the comparison to disk.

The runner operates exclusively in "europe" mode (solo mode is not supported by the runner).

Usage:
    # Standard execution: collect all files in input/runner/ (.txt) and run European simulations:
    python main_runner.py
    python main_runner.py txt

    # JSON execution: use JSON files in input/runner/ or input/long_term_uc/countries/ (.json):
    python main_runner.py json
    python main_runner.py --source json

    # Preview mode (updates country .json files without solving):
    python main_runner.py --dry-run
    python main_runner.py json --dry-run

    # Test/debug mode (fast 2-week simulation: 1900/1/1 to 1900/1/15):
    python main_runner.py --test
    python main_runner.py --test --dry-run

    # Execute only Europe 1 or Europe 2:
    python main_runner.py --europe europe1
    python main_runner.py --europe europe2

    # Clear previous runner outputs and history:
    python main_runner.py --clean
    python main_runner.py clean

    # Specify a custom folder or files:
    python main_runner.py custom_folder/
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Dict, List, Optional

from src.include_runner.investments_runner import (
    AVAILABLE_COUNTRIES,
    DEFAULT_PERIOD_START,
    DEFAULT_PERIOD_END,
    TEST_PERIOD_START,
    TEST_PERIOD_END,
    resolve_simulation_period,
    parse_investment_file,
    parse_country_json_file,
    load_country_capacities,
    collect_investment_files,
    write_country_json,
    update_europe_params,
    run_simulation,
    find_latest_summary_file,
    find_new_summary_file,
    display_multi_country_summary,
    group_files_by_europe_with_sources,
    extract_simulation_metrics,
    compute_europe_comparison,
    format_comparison_table,
    display_europe_comparison,
    save_comparison_results,
    load_simulation_history,
    compute_simulation_evolutions,
    format_simulation_evolutions_table,
    display_simulation_evolutions,
    save_simulation_evolutions,
    clear_runner_outputs,
    load_eraa_country_capacities,
    format_installed_capacities_table,
    display_installed_capacities,
    format_capacity_comparison_table,
    get_selected_target_year,
    get_selected_climatic_year,
    AVAILABLE_CLIMATIC_YEARS,
    load_europe_interco_overwrites,
    get_or_run_eraa_baseline,
    compute_four_way_comparison,
    format_four_way_comparison_table,
    load_all_simulation_runs,
    format_incremental_capacities_trajectory_table,
    format_incremental_capacities_summary_table,
    save_incremental_capacities_trajectory,
    sync_country_json_files,
    fill_missing_countries_with_eraa,
)


def run_single_europe_simulation(
    europe_name: str,
    country_capacities: Dict[str, Dict[str, float]],
    period_start: str,
    period_end: str,
    climatic_year: Optional[int] = None,
    interco_capas: Optional[Dict[str, float]] = None,
    dry_run: bool = False,
    sim_output_base_dir: str = "output/long_term_uc",
) -> Optional[dict]:
    """
    Configure and execute simulation for one European mix, returning extracted metrics.
    """
    countries = sorted(list(country_capacities.keys()))
    print(f"\n--- [CONFIGURING {europe_name.upper()}] ({len(countries)} countries: {', '.join(countries)}) ---")
    sync_country_json_files(active_countries=countries, country_capacities=country_capacities)
    for country in countries:
        print(f"      -> input/long_term_uc/countries/{country}.json updated ({len(country_capacities[country])} assets)")

    # Update global parameters in elec-europe_params_to-be-modif.json
    update_europe_params(
        mode="europe",
        team=countries[0],
        period_start=period_start,
        period_end=period_end,
        selected_countries=countries,
        climatic_year=climatic_year,
        interco_capas=interco_capas,
    )
    is_full_year = (period_start == DEFAULT_PERIOD_START and period_end == DEFAULT_PERIOD_END)
    period_label = "full year period" if is_full_year else "simulation period"
    cy_label = f", climatic_year={climatic_year}" if climatic_year is not None else ""
    interco_count = len(interco_capas) if interco_capas else 0
    interco_label = f", {interco_count} interco override(s)" if interco_count > 0 else ", native ERAA intercos"
    print(
        f"      -> Global parameters: mode='europe', {period_label} [{period_start} - {period_end}]{cy_label}{interco_label} "
        f"countries={countries} activated in elec-europe_params_to-be-modif.json"
    )
    if interco_capas:
        print(f"      -> Overriding {len(interco_capas)} cross-border interconnection(s) w.r.t. ERAA:")
        for link_name, link_capa in sorted(interco_capas.items()):
            print(f"           * {link_name:<25}: {link_capa:>8,.1f} MW")

    if dry_run:
        print(f"      [INFO] --dry-run active for {europe_name}: files updated, solver not launched.")
        return None

    pattern = os.path.join(sim_output_base_dir, "**", "uc-summary_*.json")
    before_files = set(glob.glob(pattern, recursive=True))

    print(f"\n[RUN] Launching simulation for {europe_name.upper()} (my_little_europe_lt_uc.py)...")
    print("-" * 70)
    return_code = run_simulation()
    print("-" * 70)

    if return_code != 0:
        print(f"\n[!] Simulation for {europe_name} failed with non-zero exit code ({return_code}).", file=sys.stderr)
        sys.exit(return_code)

    summary_file = find_new_summary_file(before_files=before_files, output_base_dir=sim_output_base_dir)
    if not summary_file or not os.path.isfile(summary_file):
        print(f"[!] Warning: No summary file found for {europe_name} run.")
        return None

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        metrics = extract_simulation_metrics(
            summary_data=summary_data,
            per_country_capacities=country_capacities,
            summary_file_path=summary_file,
            countries=AVAILABLE_COUNTRIES,
        )
        return metrics
    except Exception as exc:
        logging.warning(f"Error reading summary file {summary_file}: {exc}")
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the collective runner."""
    parser = argparse.ArgumentParser(
        description="Collective Runner: collects student investments for all countries and executes European simulation(s)."
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
        help="Source format to use: 'txt' (input/runner/*.txt) or 'json' (input/runner/*.json). Default: 'txt'",
    )
    parser.add_argument(
        "--europe",
        choices=["all", "europe1", "europe2"],
        default="all",
        help="Specify which Europe to simulate: 'europe1', 'europe2', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Update country .json files without executing the PyPSA simulation",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Fast debug mode: simulate only a 2-week period ('1900/1/1' to '1900/1/15') for quick testing",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not display the comparative summary table at the end of simulation",
    )
    parser.add_argument(
        "--output-dir",
        default="output/runner",
        help="Directory where comparison reports and simulation history are saved (default: 'output/runner')",
    )
    parser.add_argument(
        "--history-europe",
        choices=["all", "europe1", "europe2"],
        default="all",
        help="Europe configuration to track across historical simulations: 'all', 'europe1', or 'europe2' (default: 'all')",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not display the simulation evolution history across successive runs",
    )
    parser.add_argument(
        "--clean",
        "--clear",
        action="store_true",
        dest="clean",
        help="Clear runner outputs and simulation history in output/runner/ (or specified --output-dir) to prevent blurring incremental results",
    )
    parser.add_argument(
        "--fill-missing-with-eraa",
        "--fill-missing-eraa",
        action="store_true",
        dest="fill_missing_with_eraa",
        help="Automatically populate any European country without a submitted investment file with official ERAA reference capacities",
    )
    parser.add_argument(
        "--no-compare-eraa",
        action="store_true",
        help="Do not compare results with pure ERAA 2025 and ERAA 2033 baselines",
    )
    parser.add_argument(
        "--force-eraa-run",
        action="store_true",
        help="Force re-running PyPSA simulation for ERAA baselines even if cached in output/runner/baselines/",
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
    parser.add_argument(
        "--climatic-year",
        "--cy",
        type=int,
        default=None,
        help="Climatic year to simulate (valid ERAA years: 1982, 1989, 1996, 2003, 2010, 2016). Defaults to selected_climatic_year in elec-europe_params_to-be-modif.json (or 1989)",
    )
    parser.add_argument(
        "--interco-europe1",
        default=None,
        help="Optional path to custom interconnection overrides JSON file for Europe 1 (default: auto-detected in input/runner/)",
    )
    parser.add_argument(
        "--interco-europe2",
        default=None,
        help="Optional path to custom interconnection overrides JSON file for Europe 2 (default: auto-detected in input/runner/)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Determine source mode and input targets
    raw_targets = list(args.folder_or_files) if args.folder_or_files else []
    source_mode = args.source

    # Check if a positional keyword was used ('clean', 'clear', 'txt', or 'json')
    is_clean_keyword = False
    if raw_targets and len(raw_targets) == 1 and raw_targets[0].lower() in ("clean", "clear"):
        args.clean = True
        is_clean_keyword = True
        raw_targets = []

    # Handle cleaning runner outputs if requested
    if args.clean:
        removed = clear_runner_outputs(output_dir=args.output_dir)
        if removed:
            print(f"\n[OK] Cleared {len(removed)} output file(s) in '{args.output_dir}':")
            for f in removed:
                print(f"      - {f}")
        else:
            print(f"\n[i] Output directory '{args.output_dir}' is already clean (no files to remove).")

        # If --clean or 'clean' was called standalone (no test, dry-run, or custom targets specified), exit cleanly
        if is_clean_keyword or (not args.folder_or_files and not args.test and not args.dry_run and args.europe == "all"):
            print("\nRunner output directory reset. Exiting.")
            sys.exit(0)

    # Resolve simulation period (full year vs. 2-week test period)
    period_start, period_end = resolve_simulation_period(
        is_test=args.test,
        period_start=args.period_start,
        period_end=args.period_end,
    )

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
        runner_dir = os.path.join("input", "runner")
        if os.path.isdir(runner_dir):
            raw_targets = [runner_dir]
        elif source_mode == "json" and os.path.isdir(os.path.join("input", "long_term_uc", "countries")):
            raw_targets = [os.path.join("input", "long_term_uc", "countries")]
        else:
            parser.print_help()
            print(f"\nError: Directory '{runner_dir}' not found.")
            sys.exit(1)

    ext = ".json" if source_mode == "json" else ".txt"
    matched_files = collect_investment_files(raw_targets, extension=ext)

    if not matched_files:
        print(f"[!] No {ext} files found in: {raw_targets}")
        print(f"    Please place country files in input/runner/ (e.g., france_europe1{ext}, france_europe2{ext}...)")
        sys.exit(1)

    # Group files by Europe identifier (europe1, europe2, ...)
    grouped_capas, grouped_srcs = group_files_by_europe_with_sources(matched_files)

    # Filter Europes based on --europe argument if specified
    if args.europe != "all":
        grouped_capas = {k: v for k, v in grouped_capas.items() if k == args.europe}
        grouped_srcs = {k: v for k, v in grouped_srcs.items() if k == args.europe}

    available_europes = sorted(list(grouped_capas.keys()))
    if not available_europes:
        print(f"[!] No files matched the requested Europe configuration ('{args.europe}').")
        sys.exit(1)

    if args.fill_missing_with_eraa:
        target_year = get_selected_target_year()
        grouped_capas, grouped_srcs = fill_missing_countries_with_eraa(
            grouped_capas=grouped_capas,
            grouped_srcs=grouped_srcs,
            target_year=target_year,
        )

    # Resolve and validate climatic year
    if args.climatic_year is not None and args.climatic_year not in AVAILABLE_CLIMATIC_YEARS:
        print(f"[!] Invalid climatic year: {args.climatic_year}.")
        print(f"    Available ERAA climatic years: {', '.join(str(y) for y in AVAILABLE_CLIMATIC_YEARS)}")
        sys.exit(1)

    climatic_year = args.climatic_year if args.climatic_year is not None else get_selected_climatic_year()

    # Load Europe-specific interconnection overrides
    runner_search_dirs = [os.path.dirname(matched_files[0])] if matched_files else ["input/runner"]
    custom_interco_files = {
        "europe1": args.interco_europe1,
        "europe2": args.interco_europe2,
    }
    europe_intercos: Dict[str, Dict[str, float]] = {}
    for eur in available_europes:
        europe_intercos[eur] = load_europe_interco_overwrites(
            europe_id=eur,
            runner_dirs=runner_search_dirs,
            custom_filepath=custom_interco_files.get(eur),
        )

    print("\n" + "=" * 76)
    print(f"   COLLECTIVE RUNNER FOR STUDENT INVESTMENTS - EUROPE MODE ({source_mode.upper()} SOURCE)")
    if args.test:
        print(f"   *** TEST / DEBUG MODE ACTIVATED: [{period_start} -> {period_end}] (2 WEEKS) ***")
    print(f"   Simulating Climatic Year: {climatic_year}")
    print("=" * 76)
    print(f"Detected Europe configuration(s): {', '.join(e.upper() for e in available_europes)}")
    for eur in available_europes:
        c_list = sorted(list(grouped_capas[eur].keys()))
        print(f"  * {eur.upper()} ({len(c_list)} countries): {', '.join(c_list)}")
        for c in c_list:
            print(f"      - {c:<18} <- {grouped_srcs[eur][c]} ({len(grouped_capas[eur][c])} assets)")
        if europe_intercos.get(eur):
            print(f"      -> {len(europe_intercos[eur])} interconnection overwrite(s):")
            for l_name, l_val in sorted(europe_intercos[eur].items()):
                print(f"           * {l_name:<25}: {l_val:>8,.1f} MW")

    # Display installed capacities and comparison with ERAA references (2025 and 2033)
    eraa_2025_capas = load_eraa_country_capacities(year=2025)
    eraa_2033_capas = load_eraa_country_capacities(year=2033)
    history_runs = load_all_simulation_runs(output_dir=args.output_dir)

    for eur in available_europes:
        capa_table = format_installed_capacities_table(
            country_capacities=grouped_capas[eur],
            title=f"INSTALLED CAPACITIES - {eur.upper()} (GW)",
        )
        display_installed_capacities(capa_table)

    # Unified Incremental Capacity Trajectory across ERAA references and successive runs
    traj_table = format_incremental_capacities_trajectory_table(
        eraa_2025_capas=eraa_2025_capas,
        eraa_2033_capas=eraa_2033_capas,
        history_runs=history_runs,
        current_europe1_capas=grouped_capas.get("europe1"),
        current_europe2_capas=grouped_capas.get("europe2"),
    )
    print("\n" + traj_table)

    summary_traj_table = format_incremental_capacities_summary_table(
        eraa_2025_capas=eraa_2025_capas,
        eraa_2033_capas=eraa_2033_capas,
        history_runs=history_runs,
        current_europe1_capas=grouped_capas.get("europe1"),
        current_europe2_capas=grouped_capas.get("europe2"),
    )
    if summary_traj_table:
        print("\n" + summary_traj_table)

    # Case A: Two Europes are present and requested -> Sequential run + Comparison
    if len(available_europes) >= 2 and "europe1" in grouped_capas and "europe2" in grouped_capas:
        print("\n" + "=" * 76)
        print("   SEQUENTIAL EXECUTION: SIMULATION EUROPE 1 THEN SIMULATION EUROPE 2")
        print("=" * 76)

        # 1. Run Europe 1
        metrics_eur1 = run_single_europe_simulation(
            europe_name="europe1",
            country_capacities=grouped_capas["europe1"],
            period_start=period_start,
            period_end=period_end,
            climatic_year=climatic_year,
            interco_capas=europe_intercos.get("europe1", {}),
            dry_run=args.dry_run,
        )

        # 2. Run Europe 2
        metrics_eur2 = run_single_europe_simulation(
            europe_name="europe2",
            country_capacities=grouped_capas["europe2"],
            period_start=period_start,
            period_end=period_end,
            climatic_year=climatic_year,
            interco_capas=europe_intercos.get("europe2", {}),
            dry_run=args.dry_run,
        )

        # Baseline ERAA simulations (cached)
        baseline_2025 = None
        baseline_2033 = None
        four_way_data = None
        four_way_report = None

        if not args.no_compare_eraa:
            target_year = get_selected_target_year()
            active_eval_countries = sorted(list(set(sum([list(grouped_capas[e].keys()) for e in available_europes], []))))
            baseline_2025 = get_or_run_eraa_baseline(
                eraa_year=2025,
                period_start=period_start,
                period_end=period_end,
                countries=active_eval_countries,
                target_year=target_year,
                climatic_year=climatic_year,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                force_run=args.force_eraa_run,
            )
            baseline_2033 = get_or_run_eraa_baseline(
                eraa_year=2033,
                period_start=period_start,
                period_end=period_end,
                countries=active_eval_countries,
                target_year=target_year,
                climatic_year=climatic_year,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                force_run=args.force_eraa_run,
            )
            if baseline_2025 or baseline_2033:
                four_way_data = compute_four_way_comparison(
                    metrics_eur1=metrics_eur1,
                    metrics_eur2=metrics_eur2,
                    metrics_eraa2025=baseline_2025,
                    metrics_eraa2033=baseline_2033,
                )
                four_way_report = format_four_way_comparison_table(four_way_data)
                if not args.no_summary:
                    print("\n" + four_way_report)

        if args.dry_run:
            print("\n[INFO] --dry-run completed: both Europe 1 and Europe 2 configurations validated.")
            return

        if metrics_eur1 and metrics_eur2:
            # 3. Compute and display comparative dashboard
            comparison = compute_europe_comparison(metrics_eur1, metrics_eur2)
            report_text = format_comparison_table(metrics_eur1, metrics_eur2, comparison)

            if not args.no_summary:
                display_europe_comparison(report_text)

            # 4. Save comparative results
            txt_file, json_file = save_comparison_results(
                comparison_data=comparison,
                formatted_report=report_text,
                output_dir=args.output_dir,
                filename_prefix="comparison_europe1_vs_europe2",
                metrics_eur1=metrics_eur1,
                metrics_eur2=metrics_eur2,
                metrics_eraa2025=baseline_2025,
                metrics_eraa2033=baseline_2033,
                four_way_comparison=four_way_data,
                four_way_report=four_way_report,
            )
            print(f"\n[OK] Comparative summary successfully saved:")
            print(f"     -> Text report : {txt_file}")
            print(f"     -> JSON metrics: {json_file}\n")
        else:
            print("\n[!] Warning: Could not generate comparison because one of the simulations did not produce metrics.")

    # Case B: Single Europe configuration
    else:
        single_eur = available_europes[0]
        metrics = run_single_europe_simulation(
            europe_name=single_eur,
            country_capacities=grouped_capas[single_eur],
            period_start=period_start,
            period_end=period_end,
            climatic_year=climatic_year,
            interco_capas=europe_intercos.get(single_eur, {}),
            dry_run=args.dry_run,
        )

        # Baseline ERAA simulations (cached)
        baseline_2025 = None
        baseline_2033 = None
        four_way_data = None
        four_way_report = None

        if not args.no_compare_eraa:
            target_year = get_selected_target_year()
            active_eval_countries = sorted(list(set(sum([list(grouped_capas[e].keys()) for e in available_europes], []))))
            baseline_2025 = get_or_run_eraa_baseline(
                eraa_year=2025,
                period_start=period_start,
                period_end=period_end,
                countries=active_eval_countries,
                target_year=target_year,
                climatic_year=climatic_year,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                force_run=args.force_eraa_run,
            )
            baseline_2033 = get_or_run_eraa_baseline(
                eraa_year=2033,
                period_start=period_start,
                period_end=period_end,
                countries=active_eval_countries,
                target_year=target_year,
                climatic_year=climatic_year,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                force_run=args.force_eraa_run,
            )
            if baseline_2025 or baseline_2033:
                four_way_data = compute_four_way_comparison(
                    metrics_eur1=metrics if single_eur == "europe1" else None,
                    metrics_eur2=metrics if single_eur == "europe2" else None,
                    metrics_eraa2025=baseline_2025,
                    metrics_eraa2033=baseline_2033,
                )
                four_way_report = format_four_way_comparison_table(four_way_data)
                if not args.no_summary:
                    print("\n" + four_way_report)

        if args.dry_run:
            print(f"\n[INFO] --dry-run completed for {single_eur}.")
            return

        if not args.no_summary and metrics and metrics.get("summary_file"):
            with open(metrics["summary_file"], "r", encoding="utf-8") as f:
                s_data = json.load(f)
            display_multi_country_summary(
                summary_data=s_data,
                countries=AVAILABLE_COUNTRIES,
                updated_countries=sorted(list(grouped_capas[single_eur].keys())),
                per_country_capacities=grouped_capas[single_eur],
                summary_file_path=metrics["summary_file"],
            )

    # 5. Simulation Evolution Tracking across successive runs (Europe 1 and Europe 2)
    if not args.dry_run and not args.no_history:
        if args.history_europe == "all":
            history_europes_to_process = [e for e in ("europe1", "europe2") if e in available_europes]
            if not history_europes_to_process:
                history_europes_to_process = ["europe1", "europe2"]
        else:
            history_europes_to_process = [args.history_europe]

        for target_eur in history_europes_to_process:
            history = load_simulation_history(output_dir=args.output_dir, target_europe=target_eur)
            if len(history) >= 2:
                evol_data = compute_simulation_evolutions(
                    history=history,
                    eraa_2025_capas=eraa_2025_capas,
                    eraa_2033_capas=eraa_2033_capas,
                )
                evol_table = format_simulation_evolutions_table(evol_data, target_europe=target_eur)
                display_simulation_evolutions(evol_table)
                hist_txt, hist_json = save_simulation_evolutions(
                    evolution_data=evol_data,
                    report_text=evol_table,
                    output_dir=args.output_dir,
                    target_europe=target_eur,
                )
                print(f"[i] Simulation evolution history ({target_eur.upper()}) saved to:")
                print(f"    -> Text report : {hist_txt}")
                print(f"    -> JSON metrics: {hist_json}\n")

    # Save the updated incremental capacity trajectory table to output/runner/
    if not args.dry_run:
        updated_history = load_all_simulation_runs(output_dir=args.output_dir)
        if updated_history:
            updated_traj_table = format_incremental_capacities_trajectory_table(
                eraa_2025_capas=eraa_2025_capas,
                eraa_2033_capas=eraa_2033_capas,
                history_runs=updated_history,
            )
            traj_txt = save_incremental_capacities_trajectory(
                report_text=updated_traj_table,
                output_dir=args.output_dir,
            )
            print(f"[i] Incremental capacity trajectory saved to:")
            print(f"    -> Text report : {traj_txt}\n")


if __name__ == "__main__":
    main()

