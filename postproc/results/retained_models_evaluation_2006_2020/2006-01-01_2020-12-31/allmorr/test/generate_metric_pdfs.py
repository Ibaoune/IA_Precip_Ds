#!/usr/bin/env python3
"""
Generate one PDF per metric from evaluation figures.

Reads pdf_config.yaml to determine which metrics, seasons, and experiments
to include. Figures are ordered: Annual → DJF → JJA → MAM → SON (configurable).

Usage:
    python generate_metric_pdfs.py                          # uses pdf_config.yaml in CWD
    python generate_metric_pdfs.py --config my_config.yaml  # custom config path
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import yaml
from PIL import Image


def load_config(config_path: str) -> dict:
    """Load and validate the YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def collect_figures_for_metric(base_path: str, metric: dict, seasons: list) -> list:
    """
    Collect figure paths for a single metric, ordered by season.

    Returns a list of tuples: (season_label, figure_path)
    """
    figures = []

    # Determine the figures root for this metric
    if metric.get("subfolder"):
        figures_root = os.path.join(base_path, metric["folder"], metric["subfolder"], "figures")
    else:
        figures_root = os.path.join(base_path, metric["folder"], "figures")

    if not os.path.isdir(figures_root):
        print(f"  [WARNING] Figures directory not found: {figures_root}")
        return figures

    # Collect figures for each season in order
    for season in seasons:
        season_dir = os.path.join(figures_root, season["id"])
        if not os.path.isdir(season_dir):
            print(f"  [WARNING] Season directory not found: {season_dir}")
            continue

        # Get all PNG files in the season directory, sorted
        season_files = sorted(glob.glob(os.path.join(season_dir, "*.png")))
        for f in season_files:
            figures.append((season["label"], f))

    return figures


def collect_extra_figures(base_path: str, extra: dict, seasons: list) -> list:
    """Collect figures from extra sections (seasonal_summary, regional_summary)."""
    figures = []
    extra_dir = os.path.join(base_path, extra["folder"])

    if not os.path.isdir(extra_dir):
        print(f"  [WARNING] Extra section directory not found: {extra_dir}")
        return figures

    pattern = extra.get("pattern", "*.png")
    extra_files = sorted(glob.glob(os.path.join(extra_dir, pattern)))

    for f in extra_files:
        # Only include image files (png, jpg, jpeg)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            figures.append((extra["name"], f))

    return figures


def create_pdf_from_figures(figure_list: list, output_path: str, metric_name: str):
    """
    Create a PDF from a list of (label, filepath) tuples.
    Each figure becomes one page in the PDF.
    """
    if not figure_list:
        print(f"  [SKIP] No figures found for metric '{metric_name}'")
        return

    images = []
    first_image = None

    for label, filepath in figure_list:
        try:
            img = Image.open(filepath).convert("RGB")
            if first_image is None:
                first_image = img
            else:
                images.append(img)
        except Exception as e:
            print(f"  [ERROR] Could not load {filepath}: {e}")

    if first_image is None:
        print(f"  [SKIP] No valid images for metric '{metric_name}'")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    first_image.save(output_path, save_all=True, append_images=images)
    total_pages = 1 + len(images)
    print(f"  [OK] Saved {output_path} ({total_pages} pages)")


def print_summary(figure_list: list, metric_name: str):
    """Print a summary of figures that will be included."""
    print(f"\n  Metric: {metric_name}")
    print(f"  {'─' * 50}")

    current_label = None
    for label, filepath in figure_list:
        if label != current_label:
            current_label = label
            print(f"    [{label}]")
        print(f"      • {os.path.basename(filepath)}")

    print(f"  Total: {len(figure_list)} figures")


def main():
    parser = argparse.ArgumentParser(description="Generate per-metric PDFs from evaluation figures.")
    parser.add_argument(
        "--config", "-c",
        default="pdf_config.yaml",
        help="Path to YAML configuration file (default: pdf_config.yaml)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be generated without creating PDFs"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Resolve base_path relative to config file location
    base_path = os.path.join(config_dir, config.get("base_path", "."))
    output_dir = os.path.join(config_dir, config.get("output_dir", "pdf_output"))
    seasons = config.get("seasons", [])
    metrics = config.get("metrics", [])
    extra_sections = config.get("extra_sections", [])
    experiments = config.get("experiments", [])

    print("=" * 60)
    print("  PDF Generation from Evaluation Figures")
    print("=" * 60)
    print(f"  Config      : {os.path.abspath(config_path)}")
    print(f"  Base path   : {os.path.abspath(base_path)}")
    print(f"  Output dir  : {os.path.abspath(output_dir)}")
    print(f"  Seasons     : {', '.join(s['label'] for s in seasons)}")
    print(f"  Metrics     : {len([m for m in metrics if m.get('enabled', True)])}")
    print(f"  Dry run     : {args.dry_run}")
    print("=" * 60)

    # Process each experiment
    for experiment in experiments:
        exp_path = os.path.join(base_path, experiment.get("path", "."))
        exp_name = experiment.get("name", "default")
        print(f"\n{'━' * 60}")
        print(f"  Experiment: {exp_name}")
        print(f"  Path: {os.path.abspath(exp_path)}")
        print(f"{'━' * 60}")

        # Process each enabled metric
        for metric in metrics:
            if not metric.get("enabled", True):
                print(f"\n  [DISABLED] {metric['name']} — skipping")
                continue

            # Collect metric figures
            figure_list = collect_figures_for_metric(exp_path, metric, seasons)

            # Collect extra sections (appended after metric figures)
            for extra in extra_sections:
                if extra.get("enabled", True):
                    extra_figs = collect_extra_figures(exp_path, extra, seasons)
                    figure_list.extend(extra_figs)

            # Print summary
            print_summary(figure_list, metric["name"])

            # Generate PDF
            if not args.dry_run:
                output_pdf = os.path.join(output_dir, metric["output_pdf"])
                create_pdf_from_figures(figure_list, output_pdf, metric["name"])

    if args.dry_run:
        print(f"\n[DRY RUN] No PDFs were created. Remove --dry-run to generate them.")
    else:
        print(f"\n[DONE] All PDFs saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
