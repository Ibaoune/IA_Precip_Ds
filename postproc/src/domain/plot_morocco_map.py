#!/usr/bin/env python3
"""
plot_morocco_map.py
-------------------
Trace une carte géographique du Maroc basée sur des shapefiles :
  - Frontières du Maroc  : morocco_unified_fixed_v2.shp
  - Sous-régions         : north.shp, north_east.shp, east.shp, south.shp
  - Fond blanc / océan coloré (pas de DEM)
  - Grille lat/lon avec étiquettes
  - Légende avec boîtes colorées (coin supérieur gauche, label '(b)')
  - 6 villes marquées (points noirs + nom de la ville)

Auteur  : M. El Aabaribaoune
Date    : 2026-09-02
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader as ShpReader
from cartopy.feature import ShapelyFeature
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS SHAPEFILES
# ─────────────────────────────────────────────────────────────────────────────
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SHP_DIR  = os.path.join(ROOT, "data", "shape_files")
OUT_DIR  = os.path.join(ROOT, "postproc", "results", "domain")

SHP_MOROCCO = os.path.join(SHP_DIR, "morocco_unified_fixed_v2.shp")

REGIONS = [
    {"name": "North-West", "shp": os.path.join(SHP_DIR, "north.shp"),      "color": "#1f77b4"},
    {"name": "North-East", "shp": os.path.join(SHP_DIR, "north_east.shp"), "color": "#ff7f0e"},
    {"name": "East",       "shp": os.path.join(SHP_DIR, "east.shp"),       "color": "#2ca02c"},
    {"name": "South",      "shp": os.path.join(SHP_DIR, "south.shp"),      "color": "#d62728"},
]

# ─────────────────────────────────────────────────────────────────────────────
# EXTENT  [lon_min, lon_max, lat_min, lat_max]
# ─────────────────────────────────────────────────────────────────────────────
EXTENT = [-17.5, 0.0, 21.0, 36.5]

# ─────────────────────────────────────────────────────────────────────────────
# STATIONS (VILLES)
# ─────────────────────────────────────────────────────────────────────────────
STATIONS = [
    {"name": "Tanger",     "lat": 35.75, "lon": -5.83},
    {"name": "Oujda",      "lat": 34.68, "lon": -1.90},
    {"name": "Fès",        "lat": 34.03, "lon": -5.00},
    {"name": "Casablanca", "lat": 33.57, "lon": -7.58},
    {"name": "Marrakech",  "lat": 31.63, "lon": -8.01},
    {"name": "Ouarzazate", "lat": 30.93, "lon": -6.89},
    {"name": "Dakhla",     "lat": 23.68, "lon": -15.93},
]

# (dx, dy, ha, va)
LABEL_CONFIG = {
    "Tanger":     ( 0.20,  0.18, "left",  "bottom"),
    "Oujda":      ( 0.20,  0.15, "left",  "bottom"),
    "Fès":        ( 0.20,  0.20, "left",  "bottom"),  # au-dessus à droite
    "Casablanca": (-0.20,  0.18, "right", "bottom"),  # à gauche du point
    "Marrakech":  ( 0.20,  0.18, "left",  "bottom"),
    "Ouarzazate": ( 0.20, -0.35, "left",  "top"),
    "Dakhla":     ( 0.25,  0.18, "left",  "bottom"),
}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(4.5, 5.5))
    ax  = fig.add_axes([0.07, 0.05, 0.88, 0.88], projection=proj)
    ax.set_extent(EXTENT, crs=proj)

    # ── Fond blanc + océan ───────────────────────────────────────────────────
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#cce6f4", zorder=1)

    # ── Frontières du Maroc (shapefile) ──────────────────────────────────────
    morocco_reader  = ShpReader(SHP_MOROCCO)
    morocco_feature = ShapelyFeature(
        morocco_reader.geometries(),
        proj,
        facecolor="whitesmoke",
        edgecolor="black",
        linewidth=1.4,
        zorder=2,
    )
    ax.add_feature(morocco_feature)

    # ── Sous-régions (shapefiles) ─────────────────────────────────────────────
    for region in REGIONS:
        reader  = ShpReader(region["shp"])
        feature = ShapelyFeature(
            reader.geometries(),
            proj,
            facecolor="none",
            edgecolor=region["color"],
            linewidth=2.2,
            linestyle="-",
            zorder=3,
        )
        ax.add_feature(feature)

    # ── Grille lat/lon ────────────────────────────────────────────────────────
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle="--",
        zorder=4,
    )
    gl.xlocator    = mticker.FixedLocator([-15, -10, -5])
    gl.ylocator    = mticker.FixedLocator([25, 30, 35])
    gl.xformatter  = LONGITUDE_FORMATTER
    gl.yformatter  = LATITUDE_FORMATTER
    gl.top_labels  = False
    gl.right_labels= False
    gl.xlabel_style= {"size": 9}
    gl.ylabel_style= {"size": 9}

    # ── Scatter des villes ────────────────────────────────────────────────────
    for stn in STATIONS:
        ax.scatter(
            stn["lon"], stn["lat"],
            s=15, c="black", marker="o",
            transform=proj, zorder=6,
        )
        cfg = LABEL_CONFIG.get(stn["name"], (0.20, 0.18, "left", "bottom"))
        dx, dy, ha, va = cfg
        ax.text(
            stn["lon"] + dx,
            stn["lat"] + dy,
            stn["name"],
            fontsize=7,
            fontweight="bold",
            transform=proj,
            zorder=7,
            ha=ha,
            va=va,
            color="black",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

    # ── Légende (coin supérieur gauche) ───────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            facecolor="white",
            edgecolor=r["color"],
            linewidth=2.0,
            label=r["name"],
        )
        for r in REGIONS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.88,
        edgecolor="gray",
        prop={"size": 7, "weight": "bold"},
        borderpad=0.7,
        handlelength=2.0,
        handleheight=1.4,
    )



    # ── Sauvegarde ────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "morocco_map.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Carte sauvegardée : {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
