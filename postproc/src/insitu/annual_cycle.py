import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

excel_file = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/insitu/ABH_stations_plusziz_and_abhbc1.xlsx"


BASE_PATH = Path(
    "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/elaabar/out_models/datasets"
)

# Model paths
paths = {
    "mswep": Path("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data/mswep_1979_2020.nc"),
    "glm": BASE_PATH / "glm/glm.nc",
    "unet": BASE_PATH / "unet/unet.nc",
    "vit": BASE_PATH / "vit/vit.nc",
    "cnn": BASE_PATH / "cnn/cnn.nc",
}

stations = ["CASABLANCA", "AGADIR", "FES"]
months = np.arange(1, 13)

# ==========================================================
# LOAD OBS
# ==========================================================
print(f"[INFO] Loading observations from: {excel_file}")
print("[INFO] Please wait, loading large Excel files can take a few minutes...")
df = pd.read_excel(excel_file)
print(f"[INFO] Successfully loaded {len(df)} rows from Excel.")

df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= "2006-01-01") & (df["Date"] <= "2020-12-31")]
df["month"] = df["Date"].dt.month
print("[INFO] Filtered observations to 2006-2020.")

# ==========================================================
# LOAD MODELS
# ==========================================================
print("[INFO] Loading NetCDF datasets...")
datasets = {}

for name, path in paths.items():
    print(f"       -> Loading {name} from {path.name}...")
    ds = xr.open_dataset(path)
    ds["time"] = pd.to_datetime(ds["time"].values)
    datasets[name] = ds

# ==========================================================
# FIGURE
# ==========================================================
print("[INFO] Initializing figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

# ==========================================================
# LOOP STATIONS
# ==========================================================

for i, station in enumerate(stations):
    print(f"[INFO] Processing station {i+1}/{len(stations)}: {station}")
    ax = axes[i]

    st = df[df["Station"] == station].copy()
    st = st.sort_values("Date")

    lat = st["Latitude"].iloc[0]
    lon = st["Longitude"].iloc[0]

    # ======================================================
    # OBS CLIMATOLOGY
    # ======================================================

    obs_clim = st.groupby("month")["Precipitation"].mean().reindex(months)

    ax.plot(months, obs_clim.values, label="OBS", color="black", linewidth=3)

    # ======================================================
    # METRICS STORAGE
    # ======================================================

    metric_text = []

    obs_series = st.set_index("Date")["Precipitation"]

    # ======================================================
    # MODELS
    # ======================================================

    for name, ds in datasets.items():
        point = ds.sel(lat=lat, lon=lon, method="nearest")

        var_name = "precipitation" if "precipitation" in point.data_vars else "precip"
        sim = pd.Series(
            point[var_name].values,
            index=pd.to_datetime(point["time"].values)
        )

        # ================= ALIGNMENT FIX =================
        sim = sim.reindex(obs_series.index)

        # climatology
        sim_df = pd.DataFrame({
            "value": sim.values,
            "month": sim.index.month
        })

        sim_clim = sim_df.groupby("month")["value"].mean().reindex(months)

        # ONLY plot if valid
        if not np.all(np.isnan(sim_clim.values)):
            ax.plot(months, sim_clim.values, label=name)

        # ================= METRICS =================

        merged = pd.concat([obs_series, sim], axis=1)
        merged.columns = ["obs", "sim"]
        merged = merged.dropna()

        if len(merged) > 0:
            rmse = np.sqrt(np.mean((merged["obs"] - merged["sim"])**2))
            bias = np.mean(merged["sim"] - merged["obs"])

            metric_text.append(f"{name}: RMSE={rmse:.2f} | Bias={bias:.2f}")

    # ======================================================
    # STYLE
    # ======================================================

    ax.set_title(station)
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Precipitation")
    ax.grid(True)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        # after plotting all subplots
        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        fontsize=10
              )

    # metrics box
    ax.text(
        0.02, 0.95,
        "\n".join(metric_text),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7)
    )

# ==========================================================
# (No empty panels needed for 3 stations)
# ==========================================================



# ==========================================================
# SAVE
# ==========================================================
print("[INFO] Saving plot to annual_cycle_FIXED.png...")
plt.tight_layout()
plt.savefig("annual_cycle_FIXED.png", dpi=300, bbox_inches="tight")
print("[INFO] Done!")
# plt.show()
