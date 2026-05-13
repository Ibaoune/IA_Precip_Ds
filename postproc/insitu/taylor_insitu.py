import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF

class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """
    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5)):
        self.refstd = refstd
        
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi/2, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1
        )

        if fig is None:
            fig = plt.figure()
            
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("bottom")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=10, label=label)
        t = np.linspace(0, np.pi/2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l
        
    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, np.pi/2))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours


excel_file = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/insitu/ABH_stations_plusziz_and_abhbc1.xlsx"
BASE_PATH = Path("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/elaabar/out_models/datasets")

# Model paths
paths = {
    "mswep": Path("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data/mswep_1979_2020.nc"),
    "glm": BASE_PATH / "glm/glm.nc",
    "unet": BASE_PATH / "unet/unet.nc",
    "vit": BASE_PATH / "vit/vit.nc",
    "cnn": BASE_PATH / "cnn/cnn.nc",
}

stations = ["CASABLANCA", "AGADIR", "FES"]

print(f"[INFO] Loading observations from: {excel_file}")
df = pd.read_excel(excel_file)
df["Date"] = pd.to_datetime(df["Date"])
df = df[(df["Date"] >= "2006-01-01") & (df["Date"] <= "2020-12-31")]
df["month"] = df["Date"].dt.month

print("[INFO] Loading NetCDF datasets...")
datasets = {}
for name, path in paths.items():
    print(f"       -> Loading {name}...")
    ds = xr.open_dataset(path)
    ds["time"] = pd.to_datetime(ds["time"].values)
    datasets[name] = ds

# Common colors
colors = {
    "mswep": "black",
    "glm": "green",
    "unet": "darkblue",
    "vit": "red",
    "cnn": "lightblue"
}

markers = {
    "mswep": "o",
    "glm": "s",
    "unet": "D",
    "vit": "^",
    "cnn": "v"
}

fig = plt.figure(figsize=(15, 6))

for i, station in enumerate(stations):
    print(f"[INFO] Processing station: {station}")
    st = df[df["Station"] == station].copy()
    st = st.sort_values("Date")

    lat = st["Latitude"].iloc[0]
    lon = st["Longitude"].iloc[0]

    obs_series = st.set_index("Date")["Precipitation"]
    # Let's compute metrics on monthly means to get clearer taylor diagrams, 
    # as daily precipitation is very noisy and correlation is usually low.
    obs_monthly = obs_series.resample("M").mean()

    # Create Taylor diagram for this station
    # Subplot position
    rect = 131 + i
    
    # Reference standard deviation
    ref_std = obs_monthly.std()
    
    # Initialize diagram
    dia = TaylorDiagram(ref_std, fig=fig, rect=rect, label='OBS', srange=(0, 2.5))
    
    for name, ds in datasets.items():
        point = ds.sel(lat=lat, lon=lon, method="nearest")
        var_name = "precipitation" if "precipitation" in point.data_vars else "precip"
        
        sim = pd.Series(
            point[var_name].values,
            index=pd.to_datetime(point["time"].values)
        )
        sim = sim.reindex(obs_series.index)
        sim_monthly = sim.resample("M").mean()
        
        # Calculate metrics
        merged = pd.concat([obs_monthly, sim_monthly], axis=1).dropna()
        merged.columns = ["obs", "sim"]
        
        if len(merged) > 0:
            stddev = merged["sim"].std()
            corrcoef = np.corrcoef(merged["obs"], merged["sim"])[0, 1]
            
            dia.add_sample(stddev, corrcoef, 
                           marker=markers[name], ms=10, ls='', 
                           mfc=colors[name], mec=colors[name], 
                           label=name.upper())

    # Add RMS contours
    contours = dia.add_contours(levels=5, colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

    # Add title
    dia._ax.set_title(station, pad=20)
    
    # Add legend to the first plot
    if i == 0:
        fig.legend(dia.samplePoints, 
                   [p.get_label() for p in dia.samplePoints], 
                   numpoints=1, prop=dict(size='small'), loc='upper right')

plt.tight_layout()
out_file = "taylor_diagram_insitu.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved Taylor diagram to {out_file}")
