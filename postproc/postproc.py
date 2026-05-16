import os
import yaml
import subprocess
import sys

def run_command(command):
    """Executes a shell command and prints output."""
    print(f"--- Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"!!! Error executing {' '.join(command)}")
    return result.returncode

def main():
    # Set the working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load global config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    os.environ["POSTPROC_MASTER_CONFIG"] = os.path.abspath(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    postproc_cfg = config.get("postproc", {})
    
    # Define metric mappings: (calc_script, plot_script)
    # Relative to postproc/
    metrics_map = {
        "mean": {
            "bias": ("precip/mean/bias/Bias.py", "precip/mean/bias/plot_bias.py"),
            "rmse": ("precip/mean/rmse/rmse.py", "precip/mean/rmse/plot_rmse.py"),
            "correlation": ("precip/mean/correlation/correlation.py", "precip/mean/correlation/plot_corr.py"),
        },
        "extreme": {
            "cdd": ("precip/extreme/cdd/cdd.py", "precip/extreme/cdd/plot_cdd.py"),
            "qqplot": ("precip/extreme/qqplot/qqplot.py", "precip/extreme/qqplot/plot_qqplot.py"),
            "r01": ("precip/extreme/r01/r01.py", "precip/extreme/r01/plot_r01.py"),
            "r99": ("precip/extreme/r99/r99.py", None), # Plotting is integrated in script
            "r95": ("precip/extreme/r95/r95.py", "precip/extreme/r95/plot_r95.py"),
            "r99_freq": ("precip/extreme/r99_freq/r99_nbEvents_freq.py", "precip/extreme/r99_freq/plot_r99_nbEvents_freq.py"),
            "r95_freq": ("precip/extreme/r95_freq/r95_nbEvents_freq.py", "precip/extreme/r95_freq/plot_r95_nbEvents_freq.py"),
            "rocss": ("precip/extreme/rocss/rocss.py", "precip/extreme/rocss/plot_rocss.py"),
        }
    }
    
    print("====================================================")
    print("   Post-Processing Master Runner (postproc.py)")
    print("====================================================")

    regions = config.get("parameters", {}).get("regions", [config.get("parameters", {}).get("region", "allmorr")])

    for region in regions:
        print(f"\n====================================================")
        print(f"🌍 Running for region: {region.upper()}")
        print(f"====================================================")
        os.environ["POSTPROC_REGION"] = region

        for category, category_metrics in metrics_map.items():
            cat_cfg = postproc_cfg.get(category, {})
            if cat_cfg.get("enable", False):
                print(f"\n🚀 Starting {category.upper()} metrics suite for {region.upper()}...")
                for metric_name, scripts in category_metrics.items():
                    if cat_cfg.get(metric_name, False):
                        calc_script, plot_script = scripts
                        print(f"\n🔹 Processing: {metric_name.upper()} ({region.upper()})")
                        
                        plot_only = config.get("parameters", {}).get("plot_only", False)
                        
                        # Run calculation
                        if not plot_only:
                            if calc_script and os.path.exists(calc_script):
                                run_command(["python3", calc_script])
                            elif calc_script:
                                print(f"   ⚠️ Warning: Script not found: {calc_script}")
                        else:
                            print(f"   ⏩ Skipping calculation for {metric_name.upper()} (plot_only=True)")
                        
                        # Run plotting
                        if plot_script and os.path.exists(plot_script):
                            run_command(["python3", plot_script])
                        elif plot_script:
                            print(f"   ⚠️ Warning: Script not found: {plot_script}")
            else:
                print(f"\n⏩ Skipping {category.upper()} metrics (disabled in config).")

    print("\n====================================================")
    print("   Post-Processing Finished for all regions!")
    print("====================================================")

if __name__ == "__main__":
    main()
