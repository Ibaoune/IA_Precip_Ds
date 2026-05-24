# Author: M. El Aabaribaoune (@um6p)
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
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Ensure PYTHONPATH includes the src/ directory so subprocesses can import utils
    src_abs_path = os.path.join(os.getcwd(), "src")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = f"{src_abs_path}:{existing_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = src_abs_path
        
    # Load global config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
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
            "bias": ("src/mean/bias/Bias.py", "src/mean/bias/plot_bias.py"),
            "rmse": ("src/mean/rmse/rmse.py", "src/mean/rmse/plot_rmse.py"),
            "correlation": ("src/mean/correlation/correlation.py", "src/mean/correlation/plot_corr.py"),
        },
        "extreme": {
            "cdd": ("src/extreme/cdd/cdd.py", "src/extreme/cdd/plot_cdd.py"),
            "qqplot": ("src/extreme/qqplot/qqplot.py", "src/extreme/qqplot/plot_qqplot.py"),
            "r01": ("src/extreme/r01/r01.py", "src/extreme/r01/plot_r01.py"),
            "r99": ("src/extreme/r99/r99.py", None), # Plotting is integrated in script
            "r95": ("src/extreme/r95/r95.py", "src/extreme/r95/plot_r95.py"),
            "r99_freq": ("src/extreme/r99_freq/r99_nbEvents_freq.py", "src/extreme/r99_freq/plot_r99_nbEvents_freq.py"),
            "r95_freq": ("src/extreme/r95_freq/r95_nbEvents_freq.py", "src/extreme/r95_freq/plot_r95_nbEvents_freq.py"),
            "rocss": ("src/extreme/rocss/rocss.py", "src/extreme/rocss/plot_rocss.py"),
        }
    }
    
    print("====================================================")
    print("   Post-Processing Master Runner (postproc.py)")
    print("====================================================")

    regions = config.get("parameters", {}).get("regions", [config.get("parameters", {}).get("region", "allmorr")])

    for region in regions:
        print(f"\n====================================================")
        print(f"Running for region: {region.upper()}")
        print(f"====================================================")
        os.environ["POSTPROC_REGION"] = region

        for category, category_metrics in metrics_map.items():
            cat_cfg = postproc_cfg.get(category, {})
            if cat_cfg.get("enable", False):
                print(f"\nStarting {category.upper()} metrics suite for {region.upper()}...")
                for metric_name, scripts in category_metrics.items():
                    if cat_cfg.get(metric_name, False):
                        calc_script, plot_script = scripts
                        print(f"\nProcessing: {metric_name.upper()} ({region.upper()})")
                        
                        plot_only = config.get("parameters", {}).get("plot_only", False)
                        
                        # Run calculation
                        if not plot_only:
                            if calc_script and os.path.exists(calc_script):
                                run_command(["python3", calc_script])
                            elif calc_script:
                                print(f"   Warning: Script not found: {calc_script}")
                        else:
                            print(f"   ⏩ Skipping calculation for {metric_name.upper()} (plot_only=True)")
                        
                        # Run plotting
                        if plot_script and os.path.exists(plot_script):
                            run_command(["python3", plot_script])
                        elif plot_script:
                            print(f"   Warning: Script not found: {plot_script}")
            else:
                print(f"\n⏩ Skipping {category.upper()} metrics (disabled in config).")

    print("\n====================================================")
    print("   Post-Processing Finished for all regions!")
    print("====================================================")

if __name__ == "__main__":
    main()
