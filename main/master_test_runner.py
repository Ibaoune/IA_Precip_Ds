# Author: M. El Aabaribaoune (@um6p)
import os
import subprocess
import yaml
from pathlib import Path

def get_save_path(config_path):
    """Predicts the save path based on config content (simplified logic)."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 1. Build Region String
    r = cfg['region']
    region_str = f"region_lat_{r['lat_min']}_{r['lat_max']}_lon_{r['lon_min']}_{r['lon_max']}"
    
    # 2. Build Date Strings
    d = cfg['dates']['train']
    train_dates = f"train_{d['start'].replace('-', '_')}_{d['end'].replace('-', '_')}"
    d_test = cfg['dates']['test']
    test_dates = f"test_{d_test['start'].replace('-', '_')}_{d_test['end'].replace('-', '_')}"
    
    # 3. Active Flags
    t = cfg['training']
    flags = []
    if t.get('LR_scheduler', {}).get('enable'): flags.append("lr_sched")
    if t.get('weight_decay', {}).get('enable'): flags.append("wd")
    if t.get('gradient_clipping', {}).get('enable'): flags.append("gc")
    if t.get('dropout', {}).get('enable'): flags.append("dropout")
    if t.get('scheduler', {}).get('enable'): flags.append(t['scheduler'].get('type', 'cosine'))
    if t.get('group_norm', {}).get('enable'): flags.append("gn")
    
    active_str = "_".join(flags) if flags else "none"
    params_str = f"{t['norm_mode']}_{t['learning_rate']}_{t['loss_type']}_{t['epochs']}ep_{active_str}"
    
    # Results dir from config
    res_dir = cfg['paths']['results_dir']
    exp = cfg['general']['experiment']
    
    return os.path.join(res_dir, exp, region_str, train_dates, test_dates, params_str)

def run_tests(arch, tests_dir, script_name):
    print(f"\n>>> Starting {arch.upper()} Tests...")
    configs = sorted([f for f in os.listdir(tests_dir) if f.endswith('.yaml')])
    results = []

    for cfg_file in configs:
        cfg_path = os.path.join(tests_dir, cfg_file)
        rel_cfg_path = os.path.join("../configs", arch, "tests", cfg_file)
        
        print(f"  [LAUNCHING] {cfg_file}...")
        
        # 1. Update the .sh script
        script_path = os.path.join("scripts", script_name)
        with open(script_path, 'r') as f:
            lines = f.readlines()
        
        with open(script_path, 'w') as f:
            for line in lines:
                if line.strip().startswith("CONFIG="):
                    f.write(f'CONFIG="{rel_cfg_path}"\n')
                elif line.strip().startswith("# CONFIG=") or (line.strip().startswith('CONFIG=') and line.strip() != f'CONFIG="{rel_cfg_path}"'):
                    # Comment out any other CONFIG lines that might have been activated
                    if not line.strip().startswith("#"):
                        f.write("# " + line)
                    else:
                        f.write(line)
                else:
                    f.write(line)
        
        # 2. Run the script
        # We use cwd=scripts because the .sh script uses ../paths
        try:
            start_time = os.times().elapsed
            # Run bash and capture output
            process = subprocess.run(["bash", script_name], cwd="scripts", capture_output=True, text=True)
            duration = os.times().elapsed - start_time
            
            save_path = get_save_path(cfg_path)
            results.append({
                "arch": arch,
                "file": cfg_file,
                "path": save_path,
                "status": "Success" if process.returncode == 0 else "Failed"
            })
            print(f"  [FINISHED] {cfg_file} in {duration:.1f}s (Status: {results[-1]['status']})")
            
        except Exception as e:
            print(f"  [ERROR] Failed to run {cfg_file}: {str(e)}")
            results.append({"arch": arch, "file": cfg_file, "path": "N/A", "status": "Error"})
            
    return results

if __name__ == "__main__":
    all_results = []
    
    # Define tasks
    tasks = [
        ("unet", "configs/unet/tests", "job_test_unet.sh"),
        ("cnn", "configs/cnn/tests", "job_test_cnn.sh"),
        ("glm", "configs/glm/tests", "job_test_glm.sh")
    ]
    
    for arch, tdir, sname in tasks:
        all_results.extend(run_tests(arch, tdir, sname))
    
    # Print Final Table
    print("\n" + "="*80)
    print(f"{'ARCH':<10} | {'CONFIG FILE':<30} | {'SAVE PATH'}")
    print("-"*80)
    for res in all_results:
        print(f"{res['arch']:<10} | {res['file']:<30} | {res['path']}")
    print("="*80)
