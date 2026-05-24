# Author: M. El Aabaribaoune (@um6p)
import os
import glob

def add_eval_to_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if "eval.py" not in content:
        # Extract the config file name from the train.py line
        # e.g., "python3 -u train.py configs/unet/tests/test_exp1.yaml"
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if "train.py" in line:
                # Add eval.py right after train.py
                config_path = line.split(' ')[-1]
                new_lines.append(f"python3 -u eval.py {config_path}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))

for i in range(1, 13):
    add_eval_to_file(f"run_test_unet_exp{i}.sh")
