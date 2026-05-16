import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix 'cd ../..'
    content = content.replace('cd ../..', 'cd ../../..')
    
    # Fix '../../configs'
    content = content.replace('../../configs', '../../../configs')
    
    # Fix '../../train.py'
    content = content.replace('../../train.py', '../../../train.py')
    
    # Fix '../../eval.py'
    content = content.replace('../../eval.py', '../../../eval.py')

    # Remove 'logs/' from SBATCH output to put logs directly in unet/ as per user "inside unet directory"
    # Actually, let's keep logs/ and just create the directory. But let's remove it if we want it directly in unet/.
    # user said: "I want to have logs inside unet directory". Let's just create the logs directory.
    
    with open(filepath, 'w') as f:
        f.write(content)

for model in ['unet', 'vit', 'cnn', 'glm']:
    for filepath in glob.glob(f'{model}/*.sh'):
        fix_file(filepath)
    for filepath in glob.glob(f'{model}/*.py'):
        fix_file(filepath)
