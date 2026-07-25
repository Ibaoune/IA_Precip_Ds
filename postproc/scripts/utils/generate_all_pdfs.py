"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""

import os
import glob
from PIL import Image

def create_pdf(base_dir, output_filename="all_figures_summary.pdf"):
    # Find all PNGs recursively in base_dir
    search_pattern = os.path.join(base_dir, '**', '*.png')
    image_files = glob.glob(search_pattern, recursive=True)
    
    # Sort files to have a logical order
    image_files.sort()
    
    if not image_files:
        print(f"No PNG files found in {base_dir}.")
        return
        
    print(f"Found {len(image_files)} PNG files in {base_dir}. Creating PDF...")
    
    images = []
    first_image = None
    
    for file in image_files:
        try:
            img = Image.open(file).convert('RGB')
            if first_image is None:
                first_image = img
            else:
                images.append(img)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if first_image is not None:
        output_path = os.path.join(base_dir, output_filename)
        first_image.save(output_path, save_all=True, append_images=images)
        print(f"Successfully saved {output_path} with {len(images) + 1} pages.")
    else:
        print(f"Failed to load any images in {base_dir}.")

def main():
    # Portable path calculation relative to the script location
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
    print(f"Scanning results directory: {results_dir}")
    
    # Find all inference directories
    inference_dirs = glob.glob(os.path.join(results_dir, 'inference_*'))
    inference_dirs.sort()
    for inference_dir in inference_dirs:
        if os.path.isdir(inference_dir):
            target_dir = os.path.join(inference_dir, '1979-01-01_2014-12-31', 'allmorr', 'test')
            if os.path.exists(target_dir):
                print(f"\nProcessing {inference_dir}...")
                create_pdf(target_dir)
            else:
                print(f"\nTarget directory {target_dir} does not exist. Skipping.")

if __name__ == '__main__':
    main()
