"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the post-processing and evaluation pipeline for the downscaling project.
"""


import os
import glob
from PIL import Image

def get_images_from_folder(base_dir, region, metric, period_folder):
    """
    Finds all PNG files in a given region/metric/period folder.
    Returns a sorted list of file paths.
    """
    # e.g., results/scenarios_comparison/2006-01-01_2020-12-31/north/test/bias/figures/all/*.png
    path_pattern = os.path.join(base_dir, region, 'test', metric, 'figures', period_folder, '*.png')
    files = glob.glob(path_pattern)
    return sorted(files)

def create_pdf(output_path, image_paths):
    """
    Combines a list of image paths into a single PDF.
    """
    if not image_paths:
        print(f"Warning: No images found for {output_path}. Skipping.")
        return False
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error opening {path}: {e}")
            
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=100.0
        )
        print(f"Created: {output_path} with {len(images)} pages.")
        return True
    return False

def main():
    base_dir = "results/scenarios_comparison/2006-01-01_2020-12-31"
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist.")
        return
        
    # Get all subregions (directories directly under base_dir)
    regions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    mean_metrics = ['bias', 'rmse', 'correlation']
    extreme_metrics = ['cdd', 'r95_nbEvents_freq']
    
    # Target periods in order. Note: Annual figures are stored under 'all'
    periods = [('Annual', 'all'), ('DJF', 'DJF'), ('JJA', 'JJA')]
    
    for region in regions:
        print(f"\nProcessing region: {region}")
        region_dir = os.path.join(base_dir, region)
        
        # 1. Mean Metrics PDF
        mean_image_paths = []
        for metric in mean_metrics:
            for period_name, period_folder in periods:
                paths = get_images_from_folder(base_dir, region, metric, period_folder)
                mean_image_paths.extend(paths)
                
        mean_pdf_path = os.path.join(region_dir, f"{region}_meanmetrics_allandseasons.pdf")
        create_pdf(mean_pdf_path, mean_image_paths)
        
        # 2. Extreme Metrics PDF
        extreme_image_paths = []
        for metric in extreme_metrics:
            for period_name, period_folder in periods:
                paths = get_images_from_folder(base_dir, region, metric, period_folder)
                extreme_image_paths.extend(paths)
                
        extreme_pdf_path = os.path.join(region_dir, f"{region}_extremesmetrics_allandseasons.pdf")
        create_pdf(extreme_pdf_path, extreme_image_paths)

if __name__ == "__main__":
    main()
