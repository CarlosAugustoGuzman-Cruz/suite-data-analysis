# Calcium Imaging Data Pipeline

This is a Jupyter Lab pipeline for processing two-photon calcium imaging data, integrating file management, image preprocessing, spatial deconvolution, and Suite2p analysis

---

## Overview

This pipeline streamlines the analysis of calcium imaging data (TIFF, HDF5) from raw files through ROI detection and spike inference. It provides an intuitive interface for:

- **File scanning and selection** across nested directories
- **Image preprocessing** with Gaussian/Median filters
- **Spatial deconvolution** using Richardson-Lucy algorithm (experimental)
- **Suite2p integration** for motion correction, ROI detection, and spike extraction
- **Interactive visualization** of detected ROIs and fluorescence traces

---

## Features

### **File Management**
- Recursive scanning for imaging files (`.tif`, `.tiff`, `.h5`, `.hdf5`, and more)
- Interactive file selection with checkbox interface
- Automatic Suite2p folder structure creation
- Support for large files (2.7-30+ GB)

### **Image Preprocessing**
- Real-time frame-by-frame preview
- Gaussian and Median blur filters (1-8 pixel range)
- Batch processing with progress tracking
- Memory-efficient handling of large datasets

### **Spatial Deconvolution** (Experimental)
- Richardson-Lucy deconvolution via [RedLionfish](https://github.com/rosalindfranklininstitute/RedLionfish)
- Automatic PSF generation from metadata or manual parameters
- CPU and GPU acceleration support (Metal on Mac M1/M2)
- Side-by-side preview of original vs. deconvolved frames

### **Suite2p Integration**
- Full parameter configuration with preset defaults
- Motion correction and registration
- Automated ROI detection
- Spike deconvolution
- Neuropil subtraction

### **Visualization**
- Interactive ROI overlay plots (all ROIs, cells, non-cells)
- Fluorescence trace viewer with dropdown ROI selection
- Max projection and mean image displays

---

## Installation

### Requirements
```bash
# Core dependencies
pip install numpy pandas matplotlib plotly scipy ipywidgets ipyfilechooser
pip install suite2p h5py tifffile

# For deconvolution (optional but recommended)
conda install -c conda-forge redlionfish

# For Mac M1/M2 users - use Miniforge
# Download from: https://github.com/conda-forge/miniforge
```

### Recommended Environment Setup (Mac M1/M2)
```bash
# Install Miniforge for Apple Silicon
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh

# Create environment
conda create -n calcium_imaging python=3.11
conda activate calcium_imaging

# Install packages
conda install numpy scipy matplotlib pandas jupyterlab ipywidgets
conda install -c conda-forge scikit-image tifffile h5py suite2p redlionfish
```

---

## Quick Start

### Basic Workflow
```python
import DataPipeLine
from DataPipeLine import (
    data_scan as scan_for_data,
    print_files_found as show_found_files,
    print_files_and_paths as show_files_and_paths,
    file_selection as choose_files_to_analyse,
    run_suite2 as run_suite2,
    gaussian_median_blur_viewer as viewer_gaussian_median_blur,
    run_deconv as image_deconvolution,
    detection_plots,
    graph_traces
)

# 1. Scan for imaging files
scan_for_data()

# Optional: verify found files
show_files_and_paths()

# 2. Select files for analysis
choose_files_to_analyse()

# 3. (Optional) Apply spatial deconvolution
image_deconvolution()

# 4. (Optional) Preview and apply Gaussian/Median filters
viewer_gaussian_median_blur()

# 5. Run Suite2p analysis
list_ops_s2p = run_suite2()

# 6. Visualize results
# Select an ops dictionary from list_ops_s2p
ops_result = list_ops_s2p['ops_your_file_name']
detection_plots(ops_result)
graph_traces(ops_result)
```

---

## Function Reference

### `scan_for_data()`
Recursively scans directories for compatible imaging files. Opens an interactive folder browser

**Supported formats:** `.tif`, `.tiff`, `.h5`, `.hdf5`, `.sbx`, `.nd2`, `.raw`, `.dcimg`, and more

### `show_found_files()` / `show_files_and_paths()`
Display scanned files in console (names only or with full paths)

### `choose_files_to_analyse()`
Interactive checkbox interface to select specific files for processing. Option to create Suite2p folder structure immediately or defer until needed

### `viewer_gaussian_median_blur()`
Load large imaging files and:
- Preview individual frames
- Apply Gaussian blur (sigma 1-8)
- Apply Median blur (size 1-8)
- Save filtered output to disk

**Note:** Only one filter type can be applied at a time

### `image_deconvolution()`
Spatial deconvolution using Richardson-Lucy algorithm:
- Load files from scanned list or manual path
- Generate theoretical PSF or load custom PSF
- Configure iterations (1-50) and method (CPU/GPU)
- Preview first frame before processing
- Batch process entire stack

**PLEASE READ: This part was experimental:** It uses RedLionfish's 3D deconvolution on 2D time-series. For publication-quality results, consider professional software (see below)

### `run_suite2()`
Run Suite2p pipeline with:
- Default or custom parameter configuration
- Motion correction options
- ROI detection settings
- Spike deconvolution
- Multiple output formats (NPY, MAT, NWB)

**Returns:** Dictionary of ops objects for each processed file

### `detection_plots(ops_dict)`
Visualize Suite2p results:
- Max projection
- All detected ROIs
- Cell vs. non-cell classification
- Mean registered image

### `graph_traces(ops_dict)`
Interactive fluorescence trace viewer:
- Cell fluorescence (F)
- Neuropil fluorescence (Fneu)
- Deconvolved spikes
- Dropdown to switch between ROIs

---

## Tips & Troubleshooting

### Jupyter Lab Usage
- **Run cells:** `Shift + Enter`
- **Restart kernel:** Kernel → Restart Kernel and Clear Outputs
- **Check status:** Bottom left corner (Busy/Idle indicator)

### HDF5 Key Discovery (Mac Users)
If you need to find the dataset key in an `.h5` file:
```bash
# In terminal from file's directory
h5dump -n filename.h5
```
Look for the `DATASET` entry (commonly `"Data/Images"` or `"data"`).

### Large File Processing
- Files are processed in batches (typically 100-500 frames)
- Progress indicators show completion percentage
- Ensure sufficient disk space (output ≈ input size)
- For 20+ GB files, processing may take 10-60+ minutes

### Deconvolution Performance
- **GPU method** (Mac M1/M2): Uses Metal, typically 3-10x faster than CPU
- **CPU method**: More compatible, works everywhere
- Start with 10 iterations; more isn't always better (can amplify noise)

---

## Professional Deconvolution Software

For publication-quality spatial deconvolution, consider commercial solutions:

| Software | Best Use | 2P-Specific | Price Range |
|----------|----------|-------------|-------------|
| **[Huygens](https://svi.nl)** | Multiple Systems | Option available | $5k - $20k |
| **[Imaris ClearView](https://imaris.oxinst.com)** | Multi-System | Partial | $10k - $30k |
| **Nikon NIS-Elements** | Nikon Systems | Option available | $5k - $15k |
| **Zeiss ZEN** | Zeiss Systems | Option available | $5k - $15k |

---

## File Outputs

### Suite2p Outputs (per file)
Located in `s2p_{filename}/s2p_save/`:
- `stat.npy` - ROI statistics and spatial footprints
- `F.npy` - Fluorescence traces (nROIs × nFrames)
- `Fneu.npy` - Neuropil fluorescence traces
- `spks.npy` - Deconvolved spike inference
- `iscell.npy` - Cell classification probabilities
- `ops.npy` - Processing parameters and metadata

### Filtered/Deconvolved Outputs
Saved in same directory as source file:
- `gaus_{original_name}` - Gaussian filtered
- `median_{original_name}` - Median filtered
- `deconv_{original_name}` - Deconvolved (experimental)

---

## Citation

If you use this pipeline in your research, please cite:

**Suite2p:**
```
Pachitariu, M., Stringer, C., Dipoppa, M. et al. 
Suite2p: beyond 10,000 neurons with standard two-photon microscopy. 
bioRxiv (2017). doi: 10.1101/061507
```

**RedLionfish (if using deconvolution):**
```
Optinav/RedLionfish: Fast Richardson-Lucy deconvolution
https://github.com/rosalindfranklininstitute/RedLionfish
```

---

## Author

**Carlos A. Guzman-Cruz**  
Email: carguz2002@gmail.com  
Version: 1.0.0  
Date: December 2025

---

## Contributing

This pipeline is designed for ease of use while maintaining essential functionality. To request additional features or report issues, please contact me!

---

## Acknowledgments

- Suite2p team 
- RedLionfish developers at Rosalind Franklin Institute
- Open-source Python scientific computing community
