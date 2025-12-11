"""
Filename: DataPipeLineV1.py

Author: Carlos A. Guzman-Cruz
Date: December 2025
Version: 1.0
Description:
This files servers to aid in incorporating file scans, file selection, image visualization, 
image pre-processing, and suite-2p. Main purpose of this is to allow for fast work flows 
and still remain easy for any scienctist to use these tools. Many of the functions have been 
reduced to the most important/relevant features needed for current project. If you want to add 
more features feel free to reach out to me and I can help in adding more features.
"""


__author__ = "Carlos A. Guzman-Cruz"
__email__ = "carguz2002@gmail.com"
__version__ = "1.0.0"



import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
from scipy.special import binom
import fnmatch
import math
import plotly.io as pio
import json
from ipyfilechooser import FileChooser 
import suite2p
from pathlib import Path
from suite2p.run_s2p import run_s2p
from suite2p.default_ops import default_ops
import h5py
import tifffile
from scipy.ndimage import gaussian_filter, median_filter



h5py_key = "Data/Images"
suite_2p_outputs = {}      # {'OPS_data_file_name': suite2pOutputdictionary}
data_file_paths = {}       # {'data_file_name': 'path_to_data'}
select_files_names = []    # ['selected_data_file_name_1','selected_data_file_name_2' , ...]
selected_files_raw = {} 
                           # { 'data_file_name': { 'data_path' : 'path_to_data',
                           #                       'home_directory_path' : 'S2p_data_file_name',
                           #                       'save_path': 's2p_sv_data_file_name',
                           #                       'fast_disk_path': 's2p_fd_data_file_name'}}



def data_scan():
    global data_file_paths

    PRIORITY_EXTS = ('.tif', '.tiff', '.h5', '.hdf5')
    OTHER_EXTS = ('.binary', '.bruker', '.sbx', '.movie',
                  '.nd2', '.mesoscan', '.raw', '.dcimg')
    PRIORITY_EXTS = tuple(ext.lower() for ext in PRIORITY_EXTS)
    OTHER_EXTS = tuple(ext.lower() for ext in OTHER_EXTS)
    ALL_EXTS = PRIORITY_EXTS + OTHER_EXTS

    def scan_folder(root_dir: str):
        global data_file_paths
        data_file_paths.clear()  
        def walk_and_add(target_exts):
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for filename in sorted(filenames):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in target_exts:
                        full_path = os.path.join(dirpath, filename)
                        base_name = os.path.splitext(filename)[0]
                        # handle possible duplicate names
                        key = base_name
                        suffix = 1
                        while key in data_file_paths:
                            key = f"{base_name}__{suffix}"
                            suffix += 1

                        data_file_paths[key] = os.path.abspath(full_path)

        walk_and_add(PRIORITY_EXTS)
        remaining_exts = tuple(ext for ext in ALL_EXTS if ext not in PRIORITY_EXTS)
        walk_and_add(remaining_exts)

    try:
        chooser = FileChooser(os.getcwd())
        chooser.title = "<b>Select a folder that contains your imaging data</b>"
        chooser.show_only_dirs = True  
        out = widgets.Output()
        last_selected_dir = {"path": None}

        def on_select(chooser_obj):
            """Callback when the user presses the 'Select' button."""
            with out:
                clear_output()
                selected_dir = chooser_obj.selected_path
                if not selected_dir or not os.path.isdir(selected_dir):
                    print("Please select a valid directory.")
                    return

                last_selected_dir["path"] = selected_dir
                scan_folder(selected_dir)

                print(f"Scanning directory:\n  {selected_dir}")
                print(f"Found {len(data_file_paths)} data file(s).")

        def on_rerun(_):
            """Re-scan the last selected directory."""
            with out:
                clear_output()
                selected_dir = last_selected_dir["path"]
                if not selected_dir or not os.path.isdir(selected_dir):
                    print("No folder cached yet. Please select a folder first.")
                    return

                scan_folder(selected_dir)
                print(f"Re-scanning directory:\n  {selected_dir}")
                print(f"Found {len(data_file_paths)} data file(s).")

        chooser.register_callback(on_select)

        rerun_button = widgets.Button(
            description="Re-Run",
            button_style="warning",
            icon="repeat",
            tooltip="Re-scan the last selected folder for new files"
        )

        rerun_button.on_click(on_rerun)

        ui = widgets.VBox([chooser, rerun_button, out])
        display(ui)

    except ImportError:
        dir_text = widgets.Text(
            value=os.getcwd(),
            description="Folder:",
            layout=widgets.Layout(width="80%")
        )
        scan_button = widgets.Button(
            description="Scan folder",
            button_style='primary'
        )
        rerun_button = widgets.Button(
            description="Re-Run",
            button_style="warning",
            icon="repeat",
            tooltip="Re-scan the same folder for new files"
        )
        out = widgets.Output()

        def perform_scan(selected_dir):
            scan_folder(selected_dir)
            print(f"Scanning directory:\n  {selected_dir}")
            print(f"Found {len(data_file_paths)} data file(s).")
            print("You can inspect them via `print_files_found()` "
                  "or by using `helper.data_file_paths`.")

        def on_click(_):
            with out:
                clear_output()
                selected_dir = dir_text.value.strip()
                if not os.path.isdir(selected_dir):
                    print(f"'{selected_dir}' is not a valid directory.")
                    return
                perform_scan(selected_dir)

        def on_rerun(_):
            with out:
                clear_output()
                selected_dir = dir_text.value.strip()
                if not os.path.isdir(selected_dir):
                    print(f"'{selected_dir}' is not a valid directory.")
                    return
                perform_scan(selected_dir)

        scan_button.on_click(on_click)
        rerun_button.on_click(on_rerun)

        buttons = widgets.HBox([scan_button, rerun_button])
        ui = widgets.VBox([dir_text, buttons, out])
        display(ui)




def print_files_found():
    global data_file_paths 
    for fileName in data_file_paths.keys():
        print(fileName)
        
def print_files_and_paths():
    global data_file_paths 
    #print(json.dumps(data_file_paths, indent=4))  
    for files in data_file_paths:
        print("File name: %s,   Path: %s"%(files, data_file_paths[files]))   



def file_selection():
    global data_file_paths, select_files_names, selected_files_raw

    if not data_file_paths:
        print("No data files found. Please run data_scan() first")
        return

    label = widgets.HTML("<b>Select the files to prep before running suite2p on:</b>")

    checkboxes = []
    for file_name in sorted(data_file_paths.keys()):
        path = data_file_paths[file_name]
        ext = os.path.splitext(path)[1].lower()
        if ext:
            desc = f"{file_name}{ext}"
        else:
            desc = file_name

        cb = widgets.Checkbox(
            value=False,
            description=desc,
            indent=False
        )
        cb._file_key = file_name
        checkboxes.append(cb)

    make_s2p_folders = widgets.Checkbox(
        value=True,
        description="Make Suite2p folders",
        indent=False
    )

    save_button = widgets.Button(
        description="save selection",
        button_style="success"
    )

    out = widgets.Output()

    def on_save_clicked(_):
        global select_files_names, selected_files_raw
        with out:
            clear_output()

            selected_names = [cb._file_key for cb in checkboxes if cb.value]

            select_files_names.clear()
            selected_files_raw.clear()

            if not selected_names:
                print("No files selected.")
                return

            select_files_names.extend(selected_names)

            for data_file_name in selected_names:
                data_path = data_file_paths.get(data_file_name)
                if data_path is None:
                    print(f"Warning: '{data_file_name}' not found in data_file_paths. Skipping.")
                    continue

                home_directory_path = os.path.dirname(data_path)

                s2p_root = os.path.join(home_directory_path, f"s2p_{data_file_name}")
                save_path = os.path.join(s2p_root, "s2p_save")
                fast_disk_path = os.path.join(s2p_root, "s2p_fd")

                if make_s2p_folders.value:
                    os.makedirs(s2p_root, exist_ok=True)
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(fast_disk_path, exist_ok=True)

                selected_files_raw[data_file_name] = {
                    'data_path': os.path.abspath(data_path),
                    'home_directory_path': os.path.abspath(home_directory_path),
                    'save_path': os.path.abspath(save_path),
                    'fast_disk_path': os.path.abspath(fast_disk_path)
                }

            print("Saved selection for the following file(s):")
            for name in selected_files_raw.keys():
                print(f"  - {name}")

            if make_s2p_folders.value:
                print("Suite2p folders were created for the selected file(s).")
            else:
                print("Suite2p folders were NOT created. They can be created later when needed.")

    save_button.on_click(on_save_clicked)

    files_box = widgets.VBox(checkboxes)
    ui = widgets.VBox([label, files_box, make_s2p_folders, save_button, out])
    display(ui)



def run_suite2():
    global selected_files_raw, suite_2p_outputs, h5py_key

    try:
        from suite2p.run_s2p import run_s2p
        try:
            from suite2p.default_ops import default_ops
        except Exception:
            import suite2p
            default_ops = suite2p.default_ops
    except ImportError:
        print("suite2p is not installed in this environment. Please install it before running run_suite2().")
        return suite_2p_outputs

    if not selected_files_raw:
        print("No files selected. Please run file_selection() first.")
        return suite_2p_outputs


    mode_selector = widgets.ToggleButtons(
        options=[("Use defaults", "default"), ("Custom", "custom")],
        description="Settings mode:",
        style={"description_width": "initial"}
    )

    fs_widget = widgets.FloatText(
        value=3.0,
        description="fs",
        description_tooltip="Sampling rate per plane (Hz). Used in deconvolution and timing.",
        style={"description_width": "initial"}
    )

    h5py_key_widget = widgets.Text(
        value=h5py_key,
        description="h5py_key",
        description_tooltip="Dataset key inside the HDF5 file that contains the imaging data.",
        style={"description_width": "initial"}
    )


    
    nplanes_w = widgets.IntText(
        value=1,
        description="nplanes",
        description_tooltip="# planes in the imaging stack. Wrong value will scramble planes.",
        style={"description_width": "initial"}
    )
    nchannels_w = widgets.IntText(
        value=1,
        description="nchannels",
        description_tooltip="# imaging channels per frame.",
        style={"description_width": "initial"}
    )
    functional_chan_w = widgets.IntText(
        value=1,
        description="functional_chan",
        description_tooltip="Which channel to detect ROIs from (1-based).",
        style={"description_width": "initial"}
    )
    tau_w = widgets.FloatText(
        value=1.0,
        description="tau",
        description_tooltip="Approximate calcium decay time constant (seconds). Shapes deconvolution kernel.",
        style={"description_width": "initial"}
    )


    
    preclassify_w = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.1,
        description="preclassify",
        description_tooltip="Probability threshold for pre-classifying ROIs as non-cells before extraction. 0.0 disables.",
        continuous_update=False,
        style={"description_width": "initial"}
    )
    save_mat_w = widgets.Checkbox(
        value=False,
        description="save_mat",
        description_tooltip="Save results in Fall.mat (MATLAB).",
        indent=False
    )
    save_NWB_w = widgets.Checkbox(
        value=False,
        description="save_NWB",
        description_tooltip="Save results as ophys.nwb (NWB format).",
        indent=False
    )
    combined_w = widgets.Checkbox(
        value=True,
        description="combined",
        description_tooltip="Combine results across planes into a combined folder.",
        indent=False
    )
    aspect_w = widgets.FloatText(
        value=1.0,
        description="aspect",
        description_tooltip="Aspect ratio (um/pixel X / um/pixel Y). Affects GUI display only.",
        style={"description_width": "initial"}
    )
    reg_tif_w = widgets.Checkbox(
        value=False,
        description="reg_tif",
        description_tooltip="Write registered binaries out as TIFFs (can be large).",
        indent=False
    )
    reg_tif_chan2_w = widgets.Checkbox(
        value=False,
        description="reg_tif_chan2",
        description_tooltip="Write registered TIFFs for channel 2 as well.",
        indent=False
    )
    delete_bin_w = widgets.Checkbox(
        value=False,
        description="delete_bin",
        description_tooltip="Delete registered binary after processing to save disk space.",
        indent=False
    )
    move_bin_w = widgets.Checkbox(
        value=False,
        description="move_bin",
        description_tooltip="Move binary from fast_disk to save_path0 when finished.",
        indent=False
    )


    
    do_registration_w = widgets.Checkbox(
        value=False,
        description="do_registration",
        description_tooltip="Whether to run motion correction.",
        indent=False
    )
    align_by_chan_w = widgets.IntText(
        value=1,
        description="align_by_chan",
        description_tooltip="Channel used for registration (1-based).",
        style={"description_width": "initial"}
    )
    nimg_init_w = widgets.IntText(
        value=300,
        description="nimg_init",
        description_tooltip="# of initial frames to build reference image.",
        style={"description_width": "initial"}
    )
    batch_size_w = widgets.IntText(
        value=500,
        description="batch_size",
        description_tooltip="# of frames per registration batch (larger = faster but more RAM).",
        style={"description_width": "initial"}
    )
    smooth_sigma_w = widgets.FloatText(
        value=1.15,
        description="smooth_sigma",
        description_tooltip="Spatial Gaussian smoothing (pixels) before phase correlation.",
        style={"description_width": "initial"}
    )
    smooth_sigma_time_w = widgets.FloatText(
        value=0.0,
        description="smooth_sigma_time",
        description_tooltip="Temporal smoothing (frames) before registration.",
        style={"description_width": "initial"}
    )
    maxregshift_w = widgets.FloatText(
        value=0.1,
        description="maxregshift",
        description_tooltip="Max rigid shift as fraction of FOV size (e.g., 0.1 = 10%).",
        style={"description_width": "initial"}
    )
    th_badframes_w = widgets.FloatText(
        value=1.0,
        description="th_badframes",
        description_tooltip="Threshold for excluding bad frames during registration (0 disables).",
        style={"description_width": "initial"}
    )
    keep_movie_raw_w = widgets.Checkbox(
        value=False,
        description="keep_movie_raw",
        description_tooltip="Keep unregistered binary as well as registered one.",
        indent=False
    )
    two_step_registration_w = widgets.Checkbox(
        value=False,
        description="two_step_registration",
        description_tooltip="Do a coarse then fine registration (for low SNR data).",
        indent=False
    )

    
    nonrigid_w = widgets.Checkbox(
        value=False,
        description="nonrigid",
        description_tooltip="Perform non-rigid (piecewise) registration.",
        indent=False
    )

    
    # detection
    roidetect_w = widgets.Checkbox(
        value=True,
        description="roidetect",
        description_tooltip="Run ROI detection and extraction.",
        indent=False
    )
    sparse_mode_w = widgets.Checkbox(
        value=True,
        description="sparse_mode",
        description_tooltip="Sparse-mode detection (faster, good when only some cells).",
        indent=False
    )
    denoise_w = widgets.Checkbox(
        value=True,
        description="denoise",
        description_tooltip="Denoise binned movie before ROI detection.",
        indent=False
    )
    spatial_scale_w = widgets.IntSlider(
        value=4,
        min=0,
        max=4,
        step=1,
        description="spatial_scale",
        description_tooltip="Cell size: 0=auto, 1=~6px, 2=~12px, 3=~24px, 4=~48px.",
        continuous_update=False,
        style={"description_width": "initial"}
    )
    connected_w = widgets.Checkbox(
        value=True,
        description="connected",
        description_tooltip="Force ROIs to be single connected component (good for somas).",
        indent=False
    )
    threshold_scaling_w = widgets.FloatText(
        value=1.0,
        description="threshold_scaling",
        description_tooltip="Multiplier on detection threshold; higher = fewer ROIs.",
        style={"description_width": "initial"}
    )
    max_overlap_w = widgets.FloatText(
        value=0.35,
        description="max_overlap",
        description_tooltip="Max fractional overlap between ROIs before discarding one.",
        style={"description_width": "initial"}
    )
    max_iterations_w = widgets.IntText(
        value=25,
        description="max_iterations",
        description_tooltip="Max iterations for ROI detection (higher finds fainter cells).",
        style={"description_width": "initial"}
    )
    high_pass_w = widgets.IntText(
        value=60,
        description="high_pass",
        description_tooltip="Window (frames) for running-mean subtraction before detection.",
        style={"description_width": "initial"}
    )
    spatial_hp_detect_w = widgets.IntText(
        value=30,
        description="spatial_hp_detect",
        description_tooltip="Filter size (pixels) for high-pass filtering before detection.",
        style={"description_width": "initial"}
    )

    # anatomical detection 
    anatomical_only_w = widgets.IntSlider(
        value=0,
        min=0,
        max=4,
        step=1,
        description="anatomical_only",
        description_tooltip=">0 uses Cellpose on anatomical images (1–4 = different images).",
        continuous_update=False,
        style={"description_width": "initial"}
    )

    # Extraction / Neuropil / Deconvolution 
    neuropil_extract_w = widgets.Checkbox(
        value=True,
        description="neuropil_extract",
        description_tooltip="Estimate and subtract neuropil signal around each ROI.",
        indent=False
    )
    allow_overlap_w = widgets.Checkbox(
        value=False,
        description="allow_overlap",
        description_tooltip="Allow overlapping ROI pixels when extracting traces.",
        indent=False
    )
    inner_neuropil_radius_w = widgets.IntText(
        value=4,
        description="inner_neuropil_radius",
        description_tooltip="Gap (pixels) between ROI border and start of neuropil ring.",
        style={"description_width": "initial"}
    )
    min_neuropil_pixels_w = widgets.IntText(
        value=500,
        description="min_neuropil_pixels",
        description_tooltip="Minimum # of pixels in neuropil annulus.",
        style={"description_width": "initial"}
    )
    soma_crop_w = widgets.Checkbox(
        value=True,
        description="soma_crop",
        description_tooltip="Crop large ROIs to soma-like shapes for classification.",
        indent=False
    )
    spikedetect_w = widgets.Checkbox(
        value=True,
        description="spikedetect",
        description_tooltip="Run spike deconvolution on extracted traces.",
        indent=False
    )
    win_baseline_w = widgets.FloatText(
        value=60.0,
        description="win_baseline",
        description_tooltip="Window length (seconds) for baseline estimation.",
        style={"description_width": "initial"}
    )
    sig_baseline_w = widgets.FloatText(
        value=10.0,
        description="sig_baseline",
        description_tooltip="Smoothing (seconds) for baseline filter.",
        style={"description_width": "initial"}
    )
    neucoeff_w = widgets.FloatText(
        value=0.7,
        description="neucoeff",
        description_tooltip="Neuropil scaling factor: F_corrected = F - neucoeff * Fneu.",
        style={"description_width": "initial"}
    )

    # widgets into sections
    main_box = widgets.VBox([nplanes_w, nchannels_w, functional_chan_w, tau_w])
    output_box = widgets.VBox([
        preclassify_w, save_mat_w, save_NWB_w, combined_w,
        aspect_w, reg_tif_w, reg_tif_chan2_w, delete_bin_w, move_bin_w
    ])
    registration_box = widgets.VBox([
        do_registration_w, align_by_chan_w, nimg_init_w, batch_size_w,
        smooth_sigma_w, smooth_sigma_time_w, maxregshift_w, th_badframes_w,
        keep_movie_raw_w, two_step_registration_w
    ])
    nonrigid_box = widgets.VBox([nonrigid_w])
    functional_box = widgets.VBox([
        roidetect_w, sparse_mode_w, denoise_w, spatial_scale_w, connected_w,
        threshold_scaling_w, max_overlap_w, max_iterations_w,
        high_pass_w, spatial_hp_detect_w
    ])
    anatomical_box = widgets.VBox([anatomical_only_w])
    extraction_box = widgets.VBox([
        neuropil_extract_w, allow_overlap_w, inner_neuropil_radius_w,
        min_neuropil_pixels_w, soma_crop_w, spikedetect_w,
        win_baseline_w, sig_baseline_w, neucoeff_w
    ])

    accordion = widgets.Accordion(children=[
        main_box,
        output_box,
        registration_box,
        nonrigid_box,
        functional_box,
        anatomical_box,
        extraction_box
    ])
    titles = [
        "Main Settings",
        "Output Settings",
        "Registration",
        "Nonrigid",
        "Functional detect",
        "Anatomical Detect",
        "Extraction / Neuropil"
    ]
    for i, t in enumerate(titles):
        accordion.set_title(i, t)

    run_button = widgets.Button(
        description="Run Suite2p on selected files",
        button_style="danger",
        icon="play"
    )
    out = widgets.Output()

    # All widgets that should be disabled when using defaults 
    custom_widgets = [
        nplanes_w, nchannels_w, functional_chan_w, tau_w,
        preclassify_w, save_mat_w, save_NWB_w, combined_w,
        aspect_w, reg_tif_w, reg_tif_chan2_w, delete_bin_w, move_bin_w,
        do_registration_w, align_by_chan_w, nimg_init_w, batch_size_w,
        smooth_sigma_w, smooth_sigma_time_w, maxregshift_w, th_badframes_w,
        keep_movie_raw_w, two_step_registration_w,
        nonrigid_w,
        roidetect_w, sparse_mode_w, denoise_w, spatial_scale_w, connected_w,
        threshold_scaling_w, max_overlap_w, max_iterations_w,
        high_pass_w, spatial_hp_detect_w,
        anatomical_only_w,
        neuropil_extract_w, allow_overlap_w, inner_neuropil_radius_w,
        min_neuropil_pixels_w, soma_crop_w, spikedetect_w,
        win_baseline_w, sig_baseline_w, neucoeff_w
    ]

    def _update_custom_widgets(mode_value):
        use_custom = (mode_value == "custom")
        for w in custom_widgets:
            w.disabled = not use_custom

    def _on_mode_change(change):
        if change["name"] == "value":
            _update_custom_widgets(change["new"])

    mode_selector.observe(_on_mode_change, names="value")
    _update_custom_widgets(mode_selector.value)

    def _build_ops_update_from_widgets(use_custom):
        base = {
            "nplanes": nplanes_w.value,
            "nchannels": nchannels_w.value,
            "functional_chan": functional_chan_w.value,
            "tau": tau_w.value,
            "fs": fs_widget.value,
            "anatomical_only": anatomical_only_w.value,
            "1Preg": False  # default requested = 0
        }

        if not use_custom:
            default_specific = {
                "do_registration": do_registration_w.value,
                "nonrigid": nonrigid_w.value,
                "roidetect": roidetect_w.value,
                "sparse_mode": sparse_mode_w.value,
                "denoise": denoise_w.value,
                "spatial_scale": spatial_scale_w.value,
                "connected": connected_w.value,
                "threshold_scaling": threshold_scaling_w.value,
                "max_overlap": max_overlap_w.value,
                "max_iterations": max_iterations_w.value,
                "high_pass": high_pass_w.value,
                "spatial_hp_detect": spatial_hp_detect_w.value,
                "neuropil_extract": neuropil_extract_w.value,
                "allow_overlap": allow_overlap_w.value,
                "inner_neuropil_radius": inner_neuropil_radius_w.value,
                "min_neuropil_pixels": min_neuropil_pixels_w.value,
                "soma_crop": soma_crop_w.value,
                "spikedetect": spikedetect_w.value,
                "win_baseline": win_baseline_w.value,
                "sig_baseline": sig_baseline_w.value,
                "neucoeff": neucoeff_w.value,
                "save_mat": save_mat_w.value,
                "save_NWB": save_NWB_w.value,
                "reg_tif": reg_tif_w.value,
                "delete_bin": delete_bin_w.value,
                "move_bin": move_bin_w.value,
                "preclassify": preclassify_w.value,
                "combined": combined_w.value,
                "aspect": aspect_w.value,
                "align_by_chan": align_by_chan_w.value,
                "nimg_init": nimg_init_w.value,
                "batch_size": batch_size_w.value,
                "smooth_sigma": smooth_sigma_w.value,
                "smooth_sigma_time": smooth_sigma_time_w.value,
                "th_badframes": th_badframes_w.value,
                "keep_movie_raw": keep_movie_raw_w.value,
                "two_step_registration": two_step_registration_w.value,
                "reg_tif_chan2": reg_tif_chan2_w.value,
            }
            base.update(default_specific)
            return base

        custom = {
            "preclassify": preclassify_w.value,
            "save_mat": save_mat_w.value,
            "save_NWB": save_NWB_w.value,
            "combined": combined_w.value,
            "aspect": aspect_w.value,
            "reg_tif": reg_tif_w.value,
            "reg_tif_chan2": reg_tif_chan2_w.value,
            "delete_bin": delete_bin_w.value,
            "move_bin": move_bin_w.value,
            "do_registration": do_registration_w.value,
            "align_by_chan": align_by_chan_w.value,
            "nimg_init": nimg_init_w.value,
            "batch_size": batch_size_w.value,
            "smooth_sigma": smooth_sigma_w.value,
            "smooth_sigma_time": smooth_sigma_time_w.value,
            "maxregshift": maxregshift_w.value,
            "th_badframes": th_badframes_w.value,
            "keep_movie_raw": keep_movie_raw_w.value,
            "two_step_registration": two_step_registration_w.value,
            "nonrigid": nonrigid_w.value,
            "roidetect": roidetect_w.value,
            "sparse_mode": sparse_mode_w.value,
            "denoise": denoise_w.value,
            "spatial_scale": spatial_scale_w.value,
            "connected": connected_w.value,
            "threshold_scaling": threshold_scaling_w.value,
            "max_overlap": max_overlap_w.value,
            "max_iterations": max_iterations_w.value,
            "high_pass": high_pass_w.value,
            "spatial_hp_detect": spatial_hp_detect_w.value,
            "neuropil_extract": neuropil_extract_w.value,
            "allow_overlap": allow_overlap_w.value,
            "inner_neuropil_radius": inner_neuropil_radius_w.value,
            "min_neuropil_pixels": min_neuropil_pixels_w.value,
            "soma_crop": soma_crop_w.value,
            "spikedetect": spikedetect_w.value,
            "win_baseline": win_baseline_w.value,
            "sig_baseline": sig_baseline_w.value,
            "neucoeff": neucoeff_w.value,
        }
        base.update(custom)
        return base

    def _on_run_clicked(_):
        global suite_2p_outputs, h5py_key
        with out:
            clear_output()
            print("Starting suite2p for", len(selected_files_raw), "file(s)...")

            h5py_key = h5py_key_widget.value

            use_custom = (mode_selector.value == "custom")
            ops_common = _build_ops_update_from_widgets(use_custom)

            results = {}

            for data_name, paths in selected_files_raw.items():
                data_path = paths["data_path"]
                save_path0 = paths["save_path"]
                fast_disk = paths["fast_disk_path"]

                ext = os.path.splitext(data_path)[1].lower()

                if ext in [".h5", ".hdf5"]:
                    db = {
                        "h5py": data_path,
                        "data_path": [],
                        "look_one_level_down": False,
                    }
                    input_ops = {
                        "input_format": "h5",
                        "h5py": data_path,
                        "h5py_key": h5py_key,
                    }
                else:
                    data_dir = os.path.dirname(data_path)
                    db = {
                        "h5py": [],
                        "data_path": [data_dir],
                        "look_one_level_down": False,
                    }
                    input_ops = {
                        "input_format": "tif",
                        "h5py": [],
                    }

                path_ops = {
                    "save_path0": str(save_path0),
                    "fast_disk": str(fast_disk),
                }

                ops = default_ops()
                ops.update(input_ops)
                ops.update(path_ops)
                ops.update(ops_common)

                print(f"\nRunning suite2p on: {data_name}")
                print(f"  data_path   = {data_path}")
                print(f"  save_path0  = {save_path0}")
                print(f"  fast_disk   = {fast_disk}")
                try:
                    ops_end = run_s2p(ops=ops, db=db)
                except Exception as e:
                    print(f"  ERROR while running suite2p on {data_name}: {e}")
                    continue

                key = f"ops_{data_name}"
                results[key] = ops_end
                save_path_report = ops_end.get("save_path0", path_ops["save_path0"])
                print(f"  Finished. Results saved to: {save_path_report}")

            suite_2p_outputs.clear()
            suite_2p_outputs.update(results)

            print("\nAll done. Collected ops dictionaries:")
            for k in suite_2p_outputs.keys():
                print(" ", k)

    run_button.on_click(_on_run_clicked)

    ui = widgets.VBox([
        widgets.HTML("<b>Suite2p run configuration</b>"),
        mode_selector,
        widgets.HTML("<b>Always set:</b>"),
        widgets.HBox([fs_widget, h5py_key_widget]),
        accordion,
        run_button,
        out
    ])
    display(ui)

    return suite_2p_outputs




def gaussian_median_blur_viewer():
    
    global selected_files_raw, h5py_key

    if not selected_files_raw:
        print("No files selected. Please run file_selection() first.")
        return

    file_dropdown = widgets.Dropdown(
        options=list(selected_files_raw.keys()),
        description='Select File:',
        style={'description_width': 'initial'}
    )

    load_btn = widgets.Button(
        description='Load & Inspect',
        button_style='info',
        icon='search'
    )
    
    key_input = widgets.Text(
        value=h5py_key,
        description='H5 Key:',
        placeholder='e.g., data',
        style={'description_width': 'initial'}
    )


    gauss_check = widgets.Checkbox(value=False, description='Gaussian Blur')
    median_check = widgets.Checkbox(value=False, description='Median Blur')
    
    gauss_slider = widgets.IntSlider(value=1, min=1, max=8, description='Sigma:', disabled=True)
    median_slider = widgets.IntSlider(value=1, min=1, max=8, description='Size:', disabled=True)
    
    frame_slider = widgets.IntSlider(value=0, min=0, max=100, step=1, description='Frame:', continuous_update=False)

    save_btn = widgets.Button(
        description='Save Changes to Drive',
        button_style='danger',
        icon='save',
        disabled=True,
        tooltip='Applies filter to the ENTIRE file and saves a copy.'
    )

    img_out = widgets.Output()
    log_out = widgets.Output()

    current_data_source = None 
    current_dataset = None     
    current_file_path = ""
    current_ext = ""
    total_frames = 0
    img_shape = (0,0)


    def toggle_controls(change):
        if change['owner'] == gauss_check and change['new']:
            median_check.value = False
            gauss_slider.disabled = False
            median_slider.disabled = True
        elif change['owner'] == median_check and change['new']:
            gauss_check.value = False
            gauss_slider.disabled = True
            median_slider.disabled = False
        
        if not gauss_check.value and not median_check.value:
            gauss_slider.disabled = True
            median_slider.disabled = True
            
        update_preview()

    def load_file(_):
        nonlocal current_data_source, current_dataset, current_file_path, current_ext, total_frames, img_shape
        
        fname = file_dropdown.value
        path_info = selected_files_raw.get(fname)
        current_file_path = path_info['data_path']
        _, current_ext = os.path.splitext(current_file_path)
        current_ext = current_ext.lower()
        
        img_out.clear_output()
        log_out.clear_output()
        
        try:
            if current_ext in ['.h5', '.hdf5']:
                if current_data_source: 
                    try: current_data_source.close()
                    except: pass
                
                current_data_source = h5py.File(current_file_path, 'r')
                
                key = key_input.value
                if key not in current_data_source:
                    keys = list(current_data_source.keys())
                    with log_out:
                        print(f"Key '{key}' not found. Available keys: {keys}")
                        if len(keys) > 0 and isinstance(current_data_source[keys[0]], h5py.Group):
                             subkeys = list(current_data_source[keys[0]].keys())
                             print(f"Subkeys in {keys[0]}: {subkeys}")
                    return

                current_dataset = current_data_source[key]
                img_shape = current_dataset.shape
                total_frames = img_shape[0]

            elif current_ext in ['.tif', '.tiff']:
                current_data_source = tifffile.memmap(current_file_path)
                current_dataset = current_data_source # functions as array
                img_shape = current_dataset.shape
                total_frames = img_shape[0]
            
            frame_slider.max = total_frames - 1
            frame_slider.value = 0
            save_btn.disabled = False
            
            with log_out:
                print(f"Loaded: {fname}")
                print(f"Shape: {img_shape}")
                print(f"Size: {os.path.getsize(current_file_path)/1e9:.2f} GB")

            update_preview()

        except Exception as e:
            with log_out:
                print(f"Error loading file: {e}")

    def get_processed_frame(idx):
        raw = current_dataset[idx, :, :]
        
        if gauss_check.value:
            sigma = gauss_slider.value
            return gaussian_filter(raw, sigma=sigma)
        elif median_check.value:
            size = median_slider.value
            return median_filter(raw, size=size)
        else:
            return raw

    def update_preview(_=None):
        if current_dataset is None: return
        
        frame_idx = frame_slider.value
        
        img_data = get_processed_frame(frame_idx)

        with img_out:
            clear_output(wait=True)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_data, cmap='gray')
            plt.title(f"Frame {frame_idx} | Filter: {'Gaussian' if gauss_check.value else 'Median' if median_check.value else 'None'}")
            plt.axis('off')
            plt.show()

    def save_changes(_):
        if current_dataset is None: return
        
        if gauss_check.value:
            prefix = "gaus_"
            param = gauss_slider.value
        elif median_check.value:
            prefix = "median_"
            param = median_slider.value
        else:
            with log_out: print("No filter selected. Nothing to save")
            return

        base_dir = os.path.dirname(current_file_path)
        base_name = os.path.basename(current_file_path)
        new_name = f"{prefix}{base_name}"
        save_path = os.path.join(base_dir, new_name)

        with log_out:
            print(f"Starting processing... Target: {new_name}")
            print("DO NOT CLOSE THIS CELL. This may take a while for large files")

        try:
            batch_size = 500 
            n_batches = int(np.ceil(total_frames / batch_size))
            
            if current_ext in ['.h5', '.hdf5']:
                with h5py.File(save_path, 'w') as f_out:
                    dset_out = f_out.create_dataset(
                        key_input.value, 
                        shape=current_dataset.shape, 
                        dtype=current_dataset.dtype,
                        chunks=True # Critical for large files
                    )
                    
                    for i in range(n_batches):
                        start = i * batch_size
                        end = min((i + 1) * batch_size, total_frames)
                        
                        block = current_dataset[start:end]
                        
                        if gauss_check.value:
                            # Apply to each frame in block
                            # Note: filtering 3D block directly implies filtering over time axis too
                            # usually we want frame-by-frame 2D spatial filtering this doesn't work. If you see this 
                            # try to make it work :)
                            processed_block = np.array([gaussian_filter(frame, sigma=param) for frame in block])
                        elif median_check.value:
                            processed_block = np.array([median_filter(frame, size=param) for frame in block])
                        
                        dset_out[start:end] = processed_block
                        
                        if i % 2 == 0:
                            with log_out:
                                print(f"Processed frames {end}/{total_frames}...")

            elif current_ext in ['.tif', '.tiff']:
                with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
                    for i in range(n_batches):
                        start = i * batch_size
                        end = min((i + 1) * batch_size, total_frames)
                        
                        block = current_dataset[start:end]
                        
                        if gauss_check.value:
                            processed_block = np.array([gaussian_filter(frame, sigma=param) for frame in block])
                        elif median_check.value:
                            processed_block = np.array([median_filter(frame, size=param) for frame in block])
                        
                        tif.write(processed_block, contiguous=True)
                        
                        with log_out:
                             print(f"Processed frames {end}/{total_frames}...")

            with log_out:
                print(f"Success! Saved to: {save_path}")
                print("You can select this file in the 'Scan' step if you need it for the next step")

        except Exception as e:
            with log_out:
                print(f"FAILED to save: {e}")
                print("Ensure you have enough disk space. Sorry :(")

    load_btn.on_click(load_file)
    gauss_check.observe(toggle_controls, names='value')
    median_check.observe(toggle_controls, names='value')
    
    gauss_slider.observe(update_preview, names='value')
    median_slider.observe(update_preview, names='value')
    frame_slider.observe(update_preview, names='value')
    
    save_btn.on_click(save_changes)

    controls_ui = widgets.VBox([
        widgets.HTML("<h3>Image Filters & Loader</h3>"),
        widgets.HBox([file_dropdown, key_input]),
        load_btn,
        widgets.HTML("<hr>"),
        widgets.HBox([gauss_check, gauss_slider]),
        widgets.HBox([median_check, median_slider]),
        widgets.HTML("<br><b>Preview Control:</b>"),
        frame_slider,
        widgets.HTML("<hr>"),
        save_btn
    ])
    
    display(widgets.HBox([controls_ui, img_out]))
    display(log_out)



def run_deconv():
    import numpy as np
    import os, re
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import tifffile
    import h5py
    
    try:
        import RedLionfishDeconv as rl
    except ImportError:
        print("The RedLionfishDeconv library is not installed. Please install it (e.g. via pip) to use run_deconv().")
        return

    global data_file_paths, selected_files_raw, h5py_key
    
    mode_select = widgets.ToggleButtons(
        options=[('Use scanned file', 'scanned'), ('Manual path', 'manual')],
        value='scanned',
        description='File Selection:',
        style={'description_width': 'initial'}
    )
    scanned_options = []
    if selected_files_raw:
        scanned_options = list(selected_files_raw.keys())
    elif data_file_paths:
        scanned_options = list(data_file_paths.keys())
    file_dropdown = widgets.Dropdown(
        options=scanned_options,
        description='Select File:',
        style={'description_width': 'initial'}
    )
    manual_path = widgets.Text(
        value='',
        description='File Path:',
        placeholder='Enter full path to file',
        style={'description_width': 'initial'}
    )
    h5_key_input = widgets.Text(
        value=h5py_key,
        description='HDF5 Key:',
        style={'description_width': 'initial'}
    )
    file_select_container = widgets.VBox([file_dropdown])
    
    auto_psf_check = widgets.Checkbox(
        value=True,
        description='Generate theoretical PSF from metadata/defaults',
        indent=False
    )
    wavelength_input = widgets.FloatText(
        value=520.0,
        description='Wavelength (nm):',
        style={'description_width': 'initial'}
    )
    na_input = widgets.FloatText(
        value=1.0,
        description='Numerical Aperture:',
        style={'description_width': 'initial'}
    )
    pixsz_input = widgets.FloatText(
        value=0.5,
        description='Pixel size (µm):',
        style={'description_width': 'initial'}
    )
    
    iter_slider = widgets.IntSlider(
        value=10, min=1, max=50,
        description='Iterations:',
        continuous_update=False,
        style={'description_width': 'initial'}
    )
    method_toggle = widgets.ToggleButtons(
        options=[('CPU (FFT)', 'cpu'), ('GPU (OpenCL)', 'gpu')],
        value='cpu',  # default to CPU for compatibility
        description='Method:',
        style={'description_width': 'initial'}
    )
    import platform
    if platform.system() == 'Darwin':
        method_note = widgets.HTML("<em>Note: macOS detected. GPU mode may not be supported; will fall back to CPU if OpenCL is unavailable.</em>")
    else:
        method_note = widgets.HTML("")  
    
    load_button = widgets.Button(
        description='Load File',
        button_style='primary',
        icon='folder-open'
    )
    preview_button = widgets.Button(
        description='Preview First Frame',
        button_style='info',
        icon='eye'
    )
    save_button = widgets.Button(
        description='Run & Save Deconvolution',
        button_style='danger',
        icon='save'
    )
    
    img_out = widgets.Output()   
    log_out = widgets.Output()    
    
    current_data_source = None   
    current_dataset = None       
    current_file_path = ""
    current_ext = ""
    total_frames = 0
    
    def on_mode_change(change):
        if change['name'] == 'value':
            with log_out:
                clear_output()
            if change['new'] == 'scanned':
                file_select_container.children = [file_dropdown]
            else:
                file_select_container.children = [manual_path]
    mode_select.observe(on_mode_change, names='value')
    
    def load_file_action(_):
        nonlocal current_data_source, current_dataset, current_file_path, current_ext, total_frames
        with log_out:
            clear_output()
        img_out.clear_output()
        
        if mode_select.value == 'scanned':
            if not file_dropdown.value:
                with log_out:
                    print("No file selected. Please select a file from the list or switch to manual mode.")
                return
            fname_key = file_dropdown.value
            if fname_key in selected_files_raw:
                current_file_path = selected_files_raw[fname_key]['data_path']
            elif fname_key in data_file_paths:
                current_file_path = data_file_paths[fname_key]
            else:
                with log_out:
                    print(f"File \"{fname_key}\" not found in scanned list. Please rescan or use manual path.")
                return
        else:
            if not manual_path.value:
                with log_out:
                    print("Please enter a file path.")
                return
            current_file_path = manual_path.value.strip()
            if not os.path.isfile(current_file_path):
                with log_out:
                    print(f"File \"{current_file_path}\" not found. Please check the path.")
                return
        
        current_ext = os.path.splitext(current_file_path)[1].lower()
        try:
            if current_ext in ['.h5', '.hdf5']:
                if current_data_source:
                    try:
                        current_data_source.close()
                    except:
                        pass
                current_data_source = h5py.File(current_file_path, 'r')
                key = h5_key_input.value or h5py_key  # use provided key or default
                if key not in current_data_source:
                    if '/' in key:
                        try:
                            dset = current_data_source
                            for k in key.split('/'):
                                if k:
                                    dset = dset[k]
                            current_dataset = dset  # found via nested path
                        except Exception:
                            with log_out:
                                available = list(current_data_source.keys())
                                print(f"HDF5 dataset '{key}' not found. Available top-level keys: {available}")
                            return
                    else:
                        with log_out:
                            available = list(current_data_source.keys())
                            print(f"HDF5 dataset '{key}' not found. Available top-level keys: {available}")
                        return
                else:
                    current_dataset = current_data_source[key]
                total_frames = current_dataset.shape[0]
            elif current_ext in ['.tif', '.tiff']:
                current_data_source = tifffile.TiffFile(current_file_path)
                try:
                    current_dataset = current_data_source.asarray(memmap=True)  # memmap the whole stack
                except Exception:
                    current_dataset = tifffile.memmap(current_file_path)
                total_frames = current_dataset.shape[0]
            else:
                with log_out:
                    print(f"Unsupported file format '{current_ext}'. Only TIFF(.tif) or HDF5(.h5) are supported.")
                return
        except Exception as e:
            with log_out:
                print(f"Error loading file: {e}")
            return
        
        h5py_key = h5_key_input.value
        
        if auto_psf_check.value and current_ext in ['.tif', '.tiff']:
            try:
                tf = current_data_source  # TiffFile object
                if hasattr(tf, 'ome_metadata') and tf.ome_metadata:
                    ome_xml = tf.ome_metadata
                    match = re.search(r'PhysicalSizeX\s*=\s*"([\d\.]+)"', ome_xml)
                    unit_match = re.search(r'PhysicalSizeXUnit\s*=\s*"([^"]+)"', ome_xml)
                    if match:
                        px_val = float(match.group(1))
                        px_unit = unit_match.group(1) if unit_match else 'µm'
                        if px_unit.lower().startswith('um') or px_unit == 'µm':
                            pixsz_input.value = px_val
                        elif px_unit.lower() == 'nm':
                            pixsz_input.value = px_val / 1000.0  # nm to µm
                        elif px_unit.lower() == 'mm':
                            pixsz_input.value = px_val * 1000.0  # mm to µm
                        na_match = re.search(r'Objective[^>]*NA\s*=\s*"([\d\.]+)"', ome_xml)
                        if na_match:
                            na_input.value = float(na_match.group(1))
                if tf.imagej_metadata:
                    ij = tf.imagej_metadata
                    if 'pixel_width' in ij and 'unit' in ij:
                        unit = ij['unit'].lower()
                        px = float(ij['pixel_width'])
                        if unit in ['um', 'micron', 'micrometer', 'micrometre']:
                            pixsz_input.value = px
                        elif unit == 'nm':
                            pixsz_input.value = px / 1000.0
                        elif unit == 'mm':
                            pixsz_input.value = px * 1000.0
                if tf.micromanager_metadata:
                    mm = tf.micromanager_metadata
                    if 'PixelSize_um' in mm:
                        pixsz_input.value = float(mm['PixelSize_um'])
                    if 'ObjectiveNA' in mm:
                        try:
                            na_input.value = float(mm['ObjectiveNA'])
                        except:
                            pass
            except Exception as e:
                pass
        
        frame_shape = None
        try:
            frame_shape = current_dataset.shape[1:]  # (Y, X)
        except Exception:
            frame_shape = None
        with log_out:
            fname = os.path.basename(current_file_path)
            print(f"Loaded file: {fname}")
            if frame_shape:
                print(f"Data shape: {total_frames} frames of size {frame_shape[0]} x {frame_shape[1]}")
            else:
                print(f"Total frames: {total_frames}")
            try:
                fsize = os.path.getsize(current_file_path) / 1e9
                print(f"File size: {fsize:.2f} GB")
            except:
                pass
            if auto_psf_check.value and current_ext in ['.tif', '.tiff']:
                px_msg = f"{pixsz_input.value} µm"
                na_msg = f"{na_input.value}"
                print(f"PSF parameters (from metadata/default): Pixel size = {px_msg}, NA = {na_msg}, Wavelength = {wavelength_input.value} nm")
            else:
                print(f"PSF parameters: Pixel size = {pixsz_input.value} µm, NA = {na_input.value}, Wavelength = {wavelength_input.value} nm")
        
        update_preview()
    
    def generate_psf_kernel():
        # Get parameters
        lam_nm = wavelength_input.value
        na_val = na_input.value
        px_um = pixsz_input.value
        if lam_nm <= 0 or na_val <= 0 or px_um <= 0:
            # Fall back to some default if any parameter is non-positive
            lam_nm = 500.0
            na_val = max(na_val, 0.8)
            px_um = max(px_um, 0.5)
        # Convert wavelength to µm
        lam_um = lam_nm / 1000.0
        # Estimate lateral PSF FWHM (using ~0.51*lambda/NA as approximation for FWHM:contentReference[oaicite:11]{index=11})
        fwhm_um = 0.51 * lam_um / na_val if na_val > 0 else 0.51 * lam_um
        # Convert to pixels
        fwhm_px = fwhm_um / px_um
        sigma_px = fwhm_px / 2.3548  # convert FWHM to Gaussian sigma
        sigma_px = max(sigma_px, 0.5)  # enforce a minimum sigma in pixels
        # Determine PSF kernel size: take ~6 sigma (3 on each side) or at least 7x7
        radius = int(np.ceil(3 * sigma_px))
        size = max(2*radius + 1, 7)  # ensure it's odd and at least 7
        # Generate 2D Gaussian kernel
        ax = np.arange(-radius, radius+1, 1)
        xx, yy = np.meshgrid(ax, ax)
        r2 = xx**2 + yy**2
        # 2D Gaussian (normalized)
        gauss = np.exp(-0.5 * r2 / (sigma_px**2))
        # Normalize to sum=1
        psf_2d = gauss / np.sum(gauss, dtype=np.float64)
        return psf_2d.astype(np.float32)
    
    def update_preview(_=None):
        img_out.clear_output(wait=True)
        if current_dataset is None:
            return
        frame_idx = 0
        try:
            raw_frame = current_dataset[frame_idx]
        except Exception as e:
            with log_out:
                print(f"Error reading frame {frame_idx}: {e}")
            return
        psf_kernel = generate_psf_kernel()
        data_vol = raw_frame[np.newaxis, ...].astype(np.float32)
        psf_vol = psf_kernel[np.newaxis, ...].astype(np.float32)
        method = method_toggle.value
        if method == 'gpu' and platform.system() == 'Darwin':
            with log_out:
                print("GPU method selected on macOS. If no compatible OpenCL GPU is present, will use CPU instead.")
        try:
            deconv_vol = rl.doRLDeconvolutionFromNpArrays(data_vol, psf_vol, niter=iter_slider.value, method=method)
        except Exception as e:
            # If GPU fails, try CPU automatically
            try:
                deconv_vol = rl.doRLDeconvolutionFromNpArrays(data_vol, psf_vol, niter=iter_slider.value, method='cpu')
                with log_out:
                    print(f"GPU deconvolution failed, used CPU. ({e})")
            except Exception as e2:
                with log_out:
                    print(f"Error in deconvolution: {e2}")
                return
        deconv_frame = deconv_vol[0]  # extract the 2D result
        
        with img_out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(raw_frame, cmap='gray')
            axes[0].set_title("Original (Frame 0)")
            axes[0].axis('off')
            axes[1].imshow(deconv_frame, cmap='gray')
            axes[1].set_title(f"Deconvolved (Frame 0, {iter_slider.value} iter)")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
    
    def save_deconvolution(_):
        if current_dataset is None:
            with log_out:
                print("No data loaded. Please load a file first.")
            return
        base_dir = os.path.dirname(current_file_path)
        base_name = os.path.basename(current_file_path)
        out_name = "deconv_" + base_name
        save_path = os.path.join(base_dir, out_name)
        if os.path.isfile(save_path):
            try:
                os.remove(save_path)
            except:
                # If remove fails (e.g., open file), we'll overwrite via writing mode
                pass
        psf_kernel = generate_psf_kernel()
        psf_vol = psf_kernel[np.newaxis, ...].astype(np.float32)
        n_iterations = iter_slider.value
        method = method_toggle.value
        if method == 'gpu' and platform.system() == 'Darwin':
            with log_out:
                print("Attempting GPU deconvolution on macOS (will fallback to CPU if unsupported).")
        with log_out:
            clear_output()
            print(f"Starting deconvolution of {total_frames} frames. This may take a while...")
            print(f"Saving result to: {out_name}")
        try:
            batch_size = 100  # choose a reasonable batch size to balance memory and speed
            n_batches = int(np.ceil(total_frames / batch_size))
            if current_ext in ['.h5', '.hdf5']:
                f_out = h5py.File(save_path, 'w')
                out_dtype = current_dataset.dtype
                dset_out = f_out.create_dataset(h5_key_input.value or h5py_key,
                                                shape=current_dataset.shape,
                                                dtype=out_dtype,
                                                chunks=True)
                for i in range(n_batches):
                    start = i * batch_size
                    end = min((i+1) * batch_size, total_frames)
                    block = current_dataset[start:end]  # this will be a numpy array (h5py returns numpy on slicing)
                    block = block.astype(np.float32, copy=False)
                    result_block = np.empty_like(block)
                    for j, frame in enumerate(block):
                        data_vol = frame[np.newaxis, ...]  # shape (1,Y,X)
                        try:
                            deconv_vol = rl.doRLDeconvolutionFromNpArrays(data_vol, psf_vol, niter=n_iterations, method=method)
                        except Exception as e:
                            deconv_vol = rl.doRLDeconvolutionFromNpArrays(data_vol, psf_vol, niter=n_iterations, method='cpu')
                        result_block[j] = deconv_vol[0]
                    dset_out[start:end] = result_block.astype(out_dtype)
                    with log_out:
                        done = end
                        print(f"Processed frames {done}/{total_frames}...")
                f_out.close()
            elif current_ext in ['.tif', '.tiff']:
                with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
                    for i in range(n_batches):
                        start = i * batch_size
                        end = min((i+1) * batch_size, total_frames)
                        block = current_dataset[start:end]  # memmap slicing returns numpy array
                        block = block.astype(np.float32, copy=False)
                        result_block = []
                        for frame in block:
                            data_vol = frame[np.newaxis, ...]
                            try:
                                deconv_vol = rl.doRLDeconvolutionFromNpArrays(data_vol, psf_vol, niter=n_iterations, method=method)
                            except Exception as e:
                                deconv_vol = rl.doRLDeconvolutionFromNpArrays(data_vol, psf_vol, niter=n_iterations, method='cpu')
                            result_block.append(deconv_vol[0].astype(current_dataset.dtype))
                        result_block = np.stack(result_block, axis=0)
                        tif.write(result_block, contiguous=True)
                        with log_out:
                            done = end
                            print(f"Processed frames {done}/{total_frames}...")
            else:
                with log_out:
                    print("Unsupported file format for saving results.")
                    return
            with log_out:
                print(f"Deconvolution complete. Saved {total_frames} frames to: {out_name}")
                print("If needed, you can scan this folder again to select the deconvolved file for further analysis.")
        except Exception as e:
            with log_out:
                print(f"Error during deconvolution or saving: {e}")
                print("Deconvolution aborted.")
    
    load_button.on_click(load_file_action)
    preview_button.on_click(update_preview)
    save_button.on_click(save_deconvolution)
    
    file_ui = widgets.VBox([
        widgets.HTML("<h4>Select Input File</h4>"),
        mode_select,
        file_select_container,
        h5_key_input,
        load_button
    ])
    psf_ui = widgets.VBox([
        widgets.HTML("<h4>Configure PSF</h4>"),
        auto_psf_check,
        widgets.HBox([wavelength_input, na_input, pixsz_input])
    ])
    deconv_ui = widgets.VBox([
        widgets.HTML("<h4>Deconvolution Settings</h4>"),
        widgets.HBox([iter_slider, method_toggle]),
        method_note
    ])
    action_ui = widgets.VBox([
        widgets.HTML("<h4>Run & Preview</h4>"),
        widgets.HBox([preview_button, save_button])
    ])
    controls_ui = widgets.VBox([file_ui, psf_ui, deconv_ui, action_ui, log_out])
    display(widgets.HBox([controls_ui, img_out]))



def detection_plots(data_OPS):
    from pathlib import Path
    import numpy as np
    import suite2p
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    save_dir = Path(data_OPS.get("save_path", data_OPS.get("save_path0")))
    stats_file = save_dir / "stat.npy"
    iscell_file = save_dir / "iscell.npy"

    stats = np.load(stats_file, allow_pickle=True)
    iscell = np.load(iscell_file, allow_pickle=True)[:, 0].astype(bool)

    im = suite2p.ROI.stats_dicts_to_3d_array(
        stats,
        Ly=data_OPS["Ly"],
        Lx=data_OPS["Lx"],
        label_id=True
    )
    im[im == 0] = np.nan

    max_proj = data_OPS["max_proj"]
    mean_img = data_OPS.get("meanImg", data_OPS.get("meanImgE", max_proj))

    all_rois = np.nanmax(im, axis=0)

    if (~iscell).any():
        noncell_rois = np.nanmax(im[~iscell], axis=0)
    else:
        noncell_rois = np.full_like(all_rois, np.nan)

    if iscell.any():
        cell_rois = np.nanmax(im[iscell], axis=0)
    else:
        cell_rois = np.full_like(all_rois, np.nan)

    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=[
            "Registered Image, Max Projection",
            "All ROIs Found",
            "All Non-Cell ROIs",
            "All Cell ROIs",
            "Mean Registered Image",
        ]
    )

    fig.add_trace(
        go.Heatmap(
            z=max_proj,
            colorscale="gray",
            showscale=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=all_rois,
            coloraxis="coloraxis",
            showscale=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(
            z=noncell_rois,
            coloraxis="coloraxis",
            showscale=False
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Heatmap(
            z=cell_rois,
            coloraxis="coloraxis",
            showscale=False
        ),
        row=1, col=4
    )

    fig.add_trace(
        go.Heatmap(
            z=mean_img,
            colorscale="gray",
            showscale=False
        ),
        row=1, col=5
    )

    fig.update_layout(
        width=2500,
        height=600,
        title_text="Suite2p Summary Views",
        title_x=0.5,
        coloraxis=dict(colorscale="Jet")
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.show()
    return fig




def graph_traces(data_OPS):
    from pathlib import Path
    import numpy as np
    import plotly.graph_objects as go

    save_dir = Path(data_OPS.get("save_path", data_OPS.get("save_path0")))
    f_path = save_dir / "F.npy"
    fneu_path = save_dir / "Fneu.npy"
    spks_path = save_dir / "spks.npy"

    if not f_path.exists() or not fneu_path.exists() or not spks_path.exists():
        print("F.npy, Fneu.npy, or spks.npy not found in", str(save_dir))
        return None

    f_cells = np.load(f_path)
    f_neuropils = np.load(fneu_path)
    spks = np.load(spks_path)

    if f_cells.ndim != 2 or f_neuropils.ndim != 2 or spks.ndim != 2:
        print("Expected F, Fneu, and spks to be 2D arrays (nROIs, nFrames).")
        return None

    if f_cells.shape != f_neuropils.shape or f_cells.shape != spks.shape:
        print("Shape mismatch between F, Fneu, and spks arrays; cannot plot.")
        return None

    n_cells, n_frames = f_cells.shape

    if n_cells <= 0:
        print("No ROIs found in F.npy.")
        return None

    if n_cells > 20:
        step = max(1, n_cells // 20)
        rois = np.arange(n_cells)[::step]
    else:
        rois = np.arange(n_cells)

    x = np.arange(n_frames)

    fig = go.Figure()

    for idx, roi in enumerate(rois):
        f = f_cells[roi]
        f_neu = f_neuropils[roi]
        sp = spks[roi].astype(float)

        fmax = np.maximum(f.max(), f_neu.max())
        fmin = np.minimum(f.min(), f_neu.min())
        frange = fmax - fmin if fmax > fmin else 1.0

        if sp.max() > 0:
            sp = sp / sp.max() * frange + fmin
        else:
            sp = np.full_like(sp, fmin)

        visible_state = (idx == 0)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=f,
                mode="lines",
                name="Cell Fluorescence",
                legendgroup="cell",
                visible=visible_state
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=f_neu,
                mode="lines",
                name="Neuropil Fluorescence",
                legendgroup="neuropil",
                visible=visible_state
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=sp,
                mode="lines",
                name="Deconvolved",
                legendgroup="deconv",
                visible=visible_state
            )
        )

    buttons = []
    n_rois = len(rois)

    for idx, roi in enumerate(rois):
        vis = []
        for k in range(n_rois):
            is_active = (k == idx)
            vis.extend([is_active, is_active, is_active])
        buttons.append(
            dict(
                label=f"ROI {roi}",
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": f"Fluorescence and Deconvolved Traces for ROI {roi}"
                    },
                ],
            )
        )

    fig.update_layout(
        title="Fluorescence and Deconvolved Traces for Different ROIs",
        xaxis_title="Frame",
        yaxis_title="Fluorescence (a.u.) / scaled spikes",
        width=1200,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        ),
        updatemenus=[
            dict(
                type="dropdown",
                x=0.01,
                y=1.18,
                showactive=True,
                buttons=buttons
            )
        ],
    )

    fig.show()
    return fig


