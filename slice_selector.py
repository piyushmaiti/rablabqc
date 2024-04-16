"""
slice_selector.py

Author: Piyush Maiti
Email: piyush.maiti@ucsf.edu
Date: 2024-04-08
Created at: RabLab, University of California, San Francisco

Description:
This script provides functionality to select slices for axial, sagittal, and coronal views from a 3D volume NIfTI file represented as a numpy array. 
This script is a part of the "RabLab Quality Control" package.

Usage:
- Import this script into your project to utilize the slice selection functionality.
- Ensure that the necessary dependencies (such as numpy and nibabel) are installed.
- Example usage:
    ```python
    from slice_selector import select_axial_slices

    # Load your 3D volume NIfTI file as a numpy array
    nifti_array = load_nifti_as_numpy('your_nifti_file.nii.gz')

    # Select default slices for axial
    axial_slices = select_axial_slices(nifti_array)

    # Anchor points for all slices based on the aparc+aseg atlas

    ```
"""
import numpy as np

freesurfer_lut = {
    "hippocampus": [17, 53],
    "posterior cingulate": [1010, 1022, 1023, 1026, 2023, 3023, 4023],
    "cerebellum": [7, 8, 46, 47],
}

def roi_mask(threed_image_array, roi):
    """
    Generate a mask for the specified region of interest (ROI) based on FreeSurfer LUT.

    Args:
        threed_image_array (np.ndarray): 3D numpy array representing the image.
        roi (str): Region of interest (ROI) name.

    Returns:
        np.ndarray: Mask representing the ROI.
    """

    if roi in freesurfer_lut:
        mask = np.zeros_like(threed_image_array, dtype=np.uint8)
        for value in freesurfer_lut[roi]:
            mask[threed_image_array == value] = 1
        return mask
    else:
        raise ValueError("The specified ROI is not available in the FreeSurfer LUT.")


def select_coronal_slice(threed_image_array, roi=None, percentile=None):
    """
    Select a coronal slice based on ROI: hippocampus and cerebellum.
    1. Percentile form
    2. Maximum area form

    Args:
        threed_image_array (np.ndarray): 3D numpy array representing the image.
        roi (str): Region of interest (ROI) name.
        percentile (float): Percentile value for selecting the slice.

    Returns:
        int: Index of the selected coronal slice.
    """

    if roi is None:
        raise ValueError("Please specify a region of interest (ROI) for selecting the coronal slice.")

    mask = roi_mask(threed_image_array, roi)

    if percentile is not None:
        print(f"Selecting {percentile*100}th percentile slice for {roi}...")
        coronal_indices = np.where(np.any(mask, axis=(0, 2)))[0]
        selected_slice = np.percentile(coronal_indices, percentile*100)
    else:
        print(f"Selecting the coronal slice with the maximum area for {roi}...")
        mask_sum = np.sum(mask, axis=(0, 2))
        selected_slice = np.argmax(mask_sum)

    return int(selected_slice)

def select_two_coronal_slices(threed_image_array, percentile=None):
    """
    Select two coronal slices based on ROI: hippocampus and cerebellum.
    1. Percentile form
    2. Maximum area form

    Args:
        threed_image_array (np.ndarray): 3D numpy array representing the image.
        percentile (float): Percentile value for selecting the slices.

    Returns:
        tuple: Indices of the selected coronal slices.
    """
    if percentile is not None:
        print(f"Selecting the coronal slices with {percentile*100}th percentile for hippocampus and cerebellum...")
        hippocampus_slice = select_coronal_slice(threed_image_array, roi="hippocampus", percentile=percentile)
        cerebellum_slice = select_coronal_slice(threed_image_array, roi="cerebellum", percentile=percentile)
    else:
        print(f"Selecting the coronal slices with the maximum area for hippocampus and cerebellum...")
        hippocampus_slice = select_coronal_slice(threed_image_array, roi="hippocampus")
        cerebellum_slice = select_coronal_slice(threed_image_array, roi="cerebellum")

    return tuple([cerebellum_slice, hippocampus_slice])

def select_sagittal_slice(threed_image_array, roi=None, percentile=None):
    """
    Select the sagittal slice with the maximum area for the specified region of interest (ROI).

    Args:
        threed_image_array (np.ndarray): 3D numpy array representing the image.
        roi (str): Region of interest (ROI) name.
        percentile (float): Percentile value for selecting the slice.

    Returns:
        int: Index of the selected sagittal slice.
    """

    if roi is None:
        raise ValueError("Please specify a region of interest (ROI) for selecting the sagittal slice.")

    mask = roi_mask(threed_image_array, roi)

    if percentile is not None:
        print(f"Selecting {percentile*100}th percentile slice for {roi}...")
        sagittal_indices = np.where(np.any(mask, axis=(0, 1)))[0]
        selected_slice = np.percentile(sagittal_indices, percentile*100)
    else:
        print(f"Selecting the sagittal slice with the maximum area for {roi}...")
        mask_sum = np.sum(mask, axis=(0, 1))
        selected_slice = np.argmax(mask_sum)

    return int(selected_slice)