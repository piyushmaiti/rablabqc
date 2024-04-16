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
from freesurferlut import FreeSurferColorLUT

lut_parser = FreeSurferColorLUT()

freesurfer_lut = {
    "hippocampus": [17, 53],
    "cerebellum": [7, 8, 46, 47],
    "superiorparietal": [1029, 2029],    
    "cingulate": [1002, 1026, 1023, 1010, 2002, 2026, 2023, 2010],
}

def fs_roi_mask(threed_image_array, roi=None, label_indices=None):
    """
    Generate a mask for the specified region of interest (ROI) based on FreeSurfer LUT.
    Parameters
    ----------
    threed_image_array : np.ndarray
        3D numpy array representing the aparc+aseg image from FreeSurfer
    
    roi : str
        Region of interest (ROI) name.
    
    label_indices : list of int
        List of label indices for the ROI.

    Returns
    -------
    mask : np.ndarray
        3D numpy array representing the mask for the specified ROI containing 1s and 0s.

    Usage
    -----
    mask = fs_roi_mask(threed_image_array, roi='hippocampus')
    mask = fs_roi_mask(threed_image_array, label_indices=[17, 53])

    """
    mask = np.zeros_like(threed_image_array, dtype=np.uint8)

    if roi is not None and label_indices is not None:
        raise ValueError("Only one of roi or label_indices should be specified.")
    
    elif roi is not None and label_indices is None:
        for value in freesurfer_lut[roi]:
            mask[threed_image_array == value] = 1
        return mask
    
    elif label_indices is not None and roi is None:
        # Check if the label_indices are valid and present in the threed_image_array
        if not all([value in threed_image_array for value in label_indices]):
            raise ValueError(f"The ROI with the label indices {label_indices} is not available in the QC FreeSurfer LUT Dictionary.")
        else:
            for value in label_indices:
                mask[threed_image_array == value] = 1
            return mask

    else:
        raise ValueError(f"The ROI {roi} is not available in the QC FreeSurfer LUT Dictionary.")
class SliceSelector:
    def __init__(self, aparc):
        self.aparc = aparc

    def _select_slices(self, axis, roi=None, label_indices=None, percentile=None, num_slices=None):
        if roi is not None and label_indices is not None:
            raise ValueError("Only one of roi or label_indices should be specified.")
        
        elif roi is not None and label_indices is None:
            mask = fs_roi_mask(self.aparc, roi=roi)

        elif label_indices is not None and roi is None:
            mask = fs_roi_mask(self.aparc, label_indices=label_indices)

        # Setting the axis to 0/sagittal/x or 1/coronal/y or 2/axial/z
        axis_slices = np.where(np.any(mask, axis=tuple(set(range(3)) - {axis})))[0]
        
        if percentile is not None and num_slices is not None:
            raise ValueError("Please specify either a percentile value or the number of slices, not both.")
        
        elif percentile is not None:
            selected_slice = np.percentile(axis_slices, percentile*100)
            return int(selected_slice)
        
        elif num_slices is not None:
            print(" Work in progress for the opttion num_slices, please check back later.")
            #return [int(slice) for slice in slice_range]
        
        else:
            raise ValueError("Please specify either a percentile value or the number of slices.")

    def select_slices(self, axis, roi=None, label_indices=None, percentile=None, num_slices=None):
        if isinstance(axis, str):
            _axis = axis.lower()[0]
            if _axis in ("x", "s"):
                print("Selecting sagittal slice/s")
                axis = 0
            elif _axis in ("y", "c"):
                print("Selecting coronal slice/s")
                axis = 1
            elif _axis in ("z", "a"):
                print("Selecting axial slice/s")
                axis = 2
            else:
                raise ValueError(f"axis {axis} not recognized")
        return self._select_slices(axis, roi, label_indices, percentile, num_slices)
    
    def select_leads_slices(self):
        # Selecting 6 axial slices. One with 50 percentile to the cerebellum, and one with 50 percentile to the superiorparietal.
        # Equally spaced between the start, 50 percentile to the cerebellum, 50 percentile to the superiorparietal, and the end.
        cerebellum_slice_1 = self._select_slices(axis=2, roi='cerebellum', percentile=0.25)
        cerebellum_slice_2 = self._select_slices(axis=2, roi='cerebellum', percentile=0.5)
        superiorparietal_slices = self._select_slices(axis=2, roi='superiorparietal', percentile=0.5)
        cerebral_white_matter_2 = self._select_slices(axis=2, label_indices = [2, 41], percentile=0.5)
        cerebral_white_matter_3 = self._select_slices(axis=2, label_indices = [2, 41], percentile=0.75)
        cerebral_white_matter_4 = self._select_slices(axis=2, label_indices = [2, 41], percentile=0.95)

        axial_slices = [cerebellum_slice_1, cerebellum_slice_2, superiorparietal_slices, cerebral_white_matter_2, cerebral_white_matter_3, cerebral_white_matter_4]
        # Reordering the slices in ascending order
        axial_slices.sort()

        # Selecting 2 coronal slices. First slice with 50 percentile to the cerebellum and the second slice with 50 percentile to the hippocampus.
        cerebellum_slice = self._select_slices(axis=1, roi='cerebellum', percentile=0.5)
        hippocampus_slice = self._select_slices(axis=1, roi='hippocampus', percentile=0.5)

        coronal_slices = [cerebellum_slice, hippocampus_slice]

        # Selecting 2 sagittal slices. One with 50 percentile to the left cingulate and 50 percentile to the left hippocampus.
        left_cingulate_slice = self._select_slices(axis=0, label_indices = [1002, 1026, 1023, 1010], percentile=0.5)
        left_hippocampus_slice = self._select_slices(axis=0, label_indices = [17], percentile=0.5)

        sagittal_slices = [left_cingulate_slice, left_hippocampus_slice]

        return axial_slices, coronal_slices, sagittal_slices



