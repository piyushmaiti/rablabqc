import numpy as np
import matplotlib.pyplot as plt
from processing import ImageProcessor

def pad_image(image, max_height):
    """
    Padding the image for equal image height to avoid any errors occurring while concatenating the images.
    Parameters
    ----------
    image : np.ndarray
        2D image array.

    max_height : int
        Maximum height of images in all slices.

    Returns
    -------
    np.ndarray
        Padded 2D image array.
    
    """
    pad_size = max_height - image.shape[0]
    pad_top = pad_size // 2
    pad_bottom = pad_size - pad_top
    # Pad the image using a single function call
    return np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

def generate_qc_images(underlay_img, select_axial_slices, select_sagittal_slices, select_coronal_slices, height_padding=5, width_padding=5, mask_lower_threshold = None, mask_upper_threshold=None, overlay_img=None):
    # Combine all selected slices into one list
    all_slices = select_axial_slices + select_sagittal_slices + select_coronal_slices
    
    # Calculate the maximum height of images in all slices
    max_height = max(underlay_img[:, :, slice_number].shape[0] for slice_number in all_slices)
    
    # Create lists to hold each image
    combined_underlay_images = []

    if overlay_img is not None:
        combined_overlay_images = []
        # Loop through all slices and process them
        for slice_number in all_slices:
            # Determine slice dimension (axial, sagittal, or coronal)
            if slice_number in select_axial_slices:
                underlay, overlay = ImageProcessor.brain_padding(underlay_img[:, :, slice_number], overlay_img[:, :, slice_number], height_padding=height_padding, width_padding=width_padding)
            elif slice_number in select_sagittal_slices:
                underlay, overlay = ImageProcessor.brain_padding(underlay_img[slice_number, :, :], overlay_img[slice_number, :, :], height_padding=height_padding, width_padding=width_padding)
            elif slice_number in select_coronal_slices:
                underlay, overlay = ImageProcessor.brain_padding(underlay_img[:, slice_number, :], overlay_img[:, slice_number, :], height_padding=height_padding, width_padding=width_padding)
            
            # Padding the underlay and overlay images to the same height
            underlay_padded = pad_image(underlay, max_height)
            overlay_padded = pad_image(overlay, max_height)
            
            # Mask the overlay image if thresholds are provided
            if mask_lower_threshold is not None and mask_upper_threshold is not None:
                overlay_masked = ImageProcessor.mask_image(overlay_padded, lower_threshold=mask_lower_threshold, upper_threshold=mask_upper_threshold)
            else:
                overlay_masked = overlay_padded

            # Append images to lists
            combined_underlay_images.append(underlay_padded)
            combined_overlay_images.append(overlay_masked)
        
        # Combine all images in the list
        return np.hstack(combined_underlay_images), np.hstack(combined_overlay_images)
    
    else:
        # Loop through all slices and process them
        for slice_number in all_slices:
            # Determine slice dimension (axial, sagittal, or coronal)
            if slice_number in select_axial_slices:
                underlay = ImageProcessor.brain_padding(underlay_img[:, :, slice_number], height_padding=height_padding, width_padding=width_padding)
            elif slice_number in select_sagittal_slices:
                underlay = ImageProcessor.brain_padding(underlay_img[slice_number, :, :], height_padding=height_padding, width_padding=width_padding)
            elif slice_number in select_coronal_slices:
                underlay = ImageProcessor.brain_padding(underlay_img[:, slice_number, :], height_padding=height_padding, width_padding=width_padding)
            
            # Pad the image
            underlay_padded = pad_image(underlay, max_height)
            # Append the image to the list
            combined_underlay_images.append(underlay_padded)
        
        # Combine all images in the list
        return np.hstack(combined_underlay_images)
