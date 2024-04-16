import numpy as np
import matplotlib.pyplot as plt
from processing import ImageProcessor

#def pad_image(image, max_height):
#    """Pad an image to a specified maximum height, equally padding at the top and bottom."""
#    pad_size = (max_height - image.shape[0]) // 2
#    return np.pad(image, ((pad_size, max_height - image.shape[0] - pad_size), (0, 0)), mode='constant', constant_values=0)

def pad_image(image, max_height):
    """Pad an image to a specified maximum height."""
    pad_size = max_height - image.shape[0]
    pad_top = pad_size // 2
    pad_bottom = pad_size - pad_top
    # Pad the image using a single function call
    return np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

def generate_qc(underlay_image, select_axial_slices, select_sagittal_slices, select_coronal_slices, vmax=140):
    num_subplots = len(select_axial_slices) + len(select_sagittal_slices) + len(select_coronal_slices)

    fig, axs = plt.subplots(1, num_subplots, figsize=(15, 5))

    # Determine max height of images in all slices
    max_height = max(
        max(underlay_image[:, :, slice_number].shape[0] for slice_number in select_axial_slices),
        max(underlay_image[slice_number, :, :].shape[0] for slice_number in select_sagittal_slices),
        max(underlay_image[:, slice_number, :].shape[0] for slice_number in select_coronal_slices)
    )

    # Combine images and plot them
    combined_images = []
    all_slices = [
        (select_axial_slices, lambda s: underlay_image[:, :, s]),
        (select_sagittal_slices, lambda s: underlay_image[s, :, :]),
        (select_coronal_slices, lambda s: underlay_image[:, s, :])
    ]

    for idx, (slices, img_func) in enumerate(all_slices):
        for slice_number in slices:
            image = ImageProcessor.brain_padding(img_func(slice_number), height_padding=0)
            image = pad_image(image, max_height)
            axs[idx].imshow(image, cmap='gray', vmax=vmax)
            axs[idx].axis('off')
            combined_images.append(image)
            idx += 1

    # Combine all images in the first row into a single image
    combined_row = np.concatenate(combined_images, axis=1)
    plt.close(fig)
    
    return combined_row

def generate_qc_with_overlay(nu_img, overlay_img, select_axial_slices, select_sagittal_slices, select_coronal_slices, vmax=140):
    # Combine all selected slices into one list
    all_slices = select_axial_slices + select_sagittal_slices + select_coronal_slices
    
    # Calculate the maximum height of images in all slices
    max_height = max(nu_img[:, :, slice_number].shape[0] for slice_number in all_slices)
    
    # Create lists to hold each image
    combined_images = []
    overlay_images = []
    
    # Loop through all slices and process them
    for slice_number in all_slices:
        # Determine slice dimension (axial, sagittal, or coronal)
        if slice_number in select_axial_slices:
            underlay, overlay = ImageProcessor.brain_padding(nu_img[:, :, slice_number], overlay_img[:, :, slice_number], height_padding=0)
        elif slice_number in select_sagittal_slices:
            underlay, overlay = ImageProcessor.brain_padding(nu_img[slice_number, :, :], overlay_img[slice_number, :, :], height_padding=0)
        elif slice_number in select_coronal_slices:
            underlay, overlay = ImageProcessor.brain_padding(nu_img[:, slice_number, :], overlay_img[:, slice_number, :], height_padding=0)
        
        # Pad and mask the images
        underlay_padded = np.pad(underlay, ((0, max_height - underlay.shape[0]), (0, 0)), mode='constant', constant_values=0)
        overlay_padded = np.pad(overlay, ((0, max_height - overlay.shape[0]), (0, 0)), mode='constant', constant_values=0)
        overlay_masked = ImageProcessor.mask_image(overlay_padded)
        
        # Append images to lists
        combined_images.append(underlay_padded)
        overlay_images.append(overlay_masked)
    
    # Combine all images in the lists
    combined_row = np.hstack(combined_images)
    combined_overlay = np.hstack(overlay_images)
    
    # Create the figure and close it to free up resources
    fig, axs = plt.subplots()
    fig.patch.set_facecolor('black')
    plt.close(fig)
    
    # Return the combined row image data
    return combined_row, combined_overlay

