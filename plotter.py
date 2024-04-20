
import cv2
import numpy as np
from processing import ImageProcessor

class QCImageGenerator:
    """ 
    QCImageGenerator is a class designed to generate images combining images from the axial, sagittal, and coronal planes.
    The function "generate_qc_images" generates the images as arrays that can be used to plot images using any plotting library.
    There are two modes of operation:
    1. If an overlay image is provided, the function returns two images in the form of arrays with the overlay image combined with the underlay image.
    2. If no overlay image is provided, the function generates images in the form with only the underlay image.
    
    """

    def __init__(self, underlay_img, select_axial_slices, select_sagittal_slices, select_coronal_slices,
                 height_padding=5, width_padding=5, 
                 mask_lower_threshold=None, mask_upper_threshold=None,
                 overlay_img=None, crop_neck=None):
        
        """
        Initializes a QCImageGenerator object with the provided parameters.

        Parameters:
        ----------
            underlay_img (numpy.ndarray): The base image for example T1w, T2w, or FLAIR image or structural image.
            select_axial_slices (list): A list of axial slice numbers to include in the generated images.
            select_sagittal_slices (list): A list of sagittal slice numbers to include in the generated images.
            select_coronal_slices (list): A list of coronal slice numbers to include in the generated images.

            height_padding (int, optional): Padding value to add to the height of each slice. Default is 5.
            width_padding (int, optional): Padding value to add to the width of each slice. Default is 5.

            mask_lower_threshold (int, optional): Lower threshold value for masking the overlay images. Default is None.
                                                Use the mask_lower_threshold to set threshold values for data like the raparc image or the Tissue Probability Maps.
            mask_upper_threshold (int, optional): Upper threshold value for masking the overlay images. Default is None.
                                                Use the mask_upper_threshold to set threshold values for data like the raparc image or the Tissue Probability Maps.
            
            overlay_img (numpy.ndarray, optional): Optional overlay image data to be combined with the underlay image.
                                                    The overlay image can be the raparc + aseg image or the Tissue Probability Maps.
                                                     If None, only underlay images will be processed and generated. Default is None.

            crop_neck (numpy.ndarray, optional): Optional array used for cropping neck regions from images.
                                                   If provided, it should match the shape of the underlay_img. Default is None.
        """

        # Initialization of attributes
        self.underlay_img = underlay_img
        self.overlay_img = overlay_img
        self.select_axial_slices = select_axial_slices
        self.select_sagittal_slices = select_sagittal_slices
        self.select_coronal_slices = select_coronal_slices
        self.height_padding = height_padding
        self.width_padding = width_padding
        self.mask_lower_threshold = mask_lower_threshold
        self.mask_upper_threshold = mask_upper_threshold
        self.crop_neck = crop_neck
        
        self.all_slices = select_axial_slices + select_sagittal_slices + select_coronal_slices
        
        # Calculate the maximum height of images in all slices
        self.max_height = max(self.calculate_max_height())
    
    
    ###############################################################################
    # Functions for Padding the images for plotting
    ###############################################################################

    def calculate_max_height(self):
        max_height = []
        for slice_number in self.all_slices:
            if slice_number in self.select_axial_slices:
                img_slice = self.underlay_img[:, :, slice_number]
            elif slice_number in self.select_sagittal_slices:
                img_slice = self.underlay_img[slice_number, :, :]
            elif slice_number in self.select_coronal_slices:
                img_slice = self.underlay_img[:, slice_number, :]
            
            padded_slice = ImageProcessor.brain_padding(img_slice, height_padding=self.height_padding, width_padding=self.width_padding)
            max_height.append(padded_slice.shape[0])
        
        return max_height
    
    def pad_image(self, image):
        pad_size = self.max_height - image.shape[0]
        pad_top = pad_size // 2
        pad_bottom = pad_size - pad_top
        return np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        
    def cropped_neck(self, image_to_crop, raparc_img_slice, neck_height_padding=50, neck_width_padding=60):

        image_array = raparc_img_slice.copy()
        # Converting the image to uint8
        image_array_u8 = np.uint8(image_array)

        # Find the contours of the brain using opencv findContours
        contours, _ = cv2.findContours(image_array_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest bounding box which is the brain bounding box. The bounding box with the largest area is the brain
        area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > area:
                area = w * h
                x_max, y_max, w_max, h_max = x, y, w, h
        
        x_max -= neck_width_padding # left
        y_max -= neck_height_padding # top
        w_max += 2 * neck_width_padding # right
        h_max += 1 * neck_height_padding # Note: this h_max is different from the h_max in the processing.py
        
        # Check if the padding is within the image boundaries
        x_max = max(0, x_max)
        y_max = max(0, y_max)
        w_max = min(image_array.shape[1] - x_max, w_max)
        h_max = min(image_array.shape[0] - y_max, h_max)
        
        cropped_image = image_to_crop[y_max:y_max + h_max, x_max:x_max + w_max]
        
        return cropped_image


    def process_slices(self):
        combined_underlay_images = []
        combined_overlay_images = []
        
        for slice_number in self.all_slices:
            if self.crop_neck is not None:
                if slice_number in self.select_axial_slices:
                    underlay, overlay = ImageProcessor.brain_padding(self.underlay_img[:, :, slice_number], self.overlay_img[:, :, slice_number],
                                                                    height_padding=self.height_padding, width_padding=self.width_padding)
                    # Axial does not need to be cropped
                elif slice_number in self.select_sagittal_slices:
                    underlay, overlay = ImageProcessor.brain_padding(self.underlay_img[slice_number, :, :], self.overlay_img[slice_number, :, :],
                                                                    height_padding=self.height_padding, width_padding=self.width_padding)
                    
                    underlay = self.cropped_neck(underlay, self.crop_neck[slice_number, :,:])
                    overlay = self.cropped_neck(overlay, self.crop_neck[slice_number, :,:])

                elif slice_number in self.select_coronal_slices:
                    underlay, overlay = ImageProcessor.brain_padding(self.underlay_img[:, slice_number, :], self.overlay_img[:, slice_number, :],
                                                                    height_padding=self.height_padding, width_padding=self.width_padding)
                    
                    underlay = self.cropped_neck(underlay, self.crop_neck[:, slice_number,:])
                    overlay = self.cropped_neck(overlay, self.crop_neck[:, slice_number,:])

            else:

                if slice_number in self.select_axial_slices:
                    underlay, overlay = ImageProcessor.brain_padding(self.underlay_img[:, :, slice_number], self.overlay_img[:, :, slice_number],
                                                                    height_padding=self.height_padding, width_padding=self.width_padding)
                elif slice_number in self.select_sagittal_slices:
                    underlay, overlay = ImageProcessor.brain_padding(self.underlay_img[slice_number, :, :], self.overlay_img[slice_number, :, :],
                                                                    height_padding=self.height_padding, width_padding=self.width_padding)
                elif slice_number in self.select_coronal_slices:
                    underlay, overlay = ImageProcessor.brain_padding(self.underlay_img[:, slice_number, :], self.overlay_img[:, slice_number, :],
                                                                    height_padding=self.height_padding, width_padding=self.width_padding)
                
            # Pad the images
            underlay_padded = self.pad_image(underlay)
            overlay_padded = self.pad_image(overlay)
            
            # Mask the overlay image if thresholds are provided
            if self.mask_lower_threshold is not None and self.mask_upper_threshold is not None:
                overlay_masked = ImageProcessor.mask_image(overlay_padded, lower_threshold=self.mask_lower_threshold, upper_threshold=self.mask_upper_threshold)
            else:
                overlay_masked = overlay_padded

            # Append images to lists
            combined_underlay_images.append(underlay_padded)
            combined_overlay_images.append(overlay_masked)
        
        return np.hstack(combined_underlay_images), np.hstack(combined_overlay_images)
    
    
    def generate_qc_images(self):
        if self.overlay_img is not None:
            return self.process_slices()
        
        else:
            combined_underlay_images = []
            if self.crop_neck is not None:
                for slice_number in self.all_slices:
                    if slice_number in self.select_axial_slices:
                        underlay = ImageProcessor.brain_padding(self.underlay_img[:, :, slice_number], height_padding=self.height_padding, width_padding=self.width_padding)
                        # Axial does not need to be cropped
                    elif slice_number in self.select_sagittal_slices:
                        underlay = ImageProcessor.brain_padding(self.underlay_img[slice_number, :, :], height_padding=self.height_padding, width_padding=self.width_padding)
                        underlay = self.cropped_neck(underlay, self.crop_neck[slice_number, :,:])
                    elif slice_number in self.select_coronal_slices:
                        underlay = ImageProcessor.brain_padding(self.underlay_img[:, slice_number, :], height_padding=self.height_padding, width_padding=self.width_padding)
                        underlay = self.cropped_neck(underlay, self.crop_neck[:, slice_number,:])
    
                    # Pad the image
                    underlay_padded = self.pad_image(underlay)
                    combined_underlay_images.append(underlay_padded)
            

            else:
                for slice_number in self.all_slices:
                    if slice_number in self.select_axial_slices:
                        underlay = ImageProcessor.brain_padding(self.underlay_img[:, :, slice_number], height_padding=self.height_padding, width_padding=self.width_padding)
                    elif slice_number in self.select_sagittal_slices:
                        underlay = ImageProcessor.brain_padding(self.underlay_img[slice_number, :, :], height_padding=self.height_padding, width_padding=self.width_padding)
                    elif slice_number in self.select_coronal_slices:
                        underlay = ImageProcessor.brain_padding(self.underlay_img[:, slice_number, :], height_padding=self.height_padding, width_padding=self.width_padding)
                    
                    # Pad the image
                    underlay_padded = self.pad_image(underlay)
                    combined_underlay_images.append(underlay_padded)
            
            return np.hstack(combined_underlay_images)
