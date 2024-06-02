import os
import warnings
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Importing the necessary classes from the rablabqc package
import sys
sys.path.append('/home/mac/pmaiti/Desktop/leads_qc/rablabqc')
from plotter import QCImageGenerator
from processing import ImageProcessor

#  _____________________________________________________________ CUSTOM COLORMAPS _____________________________________________________________ #
# Notes: 
# 1. The custom colormaps are created using the LinearSegmentedColormap class from matplotlib.colors.
# 2. The alpha = 0 is to set the transparency for the zeros in the image.

def create_colormap(color):
    """
    Parameters
    ----------
    color : tuple
        The RGB values for the color.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The custom colormap.
    """
    cmap = LinearSegmentedColormap.from_list('custom', [color, color], N=256)
    cmap.set_under(alpha=0)
    return cmap

cmap_red = create_colormap((1, 0, 0))
cmap_yellow = create_colormap((1, 1, 0))
cmap_blues = create_colormap((0, 0, 1))
cmap_pink = create_colormap((1, 0, 1))

cmap_yellow2 = LinearSegmentedColormap.from_list('custom_color', [(249/255, 195/255, 31/255), (249/255, 195/255, 31/255)], N=256)
cmap_yellow2.set_under(alpha=0)  # Set transparency for zeros
#  ____________________________________________________________ IMAGE THRESHOLDS _____________________________________________________________ #

mri_vmax = 120

# ______________________________________________________________ TEMPLATE SLICES  ____________________________________________________________ #
tpm_file = '/home/mac/pmaiti/Desktop/leads_qc/TPM.nii'
def load_tpm(path, orientation="LAS"):
        """
        Load nifti image with specified orientation
        """
        from nibabel.orientations import io_orientation, axcodes2ornt

        img = nib.load(path)
        img_ornt = io_orientation(img.affine)
        new_ornt = axcodes2ornt(orientation)
        img = img.as_reoriented(img_ornt)
        return img.get_fdata()
tpm_image = load_tpm(tpm_file)[:,:,:,0]

template_axial_slices = [15,23, 34, 45, 56, 67, 78]
template_sagittal_slices = [55,44]
template_coronal_slices = [48,70]

# ______________________________________________________________________________________________________________________________________________ #

class MRIQCplots:
    def __init__(self, nu_img, axial_slices, sagittal_slices, coronal_slices,
                 aparc_img=None, c1_img=None, affine_nu_img=None, warped_nu_img=None, 
                 crop_neck=True):
    
        self.nu_img = nu_img
        self.aparc_img = aparc_img
        self.c1_img = c1_img
        self.affine_nu_img = affine_nu_img
        self.warped_nu_img = warped_nu_img

        self.axial_slices = axial_slices
        self.sagittal_slices = sagittal_slices
        self.coronal_slices = coronal_slices
        
        self.crop_neck = crop_neck

        if self.crop_neck and self.aparc_img is None:
            self.crop_neck = False
            warnings.warn("The aparc_img is not provided. The neck will not be cropped.", UserWarning)

    
    def load_nii(self,path, orientation="LAS"):
        """
        Load nifti image with specified orientation
        """
        from nibabel.orientations import io_orientation, axcodes2ornt

        img = nib.load(path)
        img_ornt = io_orientation(img.affine)
        new_ornt = axcodes2ornt(orientation)
        img = img.as_reoriented(img_ornt)
        return img.get_fdata()

    
    
    def load_images(self):
        """
        Load the provided images and store them as class attributes.
        """
        self.basename = self.nu_img.split('/')[-1].split('_')[0]+'_'+self.nu_img.split('/')[-1].split('_')[1]+'_'+self.nu_img.split('/')[-1].split('_')[2]
        self.nu_img_filename = os.path.basename(self.nu_img)
        self.nu_img = self.load_nii(self.nu_img)
        
        if self.aparc_img is not None:
            self.aparc_img_filename = os.path.basename(self.aparc_img)
            self.aparc_img = self.load_nii(self.aparc_img)
            
        if self.c1_img is not None:
            self.c1_img_filename = os.path.basename(self.c1_img)
            self.c1_img = self.load_nii(self.c1_img)

        if self.affine_nu_img is not None:
            self.affine_nu_img_filename = os.path.basename(self.affine_nu_img)
            self.affine_nu_img = self.load_nii(self.affine_nu_img)

        if self.warped_nu_img is not None:
            self.warped_nu_img_filename = os.path.basename(self.warped_nu_img)
            self.warped_nu_img = self.load_nii(self.warped_nu_img)
    
    # ______________________________________________________________________________________________________________________________________________ #
    # The following functions generate the slices for the provided images. The slices are generated using the QCImageGenerator class from the plotter.py file.
    
    def nu_img_slices(self):
        """
        This function generates the nu_img slices.

        Returns
        -------
        numpy.ndarray : The arrays with the nu_img slices.

        """
        return QCImageGenerator(
            underlay_img=self.nu_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            crop_neck=self.aparc_img if self.crop_neck else None).generate_qc_images()
    
    def nu_img_lines(self):
        """
        This function generates the lines representing the axial, sagittal, and coronal slices on the nu_img.
        Returns
        -------
        numpy.ndarray : The arrays with the lines representing the slices.
        """
        return QCImageGenerator(
            underlay_img=self.nu_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            crop_neck=self.aparc_img if self.crop_neck else None).generate_lines()


    def aparc_img_slices(self):
        """
        1. This function generates the aparc_img slices with the cortical labels thresholded between 1000 and 4000 to be overlaid on the nu_img.
        2. The function uses nu_img as the underlay image as a base to determine the bounding box for the slices so that it is accurately cropped.
        For more information about the cortical labels(and the threshold values), refer to the following link: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
        
        Returns
        -------
        numpy.ndarray : The aparc_img slices.
        """
        _,aparc_img_slices= QCImageGenerator(
            underlay_img=self.nu_img,
            overlay_img=self.aparc_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            mask_lower_threshold=1000, mask_upper_threshold=4000,
            crop_neck=self.aparc_img if self.crop_neck else None).generate_qc_images()
    
        return aparc_img_slices
    
    def generate_subcortical_slices(self):
        """
        This function generates the subcortical regions from the aparc_img. This uses the nu_img as the underlay image to determine the bounding box for the slices so that it is accurately cropped.
        For more information regarding the subcortical regions(and the thresholds applied to get the regions), refer to the following link: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
        Returns
        -------
        numpy.ndarray : The slices with the subcortical regions.
        """
        
        def generate_subcortical(lower_threshold, upper_threshold):
            _, image = QCImageGenerator(
                underlay_img = self.nu_img,
                overlay_img = self.aparc_img,
                select_axial_slices = self.axial_slices,
                select_sagittal_slices = self.sagittal_slices,
                select_coronal_slices = self.coronal_slices,
                mask_lower_threshold = lower_threshold,
                mask_upper_threshold = upper_threshold,
                crop_neck = self.aparc_img if self.crop_neck else None).generate_qc_images()
            return image
        
        # Define threshold ranges for subcortical regions
        threshold_ranges = {
            "left_hippocampus": (17, 18),
            "right_hippocampus": (53, 54),
            "left_amygdala": (18, 19),
            "right_amygdala": (54, 55),
            "left_thalumus": (10, 11),
            "right_thalumus": (49, 50),
            "left_pallidum": (13, 14),
            "right_pallidum": (52, 53),
            "left_putamen": (12, 13),
            "right_putamen": (51, 52),
            "left_caudate": (11, 12),
            "right_caudate": (50, 51)
        }

        # Generate subcortical images
        return {key: generate_subcortical(*threshold) for key, threshold in threshold_ranges.items()}


    def c1_image_slices(self):
        """
        1. This function generates the slices for the c1(spm segmentation of the Tissue Probability Maps) i.e. the Grey matter (containing high densities of unmyelinated neurons).
        2. The c1 has been thresholded between 0.3 and 1 to be overlaid on the nu_img.
        3. The function uses nu_img as the underlay image as a base to determine the bounding box for the slices so that it is accurately cropped.
        
        Returns
        -------
        numpy.ndarray : The c1_img slices.
        """
        _,c1_img_slices = QCImageGenerator(
            underlay_img=self.nu_img,
            overlay_img=self.c1_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            mask_lower_threshold=0.8, #mask_upper_threshold=1,
            crop_neck=self.aparc_img if self.crop_neck else None).generate_qc_images()
        
        return c1_img_slices
    
    

    def affine_nu_img_slices(self):
        """
        """

        affine_nu_image_slices, tpm_img_slices = QCImageGenerator(
            underlay_img=self.affine_nu_img,
            #overlay_img=nib.load('/home/mac/pmaiti/Desktop/leads_qc/rTPM.nii').get_fdata(),
            overlay_img=tpm_image,
            select_axial_slices= template_axial_slices,
            select_sagittal_slices= template_sagittal_slices,
            select_coronal_slices= template_coronal_slices,
            mask_lower_threshold=0.3, mask_upper_threshold=1,
            width_padding = 3).generate_qc_images()
        
        return affine_nu_image_slices, tpm_img_slices
    
    def warped_nu_img_slices(self):
        """
        """
        warped_nu_image_slices, wnu_tpm_img_slices = QCImageGenerator(
            underlay_img=self.warped_nu_img,
            overlay_img=tpm_image,
            select_axial_slices= template_axial_slices,
            select_sagittal_slices= template_sagittal_slices,
            select_coronal_slices= template_coronal_slices,
            mask_lower_threshold=0.3, mask_upper_threshold=1,
            width_padding=5, height_padding=5).generate_qc_images()
        
        return warped_nu_image_slices, wnu_tpm_img_slices
    
    #  _____________________________________________________________ Defining the plotting functions _____________________________________________________________ #
    # The following functions have been defined to generate the plots for the MRI slices. 
    # The functions take the axes as input and plot the images on the axes.

    def plot_mri_slices(self, axes):
        """
        This functions plots only the nu_img slices
        """
        # Plotting the nu_img slices
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax=mri_vmax, cbar=False, ax=axes) 
        # Plotting the lines representing the slices
        sns.heatmap(self.nu_img_lines(), cmap=cmap_yellow2, vmin=0.5, cbar=False, ax=axes)
        axes.text(20, 30, 'L', fontsize=15, color='white')
        axes.text(150, 30, 'R', fontsize=15, color='white')
        axes.set_title(f"{self.nu_img_filename}", fontsize=16, color='white', loc='left')
        axes.axis('off')

    def plot_nu_img_aparc_slices(self, axes):
        """
        This function the aparc+aseg and the subcortical regions overlaid on the nu_img slices.
        """
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax=mri_vmax, cbar=False, ax=axes[0])
        sns.heatmap(self.nu_img_lines(), cmap=cmap_yellow2, vmin=0.5, cbar=False, ax=axes[0])
        axes[0].text(20, 30, 'L', fontsize=15, color='white')
        axes[0].text(150, 30, 'R', fontsize=15, color='white')
        axes[0].set_title(f"{self.nu_img_filename}", fontsize=16, color='white', loc='left')
        axes[0].axis('off')

        # nu MRI image 
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax=mri_vmax, cbar=False, ax=axes[1])
        # Adding the aparc+aseg slices
        sns.heatmap(self.aparc_img_slices(), cmap=cmap_red, vmin=0.1, cbar=False, ax=axes[1])
        # Adding the subcortical regions
        subcortical_regions = self.generate_subcortical_slices()
        for key, value in subcortical_regions.items():
            sns.heatmap(value, cmap=cmap_red, vmin=0.1, cbar=False, ax=axes[1])

        axes[1].set_title(f"Overlay: {self.aparc_img_filename}", fontsize=16, color='white', loc='left')
        axes[1].axis('off')

    def plot_c1_img_slices(self, axes):
        """
        This function plots the c1(spm segmentation of the Tissue Probability Maps) i.e. the Grey matter (containing high densities of unmyelinated neurons)
        overlaid on the nu_img slices.
        """
        # Plotting the nu_img slices
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax = mri_vmax, cbar=False, ax=axes)
        # Plotting the c1 slices
        sns.heatmap(self.c1_image_slices(), cmap=cmap_red, vmin = 0.1, cbar=False, ax=axes)
        axes.set_title(f"Overlay: {self.c1_img_filename} (voxels > 0.8)", fontsize=16, color='white', loc='left')
        axes.axis('off')

    def plot_affine_nu_img_slices(self, axes):
        affine_nu_image_slices, tpm_img_slices = self.affine_nu_img_slices()

        sns.heatmap(affine_nu_image_slices, cmap='gray', vmax = mri_vmax, cbar=False, ax=axes)
        sns.heatmap(tpm_img_slices, cmap=cmap_pink, vmin = 0.1, alpha = 0.2, cbar=False, ax=axes)
        sns.heatmap(ImageProcessor.contour_image(tpm_img_slices), cmap = cmap_pink, vmin = 0.1, cbar = False, ax=axes)
        axes.set_title(f"Underlay: {self.affine_nu_img_filename}\nOverlay: TPM.nii (c1, voxels > 0.3)", fontsize=16, color='white', loc='left')
        axes.axis('off')

    def plot_warped_nu_img_slices(self, axes):
        warped_nu_image_slices, wnu_tpm_img_slices = self.warped_nu_img_slices()
        sns.heatmap(warped_nu_image_slices, cmap='gray', vmax = mri_vmax, cbar=False, ax=axes)
        sns.heatmap(wnu_tpm_img_slices, cmap=cmap_pink, vmin = 0.1, alpha = 0.2, cbar=False, ax=axes)
        sns.heatmap(ImageProcessor.contour_image(wnu_tpm_img_slices), cmap = cmap_pink, vmin = 0.1, cbar = False, ax=axes)
        axes.set_title(f"Underlay: {self.warped_nu_img_filename}\nOverlay: TPM.nii (c1, voxels > 0.3)", fontsize=16, color='white', loc='left')
        axes.axis('off')

    #  _____________________________________________________________ Plotting the slices _____________________________________________________________ #

    def plot_slices(self,output_path):
        """
        Plot the images in the provided order.

        Parameters
        ----------
        output_path : str
            The path to save the output image.

        Returns
        -------
            Saves the output image in the provided path as a .png file.
        """
        # Load the images
        self.load_images()

        plt.figure(facecolor='black')

        # Only nu_img is provided
        if self.nu_img is not None and self.aparc_img is None and self.c1_img is None and self.affine_nu_img is None and self.warped_nu_img is None:            
            fig, axes = plt.subplots(1, 1, figsize=(26, 2.5))
            
            self.plot_mri_slices(axes)

        # nu_img and aparc_img are provided
        elif self.aparc_img is not None and self.c1_img is None and self.affine_nu_img is None and self.warped_nu_img is None:
            fig, axes = plt.subplots(2, 1, figsize=(26, 6))

            self.plot_nu_img_aparc_slices(axes)


        # nu_img, aparc_img, and c1_img are provided
        elif self.aparc_img is not None and self.c1_img is not None and self.affine_nu_img is None and self.warped_nu_img is None:            
            fig, axes = plt.subplots(3, 1, figsize=(26, 9))
            
            self.plot_nu_img_aparc_slices(axes)
            self.plot_c1_img_slices(axes[2])
        
        # If nu_img, aparc_img, c1_img, and affine_nu_img are provided
        elif self.aparc_img is not None and self.c1_img is not None and self.affine_nu_img is not None and self.warped_nu_img is None:            
            fig, axes = plt.subplots(4, 1, figsize=(25, 12))
            
            self.plot_nu_img_aparc_slices(axes)
            self.plot_c1_img_slices(axes[2])
            self.plot_affine_nu_img_slices(axes[3])

        # If nu_img, aparc_img, c1_img, affine_nu_img, and warped_nu_img are provided
        elif self.aparc_img is not None and self.c1_img is not None and self.affine_nu_img is not None and self.warped_nu_img is not None:            
            fig, axes = plt.subplots(5, 1, figsize=(26, 17))
            
            self.plot_nu_img_aparc_slices(axes)
            self.plot_c1_img_slices(axes[2])
            self.plot_affine_nu_img_slices(axes[3])
            self.plot_warped_nu_img_slices(axes[4])


        fig.patch.set_facecolor('black')
        plt.subplots_adjust(top=1, wspace=0, hspace=0.3)
       
        plt.savefig(os.path.join(output_path, self.basename + '.png'), facecolor='black', bbox_inches='tight', dpi=400)