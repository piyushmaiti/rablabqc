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
def create_colormap(color):
    cmap = LinearSegmentedColormap.from_list('custom', [color, color], N=256)
    cmap.set_under(alpha=0)
    return cmap

cmap_red = create_colormap((1, 0, 0))
cmap_yellow = create_colormap((1, 1, 0))
cmap_blues = create_colormap((0, 0, 1))
cmap_pink = create_colormap((1, 0, 1))

#  _________________________________________________________________ MRI VMAX __________________________________________________________________ #

mri_vmax = 120

# _______________________________________________________________ TEMPLATE SLICES  _____________________________________________________________ #
tpm_file = '/home/mac/pmaiti/Desktop/leads_qc/TPM.nii'
tpm_image = (nib.load(tpm_file).get_fdata())[:,:,:,0]

template_axial_slices = [23, 34, 45, 56, 67, 78]
template_sagittal_slices = [55,44]
template_coronal_slices = [48,70]

# ______________________________________________________________________________________________________________________________________________ #

class MRIQCplots:
    def __init__(self,nu_img, axial_slices, sagittal_slices, coronal_slices,
                 raparc_img=None, c1_img=None, affine_nu_img=None, warped_nu_img=None, crop_neck=True):
         
        self.nu_img = nu_img
        self.raparc_img = raparc_img
        self.c1_img = c1_img
        self.affine_nu_img = affine_nu_img
        self.warped_nu_img = warped_nu_img
        self.axial_slices = axial_slices
        self.sagittal_slices = sagittal_slices
        self.coronal_slices = coronal_slices
        self.crop_neck = crop_neck

        if self.crop_neck and self.raparc_img is None:
            self.crop_neck = False
            warnings.warn("The raparc_img is not provided. The neck will not be cropped.", UserWarning)

    def load_nii(self, img_path):
        """
        Reads the NiFTI image file and returns the image data as a numpy array.

        Parameters
        ----------
        img_path : str
            The path to the image file.

        Returns
        -------
        numpy.ndarray
            The image data as a numpy array.
        """
        return nib.load(img_path).get_fdata()
    
    
    def load_images(self):
        """
        Load the provided images and store them as class attributes.

        Returns
        -------
        None
        """
        self.basename = self.nu_img.split('/')[-1].split('_')[0]+'_'+self.nu_img.split('/')[-1].split('_')[1]+'_'+self.nu_img.split('/')[-1].split('_')[2]
        self.nu_img_filename = os.path.basename(self.nu_img)
        self.nu_img = self.load_nii(self.nu_img)
        
        if self.raparc_img is not None:
            self.raparc_img_filename = os.path.basename(self.raparc_img)
            self.raparc_img = self.load_nii(self.raparc_img)
            
        if self.c1_img is not None:
            self.c1_img_filename = os.path.basename(self.c1_img)
            self.c1_img = self.load_nii(self.c1_img)

        if self.affine_nu_img is not None:
            self.affine_nu_img_filename = os.path.basename(self.affine_nu_img)
            self.affine_nu_img = self.load_nii(self.affine_nu_img)

        if self.warped_nu_img is not None:
            self.warped_nu_img_filename = os.path.basename(self.warped_nu_img)
            self.warped_nu_img = self.load_nii(self.warped_nu_img)
                    
    def nu_img_slices(self):
        """
        """
        return QCImageGenerator(
            underlay_img=self.nu_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            crop_neck=self.raparc_img if self.crop_neck else None).generate_qc_images()
    
    def nu_img_lines(self):
        """
        """
        return QCImageGenerator(
            underlay_img=self.nu_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            crop_neck=self.raparc_img if self.crop_neck else None).generate_lines()


    def raparc_img_slices(self):
        """
        """
        _,raparc_img_slices= QCImageGenerator(
            underlay_img=self.nu_img,
            overlay_img=self.raparc_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            mask_lower_threshold=1000, mask_upper_threshold=3000,
            crop_neck=self.raparc_img if self.crop_neck else None).generate_qc_images()
    
        return raparc_img_slices
    
    def generate_subcortical_slices(self):
        """
        """
        
        def generate_subcortical(lower_threshold, upper_threshold):
            _, image = QCImageGenerator(
                underlay_img = self.nu_img,
                overlay_img = self.raparc_img,
                select_axial_slices = self.axial_slices,
                select_sagittal_slices = self.sagittal_slices,
                select_coronal_slices = self.coronal_slices,
                mask_lower_threshold = lower_threshold,
                mask_upper_threshold = upper_threshold,
                crop_neck = self.raparc_img if self.crop_neck else None).generate_qc_images()
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
        """
        _,c1_img_slices = QCImageGenerator(
            underlay_img=self.nu_img,
            overlay_img=self.c1_img,
            select_axial_slices=self.axial_slices,
            select_sagittal_slices=self.sagittal_slices,
            select_coronal_slices=self.coronal_slices,
            mask_lower_threshold=0.3, #mask_upper_threshold=1,
            crop_neck=self.raparc_img if self.crop_neck else None).generate_qc_images()
        
        return c1_img_slices
    
    

    def affine_nu_img_slices(self):
        """
        """

        affine_nu_image_slices, tpm_img_slices = QCImageGenerator(
            underlay_img=self.affine_nu_img,
            overlay_img=nib.load('/home/mac/pmaiti/Desktop/leads_qc/rTPM.nii').get_fdata(),
            select_axial_slices= [43,53,66,90, 110, 135],
            select_sagittal_slices=[83, 106],
            select_coronal_slices=[80,90],
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
    
    def plot_mri_slices(self, axes):
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax=mri_vmax, cbar=False, ax=axes)
        sns.heatmap(self.nu_img_lines(), cmap=cmap_yellow, vmin=0.5, cbar=False, ax=axes)
        axes.text(20, 30, 'L', fontsize=15, color='white')
        axes.text(150, 30, 'R', fontsize=15, color='white')
        axes.set_title(f"{self.nu_img_filename}", fontsize=16, color='white', loc='left')
        axes.axis('off')

    def plot_nu_img_aparc_slices(self, axes):
        """
        This would plot the nu_img and the raparc_img overlaid on the nu_img.
        """
        # nu MRI
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax=mri_vmax, cbar=False, ax=axes[0])
        sns.heatmap(self.nu_img_lines(), cmap=cmap_yellow, vmin=0.5, cbar=False, ax=axes[0])
        axes[0].text(20, 30, 'L', fontsize=15, color='white')
        axes[0].text(150, 30, 'R', fontsize=15, color='white')
        axes[0].set_title(f"{self.nu_img_filename}", fontsize=16, color='white', loc='left')
        axes[0].axis('off')

        # nu MRI + raparc + aseg + subcortical regions
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax=mri_vmax, cbar=False, ax=axes[1])
        sns.heatmap(self.raparc_img_slices(), cmap=cmap_red, vmin=0.1, cbar=False, ax=axes[1])

        # Adding the subcortical regions
        subcortical_regions = self.generate_subcortical_slices()
        for key, value in subcortical_regions.items():
            sns.heatmap(value, cmap=cmap_red, vmin=0.1, cbar=False, ax=axes[1])

        axes[1].set_title(f"Overlay: {self.raparc_img_filename}", fontsize=16, color='white', loc='left')
        axes[1].axis('off')

    def plot_c1_img_slices(self, axes):
        sns.heatmap(self.nu_img_slices(), cmap='gray', vmax = mri_vmax, cbar=False, ax=axes)
        sns.heatmap(self.c1_image_slices(), cmap=cmap_red, vmin = 0.1, cbar=False, ax=axes)
        axes.set_title(f"Overlay: {self.c1_img_filename}", fontsize=16, color='white', loc='left')
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
        if self.nu_img is not None and self.raparc_img is None and self.c1_img is None and self.affine_nu_img is None and self.warped_nu_img is None:
            
            fig, axes = plt.subplots(1, 1, figsize=(26, 2.5))
            
            self.plot_mri_slices(axes)


        # nu_img and raparc_img are provided
        elif self.raparc_img is not None and self.c1_img is None and self.affine_nu_img is None and self.warped_nu_img is None:
            
            fig, axes = plt.subplots(2, 1, figsize=(26, 6))

            self.plot_nu_img_aparc_slices(axes)


        # nu_img, raparc_img, and c1_img are provided
        elif self.raparc_img is not None and self.c1_img is not None and self.affine_nu_img is None and self.warped_nu_img is None:
            
            fig, axes = plt.subplots(3, 1, figsize=(26, 9))
            
            self.plot_nu_img_aparc_slices(axes)
            self.plot_c1_img_slices(axes[2])
        
        # If nu_img, raparc_img, c1_img, and affine_nu_img are provided
        elif self.raparc_img is not None and self.c1_img is not None and self.affine_nu_img is not None and self.warped_nu_img is None:
            
            fig, axes = plt.subplots(4, 1, figsize=(26, 12))
            
            self.plot_nu_img_aparc_slices(axes)
            self.plot_c1_img_slices(axes[2])
            self.plot_affine_nu_img_slices(axes[3])

        # If nu_img, raparc_img, c1_img, affine_nu_img, and warped_nu_img are provided
        elif self.raparc_img is not None and self.c1_img is not None and self.affine_nu_img is not None and self.warped_nu_img is not None:
            
            fig, axes = plt.subplots(5, 1, figsize=(26, 17))
            
            self.plot_nu_img_aparc_slices(axes)
            self.plot_c1_img_slices(axes[2])
            self.plot_affine_nu_img_slices(axes[3])
            self.plot_warped_nu_img_slices(axes[4])

        else:
            raise ValueError("Please provide the images in the correct order.")

        fig.patch.set_facecolor('black')
        plt.subplots_adjust(top=1, wspace=0, hspace=0.3)

        #plt.savefig(os.path.join(output_path, os.path.splitext(self.nu_img_filename)[0] + '.png'), facecolor='black', bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_path, self.basename + '.png'), facecolor='black', bbox_inches='tight', dpi=400)