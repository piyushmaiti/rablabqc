import os
import argparse
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/mac/pmaiti/Desktop/leads_qc/rablabqc')
from slice_selector import SliceSelector
from processing import ImageProcessor
from plotter1 import QCImageGenerator
from freesurferlut import FreeSurferColorLUT

def build_parser():
    p = argparse.ArgumentParser(description="Description : Python Script to generate Quality Control Images\n",
                                     formatter_class=argparse.RawTextHelpFormatter)
     
    ############# Required Inputs to the program #############
    p.add_argument('t1', action='store', metavar='T1 path', type=str, help='Path to the T1 Nifti file or processed T1 Nifti file')
    p.add_argument('raparc', action='store', metavar='aparc+aseg path', type=str, help='Path to the aparc+aseg Nifti file')
     
    ############# Optional Inputs to the program #############
    ##### Other MRI Arguments #####
    p.add_argument('-c1', action='store', metavar='C1 path', type=str, help='Path to the C1 Nifti file')
    p.add_argument('-m_affine', action='store', metavar='Affine Nifti Image', type=str, help='Path to the T1 warped to MNI NiFTI file')
    
    ##### PET Arguments #####
    p.add_argument('-pet', action='store', metavar='PET path', type=str, help='Path to the PET Nifti file')
    p.add_argument('-suvr', action='store', metavar='SUVR path', type=str, help='Path to the SUVR Nifti file')
    p.add_argument('-ref reg1', action='store', metavar='Reference Region 1', type=str, help='Path to the Reference Region 1 Nifti file')
    p.add_argument('-ref reg2', action='store', metavar='Reference Region 2', type=str, help='Path to the Reference Region 2 Nifti file')
    
    p.add_argument('-p_affine', action='store', metavar='Affine Nifti Image', type=str, help='Path to the PET warped to MNI NiFTI file')
    
    ##### QC Arguments #####
    p.add_argument('-hp', '--height padding', action='store', metavar='Height Padding', type=int, help='Padding at the top and bottom of individual slices')
    p.add_argument('-wp', '--width padding', action='store', metavar='Width Padding', type=int, help='Padding between the individual slices')
    p.add_argument('-crop_neck', action='store', metavar='Crop Neck', type=bool, help='Crop the neck region of the images')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

    p.add_argument('-o', '--output', action='store', metavar='output path', type=str, help='Path to the output directory')

    return p


def main():
    parser = build_parser()
    results = parser.parse_args()

    
    print("The T1 path is: ", results.t1)
    print("The aparc+aseg path is: ", results.raparc)
    
    # Extracting the Name of the MRI file
    t1_name = results.t1.split("/")[-1]
    # Extracting the Subject ID from the T1 file
    subject_id = results.t1.split("/")[-1].split("_")[1]
    # Extracting the Date from the T1 file
    date = results.t1.split("/")[-1].split("_")[3]
    print("MRI Scan Date: ", date)


    raparc = nib.load(results.raparc).get_fdata()

    print("Selecting the slices for the MRI Images based on the aparc+aseg image")
    select_axial_slices, select_coronal_slices, select_sagittal_slices = SliceSelector(raparc).select_leads_slices()
    select_tpm_axial_slices = [32,55,75,90,120, 145]
    select_tpm_coronal_slices = [83, 140]
    select_tpm_sagittal_slices = [140,150]

    # Location of the Resliced Tissue Probability Maps
    tpm_path = '/home/mac/pmaiti/Desktop/leads_qc/rablabqc/rTPM.nii'
    # Check if the TPM file exists
    if not os.path.exists(tpm_path):
        ValueError("The TPM file does not exist")
    else:
        tpm_image = nib.load(tpm_path).get_fdata()
    
    ############################ Working with the MRI Images ############################
    # Files to be used to generate the QC Images for MRI
    # T1, aparc+aseg, C1, w_affine, TPM
    
    t1 = nib.load(results.t1).get_fdata()
    
    if results.crop_neck == True:
        t1_slices = QCImageGenerator(
            underlay_img=t1,
            select_axial_slices=select_axial_slices,
            select_sagittal_slices=select_sagittal_slices,
            select_coronal_slices=select_coronal_slices,crop_neck = raparc).generate_qc_images()
    else:
        t1_slices = QCImageGenerator(
            underlay_img=t1,
            select_axial_slices=select_axial_slices,
            select_sagittal_slices=select_sagittal_slices,
            select_coronal_slices=select_coronal_slices).generate_qc_images()
                                            
    
    fig, axes = plt.subplots(1, 1, figsize=(20, 5), facecolor='black')
    plt.suptitle(f'Subject ID: {subject_id}\n', fontsize=40, color='white', x=0.5, y=1)
    plt.title(f'Scan Date: {date}', fontsize=20, color='white', x=0.5, y=0.97)

    sns.heatmap(t1_slices, cmap='gray', vmax=140, cbar=False, xticklabels=False, yticklabels=False, ax=axes)
    axes.set_title(f'Underlay: {t1_name}', fontsize=30, color='white', pad=20, loc='left')
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.savefig(f'{results.output}/{subject_id}_{date}_t1.png', facecolor='black')
    
    """
    ############################ Working with the PET Images ############################
    # Extracting the Subject ID from the PET file
    subject_id_pet = results.pet.split("/")[-1].split("_")[1]
    # Extracting the Date from the PET file
    date_pet = results.pet.split("/")[-1].split("_")[3]
    print("PET Scan Date: ", date_pet)

    # Checkinf if the MRI and PET Scans are from the same subject
    if subject_id != subject_id_pet:
        print("The MRI and PET Scans are not from the same subject")
        return
    """

if __name__ == '__main__':
      main()