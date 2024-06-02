import os
import sys
import argparse
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append('/home/mac/pmaiti/Desktop/leads_qc/rablabqc')
from mri_slices import MRIQCplots
from fbb_slices import FBBQCplots
from ftp_slices import FTPQCplots
from fdg_slices import FDGQCplots

from slice_selector import SliceSelector

def build_parser():
    p = argparse.ArgumentParser(description="Description : Python Script to generate Quality Control Images\n",
                                     formatter_class=argparse.RawTextHelpFormatter)
     
    p.add_argument('path', action='store', metavar='path', type=str, help='Path to the directory containing the MRI and PET files')
    p.add_argument('output', action='store', metavar='output', type=str, help='Path to the output directory')
    p.add_argument('-mri', '--mri', action='store_true', help='Add this flag to generate QC Images for mri')
    p.add_argument('-fbb', '--fbb', action='store_true', help='Add this flag to generate QC Images for FBB')
    p.add_argument('-ftp', '--ftp', action='store_true', help='Add this flag to generate QC Images for FTP')
    p.add_argument('-fdg', '--fdg', action='store_true', help='Add this flag to generate QC Images for FDG')
    p.add_argument('-crop_neck', '--crop_neck', action='store_true', help='Add this flag to crop the neck')
    return p

def main():
    parser = build_parser()
    results = parser.parse_args()

    if not os.path.exists(results.path):
        print("Error: Input path does not exist.")
        return

    if not os.path.exists(results.output):
        print("Error: Output directory does not exist.")
        return

    id = os.path.basename(os.path.normpath(results.path))
    print("Processing ID : ", id)
    
    ## Sorting the folders based on the modality
    mri_folders = [f for f in os.listdir(results.path) if f.startswith('MRI')]
    mri_folders.sort()

    if results.fbb:
        fbb_folders = [f for f in os.listdir(results.path) if f.startswith('FBB')]
        fbb_folders.sort()
    else:
        fbb_folders = []

    if results.ftp:
        ftp_folders = [f for f in os.listdir(results.path) if f.startswith('FTP')]
        ftp_folders.sort()
    else:
        ftp_folders = []

    if results.fdg:
        fdg_folders = [f for f in os.listdir(results.path) if f.startswith('FDG')]
        fdg_folders.sort()
    else:
        fdg_folders = []

    # Grouping the modalities by Visit
    visit_lists = {}
    for number in range(len(mri_folders)):
        visit = []
        visit.append(mri_folders[number])
        visit.append(fbb_folders[number] if number < len(fbb_folders) else None)
        visit.append(ftp_folders[number] if number < len(ftp_folders) else None)
        visit.append(fdg_folders[number] if number < len(fdg_folders) else None)
        visit_name = 'visit_' + str(number)
        visit_lists[visit_name] = visit

    print("-----------")
    
    def load_nii(path, orientation="LAS"):
        """
        Load nifti image with specified orientation
        """
        from nibabel.orientations import io_orientation, axcodes2ornt

        img = nib.load(path)
        img_ornt = io_orientation(img.affine)
        new_ornt = axcodes2ornt(orientation)
        img = img.as_reoriented(img_ornt)
        return img.get_fdata()
    
    # __________________________________________ Generating QC images ____________________________________________
    for visit_name, data in visit_lists.items():

        if data[0].startswith('MRI'):
            print(id,data[0])

            mri_date = data[0].split('_')[-1]
            nu_img = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_nu.nii')
            aparcaseg_img = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_aparc+aseg.nii')
            c1_img = os.path.join(results.path, data[0], 'c1'+id + '_MRI-T1_'+mri_date+'_nu.nii')       
            wnu_img = os.path.join(results.path, data[0], 'w'+id + '_MRI-T1_'+mri_date+'_nu.nii')
            affinenu_img = os.path.join(results.path, data[0], 'a'+id + '_MRI-T1_'+mri_date+'_nu.nii')

            # ______________________ Loading all the reference region masks ______________________
            
            # For FBB
            print(" Searching for reference region masks for FBB")
            wcbl_reference_mask = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_mask-wcbl.nii')
            brainstem_reference_mask = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_mask-brainstem.nii')
            eroded_subcortwm_reference_mask = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_mask-eroded-subcortwm.nii')

            # For FTP
            print(" Searching for reference region masks for FTP")
            infcblgm_reference_mask = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_mask-infcblgm.nii')

            # For FDG
            print(" Searching for reference region masks for FDG")
            pons_reference_mask = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_mask-pons.nii')
            
            
            if os.path.exists(nu_img) and os.path.exists(aparcaseg_img) and os.path.exists(c1_img) and os.path.exists(wnu_img) and os.path.exists(affinenu_img):
                
                print(" Slice Selection in progress")
                select_axial_slices, select_coronal_slices, select_sagittal_slices = SliceSelector(load_nii(aparcaseg_img)).select_leads_slices()

                # ______________________ Generating MRI QC Images ______________________
                if results.mri:
                    print("\n______________________")
                    print("Generating MRI QC Images for ", data[0])
                    MRIQCplots(nu_img= nu_img, aparc_img= aparcaseg_img, c1_img= c1_img, 
                            affine_nu_img= affinenu_img, warped_nu_img= wnu_img,
                            axial_slices = select_axial_slices, coronal_slices = select_coronal_slices,sagittal_slices = select_sagittal_slices).plot_slices(results.output)
                    
                    print("MRI QC Images for ", data[0], "generated.")

                # ______________________ Generating Amyloid QC Images ______________________
                if results.fbb:
                    print("\n______________________")
                    print("Genrating Amyloid QC Images for ", id,":",data[1])
                    
                    fbb_date = data[1].split('_')[-1]

                    fbb_suvr_img = os.path.join(results.path,data[1],'r'+id + '_FBB_'+fbb_date+'_suvr-compwm.nii')
                    affine_suvr_img = os.path.join(results.path,data[1],'ar'+id + '_FBB_'+fbb_date+'_suvr-compwm.nii')
                    warped_suvr_img = os.path.join(results.path,data[1],'wr'+id + '_FBB_'+fbb_date+'_suvr-compwm.nii')    

                    if os.path.exists(fbb_suvr_img) and os.path.exists(affine_suvr_img) and os.path.exists(warped_suvr_img):
                        print("All the files required to process Amyloid exist")
                        
                        FBBQCplots(suvr_img = fbb_suvr_img, 
                                   axial_slices = select_axial_slices, coronal_slices = select_coronal_slices, sagittal_slices = select_sagittal_slices,
                                   nu_img = nu_img,
                                   aparc_img = aparcaseg_img,
                                   c1_img = c1_img,
                                   affine_nu_img = affinenu_img,
                                   warped_nu_img = wnu_img,
                                   reference_region_1 = wcbl_reference_mask,
                                   reference_region_2 = brainstem_reference_mask,
                                   reference_region_3 = eroded_subcortwm_reference_mask,
                                   affine_suvr_img = affine_suvr_img,
                                   warped_suvr_img = warped_suvr_img).plot_slices(results.output)
                        
                        print("FBB QC Images for ", data[1], "generated.")

                # ______________________ Generating FTP QC Images ______________________
                if results.ftp:
                    print("\n______________________")
                    print("Genrating FTP QC Images for ", id,":",data[2])
                
                    ftp_date = data[2].split('_')[-1]
                    
                    ftp_suvr_img = os.path.join(results.path,data[2],'r'+id + '_FTP_'+ftp_date+'_suvr-infcblgm.nii')
                    affine_suvr_img = os.path.join(results.path,data[2],'ar'+id + '_FTP_'+ftp_date+'_suvr-infcblgm.nii')
                    warped_suvr_img = os.path.join(results.path,data[2],'wr'+id + '_FTP_'+ftp_date+'_suvr-infcblgm.nii')    

                    if os.path.exists(ftp_suvr_img) and os.path.exists(affine_suvr_img) and os.path.exists(warped_suvr_img):
                        print("All the files required to process FTP exist")
                        
                        FTPQCplots(suvr_img = ftp_suvr_img, 
                                   axial_slices = select_axial_slices, coronal_slices = select_coronal_slices, sagittal_slices = select_sagittal_slices,
                                   nu_img = nu_img,
                                   aparc_img = aparcaseg_img,
                                   c1_img = c1_img,
                                   affine_nu_img = affinenu_img,
                                   warped_nu_img = wnu_img,
                                   reference_region_1 = infcblgm_reference_mask,
                                   reference_region_2 = eroded_subcortwm_reference_mask,
                                   affine_suvr_img = affine_suvr_img,
                                   warped_suvr_img = warped_suvr_img).plot_slices(results.output)
                        
                        print("FTP QC Images for ", data[2], "generated.")
            
            # ______________________ Generating FDG QC Images ______________________
            if results.fdg:
                print("\n______________________")
                print("Genrating FDG QC Images for ", id,":",data[3])
                
                fdg_date = data[3].split('_')[-1]
                
                fdg_suvr_img = os.path.join(results.path,data[3],'r'+id + '_FDG_'+fdg_date+'_suvr-pons.nii')
                affine_suvr_img = os.path.join(results.path,data[3],'ar'+id + '_FDG_'+fdg_date+'_suvr-pons.nii')
                warped_suvr_img = os.path.join(results.path,data[3],'wr'+id + '_FDG_'+fdg_date+'_suvr-pons.nii')    

                if os.path.exists(fdg_suvr_img) and os.path.exists(affine_suvr_img) and os.path.exists(warped_suvr_img):
                    print("All the files required to process FDG exist")
                                        
                    FDGQCplots(suvr_img = fdg_suvr_img, 
                               axial_slices = select_axial_slices, coronal_slices = select_coronal_slices, sagittal_slices = select_sagittal_slices,
                               nu_img = nu_img,
                               aparc_img = aparcaseg_img,
                               c1_img = c1_img,
                               affine_nu_img = affinenu_img,
                               warped_nu_img = wnu_img,
                               reference_region_1 = pons_reference_mask,
                               affine_suvr_img = affine_suvr_img,
                               warped_suvr_img = warped_suvr_img).plot_slices(results.output)
                    
                    print("FDG QC Images for ", data[3], "generated.")
                    
        else:
            print("Error: Some MRI files are missing")
            return
            
if __name__ == '__main__':
    main()
