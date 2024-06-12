import os
import sys
import shutil
import argparse
import subprocess

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt

# Importing the necessary classes from the rablabqc package
rablab_pkg_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rablab_pkg_path)
print("RabLab QC path:", rablab_pkg_path)

from mri_slices import MRIQCplots
from fbb_slices import FBBQCplots
from slice_selector import SliceSelector

def build_parser():
    p = argparse.ArgumentParser(description="Description : Python Script to generate Quality Control Images\n",
                                     formatter_class=argparse.RawTextHelpFormatter)
     
    p.add_argument('path', action='store', metavar='path', type=str, help='Path to the directory containing the MRI and PET files')
    p.add_argument('-mri', '--mri', action='store_true', help='Add this flag to generate QC Images for mri')
    p.add_argument('output', action='store', metavar='output', type=str, help='Path to the output directory')
    p.add_argument('-fbp', '--fbp', action='store_true', help='Add this flag to generate QC Images for fbp')
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

    id = results.path.split('/')[-1]
    print("Processing ID: ", id)
    
    mri_folders = [f for f in os.listdir(results.path) if f.startswith('MRI')]
    mri_folders.sort()

    if results.fbp:
        fbp_folders = [f for f in os.listdir(results.path) if f.startswith('FBP')]
        fbp_folders.sort()
    else:
        fbp_folders = []

    # Grouping the modalities by Visit
    visit_lists = {}
    for number in range(len(mri_folders)):
        visit = []
        visit.append(mri_folders[number])
        visit.append(fbp_folders[number] if number < len(fbp_folders) else None)
        visit_name = 'visit_' + str(number)
        visit_lists[visit_name] = visit

    print("-----------")
    # _______________________________________________ Loading Images _______________________________________________
    rtpm_path = os.path.join(rablab_pkg_path, 'reslice', 'rT1.nii') # Resliced T1 file to 1mm isotropic resolution
    reslice_matlab_script = os.path.join(rablab_pkg_path,'reslice', 'reslice.m')
    mask_reslice_matlab_script = os.path.join(rablab_pkg_path, 'reslice', 'mask_reslice.m')

    def generate_matlab_script(path, output_script_path):
        """
        This function generates a MATLAB script to reslice the image.
        """
        #with open('/home/mac/pmaiti/Desktop/leads_qc/reslice_test/reslice.m', 'r') as template_file:
        with open(reslice_matlab_script, 'r') as template_file:
            script_content = template_file.read()

        # Replace placeholders with actual paths
        script_content = script_content.replace('<RTPM_PATH>', rtpm_path)
        script_content = script_content.replace('<DATA_PATH>', path)
        
        # Write the modified script to the output path
        with open(output_script_path, 'w') as script_file:
            script_file.write(script_content)
    
    def generate_mask_reslice_mtlb(path, output_script_path):
        """
        This function generates a MATLAB script to reslice the mask image.
        """
        
        #with open('/home/mac/pmaiti/Desktop/leads_qc/reslice_test/mask_reslice.m', 'r') as template_file:
        with open(mask_reslice_matlab_script, 'r') as template_file:
            script_content = template_file.read()

        # Replace placeholders with actual paths
        script_content = script_content.replace('<RTPM_PATH>', rtpm_path)
        script_content = script_content.replace('<DATA_PATH>', path)
        
        # Write the modified script to the output path
        with open(output_script_path, 'w') as script_file:
            script_file.write(script_content)

    def load_nii_resliced(path, orientation="LAS", mask=False):
        """
        Load nifti image with specified orientation
        """
        
        id = path.split('/')[-1].split('.')[0]

        tmp_folder = os.path.join('/shared/petcore/Projects/LEADS/data_f7p1/summary/piyush_qc/tmp/')

        resliced_image_path = os.path.join(tmp_folder, id, 'qc' + id + '.nii')
        
        if not os.path.exists(resliced_image_path):
            
            # Check if the temporary folder with the id exists
            tmp_id_folder = os.path.join(tmp_folder, id)
            if not os.path.exists(tmp_id_folder):
                os.makedirs(tmp_id_folder)
            
            # Copy the image to the temporary id folder
            tmp_file = os.path.join(tmp_id_folder, id + '.nii')
            shutil.copy2(path, tmp_file)
            print(tmp_file)
            
            if mask:
                output_script_path = os.path.join(tmp_id_folder, 'mask_reslice.m')
                generate_mask_reslice_mtlb(tmp_file, output_script_path)
            else:
                output_script_path = os.path.join(tmp_id_folder, 'reslice.m')
                generate_matlab_script(tmp_file, output_script_path)
                
            # Command to run the MATLAB script
            command = f"matlab -nodisplay -nosplash -r \"run('{output_script_path}');exit;\""
            
            # Run the command
            matprocess = subprocess.run(command, shell=True, capture_output=True, text=True)
            print("Output:\n", matprocess.stdout)

        else:
            print("Resliced image already exists for ", id, ". Loading the resliced image...")
            
        img = nib.load(resliced_image_path)
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
            raparc_aseg_img = os.path.join(results.path, data[0], id + '_MRI-T1_'+mri_date+'_aparc+aseg.nii')
            c1_img = os.path.join(results.path, data[0], 'c1'+id + '_MRI-T1_'+mri_date+'_nu.nii')
            wnu_img = os.path.join(results.path, data[0], 'w'+id + '_MRI-T1_'+mri_date+'_nu.nii')
            affinenu_img = os.path.join(results.path, data[0], 'a'+id + '_MRI-T1_'+mri_date+'_nu.nii')

            
            # Check if all the files exist
            if os.path.exists(nu_img) and os.path.exists(raparc_aseg_img) and os.path.exists(c1_img) and os.path.exists(wnu_img) and os.path.exists(affinenu_img):
                print("All the files required to process MRI exist")
                print("Now Processing", data[0])
                # Loading data through nibabel
                nu_img_array = nu_img
                raparc_aseg_img_array = raparc_aseg_img
                c1_img_array = c1_img
                wnu_img_array = wnu_img
                affinenu_img_array = affinenu_img

                axial_slices, coronal_slices, sagittal_slices = SliceSelector(load_nii_resliced(raparc_aseg_img,mask = True)).select_leads_slices()

                print("All the file paths for mri")
                print(nu_img_array)
                print(raparc_aseg_img_array)
                print(c1_img_array)
                print(wnu_img_array)
                print(affinenu_img_array)

                if results.mri:
        
                    MRIQCplots(nu_img = nu_img_array,
                            axial_slices = axial_slices, coronal_slices=coronal_slices, sagittal_slices=sagittal_slices,
                            aparc_img= raparc_aseg_img_array,
                            c1_img= c1_img_array,
                            affine_nu_img= affinenu_img_array,
                            warped_nu_img= wnu_img_array).plot_slices(results.output)

                    print("MRI processing completed for", visit_name,"\n")
                
                
                if results.fbp:
                    print(" The fbp flag is set so processing fbp")
                    print("Processing", data[1], "for", visit_name)

                    fbp_date = data[1].split('_')[-1]

                    fbp_img = os.path.join(results.path, data[1], 'rs6r'+id + '_FBP_'+fbp_date+'_suvr-wcbl.nii')
                    wcbl_ref = os.path.join(results.path, data[1], id + '_MRI-T1_'+mri_date+'_mask-wcbl.nii')
                    brain_stem_ref = os.path.join(results.path, data[1], id + '_MRI-T1_'+mri_date+'_mask-brainstem.nii')
                    eroded_subcortwm_ref = os.path.join(results.path, data[1], id + '_MRI-T1_'+mri_date+'_mask-eroded-subcortwm.nii')
                    
                    affine_suvr = os.path.join(results.path, data[1], 'ars6r'+id + '_FBP_'+fbp_date+'_suvr-wcbl.nii')
                    warped_suvr = os.path.join(results.path, data[1], 'wrs6r'+id + '_FBP_'+fbp_date+'_suvr-wcbl.nii')

    

                    if os.path.exists(fbp_img) and os.path.exists(wcbl_ref) and os.path.exists(brain_stem_ref) and os.path.exists(eroded_subcortwm_ref) and os.path.exists(affine_suvr) and os.path.exists(warped_suvr):
                    
                        print("All the files required to process fbp exist")
                        print("Now Processing", data[1])
                        
                        fbp_img_array = fbp_img
                        wcbl_ref_img_array = wcbl_ref
                        comp_ref_img_array = brain_stem_ref
                        eroded_subcortwm_ref_img_array = eroded_subcortwm_ref
                        affine_suvr_img_array = affine_suvr
                        warped_suvr_img_array = warped_suvr

                        
                        FBBQCplots(suvr_img = fbp_img_array,
                                axial_slices = axial_slices, coronal_slices=coronal_slices, sagittal_slices=sagittal_slices,
                                nu_img = nu_img_array,
                                aparc_img= raparc_aseg_img_array,
                                c1_img= c1_img_array,
                                affine_nu_img= affinenu_img_array,
                                warped_nu_img= wnu_img_array,
                                
                                reference_region_1= wcbl_ref_img_array,
                                reference_region_2= comp_ref_img_array,
                                reference_region_3= eroded_subcortwm_ref_img_array,
                                
                                affine_suvr_img= affine_suvr_img_array,
                                warped_suvr_img= warped_suvr_img_array
                                ).plot_slices(results.output)

                        print("fbp processed for", visit_name)

                

            else:
                print("Error: Some MRI files are missing")
                return
            
            

            
if __name__ == '__main__':
    main()
