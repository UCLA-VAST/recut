import argparse
import os
import re
# pip install v3d-py-helper
# python setup.py build_ext --inplace
# import v3dpy.loaders import RAW, PBD

lab_args = {
        # # "e_checked6_chick_uw" : [.103, .103, .3], # mismatches
        # "e_checked6_zebrafish_larve_RGC_UW" :
        # "e_checked7_utokyo_fly" :
        "p_checked6_frog_scrippts" : [100, 100, 10000],
        "p_checked6_fruitfly_larvae_gmu" :
        {
            "done_1_CL-III_X_LifeActRuby_vpda_membrane-GFP_actin-LifeActRuby.czi-C_1" : [],
            "done_1_CL-I_X_OREGON_R_ddaD_membrane-GFP" : [],
            "done_1_CL-I_X_OREGON_R_ddaE_membrane-GFP" : [],
            "done_2_CL-III_X_Myr-GFP_v'pda_MT-mCherry_membrane-GFP_A_1024.czi_C_1" : [], 
            "done_2_CL-I_Membrane-GFP_X_F-Actin-Red_ddaD_Membrane-GFP_F-Actin-Red.czi_C_1" : [], 
            "done_2_CL-I_Membrane-GFP_X_F-Actin-Red_ddaE_Membrane-GFP_F-Actin-Red.czi_C_1" : [], 
            "done_3_CL-III_X_LifeActRuby_vpda_membrane-GFP_actin-LifeActRuby.czi_C_1" : [], 
            "done_3_CL-IV_x_Ank2-IR_ddaC_membrane-B" : [], 
            "done_3_CL-I_MT_X_MYR-GFP_ddaD_MT-mCherry_membrane-GFP.czi_C_1" : [],
            "done_Result_of_C1-CL-I(X-Mas)_x_OregonR_ddaD_F-actinGFP_MT-Red" : [], 
            "done_Result_of_C1-CL-I(X-Mas)_x_OregonR_ddaE_F-actinGFP_MT-Red" : [], 
            "done_err_3_CL-I_MT_X_MYR-GFP_ddaE_MT-mCherry_membrane-GFP.czi_C_1" : [] 
        },
        # "p_checked6_human_allen_confocal" :
        # "p_checked6_human_allen_confocal_independent_trial2_haru" :
        "p_checked6_human_culturedcell_Cambridge_in_vitro_confocal_GFP" : 
           { "Image4" : [.497001, .4970001, .54], "Image5" : [.497001, .4970001, .54], "Image7" : [.497001, .4970001, .54], "Image10" : [.497001, .4970001, .54], "Image31" : [.397, .397, .54],
            "Series009" : [], "Series017" : [], "Series019" : [], "Series021" : []},
        "p_checked6_janelia_flylight_part2" : [.38, .38, .38],
        "p_checked6_mouse_culturedcell_Cambridge_in_vivo_2_photon_PAGFP" : { "done_neuron2" : [.2688637, .2688637, 1], "done_neuron4" : [.5377274, .5377274, 1]},
        # "p_checked6_mouse_korea" :
        # "p_checked6_mouse_RGC_uw" :
        # "p_checked6_mouse_tufts" :
        # "p_checked6_mouse_ugoettingen" :
        # "p_checked6_silkmoth_utokyo" : [.62, .62, 1.44],
        # "p_checked6_zebrafish_adult_RGC_UW" : [.21, .21, .25],
        # "p_checked6_zebrafish_horizontal_cells_UW" : [.07, .07, .25],
        "p_checked7_janelia_flylight_part1" : [.38, .38, .38],
        "p_checked7_taiwan_flycirciut" : [.32, .32, 1]
}

def file_to_voxel_size(fpath):
    with open(fpath, 'r') as f:
        txt = f.read()
        voxel_size = re.sub(r'[^0-9.]', ' ', txt).split()
        clean = list(filter(lambda sz: sz != '.', voxel_size))[:3]
        floats = map(lambda sz: float(sz), clean)
        clamps = filter(lambda value: value < 3, floats)
        return list(clamps)

# extract voxel size per lab
def extract_voxel_size(lab, paths, show=False):
    lab_voxel_size = 'invalid'
    if lab in lab_args: 
        lab_voxel_size = lab_args[lab]
    else:
        txt = list(filter(lambda path: os.path.isfile(path) and ("txt" in path), paths))
        if len(txt) > 0:
            temp = file_to_voxel_size(txt[0])
            if len(temp) == 3:
                lab_voxel_size = temp

    if show:
        if type(lab_voxel_size) == list:
            print('  ', end="") # offset
            print(lab_voxel_size)
        elif type(lab_voxel_size) == dict:
            print(lab_voxel_size)
    return lab_voxel_size

def valid(voxel_size):
    return type(voxel_size) == list and len(voxel_size) == 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    # recut arguments
    parser.add_argument('--fg-percent', type=float)
    args = parser.parse_args()
    show = True

    # print(args)
    for lab in os.listdir(args.dir):
        lab_path = os.path.join(args.dir, lab)
        print(lab)
        subsubdirs = [image for image in os.listdir(lab_path)]
        paths = list(map(lambda image: os.path.join(lab_path, image), subsubdirs))

        # TODO match the voxel size to each image of the lab
        lab_voxel_size = extract_voxel_size(lab, paths)

        # TODO extract soma from each proofread swc
        # TODO write seed
        # TODO convert file format

        # extract individual images
        image_paths = filter(lambda image: os.path.isdir(image), paths)
        for image in image_paths:
            #TODO extract voxel size from lab's dict
            image_voxel_size = 'invalid'
            if type(lab_voxel_size) == dict:
                _, tail = os.path.split(image)
                image_voxel_size = lab_voxel_size[tail]
            elif type(lab_voxel_size) == list:
                image_voxel_size = lab_voxel_size
            else:
                for file in os.listdir(image):
                    if "txt" in file:
                        fpath = os.path.join(image, file)
                        image_voxel_size = file_to_voxel_size(fpath)
            if valid(image_voxel_size):
                if show:
                    print('    ', end="")
                    print(image_voxel_size)
                # call_recut(**vars(args))

if __name__ == "__main__":
    main()
