***3d reconstruction pipeline v.3.1***
--------------------------------------------------------

*For better formatting, open this file in an .md file renderer, e.g. https://dillinger.io*

**major changes from pipeline_v.2.0**

  
* includes a neurite+soma segmentation model with higher accuracy and fewer parameters.  
  user can select between soma vs neurite+soma segmentation neural network workflow. 
  
* includes a mask option in pipeline arguments  
  masking of input data set is performed on the fly. mask is expected as a tiff image. identical masking 
  operation is applied to all planes of the image volume
  
  
**minor changes from pipeline_v.2.0**

* add unit tests

**bug fix from pipeline_v.3.1**
* fix a crash by expecting `mask.tif` in the dedicated `mask` folder


&nbsp;
-------------------------------------------------------------------
    
  
**Installation (for developers)**
*  `system requirement`:   
   cuda toolkit 10.1 + nvidia driver of high enough version + cuDNN 7.6. (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)  
   python3.7, python3.7-dev, python3.7-venv
   
* `virtual environment`:  
   You will receive the pipeline as a `.zip` file, named as   
   `pipeline_v.[major].[minor].zip`.
   let's assume the version is `v.3.0`. unzip it to `/path/to/pipeline_v.3.0`.
   you will see the following files and folders in `pipeline_v.3.0`:
   ```
   bin
   lib
   python
   trained_models
   version
   ```
   from inside `pipeline_v.3.0/python`, run `python3.7 virtual_environment.py`. 
   this creates the virtual environment at `/path/to/pipeline_v.3.0/mcp3d_venv`.



&nbsp;
-------------------------------------------------------------------
    
  
**Usage**

From inside `/path/to/pipeline_v.3.0/python`, first activate the virtual 
environment with following command:  `source ../mcp3d_venv/bin/activate`. This 
ensures the pipeline uses correct versions of interpreter and libraries.  

The pipeline is launched by `run_pipeline.py` script under this directory. There 
are two ways to do this. 

1. You can modify the `run_pipeline.py` script to provide parameters. Be careful 
to always surround the parameter values in single quotes, 
e.g.  `'--offset', '0', '0', '0'`, `'--fg_percent', '0.03'`. When you are done 
editting the parameters, use `python3 run_pipeline.py` to run the pipeline.

2. You can also run the pipeline with all required arguments 
through command line, without changing the `run_pipeline.py` file itself. Here
is an example:
```
python3 run_pipeline.py --commands connected_components app2 gcut --input_dir "/media/muyezhu/Mumu/morph_project/raw_images/d1/D1-MORF3_J-G_Muye" --image_prefix D1-MORF3_J-G_Muye --offset 10 100 100 --extent -1 -1 -1 --fg_percent 0.05 --gb_mem_limit 1.5 --app2_auto_soma False --deploy_pipeline --trained_model_name "ContextualUNetV1_200129_1051"
```
the next section explains each parameter to the pipeline.

&nbsp;
-------------------------------------------------------------------
**Pipeline Arguments**

at the end of the `run_pipeline.py` script, you will see lines as below:
  ```
  if len(sys.argv) == 1:
      # arguments to the pipeline
      arguments = [
            '--commands', 'neural_network', 'connected_components', 'app2', 'gcut',
            '--input_dir', '/media/muyezhu/Mumu/morph_project/raw_images/CamK-MORF3_MSNs/Camk2-MORF3-D1Tom-D2GFP_TGME02-1_30x_Str_01C',
            '--channel', '0',
            '--image_prefix', 'imaris_auto',
            '--offset', '0', '0', '0',
            '--extent', '-1', '-1', '-1',
            '--fg_percent', '0.01',
            '--gb_mem_limit', '1.5',
            '--app2_auto_soma', 'False',
            '--label_technique', 'morph',
            '--model_classes', 'neurite+soma',
            '--mask', 'mask.tif',
            ##################################
            #    do not change things below  #
            ##################################
            '--resume_from_dev_version', '',
            '--deploy_pipeline',
        ]
  ```    
under `arguments`, each line is generally in the form of:  
 `'--argument_name', 'argument_value(s)'`.  
to run the pipeline on selected dataset with appropriate arguments, the user 
should supply `'argument_value(s)'` accordingly.

`--commands`: 

indicates the steps of the pipeline to run. a full run of the 
pipeline has 4 steps: `'neural_network', 'connected_components', 'app2', 'gcut'`. you may 
recognize them as generally the same steps in previous pipeline. when all 4 
steps are given, the pipeline will extract soma/neurite+soma locations using neural networks 
(`neural_network`), find neural clusters and somas associated with each cluster 
(`connected_components`), run app2 on each cluster (`app2`) and use gcut to 
segment clusters into single neurons (`gcut`). no further user interaction is 
needed once the pipeline launches. if you give some unknown parameter values to 
`--commands` you will see an error raised. 

`neural_network` and `connected_components` steps are applied to the entire image volume. 
`app2` and `gcut` steps are applied to the selected roi. if the neural network is asked to 
detect soma only, every time a different `fg_percent` is wanted, `connected_component` step 
should be re-run. if the neural network is asked to detect both soma and neurite, `fg_percent` 
parameter becomes obsolete, and `connected_component` step needs to be run only once. the 
neurite+soma neural network model is still under development/improvement and not production ready.

&nbsp;  

`--input_dir`:

directory to image volume. image volume can either consists of a single imaris 
file or a sequence of tiff images. note that the imaris file must end with 
`FusionStitcher.ims`.

&nbsp;  

`--channel`:

channel id to run pipeline on (channel id is 0 indexed). this parameter is only 
relevant for imaris data. 

&nbsp;  

`--image_prefix`:

prefix used in pipeline outputs. for example if `image_prefix` is `D2-Morph`, 
the pipeline will create output images `D2-Morph_000.tif`, `D2-Morph_001.tif` etc, 
as well as swc files  `D2-Morph_z0_100_y0_1000_x0_1000.swc` etc. this parameter 
is only relevant if the input image volume is tiff sequence. for imaris data, 
this parameter will be ignored, and all outputs derive their names from imaris 
file name.

&nbsp;  

`--offset` and `--extent`:

offset and extent lets you select an roi in the data to run pipeline on.   
an roi is defined as a 3d box. every 3d box has 8 corners. both offset and extent 
must be provided as 3 integer numbers. offset defines the z, y, x coordinates 
(in this order) of the front top left corner of the box. extent defines the length 
of the depth, height, width (in this order) of the box. you can given -1 to any of
the 3 numbers in extent, which will make the roi that start at the corner defined 
by offset extends till the end of the dimension which has extent -1 extent value.
`'--offset', '0', '0', '0', '--extent', '-1', '-1', '-1'` correspond to an roi 
selection equivalent to the entire volume.

&nbsp;

`--fg_percent`:

percent of voxels to consider as foreground. this parameter is only used when `model_classes` is 
`soma`. when `model_classes` is `neurite+soma`, the neural network will determine foreground 
completely.

&nbsp;

`--gb_mem_limit`:

memory limit in GB of input image data to app2. if the data exceeds this limit, 
successively lower resolution data will be considered as input until the input 
fits within limit. adjust this according to the amount of memory available to 
your system.  

&nbsp;

`--app2_auto_soma`: 

this parameter should be `True` or `False`. if `True`, this 
tells the pipeline you want to ignore the neural network and connected component 
steps and let `app2` try to find soma automatically. this is not recommended if 
the data has hollow soma. if `soma` is one of the parameter values to `--commands`, 
the pipeline will force `--app2_auto_soma` to `False` regardless of user input.

&nbsp;

`--label_technique`: 
this parameter specifies the experimental technique by which neural cells are 
fluorescently labeld. allowed parameters are `morph` and `rabies` at the moment. 
`label_technique` and `model_classes` combination decides which trained model is 
loaded by the pipeline.

&nbsp;

`--model_classes`:

this parameter specifies the tasks of the neural network. currently allowed parameters are 
`soma` which detects soma voxels alone, and `neurite+soma` which simultaneously detect and 
differentiate neurite and soma voxels. `neurite+soma` option leads to slightly different 
pipeline workflow, as it doesn't need a user provided `fg_percent` value as well as eliminates
soma filling. it is still under active development.

&nbsp;

`--mask`: 

this parameter optionally supplies a mask to the pipeline. if no mask is needed for the dataset, 
give `none` to this argument. if a mask is provided, it should be the name of the mask image under 
`input_dir/mask`. for example, if we have an imaris file `/path/to/test_FusionStitcher.ims` and we'd like 
to mask the input file. the user should create an image of the same height and width as the 
ims file. for image regions that the user wants to mask out, color the corresponding mask region black. 
save this mask image as `/path/to/mask/mask.tif`. the argument value to `--mask` for this example is `mask.tif`. 
the masking is applied across all z planes.

&nbsp;  

**how to give arguments to pipeline**

there are two ways to do this. 

1. you can modify the `run_pipeline.py` script to provide parameters. be careful 
to always surround the parameter values in single quotes, 
e.g.  `'--offset', '0', '0', '0'`, `'--fg_percent', '0.03'`. when you are done 
editting the parameters, save the script, then use `python3 run_pipeline.py` to 
run the pipeline.

2. you can also run the pipeline with all required arguments 
through command line, without changing the `run_pipeline.py` file itself. here
is an example:
```
python3 run_pipeline.py --commands connected_components app2 gcut --input_dir "/media/muyezhu/Mumu/morph_project/raw_images/d1/D1-MORF3_J-G_Muye" --image_prefix D1-MORF3_J-G_Muye --offset 10 100 100 --extent -1 -1 -1 --fg_percent 0.05 --gb_mem_limit 1.5 --app2_auto_soma False --deploy_pipeline --trained_model_name "ContextualUNetV1_200129_1051"
```

&nbsp;


**how to manage benchmark data**  

benchmark management functionalities include: (1) automatically generate 
subvolumes for manual tracing based on the somas found by the neural network. 
preserve the zyx offsets of these subvolumes  (2) apply appropriate offset corrections between
manual reconstruction (subvolume local coordinates) and automatic reconstruction (image volume global 
coordinates) swc files (3) measure similarity of automatic tracing and manual tracing, and keep records 
of results. this update supports step 1 and 2. the other steps will be included in next patch.  
For each subvolume, alongside the tiff stack/sequence file(s), there will be two swc files ending in `local.swc` 
and `manual.swc`. the `local.swc` file is the automatic reconstruction output to be validated against, but with its 
coordinates adjusted so the file will overlay correctly into the subvolume. `manual.swc` is where the manual tracing 
should be done. it will have a single soma node, and needs to be completed by a user.   
at the end of `benchmark_manager.py`, you'll see lines below. they are `'--argument_name', 'argument_value(s)'` 
pairs as discussed above for `run_pipeline.py`.  


```
if __name__ == '__main__':
    arguments = None
    if len(sys.argv) == 1:
        # arguments to the benchmark manager
        arguments = [
            '--invocation_file_path', '/media/muyezhu/Mumu/morph_project/raw_images/CamK-MORF3_MSNs/2020-05-13_08.42.00_Camk2-MORF3-D1Tom_TME03-1_30x_Str_03A_strong_axon/pipeline_v.1.2/invocation_2020-07-26-05:32.sh',
            '--make_subvolumes', 'true',
            '--remove_existing_subvolumes', 'false',
            '--subvolume_z_levels', '120',
            '--ignore_edge_neurons', 'true',
            '--edge_extent', '0', '0', '0',
            '--soma_selection_probability', '0.3',
            '--convert_to_8bit', 'true',
            '--output_format', 'tiff_sequence'
        ]
    main(args=arguments)
```

`--invocation_file_path`  
full path to an invocation file. the invocation file records the arguments to 
the pipeline during each pipeline run. it has file name such as `invocation_2020-05-27 21:23.sh`. 
`benchmark_manager.py` will read this file, and attempts to recover these arguments, 
and generate subvolumes accordingly. different previous pipeline versions use different arguments. 
compatibility functions have been incorporated for benchmark manager to be able to work with 
outputs generated by different pipeline versions. pipeline version v.1.0, v.1.1, v.1.2, v.2.0

&nbsp;

`--make_subvolumes`  
if this parameter is true, and no subvolumes have been created before, creates subvolumes for manual tracing.

&nbsp;

`--include_all_channels`
if this parameter is true, subvolumes will be created for all channels found in an imaris file. channels not used 
for reconstruction will be placed under directory `/path/to/subvolume/ch[number]`.

&nbsp; 

`--subvolume_z_levels` 
this parameter specifies the number of z levels each subvolume should have. the xy dimensions for subvolumes 
are currently fixed at 2048 * 2048.

&nbsp;

`--remove_existing_subvolumes`  
if subvolumes for the same invocation file have been generated previously, the existing subvolumes 
will be removed and new subvolumes will be generated if both `make_subvolumes` and this parameter are 
true. otherwise existing subvolumes will not be altered. set this parameter to true with caution, since 
the manual reconstructed swc files are located within the subvolume directories, and removing subvolumes 
will therefore remove these manual reconstructions.

&nbsp;

`--ignore_edge_neurons`  
if this parameter is true, somas falling into edge regions will not have subvolumes generated for them.

&nbsp;

`--edge_extent`  
three integers expected for this parameter, representing the number of voxels to be regarded as "edge" along 
z, y, and x dimensions. if `10`, `100`, `100` is given, 10 voxels are considered to be "edge" along z axis, covering 
the first as well as the last 10 z levels.

&nbsp;

`--soma_selection_probability`  
this parameter specifies the probability that a subvolume will be generated for a soma. this probability is applied 
after edge somas are already discarded. a value of 0.5 will create subvolumes for about 50% of somas not in edge regions.

&nbsp;

`--convert_to_8bit`:  
if true, the subvolume will be converted from 16 bit to 8 bit

&nbsp;

`--output_format`:  
can be `tiff_sequence`, which generates one tiff image per plane, or `multi_page_tiff`, which generates a single tiff stack for all z planes. 

&nbsp;

Similar to usage of `run_pipeline.py`, you will also need to activate the virtual environment. You can modify the arguments 
in `benchmark_manager.py`, save the file and run `python3 benchmark_manager.py`, or give all arguments through command line. 

&nbsp;
-------------------------------------------------------------------
    
  
**Pipeline outputs**

Assuming the input .ims file is `/dir/to/inputFusionStitcher.ims`, and the 
pipeline version is `v.2.0`. the pipeline generates outputs according to the 
following layout (directories are in **bold**, file names are in `inline code`). 
Some outputs irrelevant to the user are omitted.

**/dir/to**
* `inputFusionStitcher.ims`
* **inputFusionStitcher_ch0_tif**  
  tiff sequence from channel 0 of the imaris input file
* **pipeline_v.2.0**

  - `invocation_2020-02-02-12:37.sh`

     this file saves the time at which the pipeline is run (2020-02-02 12:37), 
     as well as all the arguments given to the pipeline. you will have multiple 
     such files if the pipeline is run on input volume multiple times (potentially 
     in different channels, or with different arguments).
   
  - **ch0**  
    outputs of pipeline_v.2.0 runs for channel 0, using `model_classes = soma`

    - **morph_soma**
    
      outputs generated by and downstream of the model ContextualUNetV1_200129_1051.  
      - **soma_segmentation** 
        (1) tiff images of neural network predictions
        (2) setting files
                
      - **soma_fill**  
        (1) tiff images of the input volume where soma is filled  
        (2) `clusters_fg_0.01.csv, clusters_fg_0.02.csv`, etc: csv files recording 
         neural clusters and somas in each cluster. for each foreground percent parameter, 
         a separate cluster file is produced.
         
      - **reconstruction**   
        contains a list of directories, each corresponding to a unique combination 
        of foreground percent and roi selection. for example: 
        - **reconstruction_fg_0.01_roi_30_400_100_2048_100_2048**:  
          (1) **segmentation**: all single neurons reconstructed by the pipeline 
              in the image volume. multi-neuron neural clusters have been segmented 
              by gcut. all coordinates are global (aka with respect to the entire 
              image volume). each swc file name will include `z[zmin]_[zmax]_y[ymin]_[ymax]_x[xmin]_[xmax]`.
              the values `zmin`, `zmax`, `ymin`, `ymax`, `xmin`, `xmax` defines a 
              3d box resulting from the intersection between roi selection and the 
              spatial range of a neural cluster. if a neural cluster 
              (defined in `cluster_fg_[fg_percent].csv`) falls outside of the roi 
              selection, no swc file will be produced for it.   
          (2) intermediate swc files generated by app2   
          (3) log file
          
      - **benchmark**   
        - **subvolumes**  
          contains subvolumes of input image. each subvolume is centered at 
          detected soma, and has zyx dimensions 100 * 2048 * 2048 if within valid 
          range of the input volume.   
        - `soma_swc_pairs.swc`:  
        a csv file that records for each subvolume: if the manual reconstruction is complete, 
        soma global coordinates and its manual reconstruction swc path. the first columns is meant 
        to be entered by user.  
          

&nbsp;
          
if multiple pipeline versions have been run on an input volume, with multiple 
sets of arguments, the overall output organization will look as below.

```
input_dir
       - xxx.ims
       - ch0_tif
           Z00.tif
           Z01.tif
           ...
       - ch1_tif
       - pipeline-v.1.0
       - pipeline-v.2.0
           -invocationxxx.sh
           - ch0
               - morph_soma
                   - soma_segmentation
                   - soma_fill
                   - reconstruction
                       - reconstruction_fg_0.01_roi_xxx
                           - xxx.swc
                           - segmentation
                   - benchmark
                       - subvolumes
                       - soma_swc_pairs.csv   
                       - benchmark_manager_log
               - trained_model2
                   - soma+neurite_segmentation
                   - reconstruction
                       - reconstruction_fg_0.01_roi_xxx
                           - xxx.swc
                           - segmentation
                   - benchmark
                       - subvolumes
                       - soma_swc_pairs.csv
                       - benchmark_manager_log
           - ch1
               - morph_soma
                   - soma_segmentation
                   - soma_fill
                   - reconstruction
                       - reconstruction_fg_0.01_roi_xxx
                           - xxx.swc
                           - segmentation
```   


&nbsp;
-------------------------------------------------------------------
    
  

 
