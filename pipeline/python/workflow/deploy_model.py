from neural_networks.execution_setting import InferenceSetting
from neural_networks.deploy_net import deploy_net


def deploy_model(inference_dataset_dir, model_root_dir, trained_model_path, deploy_pipeline=False, output_prefix='', mask=None):
    """
    segment image volume at inference_dataset_dir, using pretrained model.
    the restored model will place its outputs under
    model_root_dir/model_instance_name, where model_instance_name is parsed
    from records of pretrained model. deployed pipeline skips model_instance_name
    model_root_dir/model_instance_name
        scripts: copies of scripts used in generating the pretrained model
        segmentation: sequences of following two types of images
            tiff_name.tif   8 bit normalized input images. this is generated
                            for development purposes and is irrelevant for
                            pipeline processing
            tiff_name_segmentation.tif   classification outcome
    :param inference_dataset_dirs: a list of directories. each directory
                                    should contain tif sequence, on which the
                                    soma segmentation will be performed
    :param model_root_dir: discussed above
    :param trained_model_path: path to checkpoint of pretrained model
    :return: InferenceSetting
    """
    setting = InferenceSetting(model_root_dir, inference_dataset_dir, trained_model_path)
    deploy_net(setting, deploy_pipeline=deploy_pipeline, output_prefix=output_prefix, mask=mask)
    return setting
