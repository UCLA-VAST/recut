from neural_networks.train_net import train_net
from neural_networks.execution_setting import TrainingSetting


def main():
    setting = {
        'model_root_dir': '',
        'model_name': '',
        'dataset_directories': {
            'training': ['/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/image_data/d2/training/2019-05-22_09.57.08_DLS_histogram_matched_reverse_high_sample_rate',
                         '/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/image_data/rbp4/training/rbp4_str_cropped_less_dense_resolution0_tif_high_sample_rate'],
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/d2/training/2019-05-22_09.57.08_DLS_histogram_matched_reverse_high_sample_rate'],
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/d2/training/2019-05-22_09.57.08_DMS_histogram_matched_high_sample_rate',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/rbp4/training/rbp4_str_cropped_less_dense_resolution0_tif_high_sample_rate'],
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/d2/training/2019-05-22_09.57.08_DLS_histogram_matched_reverse',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/d2/training/2019-05-22_09.57.08_DMS_histogram_matched',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/rbp4/training/rbp4_str_cropped_less_dense_resolution0_tif'],
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/cluster_msn/training/cluster_msn',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/30X_whole_slice/training/30X_whole_slice',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/camk2/training/2019-02-25_14.08.42_Protocol_FusionStitcher_CAMK2_cortex_x0_y0'
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/camk2/training/2019-02-25_14.08.42_Protocol_FusionStitcher_CAMK2_cortex_x2048_y4096',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/camk2/training/2019-02-25_14.08.42_Protocol_FusionStitcher_CAMK2_cortex_x4096_y2048',
                         #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/camk2/training/2019-02-25_14.08.42_Protocol_FusionStitcher_CAMK2_cortex_x4096_y4096'
                         #'/media/muyezhu/Mumu/nissl_segmentation/training/x2000_4048_y8000_10048',
                         #'/media/muyezhu/Mumu/nissl_segmentation/training/x4096_6144_y4096_6144'],
            'validation': [],#'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/d2/validation/2019-05-22_09.57.08_DLS_histogram_matched_reverse',
                           #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/d2/validation/2019-05-22_09.57.08_DMS_histogram_matched',
                           #'/media/muyezhu/Dima/project_files/deep_learning/soma_segmentation/image_data/rbp4/validation/rbp4_str_cropped_less_dense_resolution0_tif'],
                           #'/media/muyezhu/Mumu/nissl_segmentation/validation/x4096_6144_y4096_6144'],# ],
            'test': None,
            'inference': None
        },
        # only relevant during inference time as roi selection
        'dataset_selections': {
            'offsets': [],
            'extents': [],
        },
        'trained_model': {
            'model_path': None,
            'previous_epochs': 0,
            'reset_optimizer': False
        },
        'label_rules': {
            'label_input_colors': {
                'soma': (255, 0, 0), 'neurite': (0, 255, 0), 'background': (0, 0, 0)
            },
            'label_weights': {
                'soma': 16, 'background': 1, 'neurite': 16
            }
        },
        'batch_iterator': {
            'retrieve_pattern': 'context',
            'has_paired_data': True,
            'training': True,
            'high_sample_rate': False,
            'data_fit_in_memory': True,
            'sequence_length': 3,
            'n_shards': 1,
            'start_scale': 0,
            'patch_height': 256,
            'patch_width': 256,
            'n_scales': 1,
            'patch_x_overlap_percent': 0.25,
            'patch_y_overlap_percent': 0.25,
            'batch_size': 8,
            'preprocess_method': 'none'
        },
        'transformation': {
            'params': {
                'rotation': (8, (0, 90, 180, 270, 'random', 'random',
                                 'random', 'random')),
                'gaussian': (1, (0, 3)),
                'reverse': (1, ())
            },
            'nesting_orders': [('gaussian', 'rotation'),
                               ('gaussian', 'rotation', 'reverse')],
            'include_identity': False
        },
        'net': {
            'n_epochs': 500,
            'learning_rate': 1e-3,
            'learning_rate_decay': 0.01,
            'save_output_batch_interval': 100,
            'save_model_epoch_interval': 100,
            'model_kwargs': {}
        }
    }


    model_root_dir = '/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/model_training'
    training_data_dirs = ['/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/image_data/d2/training/2019-05-22_09.57.08_DLS_histogram_matched_reverse_high_sample_rate',
                          '/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/image_data/rbp4/training/rbp4_str_cropped_less_dense_resolution0_tif_high_sample_rate']
    validation_data_dirs = ['/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/image_data/d2/validation/2019-05-22_09.57.08_DLS_histogram_matched_reverse_high_sample_rate',
                            '/media/muyezhu/Dima/project_files/deep_learning/neurite+soma_segmentation/image_data/rbp4/validation/rbp4_str_cropped_less_dense_resolution0_tif_high_sample_rate']

    setting['model_root_dir'] = model_root_dir
    setting['model_name'] = 'ContextualUNetV2'
    setting['dataset_directories']['training'] = training_data_dirs
    setting['dataset_directories']['validation'] = validation_data_dirs
    setting['batch_iterator']['preprocess_method'] = 'norm'
    training_setting = TrainingSetting(setting)
    train_net(training_setting)


if __name__ == '__main__':
    main()
