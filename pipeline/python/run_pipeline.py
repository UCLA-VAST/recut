import sys
from pipeline_util import timer
from pipeline_arguments import PipelineArguments
from pipeline import Pipeline


# launch pipeline
@timer
def run_pipeline(args=None):
    pipeline_arguments = None
    try:
        pipeline_arguments = PipelineArguments(args_str_list=args)
    except Exception as e:
        print(e)
        print('error parsing pipeline arguments. please examine input.')
        exit(0)
    pipeline = Pipeline(pipeline_arguments)
    pipeline.run()


if __name__ == '__main__':
    # if arguments not given on command line, take them from below
    arguments = None
    if len(sys.argv) == 1:
        # arguments to the pipeline
        arguments = [
            '--commands', 'neural_network', 'connected_components',# 'app2', 'gcut',
            '--input_dir', '/mnt/d/CC_Work/Pipeline_test',
            '--channel', '0',
            '--image_prefix', 'imaris_auto',
            '--offset', '0', '0', '0',
            '--extent', '-1', '-1', '-1',
            '--fg_percent', '0.01',
            '--gb_mem_limit', '1.5',
            '--app2_auto_soma', 'False',
            '--label_technique', 'morph',
            '--model_classes', 'multi-res_soma',
            '--mask', 'none',
            ##################################
            #    do not change things below  #
            ##################################
            '--resume_from_dev_version', '',
            '--deploy_pipeline',
        ]
    run_pipeline(args=arguments)
