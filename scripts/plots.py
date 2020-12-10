import os
import argparse
# set backend that allows save plots without X11 forwarding/
# $DISPLAY set
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import json
from statistics import mean
from os import getcwd
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

extract_values = lambda dictions, extract_key, start=None, stop=None: [int(diction[extract_key]) for diction in dictions[start:stop]]
extract_postfix = lambda dictions, extract_key, start=None, stop=None: [int(diction[extract_key].split('/')[-1]) for diction in dictions[start:stop]]
filter_key_value = lambda dicts, key, value: list(filter(lambda d: d[key].count(str(value)), dicts))
remove_if = lambda dicts, key, value: list(filter(lambda d: d.get(key) != value, dicts))
printj = lambda j, indent=1: print(json.dumps(j, indent=indent))

def test_ratios(ratios, r_ratios):
    for ratio, check_ratios in zip(ratios, r_ratios):
        for check_ratio in check_ratios:
            assert (ratio == check_ratio),"ratio did not match"

def rplot(xiter, xlabel, yiter, ylabel, title, args, lineprops=['k-x', 'r-o', 'g-d', 'b->', 'm-s'],
        legends=[''], legend_metric='', yiter_secondary=[], bar=False):
    xiter == list(range(len(yiter)))
    assert(len(xiter) == len(yiter))

    # if legends[0] != '':
        # assert(len(yiter) == len(legends))
    lineprops = lineprops[:len(legends)]
    title = title.replace('_', '-')

    if bar:
        width = .25
        halfbar = width / 2
        # Get some pastel shades for the colors
        colors = plt.cm.BuPu([0.25, .5])
        x = [i - halfbar for i in range(len(xiter))]  
        x1 = [i - halfbar for i in range(len(xiter))]  
        x2 = [i + halfbar for i in range(len(xiter))] 
        plt.bar(x, yiter, width, label=legends[0], 
                color=colors[0])
        if len(yiter_secondary) != 0:
            plt.bar(x2, yiter_secondary, width, label=legends[1], color=colors[1])
    else:
        for x, y, lineprop, legend in zip(xiter, yiter, lineprops, legends):
            plt.plot(x, y, lineprop, label=str(legend) + legend_metric)
    plt.xlabel(xlabel)
    # take the first one even though they should all match
    plt.ylabel(ylabel)
    if len(xiter) > 1:
        plt.legend()
    if bar:
        plt.xticks(x, labels=xiter, rotation=75)
    else:
        plt.xticks(xiter, rotation=75)
    plt.title(title)
    plt.tight_layout()
    fig_output_path = f'{args.output}{title}.{args.type}'.replace(' ', '_').replace('$', '').replace('\\', '')

    # pu.figure_setup()
    # fig_size = pu.get_fig_size(10, 10)
    # fig = plt.figure(figsize=fig_size)
    # ax = fig.add_subplot(111)
        # pu.save_fig(fig, args.save)

    if args.save:
        plt.savefig(fig_output_path, dpi=args.dpi)
        print(fig_output_path)
    if args.show:
        plt.show()
    else:
        plt.close()

def get_desired_test_info(args, desired_test_runs, test_fn , flag):
    desired_tests = []
    for test_num in desired_test_runs:
        test_fn_specific = f'{test_fn}-{flag}-{test_num}'
        test_cmd = f'{args.binary}./recut_test --gtest_output=json:{test_fn_specific}.json --gtest_filter=*.ChecksIfFinalVerticesCorrect/{test_num}'
        desired_tests.append((test_fn_specific, test_cmd))
    return desired_tests

def radius(args):
    """ Radius Stage Plots """
    # TODO timestamp
    test_fn = args.output + 'radius_test'
    bench_fn = args.output + 'radius_bench'

    if args.rerun: # run the radius script from scratch
        if args.recompile:
            recompile(args, ['TEST_ALL_BENCHMARKS'])

        # test
        test_cmd = args.binary + './recut_test --gtest_filter=Radius.* --gtest_output=json:{test_fn}.json'
        run_with_log(test_fn, test_cmd)

        # bench
        bench_cmd = args.binary + './recut_bench --benchmark_filter=radius.* --benchmark_out_format=json:{bench_fn}.json'
        run_with_log(bench_fn, bench_cmd)

    if args.save or args.show:
        df = json.load(open(f'{test_fn}.json'))
        test= df['testsuites'][0]['testsuite'][0]
        recut_keys = [i for i in test.keys() if 'Recut' in i]
        xy_keys = [i for i in test.keys() if 'XY' in i]

        recut_errors = [test[i] for i in recut_keys]
        xy_errors = [test[i] for i in xy_keys]
        grid_sizes_recut = [int(key.split()[3]) for key in recut_keys]
        grid_sizes_xy = [int(key.split()[3]) for key in xy_keys]

        # sort by the grid_size but only keep the errors
        recut_errors = [x for _,x in sorted(zip(grid_sizes_recut,recut_errors))]
        xy_errors = [x for _,x in sorted(zip(grid_sizes_xy, xy_errors))]
        grid_sizes = sorted(grid_sizes_recut)
        assert(grid_sizes == sorted(grid_sizes_xy))
        radius_sizes = [i/4 for i in grid_sizes]

        xargs = [(radius_sizes, radius_sizes), r'Radius size (pixels)']
        yiter = (recut_errors, xy_errors)
        legends = (r'Recut', r'APP2')
        rplot(*xargs, yiter, r'Error rate (\%)', r'Calculate Radius and Prune Accuracy', args, legends=legends)

        data = json.load(open(f'{bench_fn}.json'))
        df = data['benchmarks']
        # only collect it once, the benchmark data is inherently ordered
        grid_sizes = [int(i['name'].split('/')[-1]) for i in df if 'recut' in i['name']]
        names = ['recut', 'xy', 'accurate' ]
        real_times = []
        time_unit = df[0]['time_unit']
        time_unit = 's'
        unit_conversion_factor = 1000
        for name in names:
            real_times.append( [float(i['real_time']) / unit_conversion_factor for i in df if name in i['name']])
        radius_sizes = [i/4 for i in grid_sizes]

        legends = (r'Recut $ O(n) $', r'APP2 $ O(nr^3) $', r'Accurate $ O(nr^4) $')
        rplot((radius_sizes, radius_sizes, radius_sizes), r'Radius size (pixels)', real_times,
                r'Elapsed time (%s)' % time_unit, r'Calculate Radius Performance Sequential', args, legends=legends)

def stages(args):
    ''' Show runtime comparison of stages'''
    stage_names = ['value', 'radius', 'prune', 'qc', 'g-cut']
    runtimes =     [7.4, 8.1, 9.7, 1.2, 16]
    seq_runtimes = [126.4, 72.9, 31.0, 0, 0]
    if args.save or args.show:
        field = r'Recut speedup factor %'
        ylabel = r'Runtime (mins)'
        title = r'Stage Runtime Comparison'
        # xargs = (list(range(len(stage_names))), r'Stages')
        xargs = (stage_names, r'Stages')
        yargs = (runtimes, ylabel)
        legends = (r'Recut', r'APP2')

        rplot(*xargs,
                *yargs,
                title,
                args, 
                legends=legends,
                yiter_secondary=seq_runtimes, 
                bar=True)

def value(args):
    ''' Fastmarching Performance '''
    cross_compile_flags=['NO_SCHEDULE', 'NO_INTERVAL_RV', 'SCHEDULE_INTERVAL_RV']
    baseline_flags = ['TEST_ALL_BENCHMARKS', 'USE_OMP_BLOCK']
    # desired_test_runs = [11, 12, 18, 19, 25, 26, 32, 33]
    desired_test_runs = [32, 33, 34, 35, 36, 37, 38]
    # desired_test_runs = range(11, 32)
    test_fn = args.output + 'value_test'

    for flag in cross_compile_flags:
        # wrap all test specifiers
        desired_tests = get_desired_test_info(args, desired_test_runs, test_fn, flag)

        if args.recompile:
            recompile(args, (*baseline_flags, flag))

        if args.rerun:
            for test_args in desired_tests:
                run_with_log(*test_args)

        # Gather all json files
        # load all desired test files
        test_dicts = []
        for fn, cmd in desired_tests:
            try:
                df = json.load(open(f'{fn}.json'))
                test_dicts.append(df['testsuites'][0]['testsuite'][0])
            except:
                print(f'Warning: fn:{fn}.json not found')

        # rearrange into lists by ratio
        ratios = (2, 4, 8)
        ratio_dicts = [filter_key_value(test_dicts, 'Grid / interval ratio', ratio) for ratio in ratios]

        # use this list of list structure to extract desired values by key
        r_grid_sizes = [extract_values(d, 'grid_size') for d in ratio_dicts ]
        r_ratios = [extract_values(d, 'Grid / interval ratio') for d in ratio_dicts ]
        test_ratios(ratios, r_ratios)
        r_iters = [extract_values(d, 'Iterations') for d in ratio_dicts ]
        r_total_time = [extract_values(d, 'Value update elapsed (s)') for d in ratio_dicts ]
        # comparison_test_dicts = [list(remove_if(d, 'total vertex difference vs sequential value', None)) for d in ratio_dicts]
        # for d in comparison_test_dicts:
            # vdiff = [int(d.get('total vertex difference vs sequential value')) for d in comparison_test_dicts]
        # total_seq = [int(d.get('sequential fastmarching tree size')) for d in comparison_test_dicts]
        # r_frac_difference = list(map(truediv, vdiff, total_seq))
        # print(r_frac_difference)
        # r_frac_difference = [map(truediv, extract_values(d, 'total vertex difference vs sequential value'), extract_values(d, 'sequential fastmarching tree size')) for d in comparison_test_dicts ]
        assert(len(r_grid_sizes[0]) == len(r_iters[0]))

        # throughput info
        r_selected_per_total_times = [extract_values(d, 'Selected vertices/s') for d in ratio_dicts]

        if args.save or args.show:
            local_kwargs = {}
            local_kwargs['legends'] = r_ratios
            local_kwargs['legend_metric'] = ' intervals/grid'

            xargs = (r_grid_sizes, r'Grid side length (pixels)')
            rplot(*xargs, r_iters,
                    r'Iterations', r'Intervals per grid vs. iterations {}'.format(flag), args,
                    **local_kwargs)

            rplot(*xargs,
                    r_selected_per_total_times, r'Selected vertices/s',
                    r'Intervals per grid vs. computational throughput {}'.format(flag),  args,
                    **local_kwargs)

            # rplot(*xargs,
                    # r_frac_difference, r'Error rate (\%)',
                    # r'Intervals per grid vs. error rate {}'.format(flag),
                    # args,
                    # **local_kwargs)

def get_hash():
    return subprocess.check_output(["git", "describe", "--always"]).strip().decode()

def recmake(args, flags):
    flagstr = ''
    for flag in flags:
        flagstr += f'-D{flag}=ON '

    GIT_HASH = get_hash()
    flagstr += f'-DGIT_HASH={GIT_HASH} '
    cmd = f'cd {args.project}; rm -rf build; mkdir build; cd build; ~/downloads/cmake-3.17.0-Linux-x86_64/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug {flagstr}'
    print(cmd)
    subprocess.run(cmd, shell=True)

def remake(args):
    cmd = f'cd {args.project}; cd build; ninja install'
    print(cmd)
    subprocess.run(cmd, shell=True)

def recompile(args, flags):
    ''' recompile with desired MACRO / PREPROC dfines'''
    recmake(args, flags)
    remake(args)

def run_with_log(fn, cmd):
    GIT_HASH = get_hash()
    log = f'{fn}.log'
    json_fn = f'{fn}.json'
    subprocess.run(['touch', log])
    log_cmd = f'{cmd} &>> {log}'
    # tag log with the git hash
    with open(log) as out:
        print(f'{cmd} > {log}')
        subprocess.run(cmd.split(), stdout=out)
    # print(log_cmd)
    # subprocess.run(log_cmd, shell=True)
    # subprocess.run(f'echo GIT_HASH:{GIT_HASH} >> {log}', shell=True)
    # every json file produced by this script is tagged with git hash
    subprocess.run("jq '.GIT_HASH = \"%s\"' %s" % (GIT_HASH, json_fn), shell=True)

def scalability(args):
    benchmark = True
    test = True
    name = 'scalability'
    benchmark_fn = args.output + f'{name}_bench'
    test_fn = args.output + f'{name}_test'
    desired_test_runs = range(39,46)
    baseline_flags = ['TEST_ALL_BENCHMARKS', 'USE_OMP_BLOCK']
    cross_compile_flags = ['NO_SCHEDULE']

    if test:
        desired_tests = []
        for flag in cross_compile_flags:
            # needs to be recompiled with TEST ALL BENCHMARKS preprocessor set to true
            # to gather all the correct tests
            if args.recompile:
                recompile(args, (*baseline_flags, flag))
            desired_tests.extend(get_desired_test_info(args, desired_test_runs, test_fn, flag))

    if args.rerun:
        rerun(desired_tests, test)

    if args.save or args.show:
        if test:
            test_jsons = [json.load(open(f'{args[0]}.json')) for args in desired_tests]
            test_dicts = [test_json['testsuites'][0]['testsuite'][0] for test_json in test_jsons]
            grid_sizes = (512, 1024, 2048, 4096)
            # rearrange dicts by their grid_size
            grid_dicts = [filter_key_value(test_dicts, 'grid_size',
                grid_size) for grid_size in grid_sizes]

            # extract desired test data info
            # extract block size
            block_sizes = [extract_values(grid_dict, 'block_size') for grid_dict in grid_dicts]

            ylabel = r'Selected vertices / computation (s)'
            throughput = [extract_values(grid_dict, ylabel) for grid_dict in grid_dicts]
            title = r'Image scale vs. throughput [1\% label density]'
            xargs = (block_sizes, r'Block side length (pixels)')
            yargs = (throughput, ylabel)

            rplot(*xargs,
                    *yargs,
                    title,
                    args,
                    legends=[str(g) for g in grid_sizes],
                    legend_metric=' grid size'
                    )

            # narrow down to the first two grid sizes
            # only up to 1024 has a comparison to the sequential runs
            exclude = 2
            seq_grid_sizes = grid_sizes[:exclude]
            # rearrange dicts by their grid_size
            seq_grid_dicts = grid_dicts[:exclude]
            seq_block_sizes = block_sizes[:exclude]

            field = r'Recut speedup factor %'
            speedup = [extract_values(grid_dict, field) for grid_dict in seq_grid_dicts]
            ylabel = r'Recut speedup factor (\%)'
            title = r'Speedup factor vs. sequential [1\% label density]'
            xargs = (seq_block_sizes, r'Block side length (pixels)')
            yargs = (speedup, ylabel)

            rplot(*xargs,
                    *yargs,
                    title,
                    args,
                    legends=[str(g) for g in seq_grid_sizes],
                    legend_metric=' grid size'
                    )


def rerun(desired_tests, test=False, benchmark=False, benchmark_fn='', benchmark_regex=''):
    if benchmark:
        benchmark_cmd = f'{args.binary}./recut_bench --benchmark_filter=load* --benchmark_out_format=json --benchmark_out={benchmark_fn}.json'
        run_with_log(benchmark_fn, benchmark_cmd)

    if test:
        for test_args in desired_tests:
            run_with_log(*test_args)

def read(args):
    benchmark = True
    test = True
    benchmark_fn = args.output + 'read_bench'
    test_fn = args.output + 'read_test'
    desired_test_runs= range(32,39)
    baseline_flags = ['TEST_ALL_BENCHMARKS', 'USE_OMP_BLOCK']
    cross_compile_flags = ['ALL']

    if test:
        desired_tests = []
        for flag in cross_compile_flags:
            # needs to be recompiled with TEST ALL BENCHMARKS preprocessor set to true
            # to gather all the correct tests
            if args.recompile:
                recompile(args, (*baseline_flags, flag))

            desired_tests.extend(get_desired_test_info(args, desired_test_runs, test_fn, flag))

    if args.rerun:
        rerun(desired_tests, test, benchmark, benchmark_fn, benchmark_regex)

    if args.save or args.show:

        if benchmark:
            # get benchmark component
            benchmark_data = json.load(open(f'{benchmark_fn}.json'))['benchmarks']

            # define desired fields
            fields = ["load_exact_tile", "load_tile_from_large_image"]
            pretty_name = ["Exact tile", "Tile in large image"]

            # extract desired benchmark data info
            bench_dicts = [filter_key_value(benchmark_data, 'name', field) for field in fields]
            grid_sizes = [extract_postfix(d, 'name') for d in bench_dicts ]

            #throughput
            bytes_per_second = [extract_values(d, 'bytes_per_second') for d in bench_dicts ]
            # convert to more convenient metric
            MB_per_second = [list(map(lambda x: x / 1000000, b)) for b in bytes_per_second]

            # latencies converted
            cpu_time_second = [list(map(lambda x: x / 1000, extract_values(d, 'cpu_time'))) for d in bench_dicts ]
            real_time_second = [list(map(lambda x: x / 1000, extract_values(d, 'real_time'))) for d in bench_dicts ]

            xargs = (grid_sizes, r'Grid side length (pixels)')

            rplot(*xargs,
                    MB_per_second, r'MB/s',
                    r'Image read throughput',
                    args,
                    legends=pretty_name)

            rplot(*xargs,
                    cpu_time_second, r'Time (s)',
                    r'Image read latency (CPU time)',
                    args,
                    legends=pretty_name)
            rplot(*xargs,
                    real_time_second, r'Time (s)',
                    r'Image read latency (real time)',
                    args,
                    legends=pretty_name)

            # load all desired test files
        if test:
            test_jsons = [json.load(open(f'{args[0]}.json')) for args in desired_tests]
            test_dicts = [test_json['testsuites'][0]['testsuite'][0] for test_json in test_jsons]
            filtered_test_dicts = filter_key_value(test_dicts, 'Grid / interval ratio', 1)
            assert(len(test_dicts) == len(filtered_test_dicts))

            # extract desired test data info
            ylabel = r'Value update computation (s)'
            title = r'Value update computation vs image read'
            test_grid_sizes = extract_values(filtered_test_dicts, 'grid_size')
            comp_time = extract_values(filtered_test_dicts, ylabel)

            xargs = ((test_grid_sizes, *grid_sizes), r'Grid side length (pixels)')
            print(xargs)
            yiter = [comp_time, *real_time_second]
            print(yiter)

            rplot(*xargs,
                    yiter, r'Wall time (s)',
                    title,
                    args,
                    legends=('Sequential computation', 'Exact tile read', 'Tile in large image read')
                    )

def main(args):

    if args.all:
        radius(args)
        value(args)
        read(args)
        stages(args)
        return

    if args.case == 'radius':
        radius(args)
    if args.case == 'value':
        value(args)
    if args.case == 'read':
        read(args)
    if args.case == 'scalability':
        scalability(args)
    if args.case == 'stages':
        stages(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recompile code with various preprocessor definitions and optionally run and plot all results")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--all', help="Use all known cases",
            action="store_true")
    group.add_argument('-c', '--case', help="Specify which case to use",
            choices=['stages', 'radius', 'scalability', 'value', 'read'])

    parser.add_argument('-r', '--rerun', help="Rerun all to generate new test data", action="store_true")
    parser.add_argument('-w', '--save', help="save all plots to file", action="store_true")
    parser.add_argument('-s', '--show', help="show figures interactively while generating", action="store_true")
    parser.add_argument('-t', '--type', help="image output type", choices=['eps', 'png', 'pdf'], default='eps')
    parser.add_argument('-l', '--recompile', help="recompile with CMake then Make", action="store_true")
    parser.add_argument('-d', '--dpi', help="image resolution", type=int, default=100)

    project_dir = os.path.join(os.path.dirname(__file__), '../')
    bin_dir = project_dir + 'bin/'
    data_dir = project_dir + 'data/'
    parser.add_argument('-o', '--output', help="output data directory", default=data_dir)
    parser.add_argument('-b', '--binary', help="binary directory", default=bin_dir)
    parser.add_argument('-p', '--project', help="project directory", default=project_dir)

    args = parser.parse_args()
    main(args)
