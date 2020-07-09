#!/usr/bin/env python

import argparse
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

def rplot(xiter, xlabel, yiter, ylabel, title, args, lineprops=['k-x', 'r-o', 'g-d'],
        legends=[''], legend_metric=''):
    assert(len(xiter) == len(yiter))
    assert(len(yiter) == len(legends))
    lineprops = lineprops[:len(legends)]
    title = title.replace('_', '-')

    for x, y, lineprop, legend in zip(xiter, yiter, lineprops, legends):
        plt.plot(x, y, lineprop, label=str(legend) + legend_metric)
    plt.xlabel(xlabel)
    plt.xticks(x, rotation=75)
    plt.ylabel(ylabel)
    if len(xiter) > 1:
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    fig_output_path = f'{args.output}{title}.{args.type}'

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

def radius(args):
    """ Radius Stage Plots """
    # TODO timestamp
    radius_json = args.output + 'test_radius.json'
    bench_json = args.output + 'bench_radius.json'
    if args.rerun:
        # run the radius script from scratch

        # test
        test_cmd = args.binary + './recut_test --gtest_filter=Radius.* --gtest_output=json:' + radius_json
        tlog = 'test.log'
        subprocess.run(['touch', tlog])
        with open(tlog) as out:
            subprocess.run(test_cmd.split(), stdout=out)

        # bench
        bench_cmd = args.binary + './recut_bench --benchmark_filter=radius.* --benchmark_out_format=json:../data/bench_radius.json'
        blog = 'bench.log'
        subprocess.run(['touch', blog])
        with open(blog) as out:
            subprocess.run(bench_cmd.split(), stdout=out)

    if args.save or args.show:
        df = json.load(open(radius_json))
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

        xargs = [radius_sizes, r'Radius size (pixels)']
        yiter = (recut_errors, xy_errors)
        legends = (r'Recut', r'APP2')
        rplot(*xargs, yiter, r'Error rate (\%)', r'Calculate Radius and Prune Accuracy', args, legends=legends)

        data = json.load(open(bench_json))
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
        rplot(radius_sizes, r'Radius size (pixels)', real_times,
                r'Elapsed time (%s)' % time_unit, r'Calculate Radius Performance Sequential', args, legends=legends)

        # ## Fastmarching Performance

def value(args):
    flags=['NO_SCHEDULE', 'NO_INTERVAL_RV', 'SCHEDULE_INTERVAL_RV']

    for flag in flags:
        # Gather all json files
        test_dicts = []
        # for i in range(11, 32):
        for i in range(11, 12):
            name='{}recut-test-{}-{}-2.json'.format(args.output, flag, i)
            try:
                df = json.load(open(name))
            except:
                print('Warning: fn:{} not found'.format(name))
            test_dicts.append(df['testsuites'][0]['testsuite'][0])
        len(test_dicts)

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

            rplot(*xargs,
                    r_frac_difference, r'Error rate (\%)',
                    r'Intervals per grid vs. error rate {}'.format(flag),
                    args,
                    **local_kwargs)

def read(args):
    benchmark_fn = args.output + 'load_bench'
    test_fn = args.output + 'test_detail'
    fns = (benchmark_fn, test_fn)

    if args.rerun:
        # TODO needs to be recompiled with TEST ALL BENCHMARKS preprocessor set to true
        cmd = (f'bin/./recut_test --gtest_output=json:{test_fn}.json --gtest_filter=*.ChecksIfFinalVerticesCorrect*',
              f'bin/./recut_bench --benchmark_filter=load* --benchmark_out_format=json --benchmark_out={benchmark_fn}.json')

        for fn, cmd in zip(fns, cmds):
            log = f'{fn}.log'
            subprocess.run(['touch', log])
            with open(log) as out:
                subprocess.run(cmd.split(), stdout=out)

    if args.save or args.show:

        # get benchmark component
        benchmark_data = json.load(open(f'{benchmark_fn}.json'))['benchmarks']

        # extract desired benchmark data info
        tests = ["load_exact_tile", "load_tile_from_large_image"]
        pretty_name = ["Exact tile", "Tile in large image"]
        bench_dicts = [filter_key_value(benchmark_data, 'name', test) for test in tests]
        grid_sizes = [extract_postfix(d, 'name') for d in bench_dicts ]
        bytes_per_second = [extract_values(d, 'bytes_per_second') for d in bench_dicts ]
        MB_per_second = [list(map(lambda x: x / 1000000, b)) for b in bytes_per_second]
        cpu_time_second = [list(map(lambda x: x / 1000, extract_values(d, 'cpu_time'))) for d in bench_dicts ]
        real_time_second = [list(map(lambda x: x / 1000, extract_values(d, 'real_time'))) for d in bench_dicts ]

        xargs = (grid_sizes, r'Grid side length (pixels)')

        # rplot(*xargs,
                # MB_per_second, r'MB/s',
                # r'Image read throughput',
                # args,
                # legends=pretty_name)

        # rplot(*xargs,
                # cpu_time_second, r'Time (s)',
                # r'Image read latency (CPU time)',
                # args,
                # legends=pretty_name)

        # rplot(*xargs,
                # real_time_second, r'Time (s)',
                # r'Image read latency (real time)',
                # args,
                # legends=pretty_name)

        # load test component
        desired_test_runs= [slice(11,18)]
        test_json = json.load(open(f'{test_fn}.json'))
        test_dicts = [test_json['testsuites'][0]['testsuite'][r] for r in desired_test_runs]

        # extract desired test data info
        ylabel = 'Value update computation (s)'
        grid_sizes = [extract_values(d, 'grid_size') for d in test_dicts]
        comp_time = [extract_values(d, ylabel) for d in test_dicts]
        import pdb; pdb.set_trace()

        # [ extract_values(d, 'total vertex difference vs sequential value') for d in test_dicts[0]]
# extract_values(test_dicts, 'total vertex difference vs sequential value')
# [d.get('total vertex difference vs sequential value') for d in test_dicts]
# diff = filter(lambda x: x.get('total vertex difference vs sequential value'), test_dicts)
# comparison_test_dicts = remove_if(test_dicts, 'total vertex difference vs sequential value', None)


def main(args):
    if args.all:
        radius(args)
        value(args)
        read(args)
        return

    if args.case == 'radius':
        radius(args)
    if args.case == 'case':
        value(args)
    if args.case == 'read':
        read(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recompile code with various preprocessor definitions and optionally run and plot all results")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--all', help="Use all known cases",
            action="store_true")
    group.add_argument('-c', '--case', help="Specify which case to use",
            choices=['radius', 'value', 'read'])

    parser.add_argument('-r', '--rerun', help="Rerun all to generate new test data", action="store_true")
    parser.add_argument('-w', '--save', help="save all plots to file", action="store_true")
    parser.add_argument('-s', '--show', help="show figures interactively while generating", action="store_true")
    parser.add_argument('-t', '--type', help="image output type", choices=['eps', 'png', 'pdf'], default='eps')
    parser.add_argument('-d', '--dpi', help="image resolution", type=int, default=300)

    project_dir = getcwd() + '/../'
    bin_dir = project_dir + 'bin/'
    data_dir = project_dir + 'data/'
    parser.add_argument('-o', '--output', help="output data directory", default=data_dir)
    parser.add_argument('-b', '--binary', help="binary directory", default=bin_dir)

    args = parser.parse_args()
    main(args)
