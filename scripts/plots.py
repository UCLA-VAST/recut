#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import subprocess
import json
from operator import truediv
from matplotlib import rc
from statistics import mean
from os import getcwd

# project_dir = '/curr/kdmarrett/accelerate/'
project_dir = getcwd() + '/../'
bin_dir = project_dir + 'bin/'
data_dir = project_dir + 'data/'
save_fig = False
show_fig = False
run_radius = False
plot_radius = False
run_test = True
plot_test = False
dpi=300

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

extract_values = lambda dictions, extract_key: [int(diction.get(extract_key)) for diction in dictions]
extract_values_range = lambda dictions, extract_key, start=0, stop=-1: [int(diction[extract_key]) for diction in dictions[start:stop]]
filter_key_value = lambda dicts, key, value: list(filter(lambda d: d[key] == str(value), dicts))
remove_if = lambda dicts, key, value: list(filter(lambda d: d.get(key) != value, dicts))

def test_ratios(ratios, r_ratios):
    for ratio, check_ratios in zip(ratios, r_ratios):
        for check_ratio in check_ratios:
            assert (ratio == check_ratio),"ratio did not match"

def rplot(xiter, xlabel, yiter, ylabel, title, lineprops, legenditer=[],
        legend_metric='', data_dir=getcwd(), show_fig=False, save_fig=False,
        dpi=300):
    assert(len(xiter) == len(yiter))
    assert(len(yiter) == len(lineprops))
    assert(len(lineprops) == len(legenditer))
    for x, y, lineprop, legend in zip(xiter, yiter, lineprops, legenditer):
        plt.plot(x, y, lineprop, label=str(legend) + legend_metric)
    plt.xlabel(xlabel)
    plt.xticks(x, rotation=75)
    plt.ylabel(ylabel)
    if len(xiter) > 1:
        plt.legend()
    plt.title(title.replace('_', '-'))
    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    if save_fig:
        plt.savefig(fig_output_path, dpi=dpi)
        print(fig_output_path)
    if show_fig:
        plt.show()
    else:
        plt.close()

# # Radius Stage Plots
radius_json = data_dir + 'test_radius.json'

# run the radius script from scratch
if run_radius:
    test_cmd = bin_dir + './recut_test --gtest_filter=Radius.* --gtest_output=json:' + radius_json
    tlog = 'test.log'
    subprocess.run(['touch', tlog])
    with open(tlog) as out:
        subprocess.run(test_cmd.split(), stdout=out)
else: #just load it
    df = pd.read_json(radius_json)

if plot_radius:
    test= df['testsuites'][0]['testsuite'][0]
    recut_keys = [i for i in test.keys() if 'Recut' in i]
    xy_keys = [i for i in test.keys() if 'XY' in i]

    recut_errors = [test[i] for i in recut_keys]
    xy_errors = [test[i] for i in xy_keys]
    grid_sizes_recut = [int(key.split()[3]) for key in recut_keys]
    grid_sizes_xy = [int(key.split()[3]) for key in xy_keys]

    recut_errors = [x for _,x in sorted(zip(grid_sizes_recut,recut_errors))]
    xy_errors = [x for _,x in sorted(zip(grid_sizes_xy, xy_errors))]
    grid_sizes = sorted(grid_sizes_recut)
    assert(grid_sizes == sorted(grid_sizes_xy))
    radius_sizes = [i/4 for i in grid_sizes]

    plt.plot(radius_sizes, recut_errors, 'k-x', label=r'Recut')
    plt.plot(radius_sizes, xy_errors, 'r-o', label=r'APP2')
    plt.xlabel(r'Radius size (pixels)')
    plt.xticks(radius_sizes)
    plt.ylabel(r'Error rate (\%)')
    plt.legend()
    title = r'Calculate Radius and Prune Accuracy'
    plt.title(title)
    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    if save_fig:
        plt.savefig(fig_output_path)
        print(fig_output_path)

    bench_json = data_dir + 'bench_radius.json'

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
    real_times

    plt.plot(radius_sizes, real_times[0], 'k-x', label=r'Recut $ O(n) $')
    plt.plot(radius_sizes, real_times[1], 'r-o', label=r'APP2 $ O(nr^3) $')
    plt.plot(radius_sizes, real_times[2], 'g-d', label=r'Accurate $ O(nr^4) $')
#plt.plot(radius_sizes, real_times[2], 'k-', label=r'$O(r)')
    plt.xlabel(r'Radius size (pixels)')
    plt.xticks(radius_sizes)
    plt.ylabel(r'Elapsed time (%s)' % time_unit)
    plt.legend()
# plt.yscale('log', basey=10)
#plt.xscale('log', basex=10)
    title = r'Calculate Radius Performance Sequential'
    plt.title(title)
    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    if save_fig:
        plt.savefig(fig_output_path, dpi=300)
        print(fig_output_path)


    bench_cmd = bin_dir + './recut_bench --benchmark_filter=radius.* --benchmark_out_format=json:../data/bench_radius.json'
    blog = 'bench.log'
    subprocess.run(['touch', blog])
    with open(blog) as out:
        subprocess.run(bench_cmd.split(), stdout=out)

    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    if save_fig:
        plt.savefig(fig_output_path)
        print(fig_output_path)
    if show_fig:
        plt.show()
    else:
        plt.close()


# ## Fastmarching Performance

if run_test:
    flags=['ALL', 'NO_INTERVAL_RV', 'SCHEDULE_INTERVAL_RV']
    lineprops = ['k-x', 'r-o', 'g-d']

    for flag in flags:
        # Gather all json files
        test_dicts = []
        for i in range(11, 32):
            name='{}recut-test-{}-{}-2.json'.format(data_dir, flag, i)
            try:
                df = pd.read_json(name)
            except:
                print('Warning: fn:{} not found'.format(name))
            test_dicts.append(df['testsuites'][0]['testsuite'][0])
        len(test_dicts)

        # rearrange into lists by ratio
        ratios = (2, 4, 8)
        ratio_dicts = [filter_key_value(test_dicts, 'grid / interval ratio', ratio) for ratio in ratios]

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

        if plot_test:
            rplot(r_grid_sizes, r'Grid side length (pixels)', r_iters,
                    r'Iterations', r'Intervals per grid vs. iterations {}'.format(flag),
                    lineprops, 
                    legenditer=r_ratios, legend_metric=' intervals/grid',
                    show_fig=show_fig, save_fig=save_fig, dpi=dpi)

            rplot(r_grid_sizes, r'Grid side length (pixels)',
                    r_selected_per_total_times, r'Selected vertices/s',
                    r'Intervals per grid vs. computational throughput {}'.format(flag), 
                    lineprops, legenditer=r_ratios,
                    legend_metric=' intervals/grid', show_fig=show_fig,
                    save_fig=save_fig, dpi=dpi)

            rplot(r_grid_sizes, r'Grid side length (pixels)',
                    r_frac_difference, r'Error rate (\%)',
                    r'Intervals per grid vs. error rate {}'.format(flag),
                    lineprops, legenditer=r_ratios,
                    legend_metric=' intervals/grid', show_fig=show_fig,
                    save_fig=save_fig, dpi=dpi)

# [ extract_values(d, 'total vertex difference vs sequential value') for d in test_dicts[0]]
# extract_values(test_dicts, 'total vertex difference vs sequential value')
# [d.get('total vertex difference vs sequential value') for d in test_dicts]
# diff = filter(lambda x: x.get('total vertex difference vs sequential value'), test_dicts)
# comparison_test_dicts = remove_if(test_dicts, 'total vertex difference vs sequential value', None)
