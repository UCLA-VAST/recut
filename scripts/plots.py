#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import subprocess
import json
from operator import truediv

project_dir = '/curr/kdmarrett/accelerate/'
bin_dir = project_dir + 'bin/'
data_dir = project_dir + 'data/'
save_fig = True
show_fig = True
run_radius = False

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

extract_values = lambda dictions, extract_key: [int(diction[extract_key]) for diction in dictions]
extract_values_range = lambda dictions, extract_key, start=0, stop=-1: [int(diction[extract_key]) for diction in dictions[start:stop]]
filter_key_value = lambda dicts, key, value: list(filter(lambda d: d[key] == str(value), dicts))


# # Radius Stage Plots

# In[14]:


radius_json = data_dir + 'test_radius.json'


# In[15]:


# run the radius script from scratch
if run_radius:
    test_cmd = bin_dir + './recut_test --gtest_filter=Radius.* --gtest_output=json:' + radius_json
    tlog = 'test.log'
    subprocess.run(['touch', tlog])
    with open(tlog) as out:
        subprocess.run(test_cmd.split(), stdout=out)
else: #just load it
    df = pd.read_json(radius_json)


# In[17]:


test= df['testsuites'][0]['testsuite'][0]
recut_keys = [i for i in test.keys() if 'Recut' in i]
xy_keys = [i for i in test.keys() if 'XY' in i]


# In[18]:


recut_errors = [test[i] for i in recut_keys]
xy_errors = [test[i] for i in xy_keys]
grid_sizes_recut = [int(key.split()[3]) for key in recut_keys]
grid_sizes_xy = [int(key.split()[3]) for key in xy_keys]

recut_errors = [x for _,x in sorted(zip(grid_sizes_recut,recut_errors))]
xy_errors = [x for _,x in sorted(zip(grid_sizes_xy, xy_errors))]
grid_sizes = sorted(grid_sizes_recut)
assert(grid_sizes == sorted(grid_sizes_xy))
radius_sizes = [i/4 for i in grid_sizes]


# In[10]:


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


# In[19]:


bench_json = data_dir + 'bench_radius.json'


# In[26]:


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


# In[31]:


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


# In[64]:


bench_cmd = bin_dir + './recut_bench --benchmark_filter=radius.* --benchmark_out_format=json:../data/bench_radius.json'
blog = 'bench.log'
subprocess.run(['touch', blog])
with open(blog) as out:
    subprocess.run(bench_cmd.split(), stdout=out)


# In[ ]:


fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
if save_fig:
    plt.savefig(fig_output_path)
    print(fig_output_path)
if show_fig:
    plt.show()
else:
    plt.close()


# ## Fastmarching Performance

# In[11]:


flags=['ALL', 'NO_INTERVAL_RV', 'SCHEDULE_INTERVAL_RV']
for flag in flags:
    # Gather all json files
    test_dicts = []
    for i in range(11, 32):
        name='{}recut-test-{}-{}.json'.format(data_dir, flag, i)
        try:
            df = pd.read_json(name)
        except:
            print('Warning: fn:{} not found'.format(name))
        test_dicts.append(df['testsuites'][0]['testsuite'][0])
    len(test_dicts)
    
    # Extract into lists
    ratios = (2, 4, 8)
    ratio_dicts = [filter_key_value(test_dicts, 'grid / interval ratio', ratio) for ratio in ratios]
    r_grid_sizes = [extract_values(d, 'grid_size') for d in ratio_dicts ]
    r_ratios = [extract_values(d, 'grid / interval ratio') for d in ratio_dicts ]
    r_iters = [extract_values(d, 'iterations') for d in ratio_dicts ]
    r_frac_difference = [map(truediv, extract_values(d, 'total vertex difference vs sequential value'), extract_values(d, 'sequential fastmarching tree size')) for d in ratio_dicts ]
    assert(len(r_grid_sizes[0]) == len(r_iters[0]))
    
    # throughput info
    r_selected_per_total_times = [extract_values(d, 'selected vertices / total time') for d in ratio_dicts]

    # Plot
    #fig, ax = plt.subplots()
    for ratio, lineprop, grid_size, iteration, check_ratios in zip(ratios, lineprops, r_grid_sizes, r_iters, r_ratios):
        for check_ratio in check_ratios:
            assert (ratio == check_ratio),"ratio did not match"
        plt.plot(grid_size, iteration, lineprop, label=str(ratio) + ' intervals/grid')
    plt.xlabel(r'Grid side length (pixels)')
    plt.xticks(grid_size, rotation=75)
    plt.ylabel(r'Iterations')
    # plt.ylabel(r'Elapsed time (%s)' % time_unit)
    plt.legend()
    title = r'Intervals per grid vs. iterations {}'.format(flag.replace('_', '-'))
    plt.title(title)
    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    save_fig=False
    if save_fig:
        plt.savefig(fig_output_path, dpi=300)
        print(fig_output_path)
    if show_fig:
        plt.show()
    else:
        plt.close()

    for ratio, lineprop, grid_size, selected_per_total_time, check_ratios in zip(ratios, lineprops, r_grid_sizes, r_selected_per_total_times, r_ratios):
        for check_ratio in check_ratios:
            assert (ratio == check_ratio),"ratio did not match"
        plt.plot(grid_size, selected_per_total_time, lineprop, label=str(ratio) + ' intervals/grid')
    plt.xlabel(r'Grid side length (pixels)')
    plt.xticks(grid_size, rotation=75)
    plt.ylabel(r'Selected vertices/s')
    # plt.ylabel(r'Elapsed time (%s)' % time_unit)
    plt.legend()
    title = r'Intervals per grid vs. computational throughput {}'.format(flag.replace('_', '-'))
    plt.title(title)
    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    if save_fig:
        plt.savefig(fig_output_path, dpi=300)
        print(fig_output_path)
    if show_fig:
        plt.show()
    else:
        plt.close()
         
    for ratio, lineprop, grid_size, frac_difference in zip(ratios, lineprops, r_grid_sizes, r_frac_difference):
        plt.plot(grid_size, frac_difference, lineprop, label=str(ratio) + ' intervals/grid')
    plt.xlabel(r'Grid side length (pixels)')
    plt.xticks(grid_size, rotation=75)
    plt.ylabel(r'Error rate (\%)')
    plt.legend()
    title = r'Intervals per grid vs. error rate {}'.format(flag.replace('_', '-'))
    plt.title(title)
    fig_output_path = '%s%s.png' % (data_dir, title.replace(' ', '_'))
    if save_fig:
        plt.savefig(fig_output_path, dpi=300)
        print(fig_output_path)
    if show_fig:
        plt.show()
    else:
        plt.close()
                                                     


# In[13]:


[ extract_values(d, 'total vertex difference vs sequential value') for d in test_dicts[0]]


# In[15]:


extract_values(test_dicts, 'total vertex difference vs sequential value')


# In[23]:


[d.get('total vertex difference vs sequential value') for d in test_dicts]


# In[25]:


diff = filter(lambda x: x.get('total vertex difference vs sequential value'), test_dicts)


# In[29]:


test_dicts[4]


# In[ ]:




