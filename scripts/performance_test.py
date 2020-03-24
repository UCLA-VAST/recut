import json
import glob
import os
import subprocess
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

analyze = False

# Perf vars
perf = True
perf_args = ['L1-dcache-misses', 'L1-dcache-loads', 'LLC-load-misses',
        'LLC-loads', 'faults', 'instructions', 'cycles']
# 'cache-misses', 'cache-references',
perf_args_idx = [0 for i in perf_args] # get first arg

# Performance run behavior
# Compiler variables
compile_options = ['baseline', 'domain-split', 'reject-revisits', 'no-wait', 'work-stealing', 'mmap', 'infrastructure', 'data-structures']
compile_option_idx = 2
compile_option_run = compile_options[compile_option_idx]
print('Currently running compile option: %s' % compile_option_run)
show_figs = False
parallel = True
pr = '0' # bool to conduct hierarchical pruning after FM
n_off = 0 # offset of which image to use
# n_off = 1
save_dat = False # good for long runs, bad when in active research mode
# debug = False
# revisit = True
LOG = True
N = 1 # number of runs to repeat
grid_len = 256
grid_lens = [grid_len]
# image_dims = [7680, 8448, 383]
# image dims represent the total area to cover in a run,
# note an image can be split in to subdomains for processing 
# those subdomains are currently named intervals. intervals
# are unique in that they represent the level of memory that is
# memory mapped at one time. intervals can be further divided
# into blocks for higher granularity. By default an interval
# has 1 block and these blocks can run independently on their own
# thread.
image_dims = [grid_len, grid_len, grid_len]
block_sizes = [grid_len]
# block_sizes = [grid_len, grid_len / 2, grid_len / 4, grid_len / 8]
# image_dims = [1792, 1664, 352]
image_vox_num = image_dims[0] * image_dims[1] * image_dims[2]
image_vox_num_scaled = image_vox_num / 1e7

# Test image specifiers
tcase = 0 # always 4 for synthetic neurons
if tcase < 4:
    fp = '1.0'
else:
    fp = '0.01' # Foreground Percent: 1.0 means 100% of pixels are set as foreground
# doesn't pass this in to script, instead it uses max_int
# to directly specify the max intensity value to save computation
# note this also requires passing in the bkg_thresh value
# to pass in a fp and force a calculation use "-fp .01" for the cl options
slt_pct = int(100 * float(fp)) # foreground percent to specify
correct_select_num = int(float(fp) * image_vox_num)
bkg_thresh = '0'
max_int = '1'
# by setting bkg_thresh to 0 and max_int to 1 we avoid needing
# to calculate the right bkg_thresh value via computation
# and needing to traverse the image to find the min and max values
# this should always be done for synthetic generated images since
# all values for tcase4 have 0 (bkg) or 1 (foreground)
# restart_factors = [5, 10, 30, 0][::-1]
restart_factors = [0]
threads = [56]
res_lvl = '0'
ch = '0' # always 0 for now

# output data vars
prune = []
markers = []
revisits = []
par_iteration = []
times = []
par_times = []
par_generating_times = []
par_init_times = []
update_times = []
revisits = []
perf_times = np.zeros((len(perf_args), len(block_sizes), N))

if len(block_sizes) > 1:
    par_update_times = np.zeros((len(block_sizes), N))
elif len(threads) > 1:
    par_update_times = np.zeros((len(threads), N))
elif len(restart_factors) > 1:
    par_update_times = np.zeros((len(restart_factors), N))
else:
    par_update_times = np.zeros((N))
par_markers = np.zeros(par_update_times.shape)

# Directory inputs and outputs
# curr = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.abspath(os.path.join(curr, os.pardir))
parent = '/tmp'
base_dir = parent + '/data'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
print("Output files have been written to %s" % base_dir)

run_dir = base_dir + '/run_data'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

swc_dir = base_dir + '/swcs'
if not os.path.exists(swc_dir):
    os.makedirs(swc_dir)

perf_output_dir = base_dir + '/perf'
if not os.path.exists(perf_output_dir):
    os.makedirs(perf_output_dir)

fig_output_dir = base_dir + '/fig'
if not os.path.exists(fig_output_dir):
    os.makedirs(fig_output_dir)

# img_dir = base_dir + '/2019-02-07_16.07.27_Protocol_FusionStitcher_BCL11b_KO_CAMK2_cortex_resolution0_tif/'
# segmentation_dir = base_dir + '/segmented'
# filled_soma_dir = base_dir + '/filled'
# manual_dir = base_dir + '/manual_modification'
# marker_dir = base_dir + '/marker_files'
app_suff = '.swc'
# swc_output = swc_dir + '/' + img_dir.split('/')[0]
swc_output = swc_dir;
# img_dir = '/curr/kdmarrett/mcp3d/bin/test_images/256/slt_pct1'
log_dir_path = '%s/logs/' % base_dir
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)
FORMAT = '%m_%d_%H_%M_%S'

# Optionally save data to dict to save to json
data = {}
data['grid_lens'] = grid_lens
data['block_sizes'] = block_sizes
data['restart_factors'] = restart_factors
data['threads'] = threads
data['N'] = N
data['fp'] = fp

# center the offsets around the soma location evenly
# making sure not to extend into negative territory or outside
# bounds
adjust = lambda a, b : max(a - b / 2, 0)
adjust_ie = lambda offset, dim, grid_len : min(image_dims[dim] - offset, grid_len)

def parse_time(log, phrase, token_idx=-2):
    remove_chars = ['%', '(', ')', ',']
    for line in log.split('\n'):
        if phrase in line:
            line = filter(lambda i: i not in remove_chars, line)
            if not line:
                print('WARNING: no %s found' % phrase)
            return float(line.split()[token_idx])

for grid_len in grid_lens:
    img_dir = '/curr/kdmarrett/mcp3d/bin/test_images/%d/tcase%d/slt_pct%d' % (grid_len, tcase, slt_pct)
    assert(os.path.exists(img_dir))
    marker_dir = img_dir.replace('images', 'markers')
    # print("Starting range with parallel: " + str(parallel))
    for ti, thread in enumerate(threads):
        for bi, bs in enumerate(block_sizes):
            if parallel:
                print('\n\nTesting bs: %d' % bs)
            if bs > grid_len:
                continue
            for ri, rs in enumerate(restart_factors):
                if parallel:
                    print('\nTesting rs: %.3f' % rs)
                marker_file_path = ''
                # for ni, marker_file_path in enumerate(glob.glob(marker_dir + '/marker_*')[n_off:(n_off + N)]):
                for ni in range(N):
                    marker_file_path = glob.glob(marker_dir + '/marker*')[0]
                    x, y, z = [int(i) for i in open(marker_file_path,
                            'r').read().split('\n')[1].split(',')]
                    # z, y, x, mass = [int(i) for i in marker_file_path.split('_')[-4:]] 
                    marker_file = marker_file_path.split('/')[-1]
                    za = adjust(z, grid_len)
                    ya = adjust(y, grid_len)
                    xa = adjust(x, grid_len)
                    ze = adjust_ie(za, 2, grid_len)
                    ye = adjust_ie(ya, 1, grid_len)
                    xe = adjust_ie(xa, 0, grid_len)
                    swc_output_file = swc_output + '/' +  '_'.join([ str(z), str(y), str(x)]) + app_suff
                    os.system('touch ' + swc_output_file)
                    bfilter = '--benchmark_filter=bench_critical_loop/%d' % grid_len
                    params = ['../bin/recut_bench', bfilter]
                    # params = ['../bin/recut', img_dir, ch,
                            # '-inmarker', marker_dir + '/', '-io', str(za), str(ya), str(xa),
                            # '-ie', str(ze), str(ye), str(xe),
                            # '-lt', '50', '-pr', pr, '-bs', str(bs),
                            # '-ct', '1', '-rl', res_lvl, '-outswc', 
                            # swc_output_file, '-bg-thresh',
                            # bkg_thresh, '-max', max_int]
                    # params = ['../bin/vaa3d_app2', img_dir, ch,
                            # '-inmarker', marker_file_path, '-io', str(za), str(ya), str(xa),
                            # '-ie', str(ze), str(ye), str(xe),
                            # '-lt', '50', '-fp', fp, '-pr', pr, '-bs', str(bs),
                            # '-ct', '1', '-rl', res_lvl, '-outswc',
                            # swc_output_file ]
                    # if parallel:
                        # params.append('-pl')
                        # params.append(str(thread))
                    # if rs > 0:
                        # params.append('-rs')
                        # params.append(str(rs))

                    # LOGGING
                    # perf wrap same parameters in cli call 
                    if perf:
                        # perf_params = ['perf stat --log-fd 2 --append -e' ]
                        perf_fn = perf_output_dir + '/perf_%s.log' % datetime.now().strftime(FORMAT)
                        # perf does not redirect to output file properly when called from python at this time
                        # perf_params = ['perf stat -o %s -e' % perf_fn ]
                        perf_params = ['perf stat -e']
                        perf_params.append(','.join(perf_args))
                        perf_params.extend(params)
                        # instead redirect everything, saving google benchmarks stats as well
                        # perf_params.append(' >& %s' % perf_fn)
                        jparams = ' '.join(perf_params)
                    else:
                        jparams = ' '.join(params)
                    print('\t' + jparams)

                    try:
                        log = subprocess.check_output(params)
                        print("Finished performance run... success\n")
                    except subprocess.CalledProcessError as e:
                        print(e.output)
                        print('Error running command: ' + '"' + str(e.cmd) + '"' + ' see above shell error')
                        print('Return code: ' + str(e.returncode))
                        # continue
                    slog = log.decode('ascii')
                    log_path = '%s%s_%s.log' % (log_dir_path, marker_file, datetime.now().strftime(FORMAT))
                    if LOG:
                        logfile = open(log_path, 'w+')
                        logfile.write(jparams)
                        logfile.write('\n')
                        logfile.write(slog)
                        logfile.close()
                    print('Created log for reconstruction: %s...' % log_path)
                    print('Created output: ' + swc_output_file)
                    print('Passed perf output: ' + perf_fn)

                    # PARSING LOG -> SAVE TO ARRAY
                    if parallel:
                        marker = parse_time(slog, 'Total marker size before pruning')
                        if marker:
                            par_markers[bi][ni] = marker
                            print('Total markers: %d..' % int(marker))
                        # par_iteration.append(parse_time(slog, 'Total iteration'))
                        # par_iteration.append(parse_time(slog, 'block iterations', -1))
                        # if revisit:
                            # revisits.append(parse_time(slog, 'Total rejected revisits'))
                            # revisits.append(parse_time(slog, 'Total revisits'))
                            # revisits = [i for i in revisits if i != None]
                            # if revisits:
                                # print('Total revisits: %d..' % int(revisits[-1]))
                        # print('Total iterations: %d..' % int(par_iteration[-1]))
                        # prune.append(parse_time(slog, 'Pruning neuron tree wtime'))
                        # par_times.append(parse_time(slog, 'Finished total updating'))

                        # if len(block_sizes) > 1:
                            # par_update_times[bi][ni] = parse_time(slog, 'Finished updating')
                            # print('Total time: %.1f, update time: %.1f' %
                                    # (par_times[-1], par_update_times[bi][ni]))
                        # elif len(threads) > 1:
                            # par_update_times[ti][ni] = parse_time(slog, 'Finished updating')
                            # print('Total time: %.1f, update time: %.1f' %
                                    # (par_times[-1], par_update_times[ti][ni]))
                        # elif len(restart_factors) > 1:
                            # par_update_times[ri][ni] = parse_time(slog, 'Finished updating')
                            # print('Total time: %.1f, update time: %.1f' %
                                    # (par_times[-1], par_update_times[ri][ni]))
                        # else:
                        # par_update_times[bi][ni] = parse_time(slog, 'Finished marching')

                        # perf logging
                        if perf:
                            plog = open(perf_fn, 'r').read()
                            for i, (arg, idx) in enumerate(zip(perf_args, perf_args_idx)):
                                perf_times[i][bi][ni] = parse_time(plog, arg, idx)

                        # if pr == '1':
                            # print('Total time: %.1f, update time: %.1f, prune time %.1f' % (par_times[-1], par_update_times[-1]), prune[-1])
                        # else:
                            # print('Total time: %.1f, update time: %.1f' % (par_times[-1], par_update_times[-1]))
                        # par_generating_times.append(parse_time(slog, 'Finished generating'))
                        # par_init_times.append(parse_time(slog, 'Finished initialization'))
                    # else:
                        # markers.append(parse_time(slog, 'Total marker'))
                        # print('Total markers: %d..' % int(markers[-1]))
                        # times.append(parse_time(slog, 'fastmarching_tree'))
                        # update_times.append(parse_time(slog, 'Finished updating'))
                        # prune.append(parse_time(slog, 'Pruning neuron tree wtime'))
                        # if pr == '1':
                            # print('Total time: %.1f, update time: %.1f, prune time %.1f' % (times[-1],
                                        # update_times[-1], prune[-1]))
                        # else:
                            # print('Total time: %.1f, update time: %.1f' % (times[-1], update_times[-1]))

# ANALYSIS
if analyze:
# means across N runs
    par_update_times_means = np.mean(par_update_times, axis=1)
    par_update_times_stds = np.std(par_update_times, axis=1)
    throughput = float(image_vox_num_scaled) / par_update_times
    throughput_means = np.mean(throughput, axis=1)
    throughput_stds = np.std(throughput, axis=1)
    error = 100 * ((correct_select_num - par_markers) / float(correct_select_num))
    error_means = error.mean(axis=1)
    error_stds = error.std(axis=1)

# PLOTTING ARGS
# plt.style.use('seaborn')
    capsize = 2
# plt.rcParams.update({'lines.markeredgewidth':1})
# fmt=[marker][line][color]

# PRINTING
    print('Results for %s: ' % compile_option_run)
    print(par_update_times_means)
    print(throughput_means)
    print(error_means)
    print(par_update_times_stds)
    print(throughput_stds)
    print(error_stds)

# PLOTTING
    plt.errorbar(block_sizes, par_update_times_means, yerr=par_update_times_stds, fmt='d--r', label='update time', capsize=capsize)
    plt.errorbar(block_sizes, throughput_means, yerr=throughput_stds, fmt='o--k', label='Voxel/s e7', capsize=capsize)
    plt.xlabel('Block sizes')
    plt.xticks(block_sizes, [str(i) for i in block_sizes])
    if parallel:
        title = 'Block size and parallel FM runtime'
    else:
        title = 'Block size and sequential FM runtime'
    plt.title(title)
    plt.ylabel('Mean FM times (s)')
    plt.xlim(max(block_sizes) + 5, min(block_sizes) -5)
    plt.legend()
    plt.tight_layout()
    fig_output_path = '%s/%s_%s_%s.png' % (fig_output_dir, compile_option_run, title.replace(' ', '_'), datetime.now().strftime(FORMAT))
    print(fig_output_path)
    plt.savefig(fig_output_path)
    if show_figs:
        plt.show()
    plt.close()

# PLOTTING
    plt.errorbar(block_sizes, error_means, yerr=error_stds, fmt='d--r', label='error rate', capsize=capsize)
    plt.xlabel('Block sizes')
    plt.xticks(block_sizes, [str(i) for i in block_sizes])
    title = 'Block size and error rate'
    plt.title(title)
    plt.ylabel('Error rate by voxels (%)')
    plt.xlim(max(block_sizes) + 5, min(block_sizes) -5)
# plt.legend()
    plt.tight_layout()
    fig_output_path = '%s/%s_%s_%s.png' % (fig_output_dir, compile_option_run, title.replace(' ', '_'), datetime.now().strftime(FORMAT))
    print(fig_output_path)
    plt.savefig(fig_output_path)
    if show_figs:
        plt.show()
    plt.close()

# FIXME should divide element wise across last dimension before averaging or stding
if perf:
    perf_times_means = np.mean(perf_times, axis=2)
    perf_times_stds = np.std(perf_times, axis=2)

    plt.plot(block_sizes, perf_times_means[0][:] / perf_times_means[1][:], 'r--d', label=perf_args[0])
    plt.plot(block_sizes, perf_times_means[2][:] / perf_times_means[3][:], 'k--o', label=perf_args[2])
    plt.plot(block_sizes, perf_times_means[6][:] / perf_times_means[5][:], 'g-->', label='CPI')
    plt.xlabel('Block sizes')
    plt.xticks(block_sizes, [str(i) for i in block_sizes])
    plt.xlim(max(block_sizes) + 5, min(block_sizes) -5)
    title = 'Block size and sequential cache performance'
    # title = 'Block size and parallel cache performance'
    plt.title(title)
    plt.ylabel('Miss rate %')
    plt.legend()
    plt.tight_layout()

    fig_output_path = '%s/%s_%s_%s.png' % (fig_output_dir, compile_option_run, title.replace(' ', '_'), datetime.now().strftime(FORMAT))
    print(fig_output_path)
    plt.savefig(fig_output_path)
    if show_figs:
        plt.show()
    plt.close()

# FIXME page faults

# plt.plot(block_sizes, perf_times)
# if len(threads) > 1:
# plt.plot(threads, par_update_times_means)
# plt.plot(block_sizes, par_update_times_means)
# plt.xlabel('Block size')
# plt.title('Block size vs. FM runtime N=%d' % N)
# plt.xlabel('Threads')
# plt.title('Thread count vs. runtime N=%d' % N)
# plt.savefig('%s/%s.png' % (run_dir, datetime.now().strftime(FORMAT)))
    # plt.tight_layout()
# plt.plot(block_sizes, update_times)

# if parallel:
    # data['par_update_times'] = par_update_times
    # data['par_times'] = par_times
# else:
    # data['update_times'] = update_times
    # data['times'] = times

data_path = '%s/%s.json' % (run_dir, datetime.now().strftime(FORMAT))

if save_dat:
    with open(data_path, 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()
    print('Created json for data in run at: %s...' % data_path)
