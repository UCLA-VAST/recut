import json
import os
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

eight_bit = False
cur = os.path.dirname(os.path.realpath(__file__))
base_dir = '/curr/kdmarrett/data'
run_dir = base_dir + '/run_data'
swc_dir = base_dir + '/swcs'
img_dir = base_dir + '/2019-02-07_16.07.27_Protocol_FusionStitcher_BCL11b_KO_CAMK2_cortex_resolution0_tif/'
# segmentation_dir = base_dir + '/segmented'
filled_soma_dir = base_dir + '/filled'
manual_dir = base_dir + '/manual_modification'
marker_dir = base_dir + '/marker_files'
# marker_dir = manual_dir + '/marker_files'
app_suff = '.swc'
swc_output = swc_dir + '/' + img_dir.split('/')[0]

log_dir_path = '%s/../logs/' % cur
assert(os.path.exists(filled_soma_dir))
print(filled_soma_dir)
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)
FORMAT = '%m_%d_%H_%M_%S'

parallel = True
n_off = 1
save_dat = False
debug = True
revisit = True
LOG = True
N = 1
image_dims = [7680, 8448, 383]
# image_dims = [1792, 1664, 352]
grid_dims = [512]
interval_sizes = [512]
block_sizes = [128]
# block_sizes = [32]
# restart_factors = [5, 10, 30, 0][::-1]
# restart_factors = [30]
# restart_factors = [10]
restart_factors = [0]
# threads = [1,4,8][::-1]
threads = [56 ]
# threads = [1]
fp = '0.1' # 1 means 100% of pixels are set as foreground
res_lvl = '0'
ch = '0'
pr = '0' # bool for hierarchical pruning
prune = []
markers = []
revisits = []
par_markers = []
par_iteration = []
times = []
par_times = []
par_generating_times = []
par_init_times = []
update_times = []
revisits = []
par_update_times = []
# if len(block_sizes) > 1:
    # par_update_times = np.zeros((len(block_sizes), N))
# if len(threads) > 1:
    # par_update_times = np.zeros((len(threads), N))
# if len(restart_factors) > 1:
    # par_update_times = np.zeros((len(restart_factors), N))
data = {}
data['grid_dims'] = grid_dims
data['block_sizes'] = block_sizes
data['restart_factors'] = restart_factors
data['threads'] = threads
data['N'] = N
data['fp'] = fp

# center the offsets around the soma location evenly
# making sure not to extend into negative territory or outside
# bounds
adjust = lambda a, b : max(a - b / 2, 0)
adjust_ie = lambda offset, dim, grid_dim : min(image_dims[dim] - offset, grid_dim)

def parse_time(log, phrase):
    for line in log.split('\n'):
        if phrase in line:
            return float(line.split()[-2])

seq_run = False
for grid_dim in grid_dims:
    # for i in range(0, 2):
        # # if seq_run: 
            # # seq_run = False
            # # continue # only run seq once
        # parallel = False
        # if i: 
            # parallel = True
        # else:
            # seq_run = True
    print("Starting range with parallel: " + str(parallel))
    for ti, thread in enumerate(threads):
        for bi, bs in enumerate(block_sizes):
            if parallel:
                print('\n\nTesting bs: %d' % bs)
            if bs > grid_dim:
                continue
            for ri, rs in enumerate(restart_factors):
                if parallel:
                    print('\nTesting rs: %.3f' % rs)
                for ni, marker_file in enumerate(os.listdir(marker_dir)[n_off:(n_off + N)]):
                    # fmark = marker_dir + '/' + marker_file
                    fmark = marker_dir + '/'
                    z, y, x, mass = [int(i) for i in marker_file.split('_')[1:]] 
                    za = adjust(z, grid_dim)
                    ya = adjust(y, grid_dim)
                    xa = adjust(x, grid_dim)
                    ze = adjust_ie(za, 2, grid_dim)
                    ye = adjust_ie(ya, 1, grid_dim)
                    xe = adjust_ie(xa, 0, grid_dim)
                    swc_output_file = swc_output + '_'.join([ str(z), str(y), str(x)]) + app_suff
                    os.system('touch ' + swc_output_file)
                    params = ['../bin/recut', filled_soma_dir, ch,
                            '-inmarker', fmark, '-io', str(za), str(ya), str(xa),
                            '-ie', str(ze), str(ye), str(xe),
                            '-lt', '50', '-fp', fp, '-pr', pr, '-bs', str(bs),
                            '-ct', '1', '-rl', res_lvl, '-outswc',
                            swc_output_file ]
                    # params = ['../bin/vaa3d_app2', filled_soma_dir, ch,
                            # '-inmarker', fmark, '-io', str(za), str(ya), str(xa),
                            # '-ie', str(ze), str(ye), str(xe),
                            # '-lt', '50', '-fp', fp, '-pr', pr, '-bs', str(bs),
                            # '-ct', '1', '-rl', res_lvl, '-outswc',
                            # swc_output_file ]
                    if parallel:
                        params.append('-pl')
                        params.append(str(thread))
                    if rs > 0:
                        params.append('-rs')
                        params.append(str(rs))
                    jparams = ' '.join(params)
                    print(jparams)
                    if debug:
                        os.system(jparams) # print directly to stdout
                    try:
                        # log = subprocess.run(params, stdout=subprocess.PIPE).stdout
                        log = subprocess.check_output(params)
                    except subprocess.CalledProcessError as e:
                        print(e.output)
                        print('Error running command: ' + '"' + str(e.cmd) + '"' + ' see above shell error')
                        print('Return code: ' + str(e.returncode))
                        continue
                    slog = log.decode('ascii')
                    log_path = '%s%s_%s.log' % (log_dir_path, marker_file, datetime.now().strftime(FORMAT))
                    if LOG:
                        logfile = open(log_path, 'w+')
                        logfile.write(jparams)
                        logfile.write('\n')
                        logfile.write(slog)
                        logfile.close()
                    print('Created log for reconstruction: %s...' % log_path)
                    print('Created output: ' + swc_output)
                    if parallel:
                        marker = parse_time(slog, 'Total marker')
                        if marker:
                            par_markers.append(marker)
                            print('Total markers: %d..' % int(marker))
                        par_iteration.append(parse_time(slog, 'Total iteration'))
                        if revisit:
                            revisits.append(parse_time(slog, 'Total revisits'))
                            print('Total revisits: %d..' % int(revisits[-1]))
                        print('Total iterations: %d..' % int(par_iteration[-1]))
                        prune.append(parse_time(slog, 'Pruning neuron tree wtime'))
                        par_times.append(parse_time(slog, 'fastmarching_tree'))
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
                        par_update_times.append(parse_time(slog, 'Finished updating'))
                        if pr == '1':
                            print('Total time: %.1f, update time: %.1f, prune time %.1f' % (par_times[-1], par_update_times[-1]), prune[-1])
                        else:
                            print('Total time: %.1f, update time: %.1f' % (par_times[-1], par_update_times[-1]))
                        par_generating_times.append(parse_time(slog, 'Finished generating'))
                        par_init_times.append(parse_time(slog, 'Finished initialization'))
                    else:
                        markers.append(parse_time(slog, 'Total marker'))
                        print('Total markers: %d..' % int(markers[-1]))
                        times.append(parse_time(slog, 'fastmarching_tree'))
                        update_times.append(parse_time(slog, 'Finished updating'))
                        prune.append(parse_time(slog, 'Pruning neuron tree wtime'))
                        if pr == '1':
                            print('Total time: %.1f, update time: %.1f, prune time %.1f' % (times[-1],
                                        update_times[-1], prune[-1]))
                        else:
                            print('Total time: %.1f, update time: %.1f' % (times[-1], update_times[-1]))

# par_update_times_means = np.mean(np.reshape(par_update_times,
    # (len(threads), N)), axis=1)

# if par_update_times:
    # iters = np.reshape(par_iteration, (len(par_iteration) / N, N))
    # par_update_times_means = np.mean(par_update_times, axis=1) # mean across images

# plt.plot(block_sizes, par_update_times)
# plt.xlabel('Block sizes')
# if len(threads) > 1:
# plt.plot(threads, par_update_times_means)
# plt.plot(block_sizes, par_update_times_means)
# plt.xlabel('Block size')
# plt.title('Block size vs. FM runtime N=%d' % N)
# plt.ylabel('Mean FM times (s)')
# plt.xlabel('Threads')
# plt.title('Thread count vs. runtime N=%d' % N)
# if save_dat:
    # plt.savefig('%s/%s.png' % (run_dir, datetime.now().strftime(FORMAT)))
# plt.show()
# plt.plot(block_sizes, update_times)

# if parallel:
    # data['par_update_times'] = par_update_times
    # data['par_times'] = par_times
# else:
    # data['update_times'] = update_times
    # data['times'] = times

# data_path = '%s/%s.json' % (run_dir, datetime.now().strftime(FORMAT))
# if save_dat:
    # with open(data_path, 'w') as outfile:
        # json.dump(data, outfile)
    # outfile.close()
    # print('Created json for data in run at: %s...' % data_path)
