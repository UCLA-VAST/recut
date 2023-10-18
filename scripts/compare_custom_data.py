import argparse
import os
import subprocess
from glob import glob
from pathlib import Path

def swc_to_coord(swc):
    return tuple(map(int, Path(swc).stem.split('_')[0].split('-')[-3:]))

def swcs_to_dict(swcs):
    d = {}
    for swc in swcs:
        t = swc_to_coord(swc)
        d[t] = swc
    return d

def compare_2_swcs(proof_swc, auto_swc, kwargs):
    timeout_seconds = 4 * 60

    cmd = "/home/kdmarrett/recut/result/bin/recut {} --test {} --voxel-size {} {} {}".format(proof_swc,proof_swc, kwargs['voxel_size_x'], kwargs['voxel_size_y'], kwargs['voxel_size_z'])
    print("Run: " + cmd)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, check=False, text=True,
                        timeout=timeout_seconds)
    except:
        return None
    return result

def extract_accuracy(result):
    if result:
        if result.stdout:
            print('output: ', result.stdout)
        if result.stderr:
            print('error: ', result.stderr)
        if result.stderr != "" or result.returncode != 0:
          return False
        return True
    else:
        return False

def gather_automated_swcs(path):
    pattern = path + "**/*.swc"
    all_swcs = glob(pattern, recursive=True)
    auto_swcs = filter(lambda n: '_' not in os.path.basename(n), all_swcs)
    return swcs_to_dict(auto_swcs)

def gather_proofread_swcs(path):
    proof_swcs = filter(lambda f: Path(f).suffix == ".swc", glob(path + "*"))
    return swcs_to_dict(proof_swcs)

def match_swcs(proofreads, automateds):
    # return map(lambda kv: (kv[1], automateds[kv[0]]), proofreads.items())
    l = []
    for kv in proofreads.items():
        coord = kv[0]
        proofread = kv[1]
        automated = kv[1]
        try:
            automated = automateds[coord]
        except:
            continue
            # find_close_soma
        tup = (proofread, automated)
        l.append(tup)
    return l

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proofread')
    parser.add_argument('automated')
    args = parser.parse_args()
    kwargs = {}
    kwargs['voxel_size_x'] = .4
    kwargs['voxel_size_y'] = .4
    kwargs['voxel_size_z'] = .4

    automateds = gather_automated_swcs(args.automated)
    proofreads = gather_proofread_swcs(args.proofread)

    matched = match_swcs(proofreads, automateds)
    # print("Proofread count: " + str(len(proofreads)))
    print("Match count: " + str(len(matched)) + '/' + str(len(proofreads)))
    print()

    success_count = 0
    for proofread, automated in matched:
        returncode = compare_2_swcs(proofread, automated, kwargs)
        is_success = extract_accuracy(returncode)
        if is_success:
            print("Success")
            print()
            print()
            success_count +=1

    print("Success count: " + str(len(success_count)) + '/' + str(len(proofreads)))

if __name__ == "__main__":
    main()
