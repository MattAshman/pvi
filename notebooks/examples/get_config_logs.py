import argparse
import sys

def read_confs(args, to_skip):

    with open(args.input, 'r') as f:
        lines = f.readlines()

    configs = []
    jobids = []
    # read configs
    for l in lines:
        config_file = '/'.join(l.split("/")[:-1])+'/config.json'
        with open(config_file, 'r') as f:
            for line in f.readlines():
                skip = False
                for w in to_skip:
                    if 'job_id' in line and int(line.split(":")[-1][:-2]) not in jobids:
                        jobids.append(int(line.split(":")[-1][:-2]))
                    if w in line:
                        skip = True
                        break
                if not skip:
                    configs.append(line)

    with open(args.output_jobs,'w') as f:
        for w in sorted(jobids):
            f.write(f"\n {w}")

    with open(args.output, 'w') as f:
        for w in configs:
            f.write(w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--input', type=str, default='tmp.txt')
    parser.add_argument('--output', type=str, default='tmp_confs.txt')
    parser.add_argument('--output_jobs', type=str, default='tmp_jobid.txt')
    args = parser.parse_args()

    # arguments to skip when reading configs
    to_skip = ['__doc__', 'batch_proc', 'enforce_pos', 'pbar','plot','track','"seed"']

    read_confs(args, to_skip)
