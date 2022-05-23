import argparse
import shutil
import sys

def remove_folders(args):

    with open(args.input, 'r') as f:
        lines = f.readlines()
    #lines = ["adult_1bnn_local_pvi_10clients_1seeds_eps1_runs/326/"]
    print('Ok to remove:')
    for l in lines:
        print(l)
    tmp = input("y/n?")
    if tmp.lower() != 'y':
        sys.exit('Exited without removing folders')
    #sys.exit()

    # read configs
    for l in lines:
        to_remove = '/'.join(l.split("/")[:-1])
        
        print("removing: {}".format(to_remove))
        try:
            shutil.rmtree(to_remove)
        except:
            print('Failed to remove folder: {}'.format(to_remove))
        #else:
        #    continue

    print('All done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--input', type=str, default='tmp.txt')
    args = parser.parse_args()

    # arguments to skip when reading configs
    #to_skip = ['__doc__', 'batch_proc', 'enforce_pos', 'pbar','plot','track','"seed"']

    remove_folders(args)
