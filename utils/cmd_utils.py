import sys

from data.data import create_if_not_exist_dataset


def parse_data_args(line):
    data_args = line.split('-x')
    if len(data_args) == 2:
        data_args = data_args[1]
    elif len(data_args) == 1:
        data_args = None
    else:
        raise Exception('bad format for arg line')
    return data_args


def create_dataset_before(args_file):
    with open(args_file, 'r') as f:
        for line in f:
            # args = parse_main_args(line.split())
            dargs = parse_data_args(line)
            print(dargs)
            create_if_not_exist_dataset(root='data/', arg_str=dargs)


def assign_cluster(args_file):
    with open(args_file, 'r') as f:
        fcpu = open(args_file + '.cpu', 'w')
        fgpu = open(args_file + '.gpu', 'w')
        cc = 0
        cg = 0
        for line in f:
            if '-c' in line.split() or '-cp' in line.split() or '--cuda' in line.split():
                fgpu.write(line)
                cg += 1
            else:
                fcpu.write(line)
                cc += 1
        fcpu.close()
        fgpu.close()
        print('Total args to be run on gpu: {}'.format(cg))
        print('Total args to be run on cpu: {}'.format(cc))


def seedify(arg_file):
    srange = [1, 1]
    args = sys.argv
    if len(args) == 1:
        pass
    elif len(args) == 2:
        srange = [1, int(args[1])]
    elif len(args) == 3:
        srange = [int(args[1]), int(args[2])]
    else:
        raise Exception('wrong usage')
    with open(arg_file, 'r') as f:
        seeded_name = arg_file + '.seeded'
        with open(seeded_name, 'w') as sf:
            for line in f:
                for s in range(srange[0], srange[1] + 1):
                    sf.write(line.split('\n')[0] + ' --seed ' + str(s) + '\n')
