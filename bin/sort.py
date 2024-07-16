# -*- coding: utf-8 -*-
import os
import argparse


def read_output(path,num_swarm,num_pose,step):
    results = []
    for i in range(num_swarm):
        position = os.path.join(path, "swarm_%s" % i, f"gso_%d.out"%step)
        with open(position, 'r') as f_r:
            lines = f_r.readlines()
            for line in lines:
                if line[0] == '(':
                    last = line.index(')')
                    row_data = line[1:last].split(',')
                    row_data2 = line[last + 1:].split()
                    row_data2[5] = float(row_data2[5])
                    row_data.extend(row_data2)
                    results.append(row_data)
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    filename = os.path.basename(path)
    filename ='scoring_sorted.out'#
    sorted_file = os.path.join(path, filename)

    with open(sorted_file, 'a') as f_w:
        for i in range(len(results)):
            f_w.write('(')
            for j in range(27):
                if j != 26:
                    f_w.write(str(results[i][j]) + ', ')
                else:
                    f_w.write(results[i][j])
            f_w.write(') ')
            for j in range(27, 33):
                if j == 32:
                    f_w.write(str(str(results[i][j])) + '\n')
                else:
                    f_w.write(str(results[i][j]) + ' ')
        f_w.close()




if __name__ == '__main__':
    path = "../data1"
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',type=int,required=True,help='gennerate numbers')
    parser.add_argument('-step',type=int,required=True,help='step')
    parser.add_argument('-nswarms',type=int,required=True,help='step')
    args=parser.parse_args()
    num=args.n
    num_swarms=args.nswarms
    read_output(path, num_swarms, num, args.step)


