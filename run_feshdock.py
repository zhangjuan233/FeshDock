import os
import prody
import time
import concurrent.futures
import pandas as pd
from functools import partial

def  getchain(pdb_file):
    rec_stru = prody.parsePDB(pdb_file)
    rec_id = ''
    chain_id = rec_stru.getChids()
    ids=''.join(dict.fromkeys(chain_id))
    rec_id=ids
    return rec_id

def change_str(str):
    temp = []
    for item in str:
        temp.append(item)
    new_temp = ' '.join(temp)
    return new_temp

def prepare(rec_pdb,lig_pdb):
    comd0=''
    rec_chain = getchain(rec_pdb)
    lig_chain = getchain(lig_pdb)
    if len(rec_chain) > 1 or len(lig_chain) > 1:
        if len(rec_chain) > 1:
            rec_chain = change_str(rec_chain)
        if len(lig_chain) > 1:
            lig_chain = change_str(lig_chain)
        comd0 = f'-model_chain1 {rec_chain} -model_chain2 {lig_chain} -native_chain1 {rec_chain} -native_chain2 {lig_chain}'
    return comd0



if __name__ == '__main__':

    cluster_predict_path='/home/ck/pycharm_projects/Feshdock-master2/data1/final_pdbs/'
    native_path='/home/ck/pycharm_projects/Feshdock-master2/data1/'  #pdb数据
    list=['1A2K']
    for pdbname in list:
        path_data1 = 'data1/'
        n_swarms=10
        top_p = 100
        comd_init = ''
        num_cpus=10
        num_generate = 100
        num_step = 10

        rec_pdb = pdbname + '_r_u.pdb'
        lig_pdb = pdbname + '_l_u.pdb'

        comd = f'docking/megadock-gpu -R {native_path}{rec_pdb} -L {native_path}{lig_pdb} -o docking/outputs/{pdbname}.out'
        os.system(comd)

        comd0 = f"python docking/kmeans_clustering.py ../data1/{rec_pdb} ../data1/{lig_pdb} {n_swarms} anm"
        os.system(comd0)

        comd1=f'python bin/setup.py ../data1/{rec_pdb} ../data1/{lig_pdb} -s {n_swarms} -g 200 --noh --now --noxt --anm'
        os.system(comd1)

        comd2=f'python bin/optimize.py ../data1/setup.json   {num_step}  -c {n_swarms}  -name {pdbname}'
        os.system(comd2)

        comd3=f'python bin/sort.py -n {num_generate} -step {num_step} -nswarms {n_swarms}'
        os.system(comd3)

        comd4=f'python bin/generate_conformations.py ../data1/{rec_pdb} ../data1/{lig_pdb} ../data1/scoring_sorted.out {num_generate}'
        os.system(comd4)

        com6=f'python bin/hierarchical_clustering.py -pdbname {pdbname} -n {num_generate}'  #最终的预测结构
        os.system(com6)














