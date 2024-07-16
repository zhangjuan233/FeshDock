#!/usr/bin/env python3
import argparse
import os,time
import sys
from prody import parsePDB, confProDy, calcRMSD
from feshdock.util.logger import LoggingManager
from protein.constants import CLUSTER_REPRESENTATIVES_FILE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np
from scipy.spatial.distance import squareform


# Disable ProDy output
confProDy(verbosity="info")

log = LoggingManager.get_logger("cluster_bsas")


def parse_command_line():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog="lgd_cluster_bsas")

    parser.add_argument(
        "de_output_file", help="feshdock output file", metavar="de_output_file"
    )

    return parser.parse_args()


def get_backbone_atoms(ids_list, swarm_path,pdbname):
    """Get all backbone atoms (CA or P) of the PDB files specified by the ids_list.

    PDB files follow the format lightdock_ID.pdb where ID is in ids_list
    """
    ca_atoms = {}
    try:
        for struct_id in ids_list:
            pdb_file = swarm_path+ f"{pdbname}_{struct_id}.pdb"
            log.info(f"Reading CA from {pdb_file}")
            structure = parsePDB(str(pdb_file))
            selection = structure.select("name CA P")
            ca_atoms[struct_id] = selection
    except IOError as e:
        log.error(f"Error found reading a structure: {e}")
        log.error(
            "Did you generate the LightDock structures corresponding to this output file?"
        )
        raise SystemExit()
    return ca_atoms

def hierarchical_clusterize(sorted_ids, swarm_path, pdbname):
    """Clusters the structures identified by the IDs inside sorted_ids list"""
    # Read all structures backbone atoms
    backbone_atoms = get_backbone_atoms(sorted_ids, swarm_path, pdbname)
    # Calculate pairwise RMSD matrix
    rmsd_matrix = np.zeros((len(sorted_ids), len(sorted_ids)))
    for i, id_i in enumerate(sorted_ids):
        for j, id_j in enumerate(sorted_ids[i:]):
            rmsd_value = calcRMSD(backbone_atoms[id_i], backbone_atoms[id_j]).round(4)
            rmsd_matrix[i, i + j] = rmsd_value
            rmsd_matrix[i + j, i] = rmsd_value
    # Use hierarchical clustering to compute linkage matrix
    condensed_matrix = squareform(rmsd_matrix)
    linkage_matrix = linkage(condensed_matrix, method='average')

    # Assign clusters based on a threshold
    threshold = 4.0
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Organize clustered structures into dictionary
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = [sorted_ids[i]]
        else:
            cluster_dict[cluster_id].append(sorted_ids[i])
    return cluster_dict



def write_cluster_info(clusters, gso_data, swarm_path):
    """Writes the clustering result"""
    file_name = swarm_path / CLUSTER_REPRESENTATIVES_FILE
    with open(file_name, "w") as output:
        for id_cluster, ids in clusters.items():
            output.write(
                "%d:%d:%8.5f:%d:%s\n"
                % (
                    id_cluster,
                    len(ids),
                    gso_data[ids[0]].scoring,
                    ids[0],
                    "lightdock_%d.pdb" % ids[0],
                )
            )
        log.info(f"Cluster result written to {file_name} file")

def sort_by_value_length(item):
    key, value = item
    return len(value)

def rank_cluster(pdbname,clusters):
    sorted_clusters = dict(sorted(clusters.items(), key=sort_by_value_length, reverse=True))
    return  sorted_clusters

if __name__ == "__main__":
    predict_path = '../data1/predict_pdbs/'
    oldpath = '../data1/predict_pdbs'
    newpath = '../data1/final_pdbs'

    parser = argparse.ArgumentParser(description='cluster')
    parser.add_argument('-pdbname', required=True, help='PDB')
    parser.add_argument('-n', required=True, help='nums')

    args=parser.parse_args()
    num=int(args.n)
    pdbname=args.pdbname
    cluster_ids=[(i+1) for i in range(num)]
    clusters = hierarchical_clusterize(cluster_ids, predict_path, pdbname)
    clusters=rank_cluster(pdbname, clusters)


    representative_list=[]
    for key,value in clusters.items():
        representative_id=value[0]
        representative_list.append(representative_id)

    os.mkdir(newpath)
    for i in range(len(representative_list)):
        oldpdb=f'{pdbname}_{representative_list[i]}'+'.pdb'
        newpdb=f'{pdbname}_{i+1}'+'.pdb'
        comd=f'cp {oldpath}/{oldpdb} {newpath}/{newpdb}'
        os.system(comd)

