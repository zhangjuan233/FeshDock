# -*- coding: utf-8 -*-
import os.path
import sys
import argparse

from feshdock.prep.simulation import read_input_structure
import math
from feshdock.pdbutil.PDBIO import write_pdb_to_file
import numpy as np
from sklearn.cluster import KMeans
from transforms3d.quaternions import mat2quat
from feshdock.mathutil.lrandom import MTGenerator


def rotate(psi,theta,phi):
    r11 = math.cos(psi) * math.cos(phi) - math.sin(psi) * math.cos(theta) * math.sin(phi)
    r21 = math.sin(psi) * math.cos(phi) + math.cos(psi) * math.cos(theta) * math.sin(phi)
    r31 = math.sin(theta) * math.sin(phi)

    r12 = -math.cos(psi) * math.sin(phi) - math.sin(psi) * math.cos(theta) * math.cos(phi)
    r22 = -math.sin(psi) * math.sin(phi) + math.cos(psi) * math.cos(theta) * math.cos(phi)
    r32 = math.sin(theta) * math.cos(phi)

    r13 = math.sin(psi) * math.sin(theta)
    r23 = -math.cos(psi) * math.sin(theta)
    r33 = math.cos(theta)
    matrix=np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    return matrix


def rotate_atom(atom_coordinates, data, grid_width, rec_translation, num_fft):
    psi=data[0]#旋转角度值
    theta=data[1]
    phi=data[2]
    t1=data[3]#网格索引
    t2=data[4]
    t3=data[5]
    # Z1X2Z3
    r11 = math.cos(psi) * math.cos(phi) - math.sin(psi) * math.cos(theta) * math.sin(phi)
    r21 = math.sin(psi) * math.cos(phi) + math.cos(psi) * math.cos(theta) * math.sin(phi)
    r31 = math.sin(theta) * math.sin(phi)

    r12 = -math.cos(psi) * math.sin(phi) - math.sin(psi) * math.cos(theta) * math.cos(phi)
    r22 = -math.sin(psi) * math.sin(phi) + math.cos(psi) * math.cos(theta) * math.cos(phi)
    r32 = math.sin(theta) * math.cos(phi)

    r13 = math.sin(psi) * math.sin(theta)
    r23 = -math.cos(psi) * math.sin(theta)
    r33 = math.cos(theta)

    coord_new=[]
    for i in range(len(atom_coordinates)):
        xyz = []
        x=atom_coordinates[i][0]
        y=atom_coordinates[i][1]
        z=atom_coordinates[i][2]
        x1=r11 * x + r12 * y + r13 * z
        y1=r21 * x + r22 * y + r23 * z
        z1=r31 * x + r32 * y + r33 * z
        if t1>=(num_fft/2):
            t1-=num_fft
        if t2 >= (num_fft / 2):
            t2 -= num_fft
        if t3 >= (num_fft / 2):
            t3 -= num_fft
        # 调整
        x2=x1-t1*grid_width
        y2=y1-t2*grid_width
        z2=z1-t3*grid_width
        xyz.append(round(x2,2))
        xyz.append(round(y2,2))
        xyz.append(round(z2,2))
        coord_new.append(xyz)
    return coord_new


def cluster(cluster_id,ligand_dist, read_data,rec_translation,num_clusters ):
    size=10
    coords = []
    all_coord = [[] for _ in range(num_clusters)]
    for i in cluster_id:
        center_coord = calculate_center_coordinates(ligand_dist[i],rec_translation )
        coords.append(center_coord)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(coords)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    cluster_sizes = np.bincount(labels)
    iterations = 0
    while np.any(cluster_sizes < size) and iterations < 5:
        kmeans.fit(coords)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        cluster_sizes = np.bincount(labels)  #
        iterations += 1
    if iterations == 5:
        replace_list = []
        for i in range(len(cluster_sizes)):
            if cluster_sizes[i] <size :
                other_classes = np.where(cluster_sizes >= 80)[0]
                other_class = np.random.choice(other_classes)
                indices_to_move = np.random.choice(np.where(labels == other_class)[0], size=size - cluster_sizes[i],
                                                   replace=False)
                replace_list.extend(indices_to_move)
                labels[indices_to_move] = i
        if len(replace_list) != len(set(replace_list)):
            raise ValueError('存在相同的元素！')
        cluster_sizes = np.bincount(labels)

    clusters = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(labels):
        temp=[]
        rotate_data=read_data[i][:3]#旋转角度 网格 以及分数
        mat=rotate(rotate_data[0],rotate_data[1],rotate_data[2])
        quat=mat2quat(mat)#wxyz  四元数
        temp.extend(coords[i])
        temp.extend(quat) #
        temp.append(i)
        clusters[label].append(temp)
    for i in range(num_clusters):
        temp=[]
        for j in clusters[i]:
            data=[]
            data=j[:3]
            data.append(j[-1])
            temp.append(data)
        all_coord[i]=temp
    return all_coord,clusters


def read_out(out_file,num_gennerate):
    read_data = []
    with open(out_file, 'r') as f:
        lines = f.readlines()
        num_fft=int(lines[0].split()[0])
        data = lines[4:num_gennerate+4]
        for i in range(num_gennerate):
            line = data[i].split()
            temp = []
            for j in range(len(line)):
                if j >= 3 and j <= 5:
                    temp.append(int(line[j]))
                else:
                    temp.append(float(line[j]))
            read_data.append(temp)
    return read_data,num_fft
def split_string(string):
    parts = string.split('/')
    filename = parts[-1]
    code = filename.split('_')[0]
    return code
def calculate_center_coordinates(filename,rec_translation):
    with open(filename, 'r') as file:
        lines = file.readlines()
    coordinates = []
    center_coordinates=[]
    for line in lines:
        if line.startswith('ATOM'):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coordinates.append((x, y, z))

    num_atoms = len(coordinates)
    center_x = sum(coord[0] for coord in coordinates) / num_atoms
    center_y = sum(coord[1] for coord in coordinates) / num_atoms
    center_z = sum(coord[2] for coord in coordinates) / num_atoms
    center_coordinates = [center_x, center_y,center_z]
    return center_coordinates


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        # Receptor
        parser.add_argument(
            "receptor_pdb", help="receptor structures: PDB file or list of PDB files",
            metavar="receptor_structure",
        )
        # Ligand
        parser.add_argument(
            "ligand_pdb", help="ligand structures: PDB file or list of PDB files",
            metavar="ligand_structure",
        )
        parser.add_argument(
            "n_clusters", help="num of the clusters", type=int,
            metavar='num',
        )
        parser.add_argument(
            "anm", help="num of the clusters", type=str,
            metavar='ANM',
        )
        args = parser.parse_args()
        num_clusters=args.n_clusters
        ANM=args.anm
        rng = MTGenerator(324324)
        receptor=read_input_structure(args.receptor_pdb,ignore_hydrogens=True,ignore_oxt=False)#receptor.atom_type存放的是残基原子类型
        ligand=read_input_structure(args.ligand_pdb,ignore_hydrogens=True,ignore_oxt=False)
        rec_translation = receptor.move_to_origin()
        lig_translation = ligand.move_to_origin()

        string_out = split_string(args.receptor_pdb)
        out_file = 'outputs/'+string_out + '.out'

        out_file=os.path.join(os.getcwd(),out_file)
        num_gennerate=1000
        read_data,num_fft=read_out(out_file,num_gennerate)
        mk_flie = os.path.join('../data1/' + 'generate_ligands')
        os.mkdir(mk_flie)
        ligand_dist={}
        cluster_id=[]
        for i in range(num_gennerate):
            filename= "ligand" + "_%s.pdb" % (i+1)
            des_name=os.path.join(mk_flie,filename)
            ligand_dist[i] = des_name
            cluster_id.append(i)
            changed_coord = rotate_atom(ligand.atom_coordinates[0], read_data[i], ligand.grid_width, rec_translation, num_fft)
            write_pdb_to_file(ligand,des_name,changed_coord)


        all_coord,clusters=cluster(cluster_id,ligand_dist,read_data,rec_translation,num_clusters)
        init_path='../data1/init'
        num_glowworm=120  #控制初始种群的数量
        os.mkdir(init_path)
        for i in range(num_clusters):
            saving_path='%s%d%s'%("initial_positions_",i,'.dat')
            postion_path=os.path.join(init_path,saving_path)
            with open(postion_path,'a') as f:
                every_cluster=clusters[i]
                num_cluster=len(every_cluster)
                top_element=[]
                for j in range(num_cluster):
                    if j==num_glowworm:
                        break
                    if ANM=='anm':
                        f.write(' '.join(map(str,every_cluster[j][:7]))+' ')
                        anm_list=[rng() for _ in range(20)]
                        f.write(' '.join('{:.9f}'.format(float(x))for x in anm_list)+'\n')
                    else:
                        f.write(' '.join(map(str, every_cluster[j][:7])) + '\n')
                    if j == 0:
                        top_element=every_cluster[j][:3]












