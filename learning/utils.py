import numpy as np
import os



def write_params(param_dict, dir_path, file_name):
    """Write a parameter file"""
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print ("Creation of the directory failed")
    f = open(dir_path + file_name, "w")
    for k,v in param_dict.items():
        if type(v) is list or type(v) is np.ndarray:
            f.write(k + "\t")
            for i in range(len(v)):
                f.write(str(v[i])+",")
            f.write("\n")
        else:
            f.write(k + "\t" + str(v) + "\n")
    f.close()


def read_params(path):
    """Read a parameter file"""
    params = dict()
    f = open(path, "r")
    for l in f.readlines():
        try:
            params[l.split()[0]] = float(l.split()[1])
        except ValueError:
            if ',' not in l.split()[1]:
                params[l.split()[0]] = l.split()[1]
            else:
                params[l.split()[0]] = np.array(l.split()[1].split(',')[:-1], dtype=float)
    return params


def read_traj(path):
    """Read a trajectory with headers"""
    f = open(path, "r")
    d_traj = []
    r_traj = []
    #state_labels = f.readline().split()
    for line in f.readlines():
        d_traj.append(line.split()[0])
        r_traj.append(line.split()[1])
    return np.array(d_traj, dtype="int"), np.array(r_traj, dtype="float")


def read_traj2(path):
    """Read a trajectory with headers"""
    f = open(path, "r")
    d_traj = []
    r_traj = []
    #state_labels = f.readline().split()
    for line in f.readlines():
        d_traj.append(line.split()[0])
        r_traj.append(line.split()[1])
    return np.array(d_traj, dtype="int"), np.array(d_traj, dtype="float")
