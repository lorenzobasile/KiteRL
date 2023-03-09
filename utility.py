import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def plot_average_reward(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.close()
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
def runn_average(arr,avg): 
    running_average = np.zeros(len(arr))
    for i in range(len(arr)): 
        running_average[i] = np.mean(arr[max(0, i-avg):(i+1)])
    return running_average    
    
def read_file(name = None): 
    if name is not None: 
        score = []
        with open(name) as f: 
            line = f.readline() 
            s = float(line)
            score.append(s)
            while line: 
                line = f.readline()
                if line =='':
                    break
                s = float(line)
                score.append(s)
        return score
        
        
def plot_trajectory(x,y,z,file_name): 

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, -112.5)

    startpoint = 0

    maxdata_plot = 180
    rail = np.zeros((3, 2))




    rail[0][0] = 0
    rail[0][1] = 600
    line, = ax.plot(x[::1], y[::1], z[::1], label = 'kite trajectory')

   
    ax.set_xlabel('X', fontsize=17, labelpad=10)
    ax.set_ylabel('Y', fontsize=17, labelpad=10)
    ax.set_zlabel('Z', fontsize=17, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=13)


    ax.plot(x[-1],y[-1],z[-1],
    markerfacecolor='red', markeredgecolor='k', marker = "o", markersize=20, label='final position', linestyle='None')
    ax.plot(x[0],y[0],z[0],
    markerfacecolor='green', markeredgecolor='k', marker = "o", markersize=20, label='starting position', linestyle='None')

    ax.plot(0,0,0,
    markerfacecolor='orange', markeredgecolor='k', marker = "^", markersize=20, label='ground station', linestyle='None')


    ax.plot([x[0],0],
            [y[0], 0],
            [z[0], 0], color = 'black', linewidth = .5)
    ax.plot([0, x[-1]],
            [0, y[-1]],
            [0, z[-1]], color = 'black', linewidth = .5)

    ax.legend(fontsize=15, bbox_to_anchor=(0.89, 0.3, 0.5, 0.5))
    plt.close(fig)
    ax.figure.savefig(file_name)
    
    
    
def plot_distance(x,y,z,time,file_name):
    
    lim_ = time[-1]+5
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3.5))
    
    lim = [0,lim_]

    ax1.set_xlim(lim)
    ax1.set_xlabel('time (s)', fontsize=14)
    ax1.set_ylabel('Kite x, (m)', fontsize=14)
    ax1.plot(time,x)

    ax2.set_xlim(lim)
    ax2.set_xlabel('time, (s)', fontsize=14)
    ax2.set_ylabel('Kite y, (m)', fontsize=14)
    ax2.plot(time,y)

    ax3.set_xlim(lim)
    ax3.set_xlabel('time, (s)', fontsize=14)
    ax3.set_ylabel('Kite z, (m)', fontsize=14)
    ax3.plot(time,z)

    plt.tight_layout()

    #plt.show()
    #plt.close()
    plt.savefig(file_name)
    
    
    


