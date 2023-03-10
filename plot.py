import numpy as np
from learning.algorithms import *
from argparse import ArgumentParser
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def main(args):
    np.random.seed(0)
    path=args.path
    if not os.path.exists(path):
        exit()
    durations=np.load(args.path+"durations.npy")
    cumulative_durations=np.load(args.path+"cumulative_durations.npy")
    power=np.load(args.path+"power.npy")
    x=np.load(args.path+"x.npy")[:cumulative_durations[0]]
    y=np.load(args.path+"y.npy")[:cumulative_durations[0]]
    z=np.load(args.path+"z.npy")[:cumulative_durations[0]]
    alpha=np.load(args.path+"alpha.npy")
    bank=np.load(args.path+"bank.npy")
    beta=np.load(args.path+"beta.npy")
    wind=np.load(args.path+"wind.npy")


    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, -112.5)

    startpoint = 0

    maxdata_plot = 180


    rail = np.zeros((3, 2))
    rail[0][0] = 0
    rail[0][1] = 600
    line, = ax.plot(x[::1], y[::1], z[::1], label = 'kite trajectory')

    #ax.set_ylim(0, 400)
    #ax.set_xlim(0, 4000)
    #ax.set_zlim(0, 140)
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

    #a = Arrow3D([3000, 4000], [50, 50], [60, 60], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    #b = Arrow3D([3000, 3800], [50, 50], [50, 50], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    #c = Arrow3D([3000, 4200], [50, 50], [70, 70], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")

    #ax.add_artist(a)
    #ax.add_artist(b)
    #ax.add_artist(c)
    #ax.arrow(3,1,-0.89,0, length_includes_head=True,head_width=0.1, head_length=0.05, color="black")
    #ax.annotate("", xy=(-0.07, 2.07), xytext=(0.95, 1.05), arrowprops=dict(arrowstyle="<->"))
    #ax.text(3000,50,80, "Wind", size=16)
    ax.plot([x[0],0],
            [y[0], 0],
            [z[0], 0], color = 'black', linewidth = .5)
    ax.plot([0, x[-1]],
            [0, y[-1]],
            [0, z[-1]], color = 'black', linewidth = .5)

    ax.legend(fontsize=15, bbox_to_anchor=(0.89, 0.3, 0.5, 0.5))
    plt.savefig(path+"3d_trajectory_frame.png", bbox_inches='tight', dpi=200)
    #plt.show()

    


    # References
    # https://gist.github.com/neale/e32b1f16a43bfdc0608f45a504df5a84
    # https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
    # https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation

    # ANIMATION FUNCTION
    def func(num, dataSet, line,redDots):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        redDots.set_data(dataSet[0:2, num-1])
        redDots.set_3d_properties(dataSet[2, num-1])
        '''
        if num>0:
            vect=np.ones(num)*100
            vect[-1]=1
            redDots.set_data(dataSet[0:2, :num])
            redDots.set_3d_properties(np.multiply(dataSet[2, :num], vect))
        else:
            redDots.set_data(dataSet[0:2, :num])
            redDots.set_3d_properties(dataSet[2, :num])
        '''
        return line


    # THE DATA POINTS
    
    dataSet = np.array([x, y, z])
    numDataPoints = len(z)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure(figsize=(16, 10))
    ax = Axes3D(fig)
    ax.view_init(20, -112.5)


    #ax.set_ylim(0, 400)
    #ax.set_xlim(0, 4000)
    #ax.set_zlim(0, 150)
    line = ax.plot(dataSet[0], dataSet[1], dataSet[2], '-', lw=2)[0] # For line plot
    redDots = plt.plot(dataSet[0], dataSet[1], dataSet[2], markerfacecolor='red', markeredgecolor='k', marker = "o", markersize=20, label='current position', linestyle='None')[0]
    # NOTE: Can't pass empty arrays into 3d version of plot()


    # AXES PROPERTIES]
    # ax.set_xlim3d([limit0, limit1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Kite trajectory')
    #a = Arrow3D([3000, 4000], [50, 50], [60, 60], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    #b = Arrow3D([3000, 3800], [50, 50], [50, 50], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    #c = Arrow3D([3000, 4200], [50, 50], [70, 70], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")

    #ax.add_artist(a)
    #ax.add_artist(b)
    #ax.add_artist(c)
    #ax.arrow(3,1,-0.89,0, length_includes_head=True,head_width=0.1, head_length=0.05, color="black")
    #ax.annotate("", xy=(-0.07, 2.07), xytext=(0.95, 1.05), arrowprops=dict(arrowstyle="<->"))
    #ax.text(3000,50,80, "Wind", size=16)
    #ax.plot(x[-1],y[-1],z[-1],
            #markerfacecolor='red', markeredgecolor='k', marker = "o", markersize=20, label='final position', linestyle='None')
    ax.plot(x[0],y[0],z[0],
            markerfacecolor='green', markeredgecolor='k', marker = "o", markersize=20, label='starting position', linestyle='None')

    ax.plot(0,0,0,
            markerfacecolor='orange', markeredgecolor='k', marker = "^", markersize=20, label='ground station', linestyle='None')
    ax.legend(fontsize=15, bbox_to_anchor=(0.74, 0.3, 0.5, 0.5))
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line,redDots), interval=200, blit=False)
    line_ani.save(path+'animation.gif')
    plt.close()


    plt.style.use(['seaborn-whitegrid','tableau-colorblind10'])
    plt.figure(figsize=(10,6))
    plt.title("Control variables")
    file_name = os.path.join(args.path,"policy.png")
            
    alpha_tot = alpha[cumulative_durations[5]:cumulative_durations[8]]
            
    bank_tot = bank[cumulative_durations[5]:cumulative_durations[8]]
            
    beta_tot = beta[cumulative_durations[5]:cumulative_durations[8]]
    durat = [np.sum(durations[6:9])/len(beta_tot)*i for i in range(len(beta_tot))]
    plt.plot(durat,alpha_tot,linewidth=1.5)
    plt.plot(durat,bank_tot,linewidth=1.5)
    plt.vlines(np.cumsum(durations[6:9]), -100, 100, colors='k')
    plt.xlabel("time (s)", fontsize=16)
    plt.ylabel("Angle (deg)", fontsize=16)
    plt.ylim([-5,15])
    plt.xlim([0, durat[-1]])

    plt.legend(['Attack angle', 'Bank Angle',], loc='upper right')
            
    plt.savefig(file_name, dpi=200)
            
    plt.close()

    plt.figure(figsize=(10,6))
    plt.title("Produced power")
    file_name = os.path.join(args.path,"power.png")
            
    power_tot = power[cumulative_durations[5]:cumulative_durations[8]]
            
            
    durat = [np.sum(durations[6:9])/len(beta_tot)*i for i in range(len(power_tot))]
    plt.plot(durat,power_tot,linewidth=1.5)
    xmin, xmax, ymin, ymax=plt.axis()
    plt.vlines(np.cumsum(durations[6:9]), ymin, ymax, colors='k')
    plt.xlabel("time (s)", fontsize=16)
    plt.ylabel("Power (kW)", fontsize=16)
    plt.xlim([0, durat[-1]])
    plt.ylim([ymin, ymax])

            
    plt.savefig(file_name, dpi=200)
            
    plt.close()

    plt.figure(figsize=(10,6))
    
    plt.title("Effective wind speed")
    file_name = os.path.join(args.path,"wind.png")
            
    wind_tot = wind[cumulative_durations[5]:cumulative_durations[8]]
            
            
    durat = [np.sum(durations[6:9])/len(beta_tot)*i for i in range(len(wind_tot))]
    plt.plot(durat,wind_tot,linewidth=1.5)
    xmin, xmax, ymin, ymax=plt.axis()
    plt.vlines(np.cumsum(durations[6:9]), ymin, ymax, colors='k')
    plt.xlabel("time (s)", fontsize=16)
    plt.ylabel("Wind (m/s)", fontsize=16)
    plt.xlim([0, durat[-1]])
    plt.ylim([ymin, ymax])

            
    plt.savefig(file_name, dpi=200)
            
    plt.close()

    #plt.show()

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/const/")
    args = parser.parse_args()
    main(args)        
         
            
        
