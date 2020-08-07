import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np

iterate = np.arange(2000,50001,2000)

for joint in range(0,7):
    pl.close()
    #files = "all_plots.csv"
    files = "AE_joint_positions_1.csv"
    x = []
    y = []
    z = []
    fig = pl.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Scatter plot of " + str(joint+1) + " . joint")


    with open(files) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for idx,row in enumerate(csv_reader):
                
            if idx > 1 and row != []:
                array = []
                string = row[joint].replace("[","")
                string = string.replace("]","")
                string = string.split(" ")
                for item in string:
                    if item != '':
                        array.append(float(item))
                for idx,item in enumerate(array):
                    if idx == 0 and item != 0:
                        x.append(item)
                    elif idx == 1 and item != 0:
                        y.append(item)
                    elif idx == 2 and item != 0:
                        z.append(item)
    
    for i in iterate:
        fig = pl.figure()
        ax = fig.add_subplot(111,projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Scatter plot of " + str(joint+1) + " . joint")
        print(i)
        a = x[0:i]
        b = y[0:i]
        c = z[0:i]
        print(len(a))
        ax.scatter(a,b,c,marker='o',s=5)
        name = str(int(i/100)) + '_' + str(joint+1) + '.png'
        pl.show()
        pl.savefig('absolute_position_autoencoder_1/' + name)
        pl.close()
        
        







                        

                    


