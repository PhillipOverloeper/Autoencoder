import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from sklearn.neighbors import KernelDensity
import sys

iterate = np.arange(2000,20001,2000)

def show_position(episode,iterations):
    for joint in range(0,7):
        pl.close()
    
        x = np.array([],dtype="f")
        y = np.array([],dtype="f")
        z = np.array([],dtype="f")
        files = "BL_joint_positions_2.csv"
        
        fig = pl.figure()
        ax = fig.add_subplot(111,projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Scatter plot of " + str(joint+1) + " . joint")

        with open(files) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for idx,row in enumerate(csv_reader):      
                if idx > 0 and row != []:
                    array = []
                    idy = (joint+1)*3

                    for idz,data in enumerate(row):
                        string  = data.replace("[","")
                        string = string.replace("]","")
                        string = string.split(" ")
                        for item in string:
                            if idz < idy  and idz >= idy - 3:
                                array.append(float(item))
                            
                        
                        
                            
                        
                    x = np.append(x,array[0])
                    y = np.append(y,array[1])
                    z = np.append(z,array[2])

                    
                  


                elif idx == 0 and row != [] and idx != 0:
                    array = []
                    for data in row:
                        string  = data.replace("[","")
                        string = string.replace("]","")
                        string = string.split(" ")
                        for item in string:
                            array.append(float(item))
                    values = np.array([array[joint]],dtype="f")

        
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
            #pl.show()
            pl.savefig('absolute_position_baseline_2/' + name)
            pl.close()


           
        


show_position(0,0)