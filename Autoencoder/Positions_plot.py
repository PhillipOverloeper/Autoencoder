import matplotlib.pyplot as pl
import csv
import numpy as np

def show_position(episode,iterations):
    for joint in range(0,7):
        pl.close()
        values = np.array([])
        files = "joint_position.csv"

        with open(files) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for idx,row in enumerate(csv_reader):      
                if idx > 0 and row != []:
                    array = []
                    for data in row:
                        string  = data.replace("[","")
                        string = string.replace("]","")
                        string = string.split(" ")
                        for item in string:
                            array.append(float(item))

                    values = np.append(values,array[joint])
                elif idx == 0 and row != []:
                    array = []
                    for data in row:
                        string  = data.replace("[","")
                        string = string.replace("]","")
                        string = string.split(" ")
                        for item in string:
                            array.append(float(item))
                    values = np.array([array[joint]],dtype="f")

        time = np.arange(0,iterations,1)



        pl.title("Position of " + str(joint + 1) + ". joint")
        pl.xlabel("Time [s]")
        pl.ylabel("Positions [radians]")
        pl.plot(time,values)
        #pl.show()
        name = str(episode) + '_' + str(joint) + '.png'
        pl.savefig('position_plots/' +name)
        pl.close()


