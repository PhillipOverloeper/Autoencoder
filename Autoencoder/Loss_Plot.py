import matplotlib.pyplot as pl
import csv
import numpy as np

files = "loss.csv"
values = np.array([])
loss = np.array([],dtype="f")
time = np.arange(0,20)

with open(files) as csvDataFile:
    csv_reader = csv.reader(csvDataFile)
    for data in csv_reader:
        values = np.append(values,data)
for idx,data in enumerate(values):
    loss = np.append(loss,float(data))

pl.title("Loss w.r.t epochs")
pl.xlabel("Epochs")
pl.ylabel("Loss")
pl.plot(time,loss)
pl.show()
