import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from kNN import KNN
import random

class Test:
    def __init__(self, data_name="labeled-examples.txt"):
        self.Testing_Data = None
        self.Testing_Labels = None
        self.Training_Labels = None
        self.Training_Data = None
        self.data = None
        self.data_name = data_name

    # imports data, reads in from text file in format: [class, x, y, name]
    def import_data(self):
        dummyData = open(self.data_name, "r")
        data_raw = dummyData.readlines()
        self.data = [line.split() for line in data_raw]
        random.shuffle(self.data)
        self.Training_Data = [[float(self.data[i][1]), float(self.data[i][2])] for i in range(len(self.data))]
        self.Training_Labels = [self.data[i][0] for i in range(len(self.data))]
        self.Testing_Labels = self.Training_Labels
        self.Testing_Data = self.Training_Data

    def import_data_iris(self):
        dummyData = open(self.data_name, "r")
        data_raw = dummyData.readlines()
        self.data = [line.split(",") for line in data_raw]
        random.shuffle(self.data)
        self.Training_Data = [[float(self.data[i][0]), float(self.data[i][1]), float(self.data[i][2]), float(self.data[i][3])] for i in range(len(self.data))]
        self.Training_Labels = [self.data[i][4] for i in range(len(self.data))]
        self.Testing_Labels = self.Training_Labels
        self.Testing_Data = self.Training_Data
    # Splits data set into N chunks. Loops and selects each chunk to be test data and lets rest of data be the training
    # data. Runs algorithm on both training and test data and calculates average accuracy.
    def cross_validation(self, N=5, k_value=5):
        chunk_length = len(self.Training_Data) // N
        acc_train = 0
        acc_test = 0
        for j in range(N):
            chunks_Training_Data = [self.Training_Data[i:i + chunk_length] for i in range(0, len(self.Training_Data), chunk_length)]
            chunks_Training_Labels = [self.Training_Labels[i:i + chunk_length] for i in range(0, len(self.Training_Labels), chunk_length)]
            self.Testing_Data = chunks_Training_Data[j]
            self.Testing_Labels = chunks_Training_Labels[j]
            new_Training_Data = []
            new_Training_Labels = []
            for k, chunk in enumerate(chunks_Training_Data):
                if k != j:
                    new_Training_Data.extend(chunk)
                    new_Training_Labels.extend(chunks_Training_Labels[k])
            pred_train, pred_test = self.classify(new_Training_Data, new_Training_Labels, self.Testing_Data, self.Testing_Labels, k_value)

            count = 0
            for i in range(len(pred_train)):
                count += (pred_train[i] == new_Training_Labels[i])
            acc_train += (count / len(pred_train))

            count = 0
            for i in range(len(pred_test)):
                count += (pred_test[i] == self.Testing_Labels[i])
            acc_test += (count / len(pred_test))
        return [acc_train / N, acc_test / N]

    # Plots data
    def plot_data(self):
        classification = [int(point[0]) for point in self.data]
        colors = cm.rainbow([classw / max(classification) for classw in classification])
        x_coords = [float(point[1]) for point in self.data]
        y_coords = [float(point[2]) for point in self.data]
        plt.scatter(x_coords, y_coords, c=colors)
        plt.show()

    # Calls KNN Algorithm
    def classify(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, k_value):
        clf = KNN(k_value)
        clf.fit(Training_Data, Training_Labels, True)
        predictions_train = clf.predict(Training_Data, Training_Labels, True)
        predictions_test = clf.predict(Testing_Data, Testing_Labels, True)
        return predictions_train, predictions_test


x = Test("bezdekIris.data")
x.import_data_iris()
for j in range(1, 100, 2):
    gaming = x.cross_validation(k_value=j)
    print(gaming[0],",", gaming[1])
