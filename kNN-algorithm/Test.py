import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from kNN import KNN


# cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# plt.figure()
# plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# clf = KNN(k=5)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# acc = np.sum(predictions == y_test) / len(y_test)
# print(acc)
class Test:
    def __init__(self, data_name="labeled-examples.txt"):
        self.Testing_Labels = None
        self.Training_Labels = None
        self.Training_Data = None
        self.data = None
        self.data_name = data_name

    def import_data(self):
        dummyData = open(self.data_name, "r")
        data_raw = dummyData.readlines()
        self.data = [line.split() for line in data_raw]
        self.Training_Data = [[float(self.data[i][1]), float(self.data[i][2])] for i in range(len(self.data))]
        self.Training_Labels = [self.data[i][0] for i in range(len(self.data))]
        self.Testing_Labels = self.Training_Labels

    def plot_data(self):
        classification = [int(point[0]) for point in self.data]
        colors = cm.rainbow([classw / max(classification) for classw in classification])
        x_coords = [float(point[1]) for point in self.data]
        y_coords = [float(point[2]) for point in self.data]
        plt.scatter(x_coords, y_coords, c=colors)
        plt.show()

    def classify(self):
        clf = KNN(k=31)
        clf.fit(self.Training_Data, self.Training_Labels, True)
        predictions = clf.predict(self.Training_Data, self.Testing_Labels, True)
        count = 0
        for i in range(len(predictions)):
            count += (predictions[i] == self.Testing_Labels[i])
        accuracy = count / len(predictions)
        print(accuracy)

test = Test()
test.import_data()
test.plot_data()
test.classify()
