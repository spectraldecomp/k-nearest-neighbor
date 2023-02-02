import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kNN import KNN

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


# clf = KNN(k=5)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# acc = np.sum(predictions == y_test) / len(y_test)
# print(acc)

dummyData = open("labeled-examples.txt", "r")
data_raw = dummyData.readlines()
data = [line.split() for line in data_raw]
Training_Data = [[float(data[i][1]), float(data[i][2])] for i in range(len(data))]
Training_Labels = [data[i][0] for i in range(len(data))]
Testing_Labels = Training_Labels

clf = KNN(k=5)
clf.fit(Training_Data, Training_Labels)
predictions = clf.predict(Training_Data)

count = 0
for i in range(len(predictions)):
    count += (predictions[i] == Testing_Labels[i])
accuracy = count/len(predictions)
print(accuracy)



