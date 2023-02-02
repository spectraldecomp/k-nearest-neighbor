import numpy as np
from collections import Counter
from collections import defaultdict


def distance(x1, x2):
    sumd = 0
    for i in range(len(x1)):
        sumd += (x1[i] - x2[i]) ** 2
    return np.sqrt(sumd)


class KNN:
    # Find distance from point1 to point2, Euclidean.
    # Works for as many features as you'd like, not just x and y-axis.

    def __init__(self, k=5):
        self.k = k

    # Load training data and labels into model, selects the variation
    def fit(self, Training_Data, Training_Labels, variation):
        if variation: self.fit_all(Training_Data, Training_Labels)
        else: self.fit_errors(Training_Data, Training_Labels)

    # Stores each training instance in the model
    def fit_all(self, Training_Data, Training_Labels):
        self.Training_Data = Training_Data
        self.Training_Labels = Training_Labels

    def fit_errors(self, Training_Data, Training_Labels):
        self.label_dict = defaultdict(list)
        for i, label in enumerate(Training_Labels):
            if len(self.label_dict[label]) < self.k:
                self.label_dict[label].append(Training_Data[i])

    def predict(self, Testing_Data, Testing_Labels, variation):
        if variation: return self.predict_all(Testing_Data)
        else: return self.predict_errors(Testing_Data, Testing_Labels)

    # Runs helper function predicth for each value in our testing data. Gets prediction for each,
    # saved in a list
    def predict_all(self, Testing_Data):
        predictions = [self.predicth(x) for x in Testing_Data]
        return predictions

    def predict_errors(self, Testing_Data, Testing_Labels):
        predictions = []
        for i, x in enumerate(Testing_Data):
            predictions.append(self.predicth_errors(x))
            if self.predicth_errors(x) != Testing_Labels[i]:
                self.label_dict[Testing_Labels[i]].append(x)
        return predictions

    def predicth_errors(self, x):
        distances = []
        for label, training_data in self.label_dict.items():
            for train in training_data:
                distances.append((distance(x, train), label))
        distances.sort(key=lambda x: x[0])
        labels = [label for _, label in distances]
        most_common = Counter(labels[:self.k]).most_common()
        return most_common[0][0]


    # Helper function. Finds distance between point x and each data point in our training set.
    # Finds the k closest points and labels and uses Counter object to find the most common label.
    # Not very useful for even values of k, but that is fine.
    def predicth(self, x):
        # Distance from x to each training point
        distances = [distance(x, training_data) for training_data in self.Training_Data]

        # Finds k closest labels
        k_closest = np.argsort(distances)[:self.k]
        k_closest_labels = [self.Training_Labels[i] for i in k_closest]

        # Returns most common label (assuming k is odd and no ties)
        most_common = Counter(k_closest_labels).most_common()
        return (most_common[0][0])
