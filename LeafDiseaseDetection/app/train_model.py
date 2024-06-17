import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle


def read_data(path="dataset/train/", size=(32, 32)):
    images, labels, label_dict = read_data_with_labels(path, size)
    return np.array(images), np.array(labels), label_dict


def read_data_with_labels(path="dataset/train/", size=(32, 32)):
    images, labels = [], []
    label_dict = {}
    lbl = 0
    for folder_name in os.listdir(path):
        lbl += 1
        label_dict[lbl] = folder_name
        p = os.path.join(path, folder_name)
        for file in os.listdir(p):
            img = cv2.imread(os.path.join(p, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(lbl)
    if not images or not labels:
        raise ValueError("No data found or error occurred while reading data.")
    return images, labels, label_dict


def train_knn(X, y):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    return knn


if __name__ == '__main__':
    X, y, label_dict = read_data()
    num_samples, width, height = X.shape
    X = X.reshape(num_samples, width * height)
    knn = train_knn(X, y)
    with open('models/knn_model.pkl', 'wb') as f:
        pickle.dump((knn, label_dict), f)
