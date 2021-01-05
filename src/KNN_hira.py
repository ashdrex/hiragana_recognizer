# accuracy too low

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt


train_X = np.load("hiragana_train_X.npz")['arr_0']
train_y = np.load("hiragana_train_y.npz")['arr_0']
test_X = np.load("hiragana_test_X.npz")['arr_0']
test_y = np.load("hiragana_test_y.npz")['arr_0']


plt.figure(figsize=(6,6)).patch.set_facecolor('black')
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.reshape(train_X[i], (48,48)), cmap=plt.cm.binary)
plt.show()

# n = 15

# knn = KNeighborsClassifier(n_neighbors=n)   # can change neighbors to see accuracy change
# knn.fit(train_X, train_y)

# pred_y = knn.predict(test_X)

# print(metrics.balanced_accuracy_score(test_y, pred_y))