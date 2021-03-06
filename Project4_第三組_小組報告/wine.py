from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

# Project4_第三組_小組報告
wine = datasets.load_wine()

X = wine.data
y = wine.target

# method to choose assigned feature in wine.data
################################# 
# # s[start:stop:step]
# # X = wine.data[:, 0:7:2]

# X = list(wine.data[:,0:3])
# t = list(wine.data[:,8:9])

# for i in range(178):
#   X[i] = np.append(X[i], t[i])

# X = np.array(X)
#################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

clf = svm.SVC(C=1, kernel='linear', gamma='auto')
clf.fit(X_train, y_train) # 將訓練集送入訓練(fit)

# print("predict")
# print(clf.predict(X_train)) #target=y_train
# print(clf.predict(X_test))  #target=y_test

# clf.score(data, target)
# 輸出"以data進行predict後的結果"與"target進行比對"計算準確率
print("Accuracy")
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))