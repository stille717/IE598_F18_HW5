#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:56:15 2018

@author: stille
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
#df_wine = pd.read_csv('/Users/stille/Desktop/UIUC/MachineLearning/HW5/wine.csv')
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', \
                   'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', \
                   'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
cols=['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', \
                   'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', \
                   'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

#draw statistical plots
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df_wine[cols], size=2.5);
plt.show()

#heatmaps
cm = np.corrcoef(df_wine[cols].values.T) 
sns.set(font_scale=1.5) 
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols, xticklabels=cols) 
plt.show() 

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w= np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n',w)

X_train_std[0].dot(w)
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):plt.scatter(X_train_pca[y_train==l, 0], \
                  X_train_pca[y_train==l, 1],c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

lr = LogisticRegression()
lr.fit(X_train_std, y_train)
y_lr_train_pred = lr.predict(X_train_std)
print("Baseline Logistic Regression Train Acc:")
print(accuracy_score(y_train,y_lr_train_pred))

y_lr_test_pred = lr.predict(X_test_std)
print("Baseline Logistic Regression Test Acc:")
print(accuracy_score(y_test,y_lr_test_pred))

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
y_svm_train_pred = svm.predict(X_train_std)
print("Baseline SVM Train Acc:")
print(accuracy_score(y_train,y_svm_train_pred))

y_svm_test_pred = svm.predict(X_test_std)
print("Baseline SVM Test Acc:")
print(accuracy_score(y_test,y_svm_test_pred))



#******************PCA********************
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
y_lr_PCA_train_pred = lr.predict(X_train_pca)
print("PCA Logistic Regression Train Acc:")
print(accuracy_score(y_train,y_lr_PCA_train_pred))

y_lr_PCA_test_pred = lr.predict(X_test_pca)
print("PCA Logistic Regression Test Acc:")
print(accuracy_score(y_test,y_lr_PCA_test_pred))

svm.fit(X_train_pca, y_train)
y_SVM_PCA_train_pred = svm.predict(X_train_pca)
print("PCA SVM Train Acc:")
print(accuracy_score(y_train,y_SVM_PCA_train_pred))

y_SVM_PCA_test_pred = svm.predict(X_test_pca)
print("PCA SVM Test Acc:")
print(accuracy_score(y_test,y_SVM_PCA_test_pred))


#******************LDA********************
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train_std,y_train)
X_test_lda = lda.transform(X_test_std)
lr.fit(X_train_lda, y_train)
y_lr_LDA_train_pred = lr.predict(X_train_lda)
print("LDA Logistic Regression Train Acc:")
print(accuracy_score(y_train,y_lr_LDA_train_pred))

y_lr_LDA_test_pred = lr.predict(X_test_lda)
print("LDA Logistic Regression Test Acc:")
print(accuracy_score(y_test,y_lr_LDA_test_pred))

svm.fit(X_train_lda, y_train)
y_SVM_LDA_train_pred = svm.predict(X_train_lda)
print("LDA SVM Train Acc:")
print(accuracy_score(y_train,y_SVM_LDA_train_pred))

y_SVM_LDA_test_pred = svm.predict(X_test_lda)
print("LDA SVM Test Acc:")
print(accuracy_score(y_test,y_SVM_LDA_test_pred))

#******************kPCA********************
parameters = [0.01, 0.1, 1,10,100]
for i in range(5):
    kpca = KernelPCA(n_components=2,kernel='rbf')
    kpca.gamma=parameters[i]
    print("gamma=",parameters[i])
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    

    lr.fit(X_train_kpca, y_train)
    y_lr_kPCA_train_pred = lr.predict(X_train_kpca)
    print("kPCA Logistic Regression Train Acc:")
    print(accuracy_score(y_train,y_lr_kPCA_train_pred))
    
    y_lr_kPCA_test_pred = lr.predict(X_test_kpca)
    print("kPCA Logistic Regression Test Acc:")
    print(accuracy_score(y_test,y_lr_kPCA_test_pred))

    svm.fit(X_train_kpca, y_train)
    y_SVM_kPCA_train_pred = svm.predict(X_train_kpca)
    print("kPCA SVM Train Acc:")
    print(accuracy_score(y_train,y_SVM_kPCA_train_pred))
    
    y_SVM_kPCA_test_pred = svm.predict(X_test_kpca)
    print("kPCA SVM Test Acc:")
    print(accuracy_score(y_test,y_SVM_kPCA_test_pred))

print("My name is {Wenyu Ni}")
print("My NetID is: {wenyuni2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



