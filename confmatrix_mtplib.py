# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 20:48:26 2020

@author: ddelgadillo
"""

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

'''
'''



def plotCM(y_true, y_pred, matrixName, figsize=(5,5), save = False):
    '''
        Función para graficar una mátriz de confusion 
        a partir de las listas y_true, y_pred
        
        
    Parametros
    ----------
    y_true : list
    y_pred : list
    matrixName : title, file name
    figsize : TYPE, optional
        DESCRIPTION. The default is (5,5).
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.
    '''
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'True'
    cm.columns.name = 'Predicted'
    #fig, ax = plt.subplots(figsize=figsize, dpi=500)
    plt.figure(dpi=300, figsize = figsize)
    sns.heatmap(cm, cmap= "GnBu", annot=annot, annot_kws = {'fontsize':14}, fmt='', cbar=False)
    plt.xlabel('Predicted', fontdict = {'fontsize': 14})
    plt.ylabel('True', fontdict = {'fontsize': 14})
    plt.title(matrixName, fontdict = {'fontsize': 21})
    if save:
        nameExport = matrixName.replace(' ','_')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 900, quality = 100)
        plt.savefig(svgp, dpi = 900, quality = 100)
        plt.show()



def plotCM2(cm, tags,matrixName, save = False):
    cm = np.asarray(cm) 
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype('U25')
    annot
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                print(s)
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index = tags, columns=tags)
    cm.index.name = 'True'
    cm.columns.name = 'Predicted'
    #fig, ax = plt.subplots(figsize=figsize, dpi=500)
    

    #ax = plt.subplots()
    plt.figure(dpi=500, figsize = (5,5))
    sns.heatmap(cm, cmap= "GnBu", annot=annot, annot_kws = {'fontsize':9}, fmt='', cbar=False)
    plt.xlabel('Predicted', fontdict = {'fontsize': 9})
    plt.ylabel('True', fontdict = {'fontsize': 9})
    plt.title(matrixName, fontdict = {'fontsize': 10})
    if save:
        nameExport = matrixName.replace(' ','_')
        nameExport = nameExport.replace(',','')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 900, quality = 100)
        plt.savefig(svgp, dpi = 900, quality = 100)
        #plt.savefig(tiffp, dpi = 900, quality = 100)
        plt.show()
    return



def arrayFix(cm):
    '''
    Reoordenar arreglo o lista 3x3 
    (cambio Medio por bajo y vsa.)

    Parameters
    ----------
    cm : arreglo 3x3 o lista de 3 listas, 
         cada una de 3 elementos 

    Returns
    -------
    lista de 3 listas, 
    cada una de 3 elementos

    '''
    cm = np.asarray(cm)
    tmp_cm = cm.copy()
    tmp_cm[:,1] = cm[:,2]
    tmp_cm[:,2] = cm[:,1]
    cm = tmp_cm.copy()
    cm[1,:] = tmp_cm[2,:]
    cm[2,:] = tmp_cm[1,:]
    return list(cm)



y_true = ['a','a','b','b','a','a','c','b','b','a','a','b','c','b']
y_pred = ['a','c','b','a','c','a','c','b','b','a','a','a','c','b']

matrixName = 'test'



tags = ['High','Medium','Low']
matrixName = 'Linear SVM classifier, best by Mattews Correlation Coefficient'
cm=[[0,18,98],
[0,98,85],
[0,97,257]]

#plotCM2(cm, tags,matrixName, save = True)

n1 = 'K Linear SVM classifier, best by Mattews Correlation Coefficient2'
cm1=[[0,98,18],
     [0,257,97],
     [0,85,98]]

n2 = 'K Linear SVM classifier, best by F1 score'
cm2=[[0,14,102],
[0,93,90],
[0,93,261]]

n3 = 'K Linear SVM classifier, best by Accuracy'
cm3=[[0,12,104],
[0,82,101],
[0,82,272]]

n4 = 'K Linear SVM Classifier, best by NPV metric'
cm4=[[0,0,116],
[0,5,178],
[0,2,352]]

n5 = 'K Linear SVM Classifier, best by PPV metric'
cm5=[[0,0,116],
[0,8,175],
[0,6,348]]

n6 = 'K Linear SVM Classifier, best by TNR metric'
cm6=[[59,27,30],
[29,110,44],
[101,111,142]]

n7 = 'K Linear SVM Classifier, best by TPR metric'
cm7=[[59,27,30],
[29,110,44],
[101,111,142]]



cm = [[1,4,5],
      [6,2,7],
      [8,9,3]]

cm1 = arrayFix(cm1)
cm2 = arrayFix(cm2)
cm3 = arrayFix(cm3)
cm4 = arrayFix(cm4)
cm5 = arrayFix(cm5)
cm6 = arrayFix(cm6)
cm7 = arrayFix(cm7)


plotCM2(cm1, tags, n1, save = True)
plotCM2(cm2, tags, n2, save = True)
plotCM2(cm3, tags, n3, save = True)
plotCM2(cm4, tags, n4, save = True)
plotCM2(cm5, tags, n5, save = True)
plotCM2(cm6, tags, n6, save = True)
plotCM2(cm7, tags, n7, save = True)


