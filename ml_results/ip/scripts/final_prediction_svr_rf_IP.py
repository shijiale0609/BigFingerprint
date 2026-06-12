import copy
import json
import random
import pickle
from pathlib import Path

from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
plt.rcParams['text.usetex'] = False
# Setting the default font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.stats import bernoulli
from sklearn.decomposition import FastICA, FactorAnalysis, PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.manifold import Isomap
from sklearn.metrics import (confusion_matrix, pairwise_distances, r2_score)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_moons
from matplotlib import rcParams


# Suppress future warnings
simplefilter(action='ignore', category=FutureWarning)

psmiles_MorganFP_Matrix = np.load("./datasets/vipea/psmiles_mfp_arr.npy")
print(psmiles_MorganFP_Matrix.shape)

ECFP_Matrix = np.load("./datasets/vipea/EVIP_pecfp_arr.npy")
print(ECFP_Matrix.shape)

BigFP_Matrix = np.load("./datasets/vipea/EVIP_bigfp.npy")
print(BigFP_Matrix.shape)

from sklearn.model_selection import KFold

# Sample list
#data = list(range(682*3))  # Example list
data = []

monoA = list(range(9))
monoB = list(range(682))
top = list(range(3))

random.seed(0)
monoB_sublist = random.sample(monoB, 100)

for i in monoA:
    for j in monoB_sublist:
        for k in top:
            data.append(i*682*3+j*3+k)


# Create KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create variables to hold each fold's test indices
split0, split1, split2, split3, split4 = [], [], [], [], []

# Collect indices
splits = [split0, split1, split2, split3, split4]

for i, (_, test_index) in enumerate(kf.split(data)):
    splits[i].extend(test_index)


import pandas as pd
df = pd.read_csv('./datasets/vipea/dataset.csv')

print(max(df['EA vs SHE (eV)']), min(df['EA vs SHE (eV)']))

#EV = list(df['EA vs SHE (eV)'][0:18414])
IP = list(df['IP vs SHE (eV)'][0:18414])

train_ix = np.concatenate((np.concatenate((np.concatenate((split1, split2)), split3)), split4))
test_ix = split0

ip_values = np.array(IP)
ip_train = ip_values[train_ix]
ip_test = ip_values[test_ix]

### SVR with different fingerprints


regr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
regr.fit(psmiles_MorganFP_Matrix[train_ix],ip_values[train_ix])
ip_train_predict = regr.predict(psmiles_MorganFP_Matrix[train_ix])
ip_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])

R2_train = r2_score(ip_values[train_ix], ip_train_predict)
R2_test = r2_score(ip_values[test_ix], ip_test_predict)
print( "Morgan FP: Train R2 score:",R2_train, 
  "Test R2 score:",R2_test)
MSE_train = mean_squared_error(ip_values[train_ix], ip_train_predict)
MSE_test = mean_squared_error(ip_values[test_ix], ip_test_predict)
print("Morgan FP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("Morgan FP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
plt.tick_params(axis='x', labelbottom=False)
plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("psmiles_MorganFP_IP_prediction_SVR.svg", transparent=True, bbox_inches='tight')



plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
plt.ylabel("Predicted IP (eV)", fontsize= 18)
plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
#plt.tick_params(axis='x', labelbottom=False)
#plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("axis_IP.svg",  transparent=True, bbox_inches='tight')


regr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
regr.fit(ECFP_Matrix[train_ix],ip_values[train_ix])
ip_train_predict = regr.predict(ECFP_Matrix[train_ix])
ip_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(ip_values[train_ix], ip_train_predict)
R2_test = r2_score(ip_values[test_ix], ip_test_predict)
print("ECFP: Train R2 score:",R2_train, 
  "Test R2 score:",R2_test)
MSE_train = mean_squared_error(ip_values[train_ix], ip_train_predict)
MSE_test = mean_squared_error(ip_values[test_ix], ip_test_predict)
print("ECFP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("ECFP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)


plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
plt.tick_params(axis='x', labelbottom=False)
plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("ECFP_IP_prediction_SVR.svg", transparent=True, bbox_inches='tight')


regr = SVR(kernel='rbf', C=10.0, epsilon=0.001)
regr.fit(BigFP_Matrix[train_ix],ip_values[train_ix])
ip_train_predict = regr.predict(BigFP_Matrix[train_ix])
ip_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(ip_values[train_ix], ip_train_predict)
R2_test = r2_score(ip_values[test_ix], ip_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
  "Test R2 score:",R2_test)
MSE_train = mean_squared_error(ip_values[train_ix], ip_train_predict)
MSE_test = mean_squared_error(ip_values[test_ix], ip_test_predict)
print("BigFP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("BigFP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
plt.tick_params(axis='x', labelbottom=False)
plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("BigFP_IP_prediction_SVR.svg", transparent=True, bbox_inches='tight')




### Random Forest with different fingerprints
regr = RandomForestRegressor(n_estimators=150, 
                                max_depth=100, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=600,
                                random_state=0) 
regr.fit(psmiles_MorganFP_Matrix[train_ix],ip_values[train_ix])
regr.fit(psmiles_MorganFP_Matrix[train_ix],ip_values[train_ix])
ip_train_predict = regr.predict(psmiles_MorganFP_Matrix[train_ix])
ip_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])

R2_train = r2_score(ip_values[train_ix], ip_train_predict)
R2_test = r2_score(ip_values[test_ix], ip_test_predict)
print( "Morgan FP: Train R2 score:",R2_train, 
    "Test R2 score:",R2_test)
MSE_train = mean_squared_error(ip_values[train_ix], ip_train_predict)
MSE_test = mean_squared_error(ip_values[test_ix], ip_test_predict)
print("Morgan FP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("Morgan FP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
plt.tick_params(axis='x', labelbottom=False)
plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("psmiles_MorganFP_IP_prediction_RF.svg", transparent=True,  bbox_inches='tight')


regr = RandomForestRegressor(n_estimators=200, 
                                max_depth=100, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=800,
                                random_state=0)

regr.fit(ECFP_Matrix[train_ix],ip_values[train_ix])
ip_train_predict = regr.predict(ECFP_Matrix[train_ix])
ip_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(ip_values[train_ix], ip_train_predict)
R2_test = r2_score(ip_values[test_ix], ip_test_predict)
print( "ECFP: Train R2 score:",R2_train, 
    "Test R2 score:",R2_test)
MSE_train = mean_squared_error(ip_values[train_ix], ip_train_predict)
MSE_test = mean_squared_error(ip_values[test_ix], ip_test_predict)
print("ECFP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("ECFP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
plt.tick_params(axis='x', labelbottom=False)
plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("ECFP_IP_prediction_RF.svg",  transparent=True, bbox_inches='tight')



regr = RandomForestRegressor(n_estimators=200, 
                                max_depth=100, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=1000,
                                random_state=0)

regr.fit(BigFP_Matrix[train_ix],ip_values[train_ix])
ip_train_predict = regr.predict(BigFP_Matrix[train_ix])
ip_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(ip_values[train_ix], ip_train_predict)
R2_test = r2_score(ip_values[test_ix], ip_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
    "Test R2 score:",R2_test)
MSE_train = mean_squared_error(ip_values[train_ix], ip_train_predict)
MSE_test = mean_squared_error(ip_values[test_ix], ip_test_predict)
print("BigFP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("BigFP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0,4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0, 1, 2, 3, 4], fontsize=14)
plt.yticks([0, 1, 2, 3, 4], fontsize=14)
plt.tick_params(axis='x', direction='in')  # x-axis only
plt.tick_params(axis='y', direction='in')  # y-axis only
plt.tick_params(axis='x', labelbottom=False)
plt.tick_params(axis='y', labelleft=False)
legend = plt.legend(loc="lower right",fontsize=16)
legend.get_frame().set_facecolor('none')  # Transparent background
legend.get_frame().set_edgecolor('none')  # Remove border
plt.margins(0)
plt.tight_layout()
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.savefig("BigFP_IP_prediction_RF.svg",  transparent=True, bbox_inches='tight')
