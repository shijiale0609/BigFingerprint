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
plt.rcParams['text.usetex'] = True
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

#print(max(df['EA vs SHE (eV)']), min(df['EA vs SHE (eV)']))

EV = list(df['EA vs SHE (eV)'][0:18414])
IP = list(df['IP vs SHE (eV)'][0:18414])
print(max(IP), min(IP))

train_ix = np.concatenate((np.concatenate((np.concatenate((split1, split2)), split3)), split4))
test_ix = split0

ip_values = np.array(IP)
ip_train = ip_values[train_ix]
ip_test = ip_values[test_ix]

### SVR with different fingerprints


ip_train_predict = np.load("./fold_ip_0_gp_train_psmiles_mfp.npy")
ip_test_predict = np.load("./fold_ip_0_gp_test_psmiles_mfp.npy")

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0, 4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0,1,2,3,4], fontsize=14)
plt.yticks([0,1,2,3,4], fontsize=14)
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
plt.savefig("psmiles_MorganFP_IP_prediction_GP.svg", transparent=True, bbox_inches='tight')





ip_train_predict = np.load("./fold_ip_0_gp_train_pecfp.npy")
ip_test_predict = np.load("./fold_ip_0_gp_test_pecfp.npy")

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0, 4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0,1,2,3,4], fontsize=14)
plt.yticks([0,1,2,3,4], fontsize=14)
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
plt.savefig("ECFP_IP_prediction_GP.svg", transparent=True, bbox_inches='tight')


ip_train_predict = np.load("./fold_ip_0_gp_train_bigfp.npy")
ip_test_predict = np.load("./fold_ip_0_gp_test_bigfp.npy")

plt.figure(figsize = (4,4),dpi=300)
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.tick_params(length=8, width=2)
line = np.linspace(0, 4)
plt.plot(line, line, c='k')
plt.scatter(ip_values[train_ix], ip_train_predict,  c='b', edgecolors='black',label="Train")
plt.scatter(ip_values[test_ix], ip_test_predict, c='r', edgecolors='black',marker='s', label="Test")
plt.xlim(0,4)
plt.ylim(0,4)
#plt.ylabel("Predicted IP (eV)", fontsize= 18)
#plt.xlabel("Simulated IP (eV)", fontsize = 18)
plt.xticks([0,1,2,3,4], fontsize=14)
plt.yticks([0,1,2,3,4], fontsize=14)
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
plt.savefig("BigFP_IP_prediction_GP.svg", transparent=True, bbox_inches='tight')









