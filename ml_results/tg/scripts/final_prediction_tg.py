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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_moons

# Suppress future warnings
simplefilter(action='ignore', category=FutureWarning)

psmiles_MorganFP_Matrix = np.load("psmiles_radius_2_2048_morganfp.npy")
print(psmiles_MorganFP_Matrix.shape)

ECFP_Matrix = np.load("bigfingerprint_matrix_Bicerano_local_7property_2048.npy")
print(ECFP_Matrix.shape)

BigTOP_Matrix = np.load("Bicerano_bigsmiles_topfp_update_40d.npy")
print(BigTOP_Matrix.shape)

BigFP_Matrix = np.copy(np.concatenate((ECFP_Matrix, BigTOP_Matrix), axis=1))
print(BigFP_Matrix.shape)


split0 = np.load("./tg_315_indices_folder1.npy")
split1 = np.load("./tg_315_indices_folder2.npy")
split2 = np.load("./tg_315_indices_folder3.npy")
split3 = np.load("./tg_315_indices_folder4.npy")
split4 = np.load("./tg_315_indices_folder5.npy")

import pandas as pd
df1 = pd.read_excel('../ap0c00524_si_001.xlsx')
df1
columns = list(df1.columns)
tg_exp = list(df1[columns[3]])
rho_cal = list(df1[columns[7]])
rubber_cte = list(df1[columns[11]])

train_ix = np.concatenate((np.concatenate((np.concatenate((split1, split2)), split3)), split4))
test_ix = split0


from sklearn.model_selection import train_test_split

train_ix, test_ix = train_test_split(np.arange(0,305), random_state=0, test_size=0.2, shuffle=True)



tg_exp = np.array(tg_exp)
tg_exp_train = tg_exp[train_ix]
tg_exp_test = tg_exp[test_ix]

rho_cal = np.array(rho_cal)
rho_cal_train = rho_cal[train_ix]
rho_cal_test = rho_cal[train_ix]

rubber_cte = np.array(rubber_cte)
rubber_cte_train = rubber_cte[train_ix]
rubber_cte_test = rubber_cte[train_ix]



### SVR with different fingerprints

regr = SVR(kernel='rbf', C=800, epsilon=0.1)
regr.fit(psmiles_MorganFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(psmiles_MorganFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "Morgan FP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)


regr = SVR(kernel='rbf', C=1500, epsilon=0.1)
regr.fit(ECFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(ECFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print("ECFP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)

regr = SVR(kernel='rbf', C=1000, epsilon=0.1)
regr.fit(BigFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(BigFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)


### gaussian process with different fingerprints
kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
regr = GaussianProcessRegressor(kernel=kernel, alpha= 500, n_restarts_optimizer=20, random_state=42)
regr.fit(psmiles_MorganFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(psmiles_MorganFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "Morgan FP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)


kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
regr = GaussianProcessRegressor(kernel=kernel, alpha= 300, n_restarts_optimizer=20, random_state=42)
regr.fit(ECFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(ECFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "ECFP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)


kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
regr = GaussianProcessRegressor(kernel=kernel, alpha= 300, n_restarts_optimizer=20, random_state=42)
regr.fit(BigFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(BigFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)



### Random Forest with different fingerprints
regr = RandomForestRegressor(n_estimators = 1200, 
                              max_depth=50, 
                              min_samples_split=2, 
                              min_samples_leaf=1, 
                              max_features=1.0,
                              max_leaf_nodes=100,
                              random_state=42) 
regr.fit(psmiles_MorganFP_Matrix[train_ix],tg_exp[train_ix])
regr.fit(psmiles_MorganFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(psmiles_MorganFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "Morgan FP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)


regr = RandomForestRegressor(n_estimators = 2000, 
                              max_depth=20, 
                              min_samples_split=2, 
                              min_samples_leaf=1, 
                              max_features=1.0,
                              max_leaf_nodes=100,
                              random_state=42)

regr.fit(ECFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(ECFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "ECFP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)

regr = RandomForestRegressor(n_estimators = 2000, 
                              max_depth=20, 
                              min_samples_split=2, 
                              min_samples_leaf=1, 
                              max_features=1.0,
                              max_leaf_nodes=100,
                              random_state=42)

regr.fit(BigFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(BigFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
      "Test R2 score:",R2_test)





