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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_moons

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

EV = list(df['EA vs SHE (eV)'][0:18414])
IP = list(df['IP vs SHE (eV)'][0:18414])


# Define all splits
#splits = [split0, split1, split2, split3, split4]

# Initialize dictionaries to store scores for each model and fingerprint combination
results = {
    'SVR_Morgan': {'R2': [], 'MSE': [], 'RMSE': []},
    'SVR_ECFP': {'R2': [], 'MSE': [], 'RMSE': []},
    'SVR_BigFP': {'R2': [], 'MSE': [], 'RMSE': []},
    #'GP_Morgan': {'R2': [], 'MSE': [], 'RMSE': []},
    #'GP_ECFP': {'R2': [], 'MSE': [], 'RMSE': []},
    #'GP_BigFP': {'R2': [], 'MSE': [], 'RMSE': []},
    'RF_Morgan': {'R2': [], 'MSE': [], 'RMSE': []},
    'RF_ECFP': {'R2': [], 'MSE': [], 'RMSE': []},
    'RF_BigFP': {'R2': [], 'MSE': [], 'RMSE': []}
}

# 5-fold cross-validation loop
for fold in range(5):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}: Test split = split{fold}")
    print(f"{'='*50}")
    
    # Set test index to current fold
    test_ix = splits[fold]
    
    # Create train index by concatenating all other splits
    train_splits = [splits[i] for i in range(5) if i != fold]
    train_ix = np.concatenate(train_splits)
    
    # Prepare data (fix the indexing error in original code)
    EV = np.array(EV)
    EV_train = EV[train_ix]
    EV_test = EV[test_ix]
    

    
    print(f"Train size: {len(train_ix)}, Test size: {len(test_ix)}")
    
    # SVR with different fingerprints
    print("\n--- SVR Models ---")
    
    # Morgan FP
    regr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    regr.fit(psmiles_MorganFP_Matrix[train_ix], EV[train_ix])
    EV_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])
    
    R2_test = r2_score(EV[test_ix], EV_test_predict)
    MSE_test = mean_squared_error(EV[test_ix], EV_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['SVR_Morgan']['R2'].append(R2_test)
    results['SVR_Morgan']['MSE'].append(MSE_test)
    results['SVR_Morgan']['RMSE'].append(RMSE_test)
    
    print(f"Morgan FP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # ECFP
    regr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    regr.fit(ECFP_Matrix[train_ix], EV[train_ix])
    EV_test_predict = regr.predict(ECFP_Matrix[test_ix])
    
    R2_test = r2_score(EV[test_ix], EV_test_predict)
    MSE_test = mean_squared_error(EV[test_ix], EV_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['SVR_ECFP']['R2'].append(R2_test)
    results['SVR_ECFP']['MSE'].append(MSE_test)
    results['SVR_ECFP']['RMSE'].append(RMSE_test)
    
    print(f"ECFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # BigFP
    regr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    regr.fit(BigFP_Matrix[train_ix], EV[train_ix])
    EV_test_predict = regr.predict(BigFP_Matrix[test_ix])
    
    R2_test = r2_score(EV[test_ix], EV_test_predict)
    MSE_test = mean_squared_error(EV[test_ix], EV_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['SVR_BigFP']['R2'].append(R2_test)
    results['SVR_BigFP']['MSE'].append(MSE_test)
    results['SVR_BigFP']['RMSE'].append(RMSE_test)
    
    print(f"BigFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")


   
    # Random Forest with different fingerprints
    print("\n--- Random Forest Models ---")
    
    # Morgan FP
    regr = RandomForestRegressor(n_estimators=50, 
                                max_depth=200, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=600,
                                random_state=0)
    regr.fit(psmiles_MorganFP_Matrix[train_ix], EV[train_ix])
    EV_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])
    
    R2_test = r2_score(EV[test_ix], EV_test_predict)
    MSE_test = mean_squared_error(EV[test_ix], EV_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['RF_Morgan']['R2'].append(R2_test)
    results['RF_Morgan']['MSE'].append(MSE_test)
    results['RF_Morgan']['RMSE'].append(RMSE_test)
    
    print(f"Morgan FP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # ECFP
    regr = RandomForestRegressor(n_estimators=150, 
                                max_depth=100, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=600,
                                random_state=0)
    regr.fit(ECFP_Matrix[train_ix], EV[train_ix])
    EV_test_predict = regr.predict(ECFP_Matrix[test_ix])
    
    R2_test = r2_score(EV[test_ix], EV_test_predict)
    MSE_test = mean_squared_error(EV[test_ix], EV_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['RF_ECFP']['R2'].append(R2_test)
    results['RF_ECFP']['MSE'].append(MSE_test)
    results['RF_ECFP']['RMSE'].append(RMSE_test)
    
    print(f"ECFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # BigFP
    regr = RandomForestRegressor(n_estimators=50, 
                                max_depth=100, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=600,
                                random_state=0)
    regr.fit(BigFP_Matrix[train_ix], EV[train_ix])
    EV_test_predict = regr.predict(BigFP_Matrix[test_ix])
    
    R2_test = r2_score(EV[test_ix], EV_test_predict)
    MSE_test = mean_squared_error(EV[test_ix], EV_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['RF_BigFP']['R2'].append(R2_test)
    results['RF_BigFP']['MSE'].append(MSE_test)
    results['RF_BigFP']['RMSE'].append(RMSE_test)
    
    print(f"BigFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
 
# Calculate and display final statistics
print(f"\n{'='*70}")
print("FINAL CROSS-VALIDATION RESULTS (Mean ± Std)")
print(f"{'='*70}")

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, scores in metrics.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {metric_name}: {mean_score:.4f} ± {std_score:.4f}")

# Optional: Create a summary DataFrame for better visualization
import pandas as pd

summary_data = []
for model_name, metrics in results.items():
    for metric_name, scores in metrics.items():
        summary_data.append({
            'Model': model_name,
            'Metric': metric_name,
            'Mean': np.mean(scores),
            'Std': np.std(scores),
            'Fold_1': scores[0],
            'Fold_2': scores[1],
            'Fold_3': scores[2],
            'Fold_4': scores[3],
            'Fold_5': scores[4]
        })

summary_df = pd.DataFrame(summary_data)
print(f"\n{'='*70}")
print("DETAILED RESULTS TABLE:")
print(f"{'='*70}")
print(summary_df.to_string(index=False))

