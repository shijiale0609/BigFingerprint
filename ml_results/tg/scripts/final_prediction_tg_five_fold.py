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


# Define all splits
splits = [split0, split1, split2, split3, split4]

# Initialize dictionaries to store scores for each model and fingerprint combination
results = {
    'SVR_Morgan': {'R2': [], 'MSE': [], 'RMSE': []},
    'SVR_ECFP': {'R2': [], 'MSE': [], 'RMSE': []},
    'SVR_BigFP': {'R2': [], 'MSE': [], 'RMSE': []},
    'GP_Morgan': {'R2': [], 'MSE': [], 'RMSE': []},
    'GP_ECFP': {'R2': [], 'MSE': [], 'RMSE': []},
    'GP_BigFP': {'R2': [], 'MSE': [], 'RMSE': []},
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
    tg_exp = np.array(tg_exp)
    tg_exp_train = tg_exp[train_ix]
    tg_exp_test = tg_exp[test_ix]
    
    rho_cal = np.array(rho_cal)
    rho_cal_train = rho_cal[train_ix]
    rho_cal_test = rho_cal[test_ix]  # Fixed: was [train_ix] in original
    
    rubber_cte = np.array(rubber_cte)
    rubber_cte_train = rubber_cte[train_ix]
    rubber_cte_test = rubber_cte[test_ix]  # Fixed: was [train_ix] in original
    
    print(f"Train size: {len(train_ix)}, Test size: {len(test_ix)}")
    
    # SVR with different fingerprints
    print("\n--- SVR Models ---")
    
    # Morgan FP
    regr = SVR(kernel='rbf', C=800, epsilon=0.1)
    regr.fit(psmiles_MorganFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['SVR_Morgan']['R2'].append(R2_test)
    results['SVR_Morgan']['MSE'].append(MSE_test)
    results['SVR_Morgan']['RMSE'].append(RMSE_test)
    
    print(f"Morgan FP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # ECFP
    regr = SVR(kernel='rbf', C=1500, epsilon=0.1)
    regr.fit(ECFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['SVR_ECFP']['R2'].append(R2_test)
    results['SVR_ECFP']['MSE'].append(MSE_test)
    results['SVR_ECFP']['RMSE'].append(RMSE_test)
    
    print(f"ECFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # BigFP
    regr = SVR(kernel='rbf', C=1000, epsilon=0.1)
    regr.fit(BigFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['SVR_BigFP']['R2'].append(R2_test)
    results['SVR_BigFP']['MSE'].append(MSE_test)
    results['SVR_BigFP']['RMSE'].append(RMSE_test)
    
    print(f"BigFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # Gaussian Process with different fingerprints
    print("\n--- Gaussian Process Models ---")
    
    # Morgan FP
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
    regr = GaussianProcessRegressor(kernel=kernel, alpha=500, n_restarts_optimizer=20, random_state=42)
    regr.fit(psmiles_MorganFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['GP_Morgan']['R2'].append(R2_test)
    results['GP_Morgan']['MSE'].append(MSE_test)
    results['GP_Morgan']['RMSE'].append(RMSE_test)
    
    print(f"Morgan FP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # ECFP
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
    regr = GaussianProcessRegressor(kernel=kernel, alpha=300, n_restarts_optimizer=20, random_state=42)
    regr.fit(ECFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['GP_ECFP']['R2'].append(R2_test)
    results['GP_ECFP']['MSE'].append(MSE_test)
    results['GP_ECFP']['RMSE'].append(RMSE_test)
    
    print(f"ECFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # BigFP
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
    regr = GaussianProcessRegressor(kernel=kernel, alpha=300, n_restarts_optimizer=20, random_state=42)
    regr.fit(BigFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['GP_BigFP']['R2'].append(R2_test)
    results['GP_BigFP']['MSE'].append(MSE_test)
    results['GP_BigFP']['RMSE'].append(RMSE_test)
    
    print(f"BigFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # Random Forest with different fingerprints
    print("\n--- Random Forest Models ---")
    
    # Morgan FP
    regr = RandomForestRegressor(n_estimators=1200, 
                                max_depth=50, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=100,
                                random_state=42)
    regr.fit(psmiles_MorganFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(psmiles_MorganFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['RF_Morgan']['R2'].append(R2_test)
    results['RF_Morgan']['MSE'].append(MSE_test)
    results['RF_Morgan']['RMSE'].append(RMSE_test)
    
    print(f"Morgan FP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # ECFP
    regr = RandomForestRegressor(n_estimators=2000, 
                                max_depth=20, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=100,
                                random_state=42)
    regr.fit(ECFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
    RMSE_test = np.sqrt(MSE_test)
    
    results['RF_ECFP']['R2'].append(R2_test)
    results['RF_ECFP']['MSE'].append(MSE_test)
    results['RF_ECFP']['RMSE'].append(RMSE_test)
    
    print(f"ECFP: Test R2={R2_test:.4f}, MSE={MSE_test:.4f}, RMSE={RMSE_test:.4f}")
    
    # BigFP
    regr = RandomForestRegressor(n_estimators=2000, 
                                max_depth=20, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                max_features=1.0,
                                max_leaf_nodes=100,
                                random_state=42)
    regr.fit(BigFP_Matrix[train_ix], tg_exp[train_ix])
    tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])
    
    R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
    MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
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


""" train_ix = np.concatenate((np.concatenate((np.concatenate((split1, split2)), split3)), split4))
test_ix = split0



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
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("Morgan FP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("Morgan FP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)



regr = SVR(kernel='rbf', C=1500, epsilon=0.1)
regr.fit(ECFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(ECFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print("ECFP: Train R2 score:",R2_train, 
  "Test R2 score:",R2_test)
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("ECFP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("ECFP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)





regr = SVR(kernel='rbf', C=1000, epsilon=0.1)
regr.fit(BigFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(BigFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
  "Test R2 score:",R2_test)
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("BigFP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("BigFP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)





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
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("Morgan FP: Train MSE:",MSE_train, 
  "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("Morgan FP: Train RMSE:",RMSE_train, 
  "Test RMSE:",RMSE_test)





kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
regr = GaussianProcessRegressor(kernel=kernel, alpha= 300, n_restarts_optimizer=20, random_state=42)
regr.fit(ECFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(ECFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(ECFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "ECFP: Train R2 score:",R2_train, 
    "Test R2 score:",R2_test)
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("ECFP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("ECFP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)



kernel =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
regr = GaussianProcessRegressor(kernel=kernel, alpha= 300, n_restarts_optimizer=20, random_state=42)
regr.fit(BigFP_Matrix[train_ix],tg_exp[train_ix])
tg_exp_train_predict = regr.predict(BigFP_Matrix[train_ix])
tg_exp_test_predict = regr.predict(BigFP_Matrix[test_ix])

R2_train = r2_score(tg_exp[train_ix], tg_exp_train_predict)
R2_test = r2_score(tg_exp[test_ix], tg_exp_test_predict)
print( "BigFP: Train R2 score:",R2_train, 
    "Test R2 score:",R2_test)
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("BigFP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("BigFP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)




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
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("Morgan FP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("Morgan FP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)




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
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("ECFP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("ECFP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test)




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
MSE_train = mean_squared_error(tg_exp[train_ix], tg_exp_train_predict)
MSE_test = mean_squared_error(tg_exp[test_ix], tg_exp_test_predict)
print("BigFP: Train MSE:",MSE_train, 
    "Test MSE:",MSE_test)
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print("BigFP: Train RMSE:",RMSE_train, 
    "Test RMSE:",RMSE_test) """




