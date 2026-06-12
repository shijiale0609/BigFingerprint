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

psmiles_MorganFP_Matrix = np.load("./datasets/vipea/psmiles_mfp_arr.npy")
print(psmiles_MorganFP_Matrix.shape)

ECFP_Matrix = np.load("./datasets/vipea/EVIP_pecfp_arr.npy")
print(ECFP_Matrix.shape)

BigFP_Matrix = np.load("./datasets/vipea/EVIP_bigfp.npy")
print(BigFP_Matrix.shape)


from sklearn.model_selection import KFold

# Sample list
data = list(range(18414))  # Example list

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

EV = df['EA vs SHE (eV)'][0:18414]
IP = df['IP vs SHE (eV)'][0:18414]
#ev_values = df['EA vs SHE (eV)'][0:18414]
#rho_cal = list(df1[columns[7]])


train_ix = np.concatenate((np.concatenate((np.concatenate((split1, split2)), split3)), split4))
test_ix = split0

EV = np.array(EV)
EV_train = EV[train_ix]
EV_test = EV[train_ix]
IP = np.array(IP)
IP_train = IP[train_ix]
IP_test = IP[test_ix]


def random_forest_regression_hyperparameter_optimization(Input_X = None, 
                                                 Output_Y = None, 
                                                 split1_x = None, 
                                                 split2_x = None,
                                                 split3_x = None,
                                                 split4_x = None,
                                                 n_estimators_list = np.array([2000]),
                                                 max_depth_list = np.array([20]),
                                                 min_samples_split_list = np.array([2,3]),
                                                 min_samples_leaf_list = np.array([1,2,3]),
                                                 max_features_list = np.array([1.0]),
                                                 max_leaf_nodes_split = 10*np.array([7,8,9,10]),
                                                 results_record_dict = None,
                                                 outputfile_name = "ev_rf_results_record_dict.pkl",
                                                ):

    r2_score_mean_list = []
    r2_score_std_list = []
    
    file_path = Path(outputfile_name)

    if file_path.is_file():
        print(f"File '{file_path}' exists.")
        with open(file_path, 'rb') as f:
            results_record_dict = pickle.load(f)
    else:
        print(f"File '{file_path}' does not exist.")
        results_record_dict = None
    #Matrix = np.copy(Matrix_FP)
    if results_record_dict == None:
        results_record_dict = {}
        max_mean = 0.0
        
    else:
        max_mean = max([list(results_record_dict.values())[x][0] for x in range(0, len(list(results_record_dict.values())))])
        
    for n_estimators_i in n_estimators_list:
        for max_depth_i in max_depth_list:
            for min_samples_split_i in min_samples_split_list:
                for min_samples_leaf_i in min_samples_leaf_list:
                    for max_features_i in max_features_list:
                        for max_leaf_nodes_i in max_leaf_nodes_split:
                            
                            result_key_list = [n_estimators_i, 
                                   max_depth_i, 
                                   min_samples_split_i,
                                   min_samples_leaf_i,
                                   max_features_i, 
                                   max_leaf_nodes_i,
                                  ]
                            result_key = str(result_key_list)
                            #如果已经算过了，就直接跳过
                            if result_key in results_record_dict:
                                print("This set has been calculated")
                                print(result_key, results_record_dict[result_key])
                                continue
            
                            train_ix_1 = np.concatenate((np.concatenate((split2_x, split3_x)), split4_x))
                            valid_ix_1 = split1_x
        
                            regr1 = RandomForestRegressor(n_estimators = n_estimators_i, 
                                          max_depth=max_depth_i, 
                                          min_samples_split=min_samples_split_i, 
                                          min_samples_leaf=min_samples_leaf_i,
                                          max_features = max_features_i,                                                                                    
                                          max_leaf_nodes=max_leaf_nodes_i, 
                                          random_state=0)

        
                            regr1.fit(Input_X[train_ix_1],Output_Y[train_ix_1])
        
                            Output_Y_valid_predict1 = regr1.predict(Input_X[valid_ix_1])
        
                            R2_score1 = r2_score(Output_Y[valid_ix_1], Output_Y_valid_predict1)
        

                            train_ix_2 = np.concatenate((np.concatenate((split1_x, split3_x)), split4_x))
                            valid_ix_2 = split2_x

                            regr2 = RandomForestRegressor(n_estimators = n_estimators_i, 
                                          max_depth=max_depth_i, 
                                          min_samples_split=min_samples_split_i, 
                                          min_samples_leaf=min_samples_leaf_i,
                                          max_features = max_features_i,                                                                                    
                                          max_leaf_nodes=max_leaf_nodes_i, 
                                          random_state=0)

                            regr2.fit(Input_X[train_ix_2],Output_Y[train_ix_2])
        
                            Output_Y_valid_predict2 = regr2.predict(Input_X[valid_ix_2])
        
                            R2_score2 = r2_score(Output_Y[valid_ix_2], Output_Y_valid_predict2)
        

                            train_ix_3 = np.concatenate((np.concatenate((split1_x, split2_x)), split4_x))
                            valid_ix_3 = split3_x
        
               
                            regr3 = RandomForestRegressor(n_estimators = n_estimators_i, 
                                          max_depth=max_depth_i, 
                                          min_samples_split=min_samples_split_i, 
                                          min_samples_leaf=min_samples_leaf_i,
                                          max_features = max_features_i,                                                                                    
                                          max_leaf_nodes=max_leaf_nodes_i, 
                                          random_state=0)
        
                            regr3.fit(Input_X[train_ix_3],Output_Y[train_ix_3])
        
                            Output_Y_valid_predict3 = regr3.predict(Input_X[valid_ix_3])
        
                            R2_score3 = r2_score(Output_Y[valid_ix_3], Output_Y_valid_predict3)
        
                            train_ix_4 = np.concatenate((np.concatenate((split1_x, split2_x)), split3_x))
                            valid_ix_4 = split4_x
        
            
                            regr4 = RandomForestRegressor(n_estimators = n_estimators_i, 
                                          max_depth = max_depth_i, 
                                          min_samples_split = min_samples_split_i, 
                                          min_samples_leaf = min_samples_leaf_i,
                                          max_features = max_features_i,                                                                                    
                                          max_leaf_nodes = max_leaf_nodes_i, 
                                          random_state=0)
                            
                            regr4.fit(Input_X[train_ix_4],Output_Y[train_ix_4])
        
                            Output_Y_valid_predict4 = regr4.predict(Input_X[valid_ix_4])
                            
                            R2_score4 = r2_score(Output_Y[valid_ix_4], Output_Y_valid_predict4)

                            r2_score_list = [R2_score1,R2_score2,R2_score3,R2_score4]
                            mean = np.mean(r2_score_list)
                            std = np.std(r2_score_list)

                            #记入计算结果
                            result_value = [mean, std, [R2_score1,R2_score2,R2_score3,R2_score4]]

                            results_record_dict[result_key] = result_value

                            
                            with open(outputfile_name, 'wb') as f:
                                pickle.dump(results_record_dict, f)
        
                            print("##", n_estimators_i, 
                                  "##", max_depth_i, 
                                  "##", min_samples_split_i,
                                  "##", min_samples_leaf_i,
                                  "##", max_features_i, 
                                  "##", max_leaf_nodes_i,
                                  "##", mean, 
                                  "##", std)
        
                            if mean >= max_mean:
                                max_mean = mean
                                print("Update max mean")


def support_vector_regression_hyperparameter_optimization(Input_X = None, 
                                                 Output_Y = None, 
                                                 split1_x = None, 
                                                 split2_x = None,
                                                 split3_x = None,
                                                 split4_x = None,
                                                 C_list = None,
                                                 epsilon_list = None,
                                                 results_record_dict = None,
                                                 outputfile_name = "ev_svr_results_record_dict.pkl",
                                                ):

    r2_score_mean_list = []
    r2_score_std_list = []
    
    file_path = Path(outputfile_name)

    if file_path.is_file():
        print(f"File '{file_path}' exists.")
        with open(file_path, 'rb') as f:
            results_record_dict = pickle.load(f)
    else:
        print(f"File '{file_path}' does not exist.")
        results_record_dict = None
    #Matrix = np.copy(Matrix_FP)
    if results_record_dict == None:
        results_record_dict = {}
        max_mean = 0.0
        
    else:
        max_mean = max([list(results_record_dict.values())[x][0] for x in range(0, len(list(results_record_dict.values())))])
        
    for C_i in C_list:
        for epsilon_i in epsilon_list:
                            
                            result_key_list = [C_i, 
                                   epsilon_i, 
                                  ]
                            result_key = str(result_key_list)
                            #如果已经算过了，就直接跳过
                            if result_key in results_record_dict:
                                print("This set has been calculated")
                                print(result_key, results_record_dict[result_key])
                                continue
            
                            train_ix_1 = np.concatenate((np.concatenate((split2_x, split3_x)), split4_x))
                            valid_ix_1 = split1_x
        
                            regr1 = SVR(kernel='rbf', C = C_i, epsilon = epsilon_i)

                            regr1.fit(Input_X[train_ix_1],Output_Y[train_ix_1])
        
                            Output_Y_valid_predict1 = regr1.predict(Input_X[valid_ix_1])
        
                            R2_score1 = r2_score(Output_Y[valid_ix_1], Output_Y_valid_predict1)
        

                            train_ix_2 = np.concatenate((np.concatenate((split1_x, split3_x)), split4_x))
                            valid_ix_2 = split2_x

                            regr2 = SVR(kernel='rbf', C = C_i, epsilon = epsilon_i)


                            regr2.fit(Input_X[train_ix_2],Output_Y[train_ix_2])
        
                            Output_Y_valid_predict2 = regr2.predict(Input_X[valid_ix_2])
        
                            R2_score2 = r2_score(Output_Y[valid_ix_2], Output_Y_valid_predict2)
        

                            train_ix_3 = np.concatenate((np.concatenate((split1_x, split2_x)), split4_x))
                            valid_ix_3 = split3_x
        
               
                            regr3 =  SVR(kernel='rbf', C = C_i, epsilon = epsilon_i)
        
                            regr3.fit(Input_X[train_ix_3],Output_Y[train_ix_3])
        
                            Output_Y_valid_predict3 = regr3.predict(Input_X[valid_ix_3])
        
                            R2_score3 = r2_score(Output_Y[valid_ix_3], Output_Y_valid_predict3)
        
                            train_ix_4 = np.concatenate((np.concatenate((split1_x, split2_x)), split3_x))
                            valid_ix_4 = split4_x
        
            
                            regr4 = SVR(kernel='rbf', C = C_i, epsilon = epsilon_i)
                            
                            regr4.fit(Input_X[train_ix_4],Output_Y[train_ix_4])
        
                            Output_Y_valid_predict4 = regr4.predict(Input_X[valid_ix_4])
                            
                            R2_score4 = r2_score(Output_Y[valid_ix_4], Output_Y_valid_predict4)

                            r2_score_list = [R2_score1,R2_score2,R2_score3,R2_score4]
                            mean = np.mean(r2_score_list)
                            std = np.std(r2_score_list)

                            #记入计算结果
                            result_value = [mean, std, [R2_score1,R2_score2,R2_score3,R2_score4]]

                            results_record_dict[result_key] = result_value

                            
                            with open(outputfile_name, 'wb') as f:
                                pickle.dump(results_record_dict, f)
        
                            print("##", C_i, 
                                  "##", epsilon_i, 
                                  "##", mean, 
                                  "##", std)
        
                            if mean >= max_mean:
                                max_mean = mean
                                print("Update max mean")


def gaussian_process_regression_hyperparameter_optimization(Input_X = None, 
                                                 Output_Y = None, 
                                                 split1_x = None, 
                                                 split2_x = None,
                                                 split3_x = None,
                                                 split4_x = None,
                                                 alpha_list = None,
                                                 n_restarts_optimizer_list = None,
                                                 results_record_dict = None,
                                                 outputfile_name = "ev_gp_results_record_dict.pkl",
                                                ):

    r2_score_mean_list = []
    r2_score_std_list = []
    
    file_path = Path(outputfile_name)

    if file_path.is_file():
        print(f"File '{file_path}' exists.")
        with open(file_path, 'rb') as f:
            results_record_dict = pickle.load(f)
    else:
        print(f"File '{file_path}' does not exist.")
        results_record_dict = None
    #Matrix = np.copy(Matrix_FP)
    if results_record_dict == None:
        results_record_dict = {}
        max_mean = 0.0
        
    else:
        max_mean = max([list(results_record_dict.values())[x][0] for x in range(0, len(list(results_record_dict.values())))])
        
    
    for alpha_i in alpha_list:
        for n_restarts_optimizer_i in n_restarts_optimizer_list:
                            
                            result_key_list = [alpha_i, n_restarts_optimizer_i]

                            result_key = str(result_key_list)

                            #如果已经算过了，就直接跳过
                            if result_key in results_record_dict:
                                print("This set has been calculated")
                                print(result_key, results_record_dict[result_key])
                                continue
            
                            train_ix_1 = np.concatenate((np.concatenate((split2_x, split3_x)), split4_x))
                            valid_ix_1 = split1_x

                            #kernel1 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6)) + WhiteKernel(noise_level=400)
                            kernel1 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
                            regr1 = GaussianProcessRegressor(
                                    kernel=kernel1, alpha= alpha_i, n_restarts_optimizer=n_restarts_optimizer_i, random_state=42
                            )

                            regr1.fit(Input_X[train_ix_1],Output_Y[train_ix_1])
        
                            Output_Y_valid_predict1 = regr1.predict(Input_X[valid_ix_1])
        
                            R2_score1 = r2_score(Output_Y[valid_ix_1], Output_Y_valid_predict1)
        

                            train_ix_2 = np.concatenate((np.concatenate((split1_x, split3_x)), split4_x))
                            valid_ix_2 = split2_x

                            #kernel2 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6)) + WhiteKernel(noise_level=400)
                            kernel2 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
                            regr2 = GaussianProcessRegressor(
                                    kernel=kernel2, alpha= alpha_i, n_restarts_optimizer=n_restarts_optimizer_i, random_state=42
                            )


                            regr2.fit(Input_X[train_ix_2],Output_Y[train_ix_2])
        
                            Output_Y_valid_predict2 = regr2.predict(Input_X[valid_ix_2])
        
                            R2_score2 = r2_score(Output_Y[valid_ix_2], Output_Y_valid_predict2)
        

                            train_ix_3 = np.concatenate((np.concatenate((split1_x, split2_x)), split4_x))
                            valid_ix_3 = split3_x
        
               
                            #kernel3 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6)) + WhiteKernel(noise_level=400)
                            kernel3 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
                            regr3 = GaussianProcessRegressor(
                                    kernel=kernel3, alpha= alpha_i, n_restarts_optimizer=n_restarts_optimizer_i, random_state=42
                            )
        
                            regr3.fit(Input_X[train_ix_3],Output_Y[train_ix_3])
        
                            Output_Y_valid_predict3 = regr3.predict(Input_X[valid_ix_3])
        
                            R2_score3 = r2_score(Output_Y[valid_ix_3], Output_Y_valid_predict3)
        
                            train_ix_4 = np.concatenate((np.concatenate((split1_x, split2_x)), split3_x))
                            valid_ix_4 = split4_x
        
            
                            #kernel4 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6)) + WhiteKernel(noise_level=400)
                            kernel4 =  ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e6))
                            regr4 = GaussianProcessRegressor(
                                    kernel=kernel4, alpha= alpha_i, n_restarts_optimizer=n_restarts_optimizer_i, random_state=42
                            )
                            
                            regr4.fit(Input_X[train_ix_4],Output_Y[train_ix_4])
        
                            Output_Y_valid_predict4 = regr4.predict(Input_X[valid_ix_4])
                            
                            R2_score4 = r2_score(Output_Y[valid_ix_4], Output_Y_valid_predict4)

                            r2_score_list = [R2_score1,R2_score2,R2_score3,R2_score4]
                            mean = np.mean(r2_score_list)
                            std = np.std(r2_score_list)

                            #记入计算结果
                            result_value = [mean, std, [R2_score1,R2_score2,R2_score3,R2_score4]]

                            results_record_dict[result_key] = result_value

                            
                            with open(outputfile_name, 'wb') as f:
                                pickle.dump(results_record_dict, f)
        
                            print("##", alpha_i,
                                  "##", n_restarts_optimizer_i,
                                  "##", mean, 
                                  "##", std,
                                  "##", [R2_score1,R2_score2,R2_score3,R2_score4])
        
                            if mean >= max_mean:
                                max_mean = mean
                                print("Update max mean")



       # Plot configuration


support_vector_regression_hyperparameter_optimization(Input_X = psmiles_MorganFP_Matrix, 
                                                 Output_Y = EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 C_list = [ 1.0],
                                                 epsilon_list = [0.001, 0.01,0.02 ],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_svr_results_record_dict_psmiles_MorganFP.pkl",
                                                )

""" support_vector_regression_hyperparameter_optimization(Input_X = ECFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 C_list = [600,800, 1000,1200,1400,1500,1600,1700,1800],
                                                 epsilon_list = [0.1,0.2,0.3],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_svr_results_record_dict_ECFP.pkl",
                                                ) """


""" support_vector_regression_hyperparameter_optimization(Input_X = BigFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 C_list = [600,800, 1000,1200,1400,1500,1600,1700,1800],
                                                 epsilon_list = [0.1,0.2,0.3],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_svr_results_record_dict_BigFP.pkl",
                                                )  """


""" gaussian_process_regression_hyperparameter_optimization(Input_X = psmiles_MorganFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 alpha_list = [10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000],
                                                 n_restarts_optimizer_list = [20],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_gp_results_record_dict_psmiles_MorganFP.pkl",
                                                ) """

""" gaussian_process_regression_hyperparameter_optimization(Input_X = ECFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 alpha_list = [100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,5000,6000,7000,8000,9000,10000,11000,12000],
                                                 n_restarts_optimizer_list = [20],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_gp_results_record_dict_ECFP.pkl",
                                                )  """

""" gaussian_process_regression_hyperparameter_optimization(Input_X = BigFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 alpha_list =  [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000],
                                                 n_restarts_optimizer_list = [20],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_gp_results_record_dict_BigFP.pkl",
                                                )  """

""" random_forest_regression_hyperparameter_optimization(Input_X = psmiles_MorganFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 n_estimators_list = [1400,1600,1800,2000,2200],
                                                 max_depth_list = [50],
                                                 min_samples_split_list = [2],
                                                 min_samples_leaf_list = [1],
                                                 max_features_list = [1.0],
                                                 max_leaf_nodes_split = [60,],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_rf_results_record_dict_psmiles_MorganFP.pkl",
                                                )    """


""" random_forest_regression_hyperparameter_optimization(Input_X = ECFP_Matrix, 
                                                 Output_Y =  EV,  
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 n_estimators_list = [1800],
                                                 max_depth_list = [60,70,80,90],
                                                 min_samples_split_list = [2],
                                                 min_samples_leaf_list = [1],
                                                 max_features_list = [1.0],
                                                 max_leaf_nodes_split = [70,80,90,100],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_rf_results_record_dict_ECFP.pkl",
                                                )  """  

""" random_forest_regression_hyperparameter_optimization(Input_X = BigFP_Matrix, 
                                                 Output_Y =  EV, 
                                                 split1_x = split1, 
                                                 split2_x = split2,
                                                 split3_x = split3,
                                                 split4_x = split4,
                                                 n_estimators_list = [1600,1800,2000],
                                                 max_depth_list = [60,70,80],
                                                 min_samples_split_list = [2],
                                                 min_samples_leaf_list = [1],
                                                 max_features_list = [1.0],
                                                 max_leaf_nodes_split = [80,90,100],
                                                 results_record_dict = None,
                                                 outputfile_name = "EV_rf_results_record_dict_BigFP.pkl",
                                                )   """


