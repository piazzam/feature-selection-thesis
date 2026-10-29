import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import cooperative_game_theory as cgt
import random
from sklearn.utils import check_random_state
import sklearn.linear_model as linmod
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import os
from scipy.stats import entropy
from sklearn.feature_selection import SequentialFeatureSelector
import scipy.io
import time
import gc

 


### ESATTO ###

# funzione per tuning best parametri random forest
def tune_random_forest(X_train_train, Y_train_train, X_val, Y_val, seed):
    N_TREES_GRID = [50, 100, 200, 300, 400, 500]
    #MAX_DEPTH_GRID = list(range(1, 51))
    MAX_DEPTH_GRID = list(range(0, 52,2))
    MAX_DEPTH_GRID[0] = 1
    best_score = -np.inf
    # best_params = None
    best_n_estimators = None
    best_max_depth = None
    for n_trees in N_TREES_GRID:
        for max_depth in MAX_DEPTH_GRID:
            model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                random_state=seed
                #n_jobs=N_CORE
            )
            model.fit(X_train_train, Y_train_train)
            y_val_pred = model.predict(X_val)
            score = accuracy_score(Y_val, y_val_pred)
            if score > best_score:
                best_score = score
                # best_params = {
                #     "n_estimators": n_trees,
                #     "max_depth": max_depth
                # }
                best_n_estimators = n_trees
                best_max_depth = max_depth

    return best_n_estimators, best_max_depth, best_score

# restituisce i parametri rf migliore per il folder='ds_name/approx_min_number/0', se ancora non esistono li salva (dovrebbe salvarli al calcolo di v_coalitions)
def get_rf_best_parametri(folder, seed_value):

    param_file = folder + "rf_best_params.npy"

    if os.path.exists(param_file):  # vengono calcolati solo quando si calcola v_coalitions, e poi vengono riusati per quando si riallena il modello
        best_n_estimators, best_max_depth = np.load(param_file)
        return int(best_n_estimators), int(best_max_depth)

    X_train_train = np.load(folder+"X_train_train.npy")
    X_val = np.load(folder+"X_val.npy")
    Y_train_train = np.load(folder+"Y_train_train.npy")
    Y_val = np.load(folder+"Y_val.npy")

    best_n_estimators, best_max_depth, best_score = tune_random_forest(X_train_train, Y_train_train, X_val, Y_val, seed=seed_value)
    np.save(param_file, np.array([best_n_estimators, best_max_depth]))

    return best_n_estimators, best_max_depth


# funzione che restituisce il modello 
def return_model(folder, string, seed_value=42):
    if string == "lr":
        return linmod.LogisticRegression(max_iter=1000, penalty=None)
    elif string == "rf":
        return RandomForestClassifier(random_state=seed_value)
    elif string == 'rf_tuning':
        # x_train_train = np.load(folder+"X_train_train.npy")
        # x_val = np.load(folder+"X_val.npy")
        # y_train_train = np.load(folder+"Y_train_train.npy")
        # y_val = np.load(folder+"Y_val.npy")
        # best_n_estimators, best_max_depth, rf_best_val_score = tune_random_forest(x_train_train, y_train_train, x_val, y_val, seed=seed_value)
        best_n_estimators, best_max_depth = get_rf_best_parametri(folder, seed_value)
        return RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=seed_value)



# seed fissato
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    rng = check_random_state(seed_value)


# genero vettore v di accuracy e f1
def generate_v_traintest_already_divided(model_string, matrix, X_train, X_test, Y_train, Y_test,
                                         store_values=True, path='folder/'):
    folder = path.rstrip('/').rsplit('/', 1)[0] + '/'   # se path=ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v/' allora folder=ds_name+'/approx_min_number_coal/'+str(i)+'/'
    v_tot = {0: [0, 0]}
    ncoal = matrix.shape[0]
    nvar = X_train.shape[1]

    acc_times = []
    f1_times = []

    for i in range(1, ncoal):
        index_selected_features = [j for j in range(nvar) if matrix[i, j] == 1]
        x_subset_train = X_train[:, index_selected_features]
        x_subset_test = X_test[:, index_selected_features]
        acc_scores = []
        f1_scores = []
        weights_list = []

        start_model = time.time()
        ml_model = return_model(folder, model_string)
        ml_model.fit(x_subset_train, Y_train)
        y_pred = ml_model.predict(x_subset_test)
        end_model = time.time()
        model_time = end_model - start_model

        start_acc = time.time()
        acc_value = accuracy_score(Y_test, y_pred)
        end_acc = time.time()
        acc_scores.append(acc_value)
        acc_times.append(model_time + (end_acc - start_acc))

        start_f1 = time.time()
        f1_value = f1_score(Y_test, y_pred, average='weighted')
        end_f1 = time.time()
        f1_scores.append(f1_value)
        f1_times.append(model_time + (end_f1 - start_f1))

        if model_string == "lr":
            weights_list.append(ml_model.coef_)
        elif model_string == "rf" or model_string == "rf_tuning":
            weights_list.append(ml_model.feature_importances_)
        avg_acc = np.mean(acc_scores)
        avg_f1 = np.mean(f1_scores)
        v_tot[i] = [avg_acc, avg_f1]

        # clean della memoria
        del ml_model, y_pred, x_subset_train, x_subset_test
        del index_selected_features
        gc.collect()

    acc = []
    f1 = []
    for coalition, value in v_tot.items():
        acc.append(value[0])
        f1.append(value[1])
        print(f"Coalition {coalition}: Accuracy = {value[0]:.4f}, F1 = {value[1]:.4f}")
    flat_weights = np.array(weights_list, dtype=object)
    flat_weights = np.vstack(flat_weights)
    df = pd.DataFrame(flat_weights)
    if store_values:
        np.save(path + model_string + "/coalitions_acc.npy", np.array(acc))
        np.save(path + model_string + "/coalitions_f1.npy", np.array(f1))
        df.to_csv(path + model_string + "/weights.csv", sep='\t', decimal=',')

    total_time_acc = sum(acc_times)
    total_time_f1 = sum(f1_times)
    np.save(path + model_string + "/time_v_acc.npy", np.array(total_time_acc))
    np.save(path + model_string + "/time_v_f1.npy", np.array(total_time_f1))

    # clean della memoria
    del acc, f1, weights_list, acc_times, f1_times, flat_weights, df
    gc.collect()

    return v_tot

# genero e salvo valore baseline mostfreq (path='iris/exact/0/v_most_freq/lr/')
def generate_baseline_most_frequent(x, y, path):
    tot_observations = x.shape[0]
    classes, obs_per_class = np.unique(y, return_counts=True)
    ratios = obs_per_class / tot_observations
    value_baseline = np.max(ratios)
    np.save(path+"value_baseline_most_freq.npy", value_baseline)

    return value_baseline

# genero e salvo valore baseline stratified (path='iris/exact/0/v_stratified/lr/')
def generate_baseline_stratified(x, y, path):
    n_simulations = 100
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(x)
    acc = np.zeros(n_simulations)
    for i in range(n_simulations):
        predictions = []
        for j in range(len(x)):
            predictions.append(np.random.choice(unique_classes, p=probabilities))
        acc[i] = metrics.accuracy_score(y, predictions)
    baseline_strat = np.mean(acc)
    np.save(path+"value_baseline_stratified.npy", baseline_strat)

    return baseline_strat


# genero e salvo shapley
def generate_shapley(v, nvar, store_values=True, file_path="shapley.npy"):
    shapley_value = cgt.shapley(v, nvar)
    if store_values:
        np.save(file_path, shapley_value)
    return shapley_value

# genero e salvo banhaf
def generate_banzhaf(v, nvar, store_values=True, file_path_banz="banzhaf.npy", file_path_banz_mult="banzhaf_mult.npy", file_path_banz_add="banzhaf_add.npy"):
    banzhaf_value, banzhaf_mult, banzhaf_add = cgt.banzhaf(v, nvar)
    if store_values:
        np.save(file_path_banz, banzhaf_value)
        np.save(file_path_banz_mult, banzhaf_mult)
        np.save(file_path_banz_add, banzhaf_add)
    return banzhaf_value, banzhaf_mult, banzhaf_add

# genero e salvo core/leastcore
def generate_core(v, matrix, store_values=True, path='folder/', file_path_x="core.npy", file_path_e="evalue.npy", file_path_ublc="ublc.npy", 
                  file_path_lblc="lblc.npy", file_path_ubc="ubc.npy", file_path_lbc="lbc.npy", file_path_eg="egalitarian.npy", metric='acc'):
    e_value, x_values, ublc, lblc, ubc, lbc, egal_values = cgt.compute_core_with_bounds(matrix, v, path=path+'/'+metric)
    if store_values:
        np.save(file_path_x, x_values)
        np.save(file_path_e, e_value)
        np.save(file_path_ublc, ublc)
        np.save(file_path_lblc, lblc)
        np.save(file_path_ubc, ubc)
        np.save(file_path_lbc, lbc)
        np.save(file_path_eg, egal_values)
    return x_values, e_value

# genero e salvo solution L1 e Linf
def generate_problem_norm_l1_linf(matrix, v, store_values=True, file_path_yl1="y_l1.npy", file_path_lb_l1="lb_l1.npy", file_path_ub_l1="ub_l1.npy", file_path_solvector_l1='solution_vector_l1.npy',
                                  file_path_ylinf="y_linf.npy", file_path_lb_linf="lb_linf.npy", file_path_ub_linf="ub_linf.npy", file_path_solvector_linf='solution_vector_linf.npy'):
    y_opt_l1, lb_l1, ub_l1, vect_l1 = cgt.compute_problem_l1(matrix, v)
    y_opt_linf, lb_linf, ub_linf, vect_linf = cgt.compute_problem_linf(matrix, v)
    if store_values:
        np.save(file_path_yl1, y_opt_l1)
        np.save(file_path_lb_l1, lb_l1)
        np.save(file_path_ub_l1, ub_l1)
        np.save(file_path_solvector_l1, vect_l1)
        np.save(file_path_ylinf, y_opt_linf)
        np.save(file_path_lb_linf, lb_linf)
        np.save(file_path_ub_linf, ub_linf)
        np.save(file_path_solvector_linf, vect_linf)
    return y_opt_l1, lb_l1, ub_l1, y_opt_linf, lb_linf, ub_linf

# salvo i valori correttamente (path='iris/exact/0/v/lr/)
def execute(nvar, matrix, path, v_acc, v_f1): 
    set_seed()

    generate_shapley(v_acc, nvar, store_values=True, file_path=path+'shap_acc.npy')
    generate_shapley(v_f1, nvar, store_values=True, file_path=path+'shap_f1.npy')

    generate_banzhaf(v_acc, nvar, store_values=True, file_path_banz=path+'banz_acc.npy', file_path_banz_mult=path+'banz_mult_acc.npy', file_path_banz_add=path+'banz_add_acc.npy')
    generate_banzhaf(v_f1, nvar, store_values=True, file_path_banz=path+'banz_f1.npy', file_path_banz_mult=path+'banz_mult_f1.npy', file_path_banz_add=path+'banz_add_f1.npy')

    generate_core(v_acc, matrix, store_values=True, path=path, file_path_x=path+'xicore_acc.npy', file_path_e=path+'evalue_acc.npy', 
                 file_path_ublc=path+'ublc_acc.npy', file_path_lblc=path+'lblc_acc.npy', file_path_ubc=path+'ubc_acc.npy', file_path_lbc=path+'lbc_acc.npy', 
                 file_path_eg=path+'egalitarian_acc.npy', metric='acc')
    generate_core(v_f1, matrix, store_values=True, path=path, file_path_x=path+'xicore_f1.npy', file_path_e=path+'evalue_f1.npy', 
                 file_path_ublc=path+'ublc_f1.npy', file_path_lblc=path+'lblc_f1.npy', file_path_ubc=path+'ubc_f1.npy', file_path_lbc=path+'lbc_f1.npy', 
                 file_path_eg=path+'egalitarian_f1.npy', metric='f1')
    
    generate_problem_norm_l1_linf(matrix, v_acc, store_values=True, file_path_yl1=path+'y_l1_acc.npy', file_path_lb_l1=path+'lb_l1_acc.npy',file_path_ub_l1=path+'ub_l1_acc.npy', 
                                  file_path_solvector_l1=path+'solution_vector_l1_acc.npy', file_path_ylinf=path+'y_linf_acc.npy', file_path_lb_linf=path+'lb_linf_acc.npy',
                                  file_path_ub_linf=path+'ub_linf_acc.npy', file_path_solvector_linf=path+'solution_vector_linf_acc.npy')
    generate_problem_norm_l1_linf(matrix, v_f1, store_values=True, file_path_yl1=path+'y_l1_f1.npy', file_path_lb_l1=path+'lb_l1_f1.npy',file_path_ub_l1=path+'ub_l1_f1.npy',
                                  file_path_solvector_l1=path+'solution_vector_l1_f1.npy', file_path_ylinf=path+'y_linf_f1.npy', file_path_lb_linf=path+'lb_linf_f1.npy',
                                  file_path_ub_linf=path+'ub_linf_f1.npy', file_path_solvector_linf=path+'solution_vector_linf_f1.npy')
    
# genero e salvo tau value leastcore (save_as='tau_lc' / 'tau_lc_min')
def generate_tauvector(v_acc, v_f1, matrix, path='iris/exact/0/v/lr/', save_as='tau_lc'):
    nvar = matrix.shape[1]
    epsilon = np.load(path+'evalue_acc.npy')    

    tauvect_exact_acc = cgt.compute_tau_leastcore(v_acc, epsilon, matrix)
    np.save(path+save_as+'_exact_acc.npy', tauvect_exact_acc)
    tauvect_exact_f1 = cgt.compute_tau_leastcore(v_f1, epsilon, matrix)
    np.save(path+save_as+'_exact_f1.npy', tauvect_exact_f1)

    if 'v_most_freq' in path:
        value_baseline = np.load(path+'value_baseline_most_freq.npy')
    elif 'v_stratified' in path:
        value_baseline = np.load(path+'value_baseline_stratified.npy')
    else:
        value_baseline = 0
    epsilon = 1 - value_baseline - v_acc[-1] / nvar      
    tauvect_approx_acc = cgt.compute_tau_leastcore(v_acc, epsilon, matrix)
    np.save(path+save_as+'_approx_acc.npy', tauvect_approx_acc)
    epsilon = 1 - value_baseline - v_f1[-1] / nvar
    tauvect_approx_f1 = cgt.compute_tau_leastcore(v_f1, epsilon, matrix)
    np.save(path+save_as+'_approx_f1.npy', tauvect_approx_f1)

# genero csv completo (path='iris/exact/0/v/lr/)
def generate_csv(path):
    matrix = np.load(path+"matrix.npy") 

    v_acc = np.load(path+"coalitions_acc.npy")
    v_f1 = np.load(path+"coalitions_f1.npy")

    shap_acc = np.load(path+"shap_acc.npy")
    shap_f1 = np.load(path+"shap_f1.npy")

    banz_mult_acc = np.load(path+"banz_mult_acc.npy")
    banz_mult_f1 = np.load(path+"banz_mult_f1.npy")

    banz_add_acc = np.load(path+"banz_add_acc.npy")
    banz_add_f1 = np.load(path+"banz_add_f1.npy")

    e_value_acc = np.load(path+"evalue_acc.npy")
    e_value_f1 = np.load(path+"evalue_f1.npy")

    xi_acc = np.load(path+"xicore_acc.npy")     
    xi_f1 = np.load(path+"xicore_f1.npy")

    lblc_acc = np.load(path+"lblc_acc.npy")
    ublc_acc = np.load(path+"ublc_acc.npy")
    lblc_f1 = np.load(path+"lblc_f1.npy")
    ublc_f1 = np.load(path+"ublc_f1.npy")

    lbc_acc = np.load(path+"lbc_acc.npy")
    ubc_acc = np.load(path+"ubc_acc.npy")
    lbc_f1 = np.load(path+"lbc_f1.npy")
    ubc_f1 = np.load(path+"ubc_f1.npy")

    egalit_acc = np.load(path+"egalitarian_acc.npy")
    egalit_f1 = np.load(path+"egalitarian_f1.npy")

    y_opt_l1_acc = np.load(path+"y_l1_acc.npy")
    lb_l1_acc = np.load(path+"lb_l1_acc.npy")
    ub_l1_acc = np.load(path+"ub_l1_acc.npy")
    sol_vect_l1_acc = np.load(path+"solution_vector_l1_acc.npy")
    y_opt_linf_acc = np.load(path+"y_linf_acc.npy")
    lb_linf_acc = np.load(path+"lb_linf_acc.npy")
    ub_linf_acc = np.load(path+"ub_linf_acc.npy")
    sol_vect_linf_acc = np.load(path+"solution_vector_linf_acc.npy")

    y_opt_l1_f1 = np.load(path+"y_l1_f1.npy")
    lb_l1_f1 = np.load(path+"lb_l1_f1.npy")
    ub_l1_f1 = np.load(path+"ub_l1_f1.npy")
    sol_vect_l1_f1 = np.load(path+"solution_vector_l1_f1.npy")
    y_opt_linf_f1 = np.load(path+"y_linf_f1.npy")
    lb_linf_f1 = np.load(path+"lb_linf_f1.npy")
    ub_linf_f1 = np.load(path+"ub_linf_f1.npy")
    sol_vect_linf_f1 = np.load(path+"solution_vector_linf_f1.npy")

    nucleolus_acc = np.loadtxt(path+'nucleolus_acc.txt', delimiter='\t')
    nucleolus_f1 = np.loadtxt(path+'nucleolus_f1.txt', delimiter='\t')

    tau_lc_exact_acc = np.load(path+'tau_lc_exact_acc.npy')
    tau_lc_approx_acc = np.load(path+'tau_lc_approx_acc.npy')
    tau_lc_exact_f1 = np.load(path+'tau_lc_exact_f1.npy')
    tau_lc_approx_f1 = np.load(path+'tau_lc_approx_f1.npy')

    shap_in_core_acc, idxfalse_shap_core_acc = cgt.vector_in_core(matrix, shap_acc, v_acc)
    shap_in_lc_acc, idxfalse_shap_lc_acc, max_excess_acc, shap_max_exc_acc = cgt.shapley_in_leastcore(matrix, shap_acc, e_value_acc, v_acc)

    banz_in_core_acc, idxfalse_banz_core_acc = cgt.vector_in_core(matrix, banz_mult_acc, v_acc)
    banz_in_lc_acc, idxfalse_banz_lc_acc = cgt.vector_in_leastcore(matrix, banz_mult_acc, e_value_acc, v_acc)

    shap_in_core_f1, idxfalse_shap_core_f1 = cgt.vector_in_core(matrix, shap_f1, v_f1)
    shap_in_lc_f1, idxfalse_shap_lc_f1, max_excess_f1, shap_max_exc_f1 = cgt.shapley_in_leastcore(matrix, shap_acc, e_value_acc, v_acc)

    banz_in_core_f1, idxfalse_banz_core_f1 = cgt.vector_in_core(matrix, banz_mult_f1, v_f1)
    banz_in_lc_f1, idxfalse_banz_lc_f1 = cgt.vector_in_leastcore(matrix, banz_mult_f1, e_value_f1, v_f1)

    tau_ex_acc_in, idxfalse_tau_ex_acc = cgt.vector_in_leastcore(matrix, tau_lc_exact_acc, e_value_acc, v_acc)
    tau_ap_acc_in, idxfalse_tau_ap_acc = cgt.vector_in_leastcore(matrix, tau_lc_approx_acc, e_value_acc, v_acc)
    tau_ex_f1_in, idxfalse_tau_ex_f1 = cgt.vector_in_leastcore(matrix, tau_lc_exact_f1, e_value_f1, v_f1)
    tau_ap_f1_in, idxfalse_tau_ap_f1 = cgt.vector_in_leastcore(matrix, tau_lc_approx_f1, e_value_f1, v_f1)

    most_diss_coals_acc = cgt.binding_constraints(matrix, v_acc, xi_acc, e_value_acc)
    most_diss_coals_f1 = cgt.binding_constraints(matrix, v_f1, xi_f1, e_value_f1)

    is_monot_acc, res_acc, max_diff_acc, mono_rate_acc = cgt.monotonicity(matrix, v_acc)
    is_monot_f1, res_f1, max_diff_f1, mono_rate_f1 = cgt.monotonicity(matrix, v_f1)

    is_superadd_acc, is_subadd_acc, super_rate_acc, sub_rate_acc = cgt.super_sub_additive(matrix, v_acc)
    is_superadd_f1, is_subadd_f1, super_rate_f1, sub_rate_f1 = cgt.super_sub_additive(matrix, v_f1)

    emptycore_acc = []
    if e_value_acc > 0:
        emptycore_acc = True
    else: 
        emptycore_acc = False

    emptycore_f1 = []
    if e_value_f1 > 0:
        emptycore_f1 = True
    else: 
        emptycore_f1 = False

    n_players = matrix.shape[1]
    n_coalitions = matrix.shape[0]

    metrics_rows = ['shap_acc', 'banz_mult_acc', 'banz_add_acc', 'egal_acc', 'evalue_acc', 'empty_core_acc', 
                    'lblc_acc', 'ublc_acc', 'lbc_acc', 'ubc_acc', 
                    'problem_norm_l1_acc', 'solution_vector_l1_acc', 'lb_norm_l1_acc', 'ub_norm_l1_acc', 
                    'problem_norm_linf_acc', 'solution_vector_linf_acc', 'lb_norm_linf_acc', 'ub_norm_linf_acc',
                    'nucl_acc', 'tau_exact_acc', 'tau_approx_acc', 'tau_ex_acc_in_lc', 'tau_ap_acc_in_lc',
                    'shap_acc_incore', 'shap_acc_inlc', 'shap_inlc_excess_acc', 'banz_acc_incore', 'banz_acc_inlc', 
                    'most_dissatisfied_coals_acc', 'game_monotonic_acc', 'game_superadd_acc', 'game_subadd_acc',
                    'shap_f1', 'banz_mult_f1', 'banz_add_f1', 'egal_f1', 'evalue_f1', 'empty_core_f1', 
                    'lblc_f1', 'ublc_f1', 'lbc_f1', 'ubc_f1', 
                    'problem_norm_l1_f1', 'solution_vector_l1_f1', 'lb_norm_l1_f1', 'ub_norm_l1_f1', 
                    'problem_norm_linf_f1', 'solution_vector_linf_f1', 'lb_norm_linf_f1', 'ub_norm_linf_f1',
                    'nucl_f1', 'tau_exact_f1', 'tau_approx_f1', 'tau_ex_f1_in_lc', 'tau_ap_f1_in_lc',
                    'shap_f1_incore', 'shap_f1_inlc', 'shap_inlc_excess_f1', 'banz_f1_incore', 'banz_f1_inlc', 
                    'most_dissatisfied_coals_f1', 'game_monotonic_f1', 'game_superadd_f1', 'game_subadd_f1']
    n_extra_rows = len(metrics_rows)

    metrics_columns = ['v_acc', 'v_f1']
    n_extra_columns = len(metrics_columns)

    evalue_acc = [e_value_acc.item()] + [None]*(n_players-1) + [None]*n_extra_columns
    evalue_f1 = [e_value_f1.item()] + [None]*(n_players-1) + [None]*n_extra_columns

    empty_core_acc = [emptycore_acc] + [None]*(n_players-1) + [None]*n_extra_columns
    empty_core_f1 = [emptycore_f1] + [None]*(n_players-1) + [None]*n_extra_columns

    probl_norm_1_acc = ["y optimal"] + [y_opt_l1_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    probl_norm_inf_acc =  ["y optimal"] + [y_opt_linf_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    probl_norm_1_f1 = ["y optimal"] + [y_opt_l1_f1] + [None]*(n_players-2) + [None]*n_extra_columns
    probl_norm_inf_f1 = ["y optimal"] + [y_opt_linf_f1] + [None]*(n_players-2) + [None]*n_extra_columns

    tau_exact_acc_in_lc = [tau_ex_acc_in] + [idxfalse_tau_ex_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    tau_apporx_acc_in_lc = [tau_ap_acc_in] + [idxfalse_tau_ap_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    tau_exact_f1_in_lc = [tau_ex_f1_in] + [idxfalse_tau_ex_f1] + [None]*(n_players-2) + [None]*n_extra_columns
    tau_apporx_f1_in_lc = [tau_ap_f1_in] + [idxfalse_tau_ap_f1] + [None]*(n_players-2) + [None]*n_extra_columns

    shap_acc_incore = [shap_in_core_acc] + [idxfalse_shap_core_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    shap_acc_inlc = [shap_in_lc_acc] + [idxfalse_shap_lc_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    banz_acc_incore = [banz_in_core_acc] + [idxfalse_banz_core_acc] + [None]*(n_players-2) + [None]*n_extra_columns
    banz_acc_inlc = [banz_in_lc_acc] + [idxfalse_banz_lc_acc] + [None]*(n_players-2) + [None]*n_extra_columns

    shap_f1_incore = [shap_in_core_f1] + [idxfalse_shap_core_f1] + [None]*(n_players-2) + [None]*n_extra_columns
    shap_f1_inlc = [shap_in_lc_f1] + [idxfalse_shap_lc_f1] + [None]*(n_players-2) + [None]*n_extra_columns
    banz_f1_incore = [banz_in_core_f1] + [idxfalse_banz_core_f1] + [None]*(n_players-2) + [None]*n_extra_columns
    banz_f1_inlc = [banz_in_lc_f1] + [idxfalse_banz_lc_f1] + [None]*(n_players-2) + [None]*n_extra_columns

    most_dissatisfied_coals_acc = [most_diss_coals_acc] + [None]*(n_players-1) + [None]*n_extra_columns
    most_dissatisfied_coals_f1 = [most_diss_coals_f1] + [None]*(n_players-1) + [None]*n_extra_columns

    shap_lc_excess_acc = ["max excess"] + [max_excess_acc] + ["idx_coal_max_excess"] + shap_max_exc_acc + [None]*(n_players + n_extra_columns - 4)
    shap_lc_excess_f1 = ["max excess"] + [max_excess_f1] + ["idx_coal_max_excess"] + [shap_max_exc_f1] + [None]*(n_players + n_extra_columns - 4)

    monot_acc = [is_monot_acc] + [max_diff_acc[:2] + [float(max_diff_acc[2])]] + ['rate_mono'] + [mono_rate_acc] + [None]*(n_players-4) + [None]*n_extra_columns
    monot_f1 = [is_monot_f1] + [max_diff_f1[:2] + [float(max_diff_f1[2])]] + ['rate_mono'] + [mono_rate_f1] + [None]*(n_players-4) + [None]*n_extra_columns

    supadd_acc = [is_superadd_acc] + ['rate_supadd'] + [super_rate_acc] + [None]*(n_players-3) + [None]*n_extra_columns
    subadd_acc = [is_subadd_acc] + ['rate_supadd'] + [sub_rate_acc] + [None]*(n_players-3) + [None]*n_extra_columns

    supadd_f1 = [is_superadd_f1] + ['rate_supadd'] + [super_rate_f1] + [None]*(n_players-3) + [None]*n_extra_columns
    subadd_f1 = [is_subadd_f1] + ['rate_supadd'] + [sub_rate_f1] + [None]*(n_players-3) + [None]*n_extra_columns

    save_full_matrix = True
    if n_coalitions > 3000:
        save_full_matrix = False

    if save_full_matrix:
        matrix_part = [list(matrix[:, i]) for i in range(n_players)]
        matrix_rows = [f'C{i}' for i in range(n_coalitions)]
    else: 
        matrix_part = [ [matrix[-1, i]] for i in range(n_players) ]
        matrix_rows = [f'C{n_coalitions-1} (last)']

    data = {f"P{i+1}": matrix_part[i] + [
            shap_acc[i], banz_mult_acc[i], banz_add_acc[i], egalit_acc[i], evalue_acc[i], empty_core_acc[i],
            lblc_acc[i], ublc_acc[i], lbc_acc[i], ubc_acc[i], 
            probl_norm_1_acc[i], sol_vect_l1_acc[i], lb_l1_acc[i], ub_l1_acc[i],
            probl_norm_inf_acc[i], sol_vect_linf_acc[i], lb_linf_acc[i], ub_linf_acc[i],
            nucleolus_acc[i], tau_lc_exact_acc[i], tau_lc_approx_acc[i], tau_exact_acc_in_lc[i], tau_apporx_acc_in_lc[i],
            shap_acc_incore[i], shap_acc_inlc[i], shap_lc_excess_acc[i], banz_acc_incore[i], banz_acc_inlc[i], 
            most_dissatisfied_coals_acc[i], monot_acc[i], supadd_acc[i], subadd_acc[i],
            shap_f1[i], banz_mult_f1[i], banz_add_f1[i], egalit_f1[i], evalue_f1[i], empty_core_f1[i], 
            lblc_f1[i], ublc_f1[i], lbc_f1[i], ubc_f1[i], 
            probl_norm_1_f1[i], sol_vect_l1_f1[i], lb_l1_f1[i], ub_l1_f1[i],
            probl_norm_inf_f1[i], sol_vect_linf_f1[i], lb_linf_f1[i], ub_linf_f1[i],
            nucleolus_f1[i], tau_lc_exact_f1[i], tau_lc_approx_f1[i], tau_exact_f1_in_lc[i], tau_apporx_f1_in_lc[i],
            shap_f1_incore[i], shap_f1_inlc[i], shap_lc_excess_f1[i], banz_f1_incore[i], banz_f1_inlc[i], 
            most_dissatisfied_coals_f1[i], monot_f1[i], supadd_f1[i], subadd_f1[i]] for i in range(n_players)}
    
    if save_full_matrix:
        data["v_acc"] = list(v_acc) + [None]*n_extra_rows
        data["v_f1"] = list(v_f1) + [None]*n_extra_rows
    else: 
        data["v_acc"] = [v_acc[-1]] + [None]*n_extra_rows
        data["v_f1"] = [v_f1[-1]] + [None]*n_extra_rows

    index = matrix_rows + metrics_rows

    df = pd.DataFrame(data)
    df.insert(0, "description", index)
    df.to_csv(path+"complete_result.csv")


# barplot solution concepts (path='iris/exact/0/v/lr/)
def generate_barplot(path, dataset_name=''):
    shapley_acc = np.load(path+"shap_acc.npy")
    banzhaf_acc = np.load(path+"banz_mult_acc.npy")
    egalitarian_acc = np.load(path+"egalitarian_acc.npy")

    lblc_acc = np.load(path+"lblc_acc.npy")
    ublc_acc = np.load(path+"ublc_acc.npy")

    num_features = len(shapley_acc)
    features = [str(i+1) for i in range(num_features)]
    bar_width = 0.25
    x = np.arange(len(features))

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - bar_width, shapley_acc, bar_width, label='Shapley', color='#B2DF8A', edgecolor='black')
    bars2 = ax.bar(x, banzhaf_acc, bar_width, label='Banzhaf norm mult', color='#FB9A99', edgecolor='black')
    bars3 = ax.bar(x + bar_width, egalitarian_acc, bar_width, label='Egalitarian Least Core', color='lightsteelblue', edgecolor='black')
   
    for i in range(len(x)):
        xmin = x[i] - bar_width
        xmax = x[i] + bar_width
        ax.hlines(lblc_acc[i], xmin, xmax, colors='#155A8A', linestyles='-.', linewidth=3.2 ,label='LB Least Core' if i == 0 else "")
        ax.hlines(ublc_acc[i], xmin, xmax, colors='#155A8A', linestyles='-.', linewidth=3.2 ,label='UB Least Core' if i == 0 else "")

    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Importance Score', fontsize=14)
    ax.set_title('Importance scores for '+dataset_name+' dataset', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path+"barplot_acc.png")
    plt.show()


# calcolo distanza e entropia per solution concept (path='iris/exact/0/v/lr/ - measure='acc','f1')
def generate_distance_entropy_matrix(path, measure):
    shap = np.load(path+'shap_'+measure+'.npy')
    banz_mult = np.load(path+'banz_mult_'+measure+'.npy')
    banz_add = np.load(path+'banz_add_'+measure+'.npy')
    egalit = np.load(path+'egalitarian_'+measure+'.npy')
    nucl = np.loadtxt(path+'nucleolus_'+measure+'.txt', delimiter='\t')
    norm_l1 = np.load(path+'solution_vector_l1_'+measure+'.npy')
    norm_linf = np.load(path+'solution_vector_linf_'+measure+'.npy')
    tau_lc_exact = np.load(path+'tau_lc_exact_'+measure+'.npy')
    tau_lc_approx = np.load(path+'tau_lc_approx_'+measure+'.npy')

    shap = shap - shap.min()
    shap = shap / shap.sum()
    banz_mult = banz_mult - banz_mult.min()
    banz_mult = banz_mult / banz_mult.sum()
    banz_add = banz_add - banz_add.min()
    banz_add = banz_add / banz_add.sum()
    egalit = egalit - egalit.min()
    egalit = egalit / egalit.sum()
    nucl = nucl - nucl.min()
    nucl = nucl / nucl.sum()
    norm_l1 = norm_l1 - norm_l1.min()
    norm_l1 = norm_l1 / norm_l1.sum()
    norm_linf = norm_linf - norm_linf.min()
    norm_linf = norm_linf / norm_linf.sum()
    tau_lc_exact = tau_lc_exact - tau_lc_exact.min()
    tau_lc_exact = tau_lc_exact / tau_lc_exact.sum()
    tau_lc_approx = tau_lc_approx - tau_lc_approx.min()
    tau_lc_approx = tau_lc_approx / tau_lc_approx.sum()

    labels = ['shapley', 'banzhaf mult', 'banzhaf add', 'egalitarian', 'nucleolus', 
              'norm l1', 'norm linf', 'tau lc exact', 'tau lc approx']
    vect = {'shapley': shap, 'banzhaf mult': banz_mult, 'banzhaf add': banz_add, 'egalitarian': egalit, 
            'nucleolus': nucl, 'norm l1': norm_l1, 'norm linf': norm_linf, 
            'tau lc exact': tau_lc_exact, 'tau lc approx': tau_lc_approx}

    # Wasserstein 
    data_wass = {}
    for r in labels:
        row = {}
        for c in labels:
            row[c] = wasserstein_distance(vect[r], vect[c])
        data_wass[r] = row
    df_wass = pd.DataFrame(data_wass)
    df_wass.reset_index(inplace=True)
    df_wass.rename(columns={'index': ''}, inplace=True)

    # Jensen Shannon
    data_js = {}
    for r in labels:
        row = {}
        for c in labels:
            row[c] = jensenshannon(vect[r], vect[c])
        data_js[r] = row
    df_js = pd.DataFrame(data_wass)
    df_js.reset_index(inplace=True)
    df_js.rename(columns={'index': ''}, inplace=True)

    # Entropy 
    entropy_shap = entropy(shap)
    entropy_shap_norm = entropy_shap / np.log(len(shap))
    entropy_banzmult = entropy(banz_mult)
    entropy_banzmult_norm = entropy_banzmult / np.log(len(banz_mult))
    entropy_banzadd = entropy(banz_add)
    entropy_banzadd_norm = entropy_banzadd / np.log(len(banz_add))
    entropy_eg = entropy(egalit)
    entropy_eg_norm = entropy_eg / np.log(len(egalit))
    entropy_nucl = entropy(nucl)
    entropy_nucl_norm = entropy_nucl / np.log(len(nucl))
    entropy_l1 = entropy(norm_l1)
    entropy_l1_norm = entropy_l1 / np.log(len(norm_l1))
    entropy_linf = entropy(norm_linf)
    entropy_linf_norm = entropy_linf / np.log(len(norm_linf))
    entropy_tauexact = entropy(tau_lc_exact)
    entropy_tauexact_norm = entropy_tauexact / np.log(len(tau_lc_exact))
    entropy_tauapprox = entropy(tau_lc_approx)
    entropy_tauapprox_norm = entropy_tauapprox / np.log(len(tau_lc_approx))

    entropy_data = {
        'Methods': ['shapley', 'banzhaf mult', 'banzhaf add', 'egalitarian', 
                    'nucleolus', 'norm l1', 'norm linf', 'tau lc exact', 'tau lc approx'],
        'Entropy Norm': [entropy_shap_norm, entropy_banzmult_norm, entropy_banzadd_norm, entropy_eg_norm,
                         entropy_nucl_norm, entropy_l1_norm, entropy_linf_norm, 
                         entropy_tauexact_norm, entropy_tauapprox_norm]}
    df_entropy = pd.DataFrame(entropy_data)
    separator_wass = pd.DataFrame([["Wasserstein distances"] + [""] * (len(labels))], columns=[''] + labels)
    separator_empty = pd.DataFrame([[""] * (len(labels) + 1)], columns=[''] + labels)
    separator_js = pd.DataFrame([["Jensen-Shannon distances"] + [""] * (len(labels))], columns=[''] + labels)
    separator_entropy = pd.DataFrame([["Entropy"] + [""] * (len(labels))], columns=[''] + labels)

    df = pd.concat([separator_wass, df_wass, separator_empty, separator_js, df_js, separator_empty, separator_entropy, df_entropy], ignore_index=True)
    df.to_csv(path+'distance_entropy_'+measure+'.csv')

# barplot entropy solution concepts (path='' se eseguo nella stessa cartella)
def generate_plot_entropy(path='folder/'):
    csv_iris = pd.read_csv('iris/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_banknote = pd.read_csv('banknote/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_breastcancer = pd.read_csv('breastcancer/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_monks1 = pd.read_csv('monks1/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_monks3 = pd.read_csv('monks3/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_raisin = pd.read_csv('raisin/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_mc10_8 = pd.read_csv('make_classification/8_informative/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_mc10_6 = pd.read_csv('make_classification/6_informative/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_mc10_4 = pd.read_csv('make_classification/4_informative/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_australian = pd.read_csv('australian/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)
    csv_magic = pd.read_csv('magic/2/v_most_freq/lr/distance_entropy_acc.csv', index_col=0)

    entropy_iris = csv_iris[csv_iris['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_banknote = csv_banknote[csv_banknote['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_breastcancer = csv_breastcancer[csv_breastcancer['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_monks1 = csv_monks1[csv_monks1['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_monks3 = csv_monks3[csv_monks3['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_raisin = csv_raisin[csv_raisin['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_mc10_8 = csv_mc10_8[csv_mc10_8['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_mc10_6 = csv_mc10_6[csv_mc10_6['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_mc10_4 = csv_mc10_4[csv_mc10_4['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_australian = csv_australian[csv_australian['Entropy Norm'].notna()]['Entropy Norm'].values
    entropy_magic = csv_magic[csv_magic['Entropy Norm'].notna()]['Entropy Norm'].values

    entropy_shap = [entropy_iris[0], entropy_banknote[0], entropy_monks1[0], entropy_monks3[0], entropy_raisin[0], entropy_breastcancer[0], entropy_magic[0], entropy_australian[0], entropy_mc10_8[0], entropy_mc10_6[0], entropy_mc10_4[0]]
    entropy_banz = [entropy_iris[1], entropy_banknote[1], entropy_monks1[1], entropy_monks3[1], entropy_raisin[1], entropy_breastcancer[1], entropy_magic[1], entropy_australian[1], entropy_mc10_8[1], entropy_mc10_6[1], entropy_mc10_4[1]]
    entropy_egalit = [entropy_iris[2], entropy_banknote[2], entropy_monks1[2], entropy_monks3[2], entropy_raisin[2], entropy_breastcancer[2], entropy_magic[2], entropy_australian[2], entropy_mc10_8[2], entropy_mc10_6[2], entropy_mc10_4[2]]

    features = ['Iris', 'Banknote', 'Monks-1', 'Monks-3', 'Raisin', 'Wisconsin', 
                'Magic', 'Australian', 'Synthetic 8', 'Synthetic 6', 'Synthetic 4']
    bar_width = 0.25
    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(17,12))
    bars1 = ax.bar(x - bar_width, entropy_shap, bar_width, label='Shapley', color='#D9D9D9', edgecolor='black')
    bars2 = ax.bar(x, entropy_banz, bar_width, label='Banzhaf', color='#FFED6F', edgecolor='black')
    bars3 = ax.bar(x + bar_width, entropy_egalit, bar_width, label='Egalitarian Least Core', color='E3BCDE', edgecolor='black')
    #ax.set_xlabel('Datasets', fontsize=18)
    ax.set_ylabel('Entropy', fontsize=20)
    ax.set_title('Entropy for all datasets', fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=16)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path+"barplot_entropy_datasets.png")
    plt.show()



# genero file .mat per eseguire codice nucleolo matlab
def generate_matlab_v(path='iris/0/v_most_freq/lr/'):
    v_acc = np.load(path+'coalitions_acc.npy')
    scipy.io.savemat(path+'v_acc.mat', {'v_acc': v_acc})
    v_f1 = np.load(path+'coalitions_f1.npy')
    scipy.io.savemat(path+'v_f1.mat', {'v_f1': v_f1})


# retrain considerando solution concepts calcolato con v acc (folder='iris/exact/0/')
def compute_accuracy_retrain(folder='iris/exact/0/', ds_name='iris', model_string='lr', nfeat=1):
    acc_row = []
    X_train = np.load(folder+"X_train.npy")
    X_test = np.load(folder+"X_test.npy")
    Y_train = np.load(folder+"Y_train.npy")
    Y_test = np.load(folder+"Y_test.npy")
    path = folder + 'v_most_freq/' + model_string + '/'

    # rialleno per shap
    shap_values = np.load(path+'shap_acc.npy')
    top_idx_shap = np.argsort(shap_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_shap]
    X_test_imp = X_test[:,top_idx_shap]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_shap = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_shap)

    # rialleno per banz mult
    banz_values = np.load(path+'banz_mult_acc.npy')
    top_idx_banz = np.argsort(banz_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_banz]
    X_test_imp = X_test[:,top_idx_banz]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_banzmult = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_banzmult)

    # rialleno per banz add
    banz_values = np.load(path+'banz_add_acc.npy')
    top_idx_banz = np.argsort(banz_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_banz]
    X_test_imp = X_test[:,top_idx_banz]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_banzadd = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_banzadd)  

    # rialleno per egalitarian
    egalit_values = np.load(path+'egalitarian_acc.npy')
    top_idx_eg = np.argsort(egalit_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_eg]
    X_test_imp = X_test[:,top_idx_eg]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_eg = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_eg) 

    # rialleno per nucl
    nucl_values = np.loadtxt(path+'nucleolus_acc.txt', delimiter='\t')
    top_idx_nucl = np.argsort(nucl_values)[-nfeat:] 
    X_train_imp = X_train[:,top_idx_nucl]
    X_test_imp = X_test[:,top_idx_nucl]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_nucl = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_nucl)

    # rialleno per tauvalue leastcore exact
    tau_exact_values = np.load(path+'tau_lc_exact_acc.npy')
    top_idx_tau = np.argsort(tau_exact_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_tau]
    X_test_imp = X_test[:,top_idx_tau]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_tau_exact = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_tau_exact)

    # rialleno per tauvalue leastcore approximated
    tau_approx_values = np.load(path+'tau_lc_approx_acc.npy')
    top_idx_tau = np.argsort(tau_approx_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_tau]
    X_test_imp = X_test[:,top_idx_tau]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_tau_approx = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_tau_approx)    

    # rialleno per tauvalue leastcore exact con N e N\i (min)
    tau_exact_min_values = np.load(path+'tau_lc_min_exact_acc.npy')   
    top_idx_tau = np.argsort(tau_exact_min_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_tau]
    X_test_imp = X_test[:,top_idx_tau]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_taumin_exact = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_taumin_exact)

    # rialleno per tauvalue leastcore approximated con N e N\i (min)
    tau_approx_min_values = np.load(path+'tau_lc_min_approx_acc.npy') 
    top_idx_tau = np.argsort(tau_approx_min_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_tau]
    X_test_imp = X_test[:,top_idx_tau]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_taumin_approx = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_taumin_approx)

    # rialleno per solution vector norm l1
    l1_values = np.load(path+'solution_vector_l1_acc.npy')
    top_idx_l1 = np.argsort(l1_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_l1]
    X_test_imp = X_test[:,top_idx_l1]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_l1 = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_l1)

    # rialleno per solution vector norm linf
    linf_values = np.load(path+'solution_vector_linf_acc.npy')
    top_idx_linf = np.argsort(linf_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_linf]
    X_test_imp = X_test[:,top_idx_linf]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_linf = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_linf)

    methods = ['shapley', 'banzhaf mult', 'banzhaf add', 'egalitarian LC', 'nucleolus', 
               'tauvalue LC exact', 'tauvalue LC approx', 'tauvalue LC min exact', 'tauvalue LC min approx',
               'solution l1', 'solution linf']
    dataset_name = [ds_name]
    acc_df = pd.DataFrame([acc_row], columns=methods, index=dataset_name)
    acc_df.to_csv(folder+'row_accuracy_retrain_'+model_string+'_n'+str(nfeat)+'.csv', index = False)

# funzione per calcolo media e std delle acc riallenate per ogni dataset
def compute_mean_acc_std_single_ds(dataset_name='iris', dim='n1', model_string='lr'):
    subfolders=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df_list = []
    for subfolder in subfolders:
        df = pd.read_csv(dataset_name+'/exact/'+subfolder+'/row_accuracy_retrain_'+model_string+'_'+dim+'.csv')
        df_list.append(df)
    concatenated_df = pd.concat(df_list)
    
    mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
    std_df = concatenated_df.std(numeric_only=True).to_frame().T
    final_df = pd.DataFrame(columns=['ds_name', 'mean_acc_shapley', 'std_shapley', 
                        'mean_acc_banzmult', 'std_banzmult', 'mean_acc_banzadd', 'std_banzadd', 
                        'mean_acc_eg', 'std_eg', 'mean_acc_nucl', 'std_nucl', 
                        'mean_acc_tau_exact', 'std_tau_exact', 'mean_acc_tau_approx', 'std_tau_approx',
                        'mean_acc_taumin_exact', 'std_taumin_exact', 'mean_acc_taumin_approx', 'std_taumin_approx', 
                        'mean_acc_l1', 'std_l1', 'mean_acc_linf', 'std_linf'])
    final_df.loc[0] = [dataset_name, 
                       mean_df['shapley'][0], std_df['shapley'][0],
                       mean_df['banzhaf mult'][0], std_df['banzhaf mult'][0],
                       mean_df['banzhaf add'][0], std_df['banzhaf add'][0],
                       mean_df['egalitarian LC'][0], std_df['egalitarian LC'][0],
                       mean_df['nucleolus'][0], std_df['nucleolus'][0],
                       mean_df['tauvalue LC exact'][0], std_df['tauvalue LC exact'][0], 
                       mean_df['tauvalue LC approx'][0], std_df['tauvalue LC approx'][0], 
                       mean_df['tauvalue LC min exact'][0], std_df['tauvalue LC min exact'][0],   
                       mean_df['tauvalue LC min approx'][0], std_df['tauvalue LC min approx'][0],
                       mean_df['solution l1'][0], std_df['solution l1'][0],
                       mean_df['solution linf'][0], std_df['solution linf'][0]]
    final_df.to_csv(dataset_name+'/exact/'+'row_mean_acc_std_retrain_'+model_string+'_'+dim+'.csv', index=False)




# LASSO
def compute_lasso(nfeat=3, th=1e-6, path='iris/exact/3/', ds_name='iris'):
    folder = path+'lasso/'
    os.makedirs(folder, exist_ok=True)

    X_train = np.load(path+'X_train.npy')
    X_test = np.load(path+'X_test.npy')
    Y_train = np.load(path+'Y_train.npy')
    Y_test = np.load(path+'Y_test.npy')

    start_lasso = time.time()
    lambda_values = np.logspace(-3, 6, 5000)
    n_features = X_train.shape[1]
    coefs_matrix = []

    for lam in lambda_values:
        model = linmod.LogisticRegression(penalty="l1", solver="saga", C=1/lam, max_iter=1000)
        model.fit(X_train, Y_train)
        if model.coef_.shape[0] == 1:
            coefs_matrix.append(model.coef_.ravel())
        else:
            coefs_matrix.append(np.max(np.abs(model.coef_), axis=0))
    coefs_matrix = np.array(coefs_matrix)

    results = {i: None for i in range(n_features)}  # feature: lamba associato
    for i in range(n_features):
        for j in range(len(lambda_values)):
            if np.all(np.abs(coefs_matrix[j:, i]) < th): # riga di matrix da un certo lambda fino alla fine e colonna specifica per ogni feature
                results[i] = lambda_values[j]   # trovo valore di lambda in cui ogni feature è 0 e resta 0
                break
    
    lambdas = list(results.values()) # lista dei valori lambda salvati in result
    duplicates = {} # se ho duplicati salvo il valore lambda duplicato e gli indici delle feature che hanno lo stesso lambda
    for i, lam in results.items():
        count = lambdas.count(lam)
        if count > 1:
            duplicates.setdefault(lam, []).append(i)

    max_iter = 10
    for lam, features in duplicates.items():        # lam = valore di lambda duplicato / features = [5,7] indici 
        if len(features) > 1:   # se ho indici della feature che hanno stesso lambda
            print('I have duplicates')
            list_feat_duplicates = set(features)
            lam_values_previous = [value for value in lambdas if value < lam]   # valori lambda più piccoli del duplicato
            lam_previous = max(lam_values_previous) if lam_values_previous else 1e-2    # estremo sinistra dellintervallo = lambda massimo tra i precedenti o se non c'è a 1e-2

            iter_count = 0
            lower = lam_previous
            upper = lam

            while iter_count < max_iter and len(list_feat_duplicates) > 0:
                middle_point = (lower + upper) / 2
                model = linmod.LogisticRegression(penalty="l1", solver="saga", C=1/middle_point, max_iter=1000)
                model.fit(X_train,Y_train)
                if model.coef_.shape[0] == 1:
                    coefs = model.coef_.ravel()
                else:
                    coefs = np.max(np.abs(model.coef_), axis=0)

                zero_flags = {f: abs(coefs[f]) < th for f in list_feat_duplicates}     # salvo true/false in base se i coeff ottenuti solo per le varibili che erano duplicate sono nulli o no
                features_zero = [f for f, is_zero in zero_flags.items() if is_zero]
                features_non_zero = [f for f in list_feat_duplicates if f not in features_zero]
                
                if len(features_zero) > 0 and len(features_non_zero) > 0: # significa che alcuni coeff sono nulli e altri no (assegno il middle point solo ad una random tra quelle zero se no ho più di una)
                    if len(features_zero) == 1 and len(features_non_zero) == 1: # ho 2 feature duplicate
                        f_to_update = features_zero[0]
                        results[f_to_update] = middle_point
                        list_feat_duplicates.remove(f_to_update)
                        break
                    else:
                        f_to_update = features_zero[0]
                        results[f_to_update] = middle_point
                        list_feat_duplicates.remove(f_to_update)
                        upper = middle_point
                elif len(features_zero) == 0: # significa che non ho differenziato e i coeff sono maggiori di 0 --> ricerca nell'intervallo sopra
                    lower = middle_point
                else: # significa che non ho differenziato e tutti i coeff sono 0 --> ricerco nell'intervallo sotto
                    upper = middle_point
                
                iter_count += 1
            
            if len(list_feat_duplicates) > 1:   # metto 1 perche se ho 2 feature duplicate, in list mi rimarrà una feature associata al valore iniziale duplicato 
                print(f'Impossible to distinguish lambda values between features {features}')
    
    data = [(i + 1, results[i]) for i in results]
    end_lasso = time.time()
    df = pd.DataFrame(data, columns=["feature", "lambda"])
    df.to_csv(folder+"lambda_values_path.csv", index=False)

    time_dict = {'time_lasso': end_lasso-start_lasso}
    time_df = pd.DataFrame([time_dict])
    time_df.to_csv(folder+'execution_time_lasso.csv', index=False)

    plt.figure(figsize=(8, 6))
    for i in range(X_train.shape[1]):
        plt.plot(lambda_values, coefs_matrix[:, i], label=f'Feature {i+1}')
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("Coefficients")
    plt.title("Logistic Regression Path "+ds_name)
    plt.legend(loc='best')
    plt.axis("tight")
    plt.savefig(folder+"plot_logistic_path.png")

    # RIALLENO
    accuracy = []
    top_rows = df.sort_values(by='lambda', ascending=True).tail(nfeat)  # lasso normale
    top_idx_path = top_rows['feature'].astype(int).values - 1
    X_train_imp = X_train[:,top_idx_path]
    X_test_imp = X_test[:,top_idx_path]
    model = linmod.LogisticRegression(max_iter=1000, penalty=None, solver='saga')   
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_lasso = accuracy_score(Y_test, y_pred)
    accuracy.append(acc_lasso)

    df_sorted = df.sort_values(by='lambda', ascending=False).reset_index(drop=True) # lasso penalizzata
    last_lambda = df_sorted.iloc[nfeat-1]['lambda']
    lambda_value = None
    for i in range(nfeat, len(df_sorted)):
        candidate_lambda = df_sorted.iloc[i]['lambda']
        if candidate_lambda != last_lambda:
            lambda_value = candidate_lambda
            break
    model = linmod.LogisticRegression(max_iter=1000, penalty='l1', solver='saga', C=1/lambda_value)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc_lasso_penalized = accuracy_score(Y_test, y_pred)
    accuracy.append(acc_lasso_penalized)

    methods = ['lasso', 'lasso penalized']
    dataset_name = [ds_name]
    accuracy_df = pd.DataFrame([accuracy], columns=methods, index=dataset_name)
    accuracy_df.to_csv(folder+"row_accuracy_retrain_lasso_n"+str(nfeat)+".csv", index=False)


def compute_mean_acc_std_lasso(nvar=4, ds_name='iris', case='approx'):
    if nvar <= 5:
        nfeat_list = [1,2,3]
    elif nvar <= 7:
        nfeat_list = [1,2,3,5]
    elif nvar <= 14: 
        nfeat_list = [1,2,3,5,7]
    elif ds_name == 'hepatitis':
        nfeat_list = [1,5,10]
    else: 
        nfeat_list = [1]
        nfeat_list += list(range(5, int(nvar/2)+1, 5))

    numbers=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for number in numbers:
        path = ds_name+'/'+case+'/'+number+'/'
        os.makedirs(path, exist_ok=True)
        for nfeat in nfeat_list:
            compute_lasso(nfeat=nfeat, th=1e-6, path=path, ds_name=ds_name)

    for nfeat in nfeat_list:
        df_list = []
        df_time_list = []
        for number in numbers:
            accuracy_df = pd.read_csv(ds_name+'/'+case+'/'+number+'/lasso/row_accuracy_retrain_lasso_n'+str(nfeat)+'.csv')
            df_list.append(accuracy_df)
        concatenated_df = pd.concat(df_list)
        mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
        std_df = concatenated_df.std(numeric_only=True).to_frame().T
        final_df = pd.DataFrame(columns=['ds_name', 
                                        'mean_acc_lasso', 'std_lasso', 
                                        'mean_acc_lasso pen', 'std_lasso_pen'])
        final_df.loc[0] = [ds_name, 
                        mean_df['lasso'][0], std_df['lasso'][0],
                        mean_df['lasso penalized'][0], std_df['lasso penalized'][0]]
        final_df.to_csv(ds_name+'/'+case+'/'+'row_mean_acc_std_retrain_lasso_n'+str(nfeat)+'.csv', index=False)

        for number in numbers:
            time_df = pd.read_csv(ds_name+'/'+case+'/'+number+'/lasso/execution_time_lasso.csv')
            df_time_list.append(time_df)
        concatenated_df_time = pd.concat(df_time_list)
        mean_df_time = concatenated_df_time.mean(numeric_only=True).to_frame().T
        std_df_time = concatenated_df_time.std(numeric_only=True).to_frame().T
        final_df_time = pd.DataFrame(columns=['ds_name', 'mean_time_lasso', 'std_lasso'])
        final_df_time.loc[0] = [ds_name, mean_df_time['time_lasso'][0], std_df_time['time_lasso'][0]]
        final_df_time.to_csv(ds_name+'/'+case+'/row_mean_std_time_lasso.csv', index=False)


# SEQUENTIAL FEATURE SELECTION SFS/SBS
# SFS con validations set
def forward_selection(X_train, y_train, X_val, y_val, n_features_to_select, model_string, folder):
    n_total_features = X_train.shape[1]
    selected_features = []
    remaining_features = list(range(n_total_features))
    for _ in range(n_features_to_select):
        best_score = 0
        best_feature = None
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            model = return_model(folder, model_string)
            model.fit(X_train[:, candidate_features], y_train)
            y_pred = model.predict(X_val[:, candidate_features])
            score = accuracy_score(y_val, y_pred)
            if score > best_score:
                best_score = score
                best_feature = feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features

# SBS con validation set
def backward_elimination(X_train, y_train, X_val, y_val, n_features_to_select, model_string, folder):
    n_total_features = X_train.shape[1]
    selected_features = list(range(n_total_features))

    while len(selected_features) > n_features_to_select:
        best_score = 0
        feature_to_remove = None

        for feature in selected_features:
            candidate_features = [f for f in selected_features if f != feature]
            model = return_model(folder, model_string)
            model.fit(X_train[:, candidate_features], y_train)
            y_pred = model.predict(X_val[:, candidate_features])
            score = accuracy_score(y_val, y_pred)

            if score > best_score:
                best_score = score
                feature_to_remove = feature

        selected_features.remove(feature_to_remove)

    return selected_features


def compute_sfs_sbs(nfeat=5, path='sonar/approx_min_number_coal/0/', string='lr', ds_name='sonar'):
    X_train = np.load(path+"X_train.npy")
    X_test = np.load(path+"X_test.npy")
    Y_train = np.load(path+"Y_train.npy")
    Y_test = np.load(path+"Y_test.npy")
    X_train_train, X_val, Y_train_train, Y_val = ms.train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)

    rows = []
    times_dict = {}
    start_sfs = time.time()
    selected_idx_sfs = forward_selection(X_train_train, Y_train_train, X_val, Y_val, nfeat, string, path)
    end_sfs = time.time()
    times_dict['time_sfs'] = end_sfs - start_sfs
    support_sfs = [i in selected_idx_sfs for i in range(X_train.shape[1])]
    row_sfs = [nfeat, 'forward'] + support_sfs
    rows.append(row_sfs)

    start_sbs = time.time()
    selected_idx_sbs = backward_elimination(X_train_train, Y_train_train, X_val, Y_val, nfeat, string, path)
    end_sbs = time.time()
    times_dict['time_sbs'] = end_sbs - start_sbs
    support_sbs = [i in selected_idx_sbs for i in range(X_train.shape[1])]
    row_sbs = [nfeat, 'backward'] + support_sbs
    rows.append(row_sbs)

    columns = ['n_features', 'direction'] + [f'P{i+1}' for i in range(X_train.shape[1])]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path+'seq_feat_selection_'+string+'.csv', index=False)

    # rialleno sfs
    accuracy = []
    row = df[(df["n_features"] == nfeat) & (df["direction"] == "forward")]
    p_cols = [col for col in df.columns if col not in ["n_features", "direction"]]
    vect_forw = row[p_cols].astype(int).values.flatten()
    top_idx_forw = np.argsort(vect_forw)[-nfeat:]
    X_train_imp = X_train[:,top_idx_forw]
    X_test_imp = X_test[:,top_idx_forw]
    model = return_model(path, string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_forw = accuracy_score(Y_test, y_pred)
    accuracy.append(acc_forw)

    # rialleno sbs
    row = df[(df["n_features"] == nfeat) & (df["direction"] == "backward")]
    p_cols = [col for col in df.columns if col not in ["n_features", "direction"]]
    vect_back = row[p_cols].astype(int).values.flatten()
    top_idx_back = np.argsort(vect_back)[-nfeat:]
    X_train_imp = X_train[:,top_idx_back]
    X_test_imp = X_test[:,top_idx_back]
    model = return_model(path, string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_back = accuracy_score(Y_test, y_pred)
    accuracy.append(acc_back)

    methods = ['sfs', 'sbs']
    dataset_name = [ds_name]
    accuracy_df = pd.DataFrame([accuracy], columns=methods, index=dataset_name)
    accuracy_df.to_csv(path+'row_accuracy_retrain_sfssbs_'+string+'_n'+str(nfeat)+".csv", index=False)

    # salvo tempi
    time_df = pd.DataFrame([times_dict])
    time_df.to_csv(path+'execution_time_sfssbs_'+string+'_n'+str(nfeat)+'.csv', index=False)


def compute_mean_acc_std_sfssbs(nvar=60, ds_name='sonar', case='approx_min_number_coal', string='lr'):
    if nvar <= 5:
        nfeat_list = [1,2,3]
    elif nvar <= 7:
        nfeat_list = [1,2,3,5]
    elif nvar <= 14: 
        nfeat_list = [1,2,3,5,7]
    elif ds_name == 'hepatitis':
        nfeat_list = [1,5,10]
    else: 
        nfeat_list = [1]
        nfeat_list += list(range(5, int(nvar/2)+1, 5))

    numbers=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for number in numbers:
        path = ds_name+'/'+case+'/'+number+'/'
        os.makedirs(path, exist_ok=True)
        for nfeat in nfeat_list:
            compute_sfs_sbs(nfeat=nfeat, path=path, string=string, ds_name=ds_name)
    
    for nfeat in nfeat_list:
        df_list = []
        df_time_list = []
        for number in numbers:
            accuracy_df = pd.read_csv(ds_name+'/'+case+'/'+number+'/row_accuracy_retrain_sfssbs_'+string+'_n'+str(nfeat)+'.csv')
            df_list.append(accuracy_df)
        concatenated_df = pd.concat(df_list)
        mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
        std_df = concatenated_df.std(numeric_only=True).to_frame().T
        final_df = pd.DataFrame(columns=['ds_name', 
                                        'mean_acc_sfs', 'std_sfs', 
                                        'mean_acc_sbs', 'std_sbs'])
        final_df.loc[0] = [ds_name, 
                        mean_df['sfs'][0], std_df['sfs'][0],
                        mean_df['sbs'][0], std_df['sbs'][0]]
        final_df.to_csv(ds_name+'/'+case+'/'+'row_mean_acc_std_retrain_sfssbs_'+string+'_n'+str(nfeat)+'.csv', index=False)

        for number in numbers:
            time_df = pd.read_csv(ds_name+'/'+case+'/'+number+'/execution_time_sfssbs_'+string+'_n'+str(nfeat)+'.csv')
            df_time_list.append(time_df)
        concatenated_df_time = pd.concat(df_time_list)
        mean_df_time = concatenated_df_time.mean(numeric_only=True).to_frame().T
        std_df_time = concatenated_df_time.std(numeric_only=True).to_frame().T
        final_df_time = pd.DataFrame(columns=['ds_name', 'mean_time_sfs', 'std_sfs', 'mean_time_sbs', 'std_sbs'])
        final_df_time.loc[0] = [ds_name, mean_df_time['time_sfs'][0], std_df_time['time_sfs'][0],
                                mean_df_time['time_sbs'][0], std_df_time['time_sbs'][0]]
        final_df_time.to_csv(ds_name+'/'+case+'/row_mean_std_time_sfssbs_'+string+'_n'+str(nfeat)+'.csv', index=False)



# RFE (al posto della LASSO per random forest)
def compute_rfe(nfeat=5, path='sonar/approx_min_number_coal/0/', string='rf', ds_name='sonar'):
    X_train = np.load(path+"X_train.npy")
    X_test = np.load(path+"X_test.npy")
    Y_train = np.load(path+"Y_train.npy")
    Y_test = np.load(path+"Y_test.npy")
    X_train_train, X_val, Y_train_train, Y_val = ms.train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)

    start_time = time.time()
    model = return_model(path, string)
    rfe = RFE(estimator=model, n_features_to_select=nfeat)
    rfe.fit(X_train_train, Y_train_train) 
    end_time = time.time()
    tot_time = end_time - start_time

    support_rfe = rfe.support_      # mi dà un array con true/false se la feature è selezionata o no
    ranking_rfe = rfe.ranking_      # mi dà un array con 1 se la feature è importante, e poi numeri crescenti per altre feature

    df_rfe = pd.DataFrame({'feature': [f'P{i+1}' for i in range(X_train.shape[1])],
                           'ranking': ranking_rfe})
    df_rfe.to_csv(path+'rfe_ranking_'+string+'_n'+str(nfeat)+'.csv', index=False)
    
    top_idx_rfe = [i for i, x in enumerate(support_rfe) if x]
    X_train_imp = X_train[:, top_idx_rfe]
    X_test_imp = X_test[:, top_idx_rfe]
    model = return_model(path, string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_rfe = accuracy_score(Y_test, y_pred)

    accuracy_df = pd.DataFrame([[acc_rfe]], columns=['rfe'], index=[ds_name])
    accuracy_df.to_csv(path+'row_accuracy_retrain_rfe_n'+str(nfeat)+".csv", index=False)

    # salvo tempi
    time_df = pd.DataFrame([{'time_rfe': tot_time}])
    time_df.to_csv(path+'execution_time_rfe_n'+str(nfeat)+'.csv', index=False)

def compute_mean_acc_std_rfe(nvar=60, ds_name='sonar', case='approx_min_number_coal', string='rf'):
    if nvar <= 5:
        nfeat_list = [1,2,3]
    elif nvar <= 7:
        nfeat_list = [1,2,3,5]
    elif nvar <= 14: 
        nfeat_list = [1,2,3,5,7]
    elif ds_name == 'hepatitis':
        nfeat_list = [1,5,10]
    else: 
        nfeat_list = [1]
        nfeat_list += list(range(5, int(nvar/2)+1, 5))

    numbers=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for number in numbers:
        path = ds_name+'/'+case+'/'+number+'/'
        os.makedirs(path, exist_ok=True)
        for nfeat in nfeat_list:
            compute_rfe(nfeat=nfeat, path=path, string=string, ds_name=ds_name)
    
    for nfeat in nfeat_list:
        df_list = []
        df_time_list = []
        for number in numbers:
            accuracy_df = pd.read_csv(ds_name+'/'+case+'/'+number+'/row_accuracy_retrain_rfe_n'+str(nfeat)+'.csv')
            df_list.append(accuracy_df)
        concatenated_df = pd.concat(df_list)
        mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
        std_df = concatenated_df.std(numeric_only=True).to_frame().T
        final_df = pd.DataFrame(columns=['ds_name', 'mean_acc_rfe', 'std_rfe'])
        final_df.loc[0] = [ds_name, mean_df['rfe'][0], std_df['rfe'][0]]
        final_df.to_csv(ds_name+'/'+case+'/'+'row_mean_acc_std_retrain_rfe_n'+str(nfeat)+'.csv', index=False)

        for number in numbers:
            time_df = pd.read_csv(ds_name+'/'+case+'/'+number+'/execution_time_rfe_n'+str(nfeat)+'.csv')
            df_time_list.append(time_df)
        concatenated_df_time = pd.concat(df_time_list)
        mean_df_time = concatenated_df_time.mean(numeric_only=True).to_frame().T
        std_df_time = concatenated_df_time.std(numeric_only=True).to_frame().T
        final_df_time = pd.DataFrame(columns=['ds_name', 'mean_time_rfe', 'std_rfe'])
        final_df_time.loc[0] = [ds_name, mean_df_time['time_rfe'][0], std_df_time['time_rfe'][0]]
        final_df_time.to_csv(ds_name+'/'+case+'/row_mean_std_time_rfe_n'+str(nfeat)+'.csv', index=False)













### APPROSSIMAZIONE ###

def generate_shapley_approx(reduced_matrix, reduced_v, store_values=True, file_path='shapley.npy'):
    start = time.time()
    shapley_values_approx = cgt.compute_shapley_approximation(reduced_matrix, reduced_v)
    end = time.time()
    if store_values:
        np.save(file_path, shapley_values_approx)
    time_shap = end-start
    return time_shap

def generate_banzadd_approx(reduced_matrix, reduced_v, store_values=True, file_path='banzhaf.npy'):
    start = time.time()
    banz_add_values_approx = cgt.compute_banzhaf_add_approximation(reduced_matrix, reduced_v)
    end = time.time()
    if store_values:
        np.save(file_path, banz_add_values_approx)
    time_banzadd = end-start
    return time_banzadd

def generate_l1_approx(reduced_matrix, reduced_v, store_values=True, file_path='solution_l1.npy'):
    start = time.time()
    l1_values_approx = cgt.compute_solution_l1_approximation(reduced_matrix, reduced_v)
    end = time.time()
    if store_values:
        np.save(file_path, l1_values_approx)
    time_l1 = end-start
    return time_l1

def generate_linf_approx(reduced_matrix, reduced_v, store_values=True, file_path='solution_linf.npy'):
    start = time.time()
    linf_values_approx = cgt.compute_solution_linf_approximation(reduced_matrix, reduced_v)
    end = time.time()
    if store_values:
        np.save(file_path, linf_values_approx)
    time_linf = end-start
    return time_linf

def generate_tau_approx(reduced_matrix, reduced_v, store_values=True, path='iris/approx/0/v/lr/', file_path='tau.npy'):
    if 'v_most_freq' in path:
        value_baseline = np.load(path+'value_baseline_most_freq.npy')
    elif 'v_stratified' in path:
        value_baseline = np.load(path+'value_baseline_stratified.npy')
    else:
        value_baseline = 0    
    epsilon = 1 - value_baseline - reduced_v[-1] / reduced_matrix.shape[1]
    start = time.time()
    tau_values_approx = cgt.compute_tau_approximation(reduced_matrix, reduced_v, epsilon)
    end = time.time()
    if store_values:
        np.save(file_path, tau_values_approx)
    time_tau = end-start
    return time_tau

def execute_approximation(basepath, model_string, reduced_matrix, reduced_v_acc, reduced_v_f1):
    set_seed()
    path = basepath+model_string+'/'
    time_shap_acc = generate_shapley_approx(reduced_matrix, reduced_v_acc, store_values=True, file_path=path+'shap_acc.npy')
    time_shap_f1 = generate_shapley_approx(reduced_matrix, reduced_v_f1, store_values=True, file_path=path+'shap_f1.npy')
    time_banzadd_acc = generate_banzadd_approx(reduced_matrix, reduced_v_acc, store_values=True, file_path=path+'banz_add_acc.npy')
    time_banzadd_f1 = generate_banzadd_approx(reduced_matrix, reduced_v_f1, store_values=True, file_path=path+'banz_add_f1.npy')
    time_l1_acc = generate_l1_approx(reduced_matrix, reduced_v_acc, store_values=True, file_path=path+'solution_vector_l1_acc.npy')
    time_l1_f1 = generate_l1_approx(reduced_matrix, reduced_v_f1, store_values=True, file_path=path+'solution_vector_l1_f1.npy')
    time_linf_acc = generate_linf_approx(reduced_matrix, reduced_v_acc, store_values=True, file_path=path+'solution_vector_linf_acc.npy')
    time_linf_f1 = generate_linf_approx(reduced_matrix, reduced_v_f1, store_values=True, file_path=path+'solution_vector_linf_f1.npy')
    time_tau_acc = generate_tau_approx(reduced_matrix, reduced_v_acc, store_values=True, path=path, file_path=path+'tau_lc_approx_acc.npy')
    time_tau_f1 = generate_tau_approx(reduced_matrix, reduced_v_f1, store_values=True, path=path, file_path=path+'tau_lc_approx_f1.npy')

    time_path = os.path.dirname(basepath.rstrip('/')) + '/'
    time_path = time_path + 'v/' + model_string + '/'
    time_v_acc = np.load(time_path + 'time_v_acc.npy') 
    time_v_f1 = np.load(time_path + 'time_v_f1.npy')

    times_dict = {
        'shap_acc': time_shap_acc + time_v_acc,
        'shap_f1': time_shap_f1 + time_v_f1,
        'banzadd_acc': time_banzadd_acc + time_v_acc,
        'banzadd_f1': time_banzadd_f1 + time_v_f1,
        'tau_acc': time_tau_acc + time_v_acc,
        'tau_f1': time_tau_f1 + time_v_f1,
        'l1_acc': time_l1_acc + time_v_acc,
        'l1_f1': time_l1_f1 + time_v_f1,
        'linf_acc': time_linf_acc + time_v_acc,
        'linf_f1': time_linf_f1 + time_v_f1}
    df = pd.DataFrame([times_dict])
    df.to_csv(path+'/execution_time_cgt_'+model_string+'.csv', index=False)


def compute_mean_std_time(dataset_name='sonar', case='approx', string='lr'):
    subfolders=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    v_folder_list = ['v', 'v_most_freq', 'v_stratified']
    for v_folder in v_folder_list:
        df_list = []
        for subfolder in subfolders:
            df = pd.read_csv(dataset_name+'/'+case+'/'+subfolder+'/'+v_folder+'/'+string+'/execution_time_cgt_'+string+'.csv')
            df_list.append(df)
        concatenated_df = pd.concat(df_list)
        
        mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
        std_df = concatenated_df.std(numeric_only=True).to_frame().T
        final_df = pd.DataFrame(columns=['ds_name', 'mean_time_shap_acc', 'std_shap_acc', 'mean_time_shap_f1', 'std_shap_f1',
                                        'mean_time_banzadd_acc', 'std_banzadd_acc', 'mean_time_banzadd_f1', 'std_banzadd_f1',
                                        'mean_time_tau_acc', 'std_tau_acc', 'mean_time_tau_f1', 'std_tau_f1',
                                        'mean_time_l1_acc', 'std_l1_acc', 'mean_time_l1_f1', 'std_l1_f1',
                                        'mean_time_linf_acc', 'std_linf_acc', 'mean_time_linf_f1', 'std_linf_f1'])
        final_df.loc[0] = [dataset_name, 
                        mean_df['shap_acc'][0], std_df['shap_acc'][0],
                        mean_df['shap_f1'][0], std_df['shap_f1'][0],
                        mean_df['banzadd_acc'][0], std_df['banzadd_acc'][0],
                        mean_df['banzadd_f1'][0], std_df['banzadd_f1'][0],
                        mean_df['tau_acc'][0], std_df['tau_acc'][0],
                        mean_df['tau_f1'][0], std_df['tau_f1'][0],
                        mean_df['l1_acc'][0], std_df['l1_acc'][0],
                        mean_df['l1_f1'][0], std_df['l1_f1'][0],
                        mean_df['linf_acc'][0], std_df['linf_acc'][0],
                        mean_df['linf_f1'][0], std_df['linf_f1'][0]]
        final_df.to_csv(dataset_name+'/'+case+'/row_mean_std_time_cgt_'+v_folder+'_'+string+'.csv', index=False)


def compute_accuracy_retrain_approx(folder='iris/approx/0/', ds_name='iris', model_string='lr', nfeat=1):
    acc_row = []
    X_train = np.load(folder+"X_train.npy")
    X_test = np.load(folder+"X_test.npy")
    Y_train = np.load(folder+"Y_train.npy")
    Y_test = np.load(folder+"Y_test.npy")
    path = folder+'v_most_freq/'+model_string+'/'

    # rialleno per shap
    shap_values = np.load(path+'shap_acc.npy')
    top_idx_shap = np.argsort(shap_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_shap]
    X_test_imp = X_test[:,top_idx_shap]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_shap = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_shap)

    # rialleno per banz add
    banz_values = np.load(path+'banz_add_acc.npy')
    top_idx_banz = np.argsort(banz_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_banz]
    X_test_imp = X_test[:,top_idx_banz]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_banzadd = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_banzadd)  

    # rialleno per tauvalue leastcore approximated
    tau_approx_values = np.load(path+'tau_lc_approx_acc.npy')
    top_idx_tau = np.argsort(tau_approx_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_tau]
    X_test_imp = X_test[:,top_idx_tau]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_tau_approx = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_tau_approx)    

    # rialleno per solution vector norm l1
    l1_values = np.load(path+'solution_vector_l1_acc.npy')
    top_idx_l1 = np.argsort(l1_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_l1]
    X_test_imp = X_test[:,top_idx_l1]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_l1 = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_l1)

    # rialleno per solution vector norm linf
    linf_values = np.load(path+'solution_vector_linf_acc.npy')
    top_idx_linf = np.argsort(linf_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_linf]
    X_test_imp = X_test[:,top_idx_linf]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_linf = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_linf)

    methods = ['shapley', 'banzhaf add',
               'tauvalue LC approx',
               'solution l1', 'solution linf']
    dataset_name = [ds_name]
    acc_df = pd.DataFrame([acc_row], columns=methods, index=dataset_name)
    acc_df.to_csv(folder+'row_accuracy_retrain_'+model_string+'_n'+str(nfeat)+".csv", index = False)

# funzione per calcolo media e std delle acc riallenate per ogni dataset
def compute_mean_acc_std_single_ds_approx(dataset_name='iris', dim='n1', case='approx', model_string='lr'):
    subfolders=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df_list = []
    for subfolder in subfolders:
        df = pd.read_csv(dataset_name+'/'+case+'/'+subfolder+'/row_accuracy_retrain_'+model_string+'_'+dim+'.csv')
        df_list.append(df)
    concatenated_df = pd.concat(df_list)
    
    mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
    std_df = concatenated_df.std(numeric_only=True).to_frame().T
    final_df = pd.DataFrame(columns=['ds_name', 'mean_acc_shapley', 'std_shapley', 
                                    'mean_acc_banzadd', 'std_banzadd', 
                                    'mean_acc_tau_approx', 'std_tau_approx',
                                    'mean_acc_l1', 'std_l1', 
                                    'mean_acc_linf', 'std_linf'])
    final_df.loc[0] = [dataset_name, 
                       mean_df['shapley'][0], std_df['shapley'][0],
                       mean_df['banzhaf add'][0], std_df['banzhaf add'][0],
                       mean_df['tauvalue LC approx'][0], std_df['tauvalue LC approx'][0], 
                       mean_df['solution l1'][0], std_df['solution l1'][0],
                       mean_df['solution linf'][0], std_df['solution linf'][0]]
    final_df.to_csv(dataset_name+'/'+case+'/row_mean_acc_std_retrain_'+model_string+'_'+dim+'.csv', index=False)


# calcolo accuracy con tutte le features 
def compute_accuracy_all_features(x_train, x_test, y_train, y_test, model_string='lr', path='sonar/approx_min_number_coal/0/'):
    model = return_model(path, model_string)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    np.save(path+'acc_all_features_'+model_string+'.npy', np.array([acc]))

    return acc

def compute_mean_acc_std_all_features(ds_name='sonar', model_string='lr'):
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    folder = ds_name+'/approx_min_number_coal/'
    accuracy_list = []
    for number in numbers:
        path = folder+number+'/'
        x_train = np.load(path+"X_train.npy")
        x_test = np.load(path+"X_test.npy")
        y_train = np.load(path+"Y_train.npy")
        y_test = np.load(path+"Y_test.npy")
        acc = compute_accuracy_all_features(x_train, x_test, y_train, y_test, model_string=model_string, path=path)
        accuracy_list.append(acc)
    accuracies = np.array(accuracy_list)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    df = pd.DataFrame([{'ds_name': ds_name, 
                        'mean_acc_all_feat': mean_acc,
                        'std_all_feat': std_acc}])
    df.to_csv(folder+'row_mean_acc_std_all_features_'+model_string+'.csv', index=False)



# funzione per il calcolo di Linf e L1 con i pesi di shap -- wLinf / wL1
def compute_wLinf_wL1(matrix, v_acc, v_f1, path, model_string):
    
    # ----------
    # print('path = ', path)
    # print('v acc = ', v_acc)
    # print('v f1 = ', v_f1)
    # ----------
    
    # path viene passato con 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/v/'+model_string+'/' ma devo togliere la parte 'RESULTS_wL1_wLinf/' per leggere correttamente quanto già calcolato
    parts = path.split("/")
    old_path = "/".join(parts[1:4]) # ds_name+'/approx_min_number_coal/'+str(i)

    # time calcoalti per il vettore delle coalizioni v
    time_v_acc = np.load(old_path+'/v/'+model_string+'/time_v_acc.npy')
    time_v_f1 = np.load(old_path+'/v/'+model_string+'/time_v_f1.npy')

    # ----------
    # print('LOAD TEMPI OK')
    # ----------

    # wLinf
    start = time.time()
    wlinf_values_acc = cgt.compute_solution_linf_approximation_weights(matrix, v_acc)
    end = time.time()
    time_wlinf_acc = end - start

    # ----------
    # print('CALCOLO WLINF CON V ACC OK')
    # ----------
    
    start = time.time()
    wlinf_values_f1 = cgt.compute_solution_linf_approximation_weights(matrix, v_f1)
    end = time.time()
    time_wlinf_f1 = end-start

    # ----------
    # print('CALCOLO WLINF CON V F1 OK')
    # ----------

    # wL1
    start = time.time()
    wl1_values_acc = cgt.compute_solution_l1_approximation_weights(matrix, v_acc)
    end = time.time()
    time_wl1_acc = end - start

    # ----------
    # print('CALCOLO WL1 CON V ACC OK')
    # ----------
    
    start = time.time()
    wl1_values_f1 = cgt.compute_solution_l1_approximation_weights(matrix, v_f1)
    end = time.time()
    time_wl1_f1 = end-start

    # ----------
    # print('CALCOLO WL1 CON V F1 OK')
    # ----------

    # salvo valori
    np.save(path+'wlinf_values_acc.npy', wlinf_values_acc)
    np.save(path+'wlinf_values_f1.npy', wlinf_values_f1)
    np.save(path+'wl1_values_acc.npy', wl1_values_acc)
    np.save(path+'wl1_values_f1.npy', wl1_values_f1)
    
    # salvo tempi
    times_dict = {
            'wl1_acc': time_wl1_acc + time_v_acc,
            'wl1_f1': time_wl1_f1 + time_v_f1,
            'wlinf_acc': time_wlinf_acc + time_v_acc,
            'wlinf_f1': time_wlinf_f1 + time_v_f1
            }
    df = pd.DataFrame([times_dict])
    df.to_csv(path+'/execution_time_cgt_'+model_string+'.csv', index=False)
    

def compute_accuracy_retrain_approx_wLinf_wL1(folder, ds_name, model_string, nfeat):
    # folder viene passato con 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/' ma devo togliere la parte 'RESULTS_wL1_wLinf/' per leggere correttamente quanto già calcolato
    parts = folder.split("/")
    old_folder = "/".join(parts[1:4]) # ds_name+'/approx_min_number_coal/'+str(i)
    
    acc_row = []
    X_train = np.load(old_folder+"/X_train.npy")
    X_test = np.load(old_folder+"/X_test.npy")
    Y_train = np.load(old_folder+"/Y_train.npy")
    Y_test = np.load(old_folder+"/Y_test.npy")
    path = folder+'v_most_freq/'+model_string+'/'

    # rialleno per wL1
    l1_values = np.load(path+'wl1_values_acc.npy')
    top_idx_l1 = np.argsort(l1_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_l1]
    X_test_imp = X_test[:,top_idx_l1]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_wl1 = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_wl1)

    # rialleno per wLinf
    linf_values = np.load(path+'wlinf_values_acc.npy')
    top_idx_linf = np.argsort(linf_values)[-nfeat:]
    X_train_imp = X_train[:,top_idx_linf]
    X_test_imp = X_test[:,top_idx_linf]
    model = return_model(folder, model_string)
    model.fit(X_train_imp, Y_train)
    y_pred = model.predict(X_test_imp)
    acc_wlinf = accuracy_score(Y_test, y_pred)
    acc_row.append(acc_wlinf)

    methods = ['solution l1 weights', 'solution linf weights']
    dataset_name = [ds_name]
    acc_df = pd.DataFrame([acc_row], columns=methods, index=dataset_name)
    acc_df.to_csv(folder+'row_accuracy_retrain_'+model_string+'_n'+str(nfeat)+".csv", index = False)






            


        
        

        