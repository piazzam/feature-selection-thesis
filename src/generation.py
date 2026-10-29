import numpy as np
import sklearn.datasets as ds
import experiment as ex
import cooperative_game_theory as cgt
import os
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as ms
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
import random
import sklearn.linear_model as linmod
from sklearn.metrics import accuracy_score
import pandas as pd
import glob
import scipy.io
import gc
from collections import Counter
import time

def _can_stratify(y, test_size):
    """Return whether a stratified hold-out is feasible for this target."""
    y = np.asarray(y).ravel()
    counts = Counter(y).values()
    n_classes = len(np.unique(y))
    n_test = int(np.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    return min(counts) >= 2 and n_test >= n_classes and n_train >= n_classes


def repeated_holdout_splits(x, y, n_repeats=10, test_size=0.1, seed=42, predefined_train_size=None):
    """Yield the outer train/test splits used by the thesis protocol.

    If ``predefined_train_size`` is given, ``x``/``y`` are expected to be the
    concatenation of the dataset's official training and test partitions.  The
    official test partition is then kept fixed across all repetitions.

    Otherwise, ten independent hold-out splits are generated.  Stratification
    is used whenever it is feasible; a plain ShuffleSplit is used as fallback
    for very small/high-cardinality datasets.
    """
    if predefined_train_size is not None:
        train_idx = np.arange(predefined_train_size)
        test_idx = np.arange(predefined_train_size, len(x))
        for _ in range(n_repeats):
            yield train_idx, test_idx
        return

    if _can_stratify(y, test_size):
        splitter = ms.StratifiedShuffleSplit(
            n_splits=n_repeats, test_size=test_size, random_state=seed
        )
        yield from splitter.split(x, np.asarray(y).ravel())
    else:
        splitter = ms.ShuffleSplit(
            n_splits=n_repeats, test_size=test_size, random_state=seed
        )
        yield from splitter.split(x)


def validation_split(X_train, Y_train, repetition, validation_size=0.1, seed=42):
    """Create the validation split used to evaluate coalition values."""
    stratify = Y_train if _can_stratify(Y_train, validation_size) else None
    return ms.train_test_split(
        X_train, Y_train, test_size=validation_size,
        random_state=seed + repetition, stratify=stratify
    )


def scale_train_test(X_train, X_test):
    """Fit MinMax scaling on training data only and transform test data."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(np.asarray(X_train, dtype=float))
    X_test_scaled = scaler.transform(np.asarray(X_test, dtype=float))
    return X_train_scaled, X_test_scaled


### DATASET FINO A 20 FEATURES ###

def generation_kfold(ds_name='iris', model_string='lr', do_exact=False):

    ex.set_seed(42)

    if ds_name == 'iris':
        data = ds.load_iris()
        x = data.data
        y = data.target

    elif ds_name == 'banknote':
        banknote_authentication = fetch_ucirepo(id=267) 
        x = np.array(banknote_authentication.data.features)
        y = np.array(banknote_authentication.data.targets)

    elif ds_name == 'monks1':
        monk = fetch_openml(name='monks-problems-1', version=1, as_frame=True)
        x = monk.data
        y = monk.target
        x = x.values
        x = x.astype(int)
        y = y.astype(int)

    elif ds_name == 'monks3':
        monk = fetch_openml(name='monks-problems-3', version=1, as_frame=True)
        x = monk.data
        y = monk.target
        x = x.values
        x = x.astype(int)
        y = y.astype(int)

    elif ds_name == 'raisin': 
        raisin = fetch_ucirepo(id=850)  
        x = raisin.data.features 
        y = raisin.data.targets 
        y = y.replace({'Besni': 0, 'Kecimen': 1})
        y = y.to_numpy()
        x = x.values

    elif ds_name == 'breastcancer':
        breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
        x = np.array(breast_cancer_wisconsin_original.data.features)
        y = np.array(breast_cancer_wisconsin_original.data.targets)
        y = (y == 4).astype(int) # le classi erano 2 e 4, così sono 0 e 1
        mask = ~np.isnan(x).any(axis=1) # avevo 16 missing values che elimino
        x = x[mask]
        y = y[mask]

    elif ds_name == 'magic':
        magic_gamma_telescope = fetch_ucirepo(id=159) 
        x = magic_gamma_telescope.data.features 
        y = magic_gamma_telescope.data.targets
        y = y.replace({'g': 0, 'h': 1})
        y = y.to_numpy()
        x = x.values 

    elif ds_name == 'australian':
        statlog_australian_credit_approval = fetch_ucirepo(id=143) 
        x = statlog_australian_credit_approval.data.features 
        y = statlog_australian_credit_approval.data.targets 
        y = y.to_numpy()
        x = x.values

    elif ds_name == 'seismic':
        seismic = fetch_openml(name='seismic-bumps', version=1, as_frame=True)
        x = seismic.data
        y = seismic.target
        x = x.values
        y = y.astype(int)
        y = np.array(y)

    elif ds_name == 'heart':
        heart = fetch_openml(name='heart-disease', version=1, as_frame=True)
        heart = heart.data
        x = heart.iloc[:, :-1]
        y = heart.iloc[:, -1]
        x = x.values
        y = y.astype(int)
        y = np.array(y)

    elif ds_name == 'make_classification/8_informative':
        flipy = random.uniform(0.01, 0.1)
        weight = np.random.uniform(0.4, 0.6)
        x, y = ds.make_classification(n_samples=1000, n_features=10, n_redundant=0, n_informative=8, flip_y=flipy, weights=[weight, 1-weight], shuffle=False)

    elif ds_name == 'make_classification/6_informative':
        flipy = random.uniform(0.01, 0.1)
        weight = np.random.uniform(0.4, 0.6)
        x, y = ds.make_classification(n_samples=1000, n_features=10, n_redundant=0, n_informative=6, flip_y=flipy, weights=[weight, 1-weight], shuffle=False)

    elif ds_name == 'make_classification/4_informative':
        flipy = random.uniform(0.01, 0.1)
        weight = np.random.uniform(0.4, 0.6)
        x, y = ds.make_classification(n_samples=1000, n_features=10, n_redundant=0, n_informative=4, flip_y=flipy, weights=[weight, 1-weight], shuffle=False)    

    x = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    protocol_splits = list(repeated_holdout_splits(x, y, n_repeats=10, test_size=0.1, seed=42))

    if do_exact:
        ## CALCOLO IN MODO ESATTO 
        for i, (train_index, test_index) in enumerate(protocol_splits):

            folder = ds_name+'/exact/'+str(i)+'/'
            basepath = ds_name+'/exact/'+str(i)+'/'+'v/'
            path = ds_name+'/exact/'+str(i)+'/v/'+model_string+'/'

            os.makedirs(basepath, exist_ok=True)
            os.makedirs(path, exist_ok=True)

            X_train = x[train_index]
            X_test = x[test_index]
            Y_train = y[train_index]
            Y_test = y[test_index]
            X_train, X_test = scale_train_test(X_train, X_test)

            np.save(folder+"X_train.npy", X_train)
            np.save(folder+"X_test.npy", X_test)
            np.save(folder+"Y_train.npy", Y_train)
            np.save(folder+"Y_test.npy", Y_test)

            nvar = x.shape[1]
            matrix = cgt.coal_player_matrix_fast(nvar)
            ncoal = matrix.shape[0] 
            np.save(path+'matrix.npy', matrix)

            # matrix ridotta per tau value con N e N\i
            mask_3_ones = np.sum(matrix, axis=1) == 3         
            mask_grand_coal = np.sum(matrix, axis=1) == nvar  
            matrix_min = matrix[mask_3_ones | mask_grand_coal]    

            ## V NORMAL
            X_train_train, X_val, Y_train_train, Y_val = validation_split(X_train, Y_train, i)
            v_tot = ex.generate_v_traintest_already_divided(model_string, matrix, X_train_train, X_val, Y_train_train, Y_val, store_values=True, path=basepath)
            v_acc = np.array([v_tot[i][0] for i in range(ncoal)])
            v_f1 = np.array([v_tot[i][1] for i in range(ncoal)])
            ex.execute(nvar, matrix, path, v_acc, v_f1)                   
            ex.generate_tauvector(v_acc, v_f1, matrix, path, 'tau_lc')    

            # tau vector con N e N\i
            v_acc_min = v_acc[mask_3_ones | mask_grand_coal]      
            v_f1_min = v_f1[mask_3_ones | mask_grand_coal]        
            ex.generate_tauvector(v_acc_min, v_f1_min, matrix_min, path, 'tau_lc_min')    

            ex.generate_distance_entropy_matrix(path, 'acc')     
            ex.generate_distance_entropy_matrix(path, 'f1')     
            ex.generate_csv(path)    
            ex.generate_barplot(path, ds_name)    

            ## V MOST FREQ
            basepath = ds_name+'/exact/'+str(i)+'/'+'v_most_freq/'
            path = ds_name+'/exact/'+str(i)+'/'+'v_most_freq/'+model_string+'/'

            os.makedirs(basepath, exist_ok=True)
            os.makedirs(path, exist_ok=True)

            np.save(path+'matrix.npy', matrix)

            value_baseline_mostfreq = ex.generate_baseline_most_frequent(X_train_train, Y_train_train, path)
            v_acc_mostfreq = v_acc - value_baseline_mostfreq
            v_acc_mostfreq[0] = 0
            np.save(path+"coalitions_acc.npy", v_acc_mostfreq)
            v_f1_mostfreq = v_f1 - value_baseline_mostfreq
            v_f1_mostfreq[0] = 0
            np.save(path+"coalitions_f1.npy", v_f1_mostfreq)
            ex.execute(nvar, matrix, path, v_acc_mostfreq, v_f1_mostfreq)
            ex.generate_tauvector(v_acc_mostfreq, v_f1_mostfreq, matrix, path, 'tau_lc')

            # tau vector con N e N\i
            v_acc_mostfreq_min = v_acc_mostfreq[mask_3_ones | mask_grand_coal]     
            v_f1_mostfreq_min = v_f1_mostfreq[mask_3_ones | mask_grand_coal]  
            ex.generate_tauvector(v_acc_mostfreq_min, v_f1_mostfreq_min, matrix_min, path, 'tau_lc_min')  

            ex.generate_distance_entropy_matrix(path, 'acc')
            ex.generate_distance_entropy_matrix(path, 'f1')
            ex.generate_csv(path)
            ex.generate_barplot(path, ds_name)

            ## V STRATIFIED
            basepath = ds_name+'/exact/'+str(i)+'/'+'v_stratified/'         
            path = ds_name+'/exact/'+str(i)+'/'+'v_stratified/'+model_string+'/'  

            os.makedirs(basepath, exist_ok=True)    
            os.makedirs(path, exist_ok=True)     

            np.save(path+'matrix.npy', matrix)

            value_baseline_strat = ex.generate_baseline_stratified(X_train_train, Y_train_train, path)    
            v_acc_strat = v_acc - value_baseline_strat
            v_acc_strat[0] = 0
            np.save(path+"coalitions_acc.npy", v_acc_strat)
            v_f1_strat = v_f1 - value_baseline_strat
            v_f1_strat[0] = 0
            np.save(path+"coalitions_f1.npy", v_acc_strat)
            ex.execute(nvar, matrix, path, v_acc_strat, v_f1_strat)
            ex.generate_tauvector(v_acc_strat, v_f1_strat, matrix, path, 'tau_lc')

            # tau vector con N e N\i
            v_acc_strat_min = v_acc_strat[mask_3_ones | mask_grand_coal]   
            v_f1_strat_min = v_f1_strat[mask_3_ones | mask_grand_coal]     
            ex.generate_tauvector(v_acc_strat_min, v_f1_strat_min, matrix_min, path, 'tau_lc_min')       

            ex.generate_distance_entropy_matrix(path, 'acc')
            ex.generate_distance_entropy_matrix(path, 'f1')
            ex.generate_csv(path)
            ex.generate_barplot(path, ds_name)

            if nvar <= 5: 
                nfeat_list = [1,2,3]
            elif nvar <= 7:
                nfeat_list = [1,2,3,5]
            else:
                nfeat_list = [1,2,3,5,7]

            for nfeat in nfeat_list:
                ex.compute_accuracy_retrain(folder=folder, ds_name=ds_name, model_string=model_string, nfeat=nfeat)

        for nfeat in nfeat_list:
            ex.compute_mean_acc_std_single_ds(dataset_name=ds_name, dim=f'n{nfeat}', model_string=model_string)      


    ## CALCOLO IN MODO APPROSSIMATO 
    for i, (train_index, test_index) in enumerate(protocol_splits):
        folder = ds_name+'/approx/'+str(i)+'/'
        basepath = ds_name+'/approx/'+str(i)+'/'+'v/'
        model_string = model_string
        path = ds_name+'/approx/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        X_train = x[train_index]
        X_test = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        X_train, X_test = scale_train_test(X_train, X_test)

        np.save(folder+"X_train.npy", X_train)
        np.save(folder+"X_test.npy", X_test)
        np.save(folder+"Y_train.npy", Y_train)
        np.save(folder+"Y_test.npy", Y_test)

        nvar = x.shape[1]
        reduced_matrix = cgt.reduced_matrix(nvar)
        ncoal = reduced_matrix.shape[0] 
        np.save(path+'matrix.npy', reduced_matrix)

        # V normale
        X_train_train, X_val, Y_train_train, Y_val = validation_split(X_train, Y_train, i)
        v_tot_red = ex.generate_v_traintest_already_divided(model_string, reduced_matrix, X_train_train, X_val, Y_train_train, Y_val, store_values=True, path=basepath)
        v_acc_red = np.array([v_tot_red[i][0] for i in range(ncoal)])
        v_f1_red = np.array([v_tot_red[i][1] for i in range(ncoal)])
        ex.execute_approximation(basepath, model_string, reduced_matrix, v_acc_red, v_f1_red)   

        # V most freq 
        basepath = ds_name+'/approx/'+str(i)+'/'+'v_most_freq/'
        path = ds_name+'/approx/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', reduced_matrix)
        value_baseline_mostfreq = ex.generate_baseline_most_frequent(X_train_train, Y_train_train, path)
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)
        ex.execute_approximation(basepath, model_string, reduced_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red)

        # V stratified
        basepath = ds_name+'/approx/'+str(i)+'/'+'v_stratified/'       
        path = ds_name+'/approx/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', reduced_matrix)
        value_baseline_strat = ex.generate_baseline_stratified(X_train_train, Y_train_train, path)
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_acc_strat_red)
        ex.execute_approximation(basepath, model_string, reduced_matrix, v_acc_strat_red, v_f1_strat_red)

        if nvar <=5:
            nfeat_list = [1,2,3]
        elif nvar <= 7:
            nfeat_list = [1,2,3,5]
        else:
            nfeat_list = [1,2,3,5,7]
        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx(folder=folder, ds_name=ds_name, model_string=model_string, nfeat=nfeat)

    for nfeat in nfeat_list:
        ex.compute_mean_acc_std_single_ds_approx(dataset_name=ds_name, dim=f'n{nfeat}', case='approx', model_string=model_string)

















### DATASET OLTRE 20 FEATURE (SOLO APPROX) ###

# ---------------------------------------------------------------------
# TUTTI I RISULTATI PRIMA DEL 24/03/26 SONO ERRATI PER F1 V STRATIFIED 
# PERCHE' VENIVA SALVATO IL VALORE CON ACCURACY E NON F1
# ---------------------------------------------------------------------

def generation_kfold_approx(ds_name='sonar', model_string='lr', fold_start=0, fold_end=10):

    ex.set_seed(42)
    predefined_train_size = None

    if ds_name == 'sonar':
        connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151) 
        x = np.array(connectionist_bench_sonar_mines_vs_rocks.data.features)
        y = connectionist_bench_sonar_mines_vs_rocks.data.targets
        y = y.replace({'R': 0, 'M': 1}).astype(int)
        y = y.to_numpy()

    elif ds_name == 'spambase':
        spambase = fetch_ucirepo(id=94) 
        x = np.array(spambase.data.features)
        y = np.array(spambase.data.targets)

    elif ds_name == 'chess':
        chess = fetch_openml(name='kr-vs-kp', version=1, as_frame=True)
        x = chess.data
        y = chess.target
        binary_columns = [col for col in x.columns if x[col].nunique() == 2]
        non_binary_columns = [col for col in x.columns if x[col].nunique() > 2]
        x_binary = x[binary_columns].replace({'f': 0, 't': 1, 'n': 0, 'l': 0, 'g': 1})
        x_non_binary = pd.get_dummies(x[non_binary_columns], drop_first=False)
        x = pd.concat([x_binary, x_non_binary], axis=1).astype(int)
        y = y.replace({'nowin': 0, 'won': 1})
        y = y.astype(int)
        y = y.values

    elif ds_name == 'valley_without_noise':
        train_df = pd.read_csv('DATA/Hill_Valley_without_noise_Training.data')
        test_df = pd.read_csv('DATA/Hill_Valley_without_noise_Testing.data')
        x_train_fixed = np.asarray(train_df.iloc[:, :-1])
        y_train_fixed = np.asarray(train_df.iloc[:, -1])
        x_test_fixed = np.asarray(test_df.iloc[:, :-1])
        y_test_fixed = np.asarray(test_df.iloc[:, -1])
        predefined_train_size = len(x_train_fixed)
        x = np.vstack([x_train_fixed, x_test_fixed])
        y = np.concatenate([y_train_fixed, y_test_fixed])

    elif ds_name == 'valley_with_noise':
        train_df = pd.read_csv('DATA/Hill_Valley_with_noise_Training.data')
        test_df = pd.read_csv('DATA/Hill_Valley_with_noise_Testing.data')
        x_train_fixed = np.asarray(train_df.iloc[:, :-1])
        y_train_fixed = np.asarray(train_df.iloc[:, -1])
        x_test_fixed = np.asarray(test_df.iloc[:, :-1])
        y_test_fixed = np.asarray(test_df.iloc[:, -1])
        predefined_train_size = len(x_train_fixed)
        x = np.vstack([x_train_fixed, x_test_fixed])
        y = np.concatenate([y_train_fixed, y_test_fixed])

    elif ds_name == 'parkinson':
        parkinson = fetch_openml(name='parkinsons', version=1, as_frame=True)
        x = parkinson.data
        y = parkinson.target
        x = x.values
        y = y.astype(int)
        y = np.array(y)

    elif ds_name == 'hepatitis':
        hepatitis = fetch_ucirepo(id=46) 
        x = hepatitis.data.features 
        y = hepatitis.data.targets
        mask = ~x.isna().any(axis=1)    # avevo 75 missing values su 155 
        x = x[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        category_columns = x.columns[1:13]      # colonne categoriche con 1 e 2 mappati in 0 e 1
        x[category_columns] = x[category_columns].apply(lambda col: col.map({1: 0, 2: 1}))
        y['Class'] = y['Class'].map({1: 0, 2: 1})
        x = x.values
        y = y.astype(int)
        y = np.array(y)
        y = y.ravel()

    elif ds_name == 'nursery':
        nursery = fetch_openml(name='nursery', version=1, as_frame=True)
        x = nursery.data
        y = nursery.target
        binary_columns = [col for col in x.columns if x[col].nunique() == 2]
        non_binary_columns = [col for col in x.columns if x[col].nunique() > 2]
        x_binary = x[binary_columns].replace({'inconv': 0, 'convenient': 1})    # una sola colonna è binaria
        x_non_binary = pd.get_dummies(x[non_binary_columns], drop_first=False)
        x = pd.concat([x_binary, x_non_binary], axis=1).astype(int)
        y = y.replace({'not_recom': 0, 'recommend': 1, 'very_recom': 2, 'priority': 3, 'spec_prior': 4})
        y = y.astype(int)
        y = y.values

    elif ds_name == 'dermatology': 
        dermatology = fetch_ucirepo(id=33)
        x = dermatology.data.features
        y = dermatology.data.targets
        mask = ~x.isna().any(axis=1) # ho 8 missing values che elimino
        x = x[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        y = y.replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5})
        y = y.values
        y = y.ravel()

    elif ds_name == 'satimage': 
        statlog_landsat_satellite = fetch_ucirepo(id=146) 
        x = statlog_landsat_satellite.data.features 
        y = statlog_landsat_satellite.data.targets 
        y = y.replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5})
        y = y.astype(int)
        y = y.values
        y = y.ravel()

    elif ds_name == 'mushrooms':
        mushrooms = fetch_openml(name='mushroom', version=1, as_frame=True)
        x = mushrooms.data
        y = mushrooms.target
        mask = ~x.isna().any(axis=1) # ho 2480 missing values che elimino
        x = x[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        constant_columns = [col for col in x.columns if x[col].nunique() == 1] # colonna che ha sempre lo stesso valore per tutte le istanze 
        x = x.drop(columns = constant_columns)
        binary_columns = [col for col in x.columns if x[col].nunique() == 2]
        for col in binary_columns:
            values = x[col].unique()
            x[col] = x[col].map({values[0]: 0, values[1]: 1})
        non_binary_columns = [col for col in x.columns if x[col].nunique() > 2]
        x_non_binary = pd.get_dummies(x[non_binary_columns], drop_first=False)
        x = pd.concat([x[binary_columns], x_non_binary], axis=1)
        x = x.astype(int)
        y = y.replace({'e': 0, 'p': 1})
        y = y.astype(int)
        y = y.values

    elif ds_name == 'splice': 
        splice = fetch_openml(name='splice', version=1, as_frame=True)
        x = splice.data
        y = splice.target
        x = pd.get_dummies(x, drop_first=False) # tutte le colonne hanno più di 2 categorie
        x = x.astype(int)
        y = y.replace({'EI': 0, 'IE': 1, 'N': 2}) 
        y = y.astype(int)
        y = y.values

    elif ds_name == 'robot':
        wall_robot = fetch_openml(name='wall-robot-navigation', version=1, as_frame=True)
        x = wall_robot.data
        y = wall_robot.target
        y = y.replace({'1': 0, '2': 1, '3': 2, '4': 3, '5': 4})
        y = y.astype(int)
        y = y.values
    
    elif ds_name == 'qsar':
        qsar = fetch_openml(name='qsar-biodeg', version=1, as_frame=True)
        x = qsar.data
        y = qsar.target
        y = y.replace({'1': 0, '2': 1})
        y = y.astype(int)
        y = y.values

    elif ds_name == 'indian_patient':
        ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225)  
        x = ilpd_indian_liver_patient_dataset.data.features 
        y = ilpd_indian_liver_patient_dataset.data.targets
        mask = ~x.isna().any(axis=1)    # ho 4 missing values che elimino
        x = x[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        x['Gender'] = x['Gender'].map({'Male': 1, 'Female': 0})
        x = x.values
        y['Selector'] = y['Selector'].map({1: 0, 2: 1})
        y = y.astype(int)
        y = np.array(y)
        y = y.ravel()

    elif ds_name == 'image':
        image_segmentation = fetch_openml(name="segment", version=1, as_frame=True)
        x = image_segmentation.data
        y = image_segmentation.target
        constant_columns = [col for col in x.columns if x[col].nunique() == 1]
        x = x.drop(columns = constant_columns)
        y = y.replace({'brickface': 0, 'cement': 1, 'foliage': 2, 'grass': 3, 'path': 4, 'sky': 5, 'window': 6}) 
        y = y.astype(int)
        y = y.values
    
    elif ds_name == 'tic_tac_toe':
        tic_tac_toe_endgame = fetch_ucirepo(id=101) 
        x = tic_tac_toe_endgame.data.features 
        y = tic_tac_toe_endgame.data.targets 
        x = pd.get_dummies(x, drop_first=False)
        x = x.astype(int)
        y = y.replace({'negative': 0, 'positive': 1})
        y = y.astype(int)
        y = y.values

    elif ds_name == 'libras':
        libras = pd.read_csv('DATA/movement_libras.data', header=None)
        x = libras.iloc[:, :-1]
        y = libras.iloc[:, -1]
        y = y.replace({i: i-1 for i in range(1, 16)})
        y = y.astype(int)
        y = y.values

    elif ds_name == 'gas_sensor':
        files = glob.glob('DATA/gas_sensor_array_drift_dataset/*.dat')
        dfs = []
        for file in files:
            df = pd.read_csv(file, header=None, sep='\\s+')
            dfs.append(df)
        gas_sensor = pd.concat(dfs, ignore_index=True)
        x_raw = gas_sensor.iloc[:, 1:]
        x = np.zeros_like(x_raw.values, dtype=float)
        for i in range(x_raw.shape[0]):
            for j in range(x_raw.shape[1]):
                x[i, j] = float(x_raw.iat[i, j].split(":")[1])
        y = gas_sensor.iloc[:, 0].astype(int)
        y = y.replace({i: i-1 for i in range(1, 7)})
        y = y.values

    elif ds_name =='musk1':
        musk_version_1 = fetch_ucirepo(id=74) 
        x = musk_version_1.data.features 
        y = musk_version_1.data.targets 
        x = x.iloc[:, 2:]  # le prime due colonne sono il nome della molecola
        y = np.array(y)
        y = y.ravel()
    
    elif ds_name == 'musk2':
        musk_version_2 = fetch_ucirepo(id=75) 
        x = musk_version_2.data.features 
        y = musk_version_2.data.targets 
        y = np.array(y)
        y = y.ravel()

    elif ds_name == 'coil':
        coil2000 = fetch_openml(name='coil2000', version=1, as_frame=True)
        x = coil2000.data
        y = coil2000.target
        x = pd.get_dummies(x, columns=['MOSHOOFD'], drop_first=False)
        x = x.astype(int)
        y = y.astype(int)
        y = y.values
        
    elif ds_name == 'myocardial':
        myocardial_infarction_complications = fetch_ucirepo(id=579) 
        x = myocardial_infarction_complications.data.features     # scelgo di tenere tutte le colonne per predirre una complicazione in y a 72h dal ricovero
        y = myocardial_infarction_complications.data.targets 
        continuous_columns = ['AGE', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'K_BLOOD', 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD',
                            'KFK_BLOOD', 'L_BLOOD', 'ROE', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_2_n', 'NOT_NA_3_n']
        categorical_cols = [col for col in x.columns if col not in continuous_columns]
        x_continuous = x[continuous_columns].fillna(x[continuous_columns].mean())
        x_categoric = x[categorical_cols].fillna(x[categorical_cols].mode().iloc[0])
        x = pd.concat([x_continuous, x_categoric], axis=1)
        y = y['FIBR_PREDS']     # ci sono 12 possibili variabili target da studiare, consideriamo atrial fibrillation
        y = y.astype(int)
        y = y.values

    elif ds_name == 'optical':
        optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
        x = optical_recognition_of_handwritten_digits.data.features 
        y = optical_recognition_of_handwritten_digits.data.targets 
        constant_columns = [col for col in x.columns if x[col].nunique() == 1] # 2 colonne hanno sempre 0 per tutte le istanze
        x = x.drop(columns = constant_columns)
        y = np.array(y)
        y = y.ravel()

    elif ds_name == 'cancer':
        lung_cancer = fetch_ucirepo(id=62)  
        x = lung_cancer.data.features 
        y = lung_cancer.data.targets
        x = x.fillna(x.mode().iloc[0])  # ho 5 missing values che sostituisco con la moda
        new_x = pd.DataFrame(index=x.index)
        for col in x.columns:
            vals = sorted(x[col].dropna().unique())
            if vals == [0,1]:
                new_x[col] = x[col]
            elif vals == [1,2]:
                new_x[col] = x[col].map({1:0, 2:1})
            else:
                dummies = pd.get_dummies(x[col], prefix=col, dtype=int)
                new_x = pd.concat([new_x, dummies], axis=1)
        x = new_x
        y = y.replace({1: 0, 2: 1, 3: 2})
        y = np.array(y)
        y = y.ravel()

    elif ds_name == 'darwin':
        darwin = fetch_ucirepo(id=732)  
        x = darwin.data.features 
        y = darwin.data.targets 
        x = x.drop(columns=['ID'])  # valori tutti diversi per ogni riga
        y = y.replace({'P': 0, 'H': 1})
        y = np.array(y)
        y = y.ravel()

    # elif ds_name == 'disease':
    #     disease = pd.read_csv('DATA/parkinson_disease_speech_features.csv', header=1)
    #     x = disease.iloc[:, :-1]
    #     y = disease.iloc[:, -1]
    #     x = x.drop(columns=['id'])
    #     scaler = MinMaxScaler()
    #     x = scaler.fit_transform(x)
    #     y = y.astype(int)
    #     y = y.values
    
    elif ds_name == 'gait':
        mat = scipy.io.loadmat('DATA/PersonGaitDataSet.mat')
        x = mat['X']
        y = mat['Y'].ravel()
        x = pd.DataFrame(x)
        x = x.fillna(x.mean())
        y = y.astype(int)

    # elif ds_name == 'secom':
    #     x = pd.read_csv('DATA/secom_dataset/secom.data', sep='\\s+', header=None)
    #     y = pd.read_csv('DATA/secom_dataset/secom_labels.data', sep='\\s+', header=None)
    #     x = x.fillna(x.mean())
    #     scaler = MinMaxScaler()
    #     x = scaler.fit_transform(x)
    #     y = y[0]
    #     y = y.replace({-1: 0, 1: 1})
    #     y = y.values

    elif ds_name == 'naticus':
        naticusdroid_android_permissions = fetch_ucirepo(id=722) 
        x = naticusdroid_android_permissions.data.features 
        y = naticusdroid_android_permissions.data.targets 
        y = y['Result']
        y = y.values

    # elif ds_name == 'madelon':
    #     x_train = pd.read_csv('DATA/madelon/madelon_train.data', sep=r'\s+', header=None)
    #     y_train = pd.read_csv('DATA/madelon/madelon_train.labels', header=None)
    #     x_valid = pd.read_csv('DATA/madelon/madelon_valid.data', sep=r'\s+', header=None)
    #     y_valid = pd.read_csv('DATA/madelon/madelon_valid.labels', header=None)
    #     x = pd.concat([x_train, x_valid], axis=0)
    #     y = pd.concat([y_train, y_valid], axis=0)
    #     scaler = MinMaxScaler()
    #     x = scaler.fit_transform(x)
    #     y = y[0]
    #     y = y.replace({-1: 0, 1: 1})
    #     y = y.values

    # elif ds_name == 'connect':
    #     connect_4 = fetch_ucirepo(id=26) 
    #     x = connect_4.data.features 
    #     y = connect_4.data.targets 
    #     x = pd.get_dummies(x, drop_first=False) # tutte le colonne hanno 3 categorie
    #     x = x.astype(int)
    #     scaler = MinMaxScaler()
    #     x = scaler.fit_transform(x)
    #     y = y['class']
    #     y = y.map({'win': 0, 'loss': 1, 'draw': 2})
    #     y = y.values

    elif ds_name == 'scene':
        scene = fetch_openml(name='scene', version=1, as_frame=True)
        x = scene.data
        y = scene.target
        x = x.iloc[:,:-5] # le ultime 5 colonne sono altre possibili variabili target
        y = y.astype(int)
        y = y.values    

    elif ds_name =='drug':
        drug = pd.read_csv('DATA/DIA_trainingset_RDKit_descriptors.csv')
        x = drug.iloc[:,1:]
        x = x.drop(columns=['SMILES'])  # colonna id
        y = drug.iloc[:,0]
        y = y.astype(int)
        y = y.values

    elif ds_name == 'voice':
        x = pd.read_excel('DATA/LSVT_voice_rehabilitation.xlsx', sheet_name='Data')
        y = pd.read_excel('DATA/LSVT_voice_rehabilitation.xlsx', sheet_name='Binary response')
        y = y['Binary class 1=acceptable, 2=unacceptable']
        y = y.map({1: 1, 2: 0})
        y = y.values

    x = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    protocol_splits = list(repeated_holdout_splits(
        x, y, n_repeats=10, test_size=0.1, seed=42,
        predefined_train_size=predefined_train_size
    ))

    for i, (train_index, test_index) in enumerate(protocol_splits):

        if i < fold_start or i >= fold_end:
            continue 

        # APPROX CON MINIMO NUMERO DI COALIZIONI (2n+2)
        folder = ds_name+'/approx_min_number_coal/'+str(i)+'/'
        basepath = ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v/'
        path = ds_name+'/approx_min_number_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        X_train = x[train_index]
        X_test = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        X_train, X_test = scale_train_test(X_train, X_test)

        np.save(folder+"X_train.npy", X_train)
        np.save(folder+"X_test.npy", X_test)
        np.save(folder+"Y_train.npy", Y_train)
        np.save(folder+"Y_test.npy", Y_test)

        nvar = x.shape[1]
        reduced_matrix = cgt.reduced_matrix(nvar)
        ncoal = reduced_matrix.shape[0] 
        np.save(path+'matrix.npy', reduced_matrix)

        # v 
        X_train_train, X_val, Y_train_train, Y_val = validation_split(X_train, Y_train, i) 
        np.save(folder+"X_train_train.npy", X_train_train)
        np.save(folder+"X_val.npy", X_val)
        np.save(folder+"Y_train_train.npy", Y_train_train)
        np.save(folder+"Y_val.npy", Y_val)
        v_tot_red = ex.generate_v_traintest_already_divided(model_string, reduced_matrix, X_train_train, X_val, Y_train_train, Y_val, store_values=True, path=basepath)
        v_acc_red = np.array([v_tot_red[i][0] for i in range(ncoal)])
        v_f1_red = np.array([v_tot_red[i][1] for i in range(ncoal)])
        ex.execute_approximation(basepath, model_string, reduced_matrix, v_acc_red, v_f1_red)

        # v most freq 
        basepath = ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v_most_freq/'
        path = ds_name+'/approx_min_number_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', reduced_matrix)
        value_baseline_mostfreq = ex.generate_baseline_most_frequent(X_train_train, Y_train_train, path)
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)
        ex.execute_approximation(basepath, model_string, reduced_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red)

        # v stratified
        basepath = ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v_stratified/'
        path = ds_name+'/approx_min_number_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', reduced_matrix)
        value_baseline_strat = ex.generate_baseline_stratified(X_train_train, Y_train_train, path)
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)
        ex.execute_approximation(basepath, model_string, reduced_matrix, v_acc_strat_red, v_f1_strat_red)

        nfeat_list = [1]
        nfeat_list += list(range(5, int(nvar/2)+1, 5))
        if ds_name == 'hepatitis' or ds_name == 'image':
            nfeat_list = [1,5,10]
        if ds_name == 'drug':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
        if ds_name == 'splice' or ds_name == 'scene' or ds_name == 'voice':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150]
        if ds_name == 'darwin' or ds_name == 'gait':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200]
        if ds_name == 'madelon':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250] 
        if ds_name == 'secom':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300]
        if ds_name == 'disease':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350] 
        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx(folder=folder, ds_name=ds_name, model_string=model_string, nfeat=nfeat)

        # clean della memoria
        del X_train, X_test, Y_train, Y_test
        del X_train_train, X_val, Y_train_train, Y_val
        del reduced_matrix
        del v_tot_red, v_acc_red, v_f1_red
        del v_acc_mostfreq_red, v_f1_mostfreq_red
        del v_acc_strat_red, v_f1_strat_red
        gc.collect()

        # APPROX CON NUMERO RANDOM UNIFORME DI COALIZIONI (2n+2+2n)
        folder = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'
        basepath = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'+'v/'
        path = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        X_train = x[train_index]
        X_test = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        X_train, X_test = scale_train_test(X_train, X_test)

        np.save(folder+"X_train.npy", X_train)
        np.save(folder+"X_test.npy", X_test)
        np.save(folder+"Y_train.npy", Y_train)
        np.save(folder+"Y_test.npy", Y_test)

        nvar = x.shape[1]
        reduced_random_matrix = cgt.reduced_random_matrix(nvar)
        ncoal = reduced_random_matrix.shape[0] 
        np.save(path+'matrix.npy', reduced_random_matrix)

        # v 
        X_train_train, X_val, Y_train_train, Y_val = validation_split(X_train, Y_train, i)
        np.save(folder+"X_train_train.npy", X_train_train)
        np.save(folder+"X_val.npy", X_val)
        np.save(folder+"Y_train_train.npy", Y_train_train)
        np.save(folder+"Y_val.npy", Y_val)
        v_tot_red = ex.generate_v_traintest_already_divided(model_string, reduced_random_matrix, X_train_train, X_val, Y_train_train, Y_val, store_values=True, path=basepath)
        v_acc_red = np.array([v_tot_red[i][0] for i in range(ncoal)])
        v_f1_red = np.array([v_tot_red[i][1] for i in range(ncoal)])
        ex.execute_approximation(basepath, model_string, reduced_random_matrix, v_acc_red, v_f1_red)

        # v most freq 
        basepath = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'+'v_most_freq/'
        path = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', reduced_random_matrix)
        value_baseline_mostfreq = ex.generate_baseline_most_frequent(X_train_train, Y_train_train, path)
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)
        ex.execute_approximation(basepath, model_string, reduced_random_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red)

        # v stratified
        basepath = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'+'v_stratified/'
        path = ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', reduced_random_matrix)
        value_baseline_strat = ex.generate_baseline_stratified(X_train_train, Y_train_train, path)
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)
        ex.execute_approximation(basepath, model_string, reduced_random_matrix, v_acc_strat_red, v_f1_strat_red)

        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx(folder=folder, ds_name=ds_name, model_string=model_string, nfeat=nfeat)

        # clean della memoria
        del X_train, X_test, Y_train, Y_test
        del X_train_train, X_val, Y_train_train, Y_val
        del reduced_random_matrix
        del v_tot_red, v_acc_red, v_f1_red
        del v_acc_mostfreq_red, v_f1_mostfreq_red
        del v_acc_strat_red, v_f1_strat_red
        gc.collect()
        
        # APPROX CON COALIZIONI DI KERNELSHAP (2n+2048+2)
        folder = ds_name+'/approx_kernelshap_coal/'+str(i)+'/'
        basepath = ds_name+'/approx_kernelshap_coal/'+str(i)+'/'+'v/'
        path = ds_name+'/approx_kernelshap_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        X_train = x[train_index]
        X_test = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        X_train, X_test = scale_train_test(X_train, X_test)

        np.save(folder+"X_train.npy", X_train)
        np.save(folder+"X_test.npy", X_test)
        np.save(folder+"Y_train.npy", Y_train)
        np.save(folder+"Y_test.npy", Y_test)

        nvar = x.shape[1]
        kernelshap_matrix = cgt.sampling_coal_kernelshap(nvar, 2*nvar+2048, seed=42)
        ncoal = kernelshap_matrix.shape[0] 
        np.save(path+'matrix.npy', kernelshap_matrix)

        # v 
        X_train_train, X_val, Y_train_train, Y_val = validation_split(X_train, Y_train, i)
        np.save(folder+"X_train_train.npy", X_train_train)
        np.save(folder+"X_val.npy", X_val)
        np.save(folder+"Y_train_train.npy", Y_train_train)
        np.save(folder+"Y_val.npy", Y_val)
        v_tot_red = ex.generate_v_traintest_already_divided(model_string, kernelshap_matrix, X_train_train, X_val, Y_train_train, Y_val, store_values=True, path=basepath)
        v_acc_red = np.array([v_tot_red[i][0] for i in range(ncoal)])
        v_f1_red = np.array([v_tot_red[i][1] for i in range(ncoal)])
        ex.execute_approximation(basepath, model_string, kernelshap_matrix, v_acc_red, v_f1_red)

        # v most freq 
        basepath = ds_name+'/approx_kernelshap_coal/'+str(i)+'/'+'v_most_freq/'
        path = ds_name+'/approx_kernelshap_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', kernelshap_matrix)
        value_baseline_mostfreq = ex.generate_baseline_most_frequent(X_train_train, Y_train_train, path)
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)
        ex.execute_approximation(basepath, model_string, kernelshap_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red)

        # v stratified
        basepath = ds_name+'/approx_kernelshap_coal/'+str(i)+'/'+'v_stratified/'
        path = ds_name+'/approx_kernelshap_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', kernelshap_matrix)
        value_baseline_strat = ex.generate_baseline_stratified(X_train_train, Y_train_train, path)
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)
        ex.execute_approximation(basepath, model_string, kernelshap_matrix, v_acc_strat_red, v_f1_strat_red)

        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx(folder=folder, ds_name=ds_name, model_string=model_string, nfeat=nfeat)

        # clean della memoria
        del X_train, X_test, Y_train, Y_test
        del X_train_train, X_val, Y_train_train, Y_val
        del kernelshap_matrix
        del v_tot_red, v_acc_red, v_f1_red
        del v_acc_mostfreq_red, v_f1_mostfreq_red
        del v_acc_strat_red, v_f1_strat_red
        gc.collect()

        # APPROX CON COALIZIONI DI BANZHAF UNIFORMI (2n+2048+2)
        folder = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'
        basepath = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'+'v/'
        path = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        X_train = x[train_index]
        X_test = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        X_train, X_test = scale_train_test(X_train, X_test)

        np.save(folder+"X_train.npy", X_train)
        np.save(folder+"X_test.npy", X_test)
        np.save(folder+"Y_train.npy", Y_train)
        np.save(folder+"Y_test.npy", Y_test)

        nvar = x.shape[1]
        banzuniforme_matrix = cgt.sampling_banzhaf_hybrid(nvar, 2*nvar+2048, seed=42)
        ncoal = banzuniforme_matrix.shape[0] 
        np.save(path+'matrix.npy', banzuniforme_matrix)

        # v 
        X_train_train, X_val, Y_train_train, Y_val = validation_split(X_train, Y_train, i)
        np.save(folder+"X_train_train.npy", X_train_train)
        np.save(folder+"X_val.npy", X_val)
        np.save(folder+"Y_train_train.npy", Y_train_train)
        np.save(folder+"Y_val.npy", Y_val)
        v_tot_red = ex.generate_v_traintest_already_divided(model_string, banzuniforme_matrix, X_train_train, X_val, Y_train_train, Y_val, store_values=True, path=basepath)
        v_acc_red = np.array([v_tot_red[i][0] for i in range(ncoal)])
        v_f1_red = np.array([v_tot_red[i][1] for i in range(ncoal)])
        ex.execute_approximation(basepath, model_string, banzuniforme_matrix, v_acc_red, v_f1_red)

        # v most freq 
        basepath = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'+'v_most_freq/'
        path = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', banzuniforme_matrix)
        value_baseline_mostfreq = ex.generate_baseline_most_frequent(X_train_train, Y_train_train, path)
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)
        ex.execute_approximation(basepath, model_string, banzuniforme_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red)

        # v stratified
        basepath = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'+'v_stratified/'
        path = ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        np.save(path+'matrix.npy', banzuniforme_matrix)
        value_baseline_strat = ex.generate_baseline_stratified(X_train_train, Y_train_train, path)
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)
        ex.execute_approximation(basepath, model_string, banzuniforme_matrix, v_acc_strat_red, v_f1_strat_red)

        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx(folder=folder, ds_name=ds_name, model_string=model_string, nfeat=nfeat)

        # clean della memoria
        del X_train, X_test, Y_train, Y_test
        del X_train_train, X_val, Y_train_train, Y_val
        del banzuniforme_matrix
        del v_tot_red, v_acc_red, v_f1_red
        del v_acc_mostfreq_red, v_f1_mostfreq_red
        del v_acc_strat_red, v_f1_strat_red
        gc.collect()


    if fold_end == 10:
        for nfeat in nfeat_list:
            ex.compute_mean_acc_std_single_ds_approx(dataset_name=ds_name, dim=f'n{nfeat}', case='approx_min_number_coal', model_string=model_string)
            ex.compute_mean_acc_std_single_ds_approx(dataset_name=ds_name, dim=f'n{nfeat}', case='approx_random_number_2n_coal', model_string=model_string)
            ex.compute_mean_acc_std_single_ds_approx(dataset_name=ds_name, dim=f'n{nfeat}', case='approx_kernelshap_coal', model_string=model_string)
            ex.compute_mean_acc_std_single_ds_approx(dataset_name=ds_name, dim=f'n{nfeat}', case='approx_banzhafuniforme_coal', model_string=model_string)
        ex.compute_mean_std_time(dataset_name=ds_name, case='approx_min_number_coal', string=model_string)
        ex.compute_mean_std_time(dataset_name=ds_name, case='approx_random_number_2n_coal', string=model_string)
        ex.compute_mean_std_time(dataset_name=ds_name, case='approx_kernelshap_coal', string=model_string)
        ex.compute_mean_std_time(dataset_name=ds_name, case='approx_banzhafuniforme_coal', string=model_string)
 
        ex.compute_mean_acc_std_all_features(ds_name=ds_name, model_string=model_string)














#### CASO PER CALCOLO LINF / L1 CON PESI DI SHAP -- 'RESULTS_wL1_wLinf/'

def generation_kfold_approx_wl1_wlinf(ds_name='sonar', model_string='lr', fold_start=0, fold_end=10):

    ex.set_seed(42)

    for i in [0,1,2,3,4,5,6,7,8,9]:

        if i < fold_start or i >= fold_end:
            continue 

    # APPROX CON MINIMO NUMERO DI COALIZIONI (2n+2)
        folder = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/'
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        reduced_matrix = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/v/'+model_string+'/matrix.npy')
        v_acc_red = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/v/'+model_string+'/coalitions_acc.npy')
        v_f1_red = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/v/'+model_string+'/coalitions_f1.npy')
        ex.compute_wLinf_wL1(reduced_matrix, v_acc_red, v_f1_red, path, model_string)

        # v most freq 
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v_most_freq/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_mostfreq = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/v_most_freq/'+model_string+'/value_baseline_most_freq.npy')
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red, path, model_string)


        # v stratified
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/'+'v_stratified/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_min_number_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_strat = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/v_stratified/'+model_string+'/value_baseline_stratified.npy')
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_strat_red, v_f1_strat_red, path, model_string)


        nfeat_list = [1]
        nvar = reduced_matrix.shape[1] 
        nfeat_list += list(range(5, int(nvar/2)+1, 5))
        if ds_name == 'hepatitis' or ds_name == 'image':
            nfeat_list = [1,5,10]
        if ds_name == 'drug':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
        if ds_name == 'splice' or ds_name == 'scene' or ds_name == 'voice':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150]
        if ds_name == 'darwin' or ds_name == 'gait':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200]
        if ds_name == 'madelon':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250] 
        if ds_name == 'secom':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300]
        if ds_name == 'disease':
            nfeat_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 250, 300, 350] 
        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx_wLinf_wL1(folder, ds_name, model_string, nfeat)
            # acc_row = []
            # X_train = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/X_train.npy')
            # X_test = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/X_test.npy')
            # Y_train = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/Y_train.npy')
            # Y_test = np.load(ds_name+'/approx_min_number_coal/'+str(i)+'/Y_test.npy')
            # wlinf_values = np.load(folder+'v_most_freq/'+model_string+'/'+'wlinf_values_acc.npy')
            # top_idx_linf = np.argsort(wlinf_values)[-nfeat:]
            # X_train_imp = X_train[:,top_idx_linf]
            # X_test_imp = X_test[:,top_idx_linf]
            # model = ex.return_model(folder, model_string)
            # model.fit(X_train_imp, Y_train)
            # y_pred = model.predict(X_test_imp)
            # acc_linf = accuracy_score(Y_test, y_pred)
            # acc_row.append(acc_linf)
            # methods = ['solution linf weights']
            # dataset_name = ds_name
            # acc_df = pd.DataFrame([acc_row], columns=methods)
            # acc_df.to_csv(folder+'row_accuracy_retrain_'+model_string+'_n'+str(nfeat)+".csv", index = False)


    # APPROX CON NUMERO RANDOM 
        folder = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'+'v/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        reduced_matrix = np.load(ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v/'+model_string+'/matrix.npy')

        v_acc_red = np.load(ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v/'+model_string+'/coalitions_acc.npy')
        v_f1_red = np.load(ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v/'+model_string+'/coalitions_f1.npy')

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_red, v_f1_red, path, model_string)


        # v most freq 
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'+'v_most_freq/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_mostfreq = np.load(ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v_most_freq/'+model_string+'/value_baseline_most_freq.npy')
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red, path, model_string)


        # v stratified
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/'+'v_stratified/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_strat = np.load(ds_name+'/approx_random_number_2n_coal/'+str(i)+'/v_stratified/'+model_string+'/value_baseline_stratified.npy')
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_strat_red, v_f1_strat_red, path, model_string)


        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx_wLinf_wL1(folder, ds_name, model_string, nfeat)


    # APPROX CON KERNLESHAP COAL
        folder = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/'
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/'+'v/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        reduced_matrix = np.load(ds_name+'/approx_kernelshap_coal/'+str(i)+'/v/'+model_string+'/matrix.npy')

        v_acc_red = np.load(ds_name+'/approx_kernelshap_coal/'+str(i)+'/v/'+model_string+'/coalitions_acc.npy')
        v_f1_red = np.load(ds_name+'/approx_kernelshap_coal/'+str(i)+'/v/'+model_string+'/coalitions_f1.npy')

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_red, v_f1_red, path, model_string)


        # v most freq 
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/'+'v_most_freq/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_mostfreq = np.load(ds_name+'/approx_kernelshap_coal/'+str(i)+'/v_most_freq/'+model_string+'/value_baseline_most_freq.npy')
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red, path, model_string)


        # v stratified
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/'+'v_stratified/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_kernelshap_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_strat = np.load(ds_name+'/approx_kernelshap_coal/'+str(i)+'/v_stratified/'+model_string+'/value_baseline_stratified.npy')
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_strat_red, v_f1_strat_red, path, model_string)


        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx_wLinf_wL1(folder, ds_name, model_string, nfeat)


    # APPROX CON BANZHAF UNIFORME
        folder = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'+'v/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v/'+model_string+'/'

        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        reduced_matrix = np.load(ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v/'+model_string+'/matrix.npy')

        v_acc_red = np.load(ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v/'+model_string+'/coalitions_acc.npy')
        v_f1_red = np.load(ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v/'+model_string+'/coalitions_f1.npy')

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_red, v_f1_red, path, model_string)


        # v most freq 
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'+'v_most_freq/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v_most_freq/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_mostfreq = np.load(ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v_most_freq/'+model_string+'/value_baseline_most_freq.npy')
        v_acc_mostfreq_red = v_acc_red - value_baseline_mostfreq
        v_acc_mostfreq_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_mostfreq_red)
        v_f1_mostfreq_red = v_f1_red - value_baseline_mostfreq
        v_f1_mostfreq_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_mostfreq_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_mostfreq_red, v_f1_mostfreq_red, path, model_string)


        # v stratified
        basepath = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/'+'v_stratified/'
        path = 'RESULTS_wL1_wLinf/'+ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v_stratified/'+model_string+'/'
        os.makedirs(basepath, exist_ok=True)
        os.makedirs(path, exist_ok=True)
        value_baseline_strat = np.load(ds_name+'/approx_banzhafuniforme_coal/'+str(i)+'/v_stratified/'+model_string+'/value_baseline_stratified.npy')
        v_acc_strat_red = v_acc_red - value_baseline_strat
        v_acc_strat_red[0] = 0
        np.save(path+"coalitions_acc.npy", v_acc_strat_red)
        v_f1_strat_red = v_f1_red - value_baseline_strat
        v_f1_strat_red[0] = 0
        np.save(path+"coalitions_f1.npy", v_f1_strat_red)

        ex.compute_wLinf_wL1(reduced_matrix, v_acc_strat_red, v_f1_strat_red, path, model_string)


        for nfeat in nfeat_list:
            ex.compute_accuracy_retrain_approx_wLinf_wL1(folder, ds_name, model_string, nfeat)


    if fold_end == 10:
        for coal_type in ['approx_min_number_coal', 'approx_random_number_2n_coal', 'approx_kernelshap_coal', 'approx_banzhafuniforme_coal']:
            for nfeat in nfeat_list:
                subfolders=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                df_list = []
                for subfolder in subfolders:
                    df = pd.read_csv('RESULTS_wL1_wLinf/'+ds_name+'/'+coal_type+'/'+subfolder+'/row_accuracy_retrain_'+model_string+'_'+f'n{nfeat}'+'.csv')
                    df_list.append(df)
                concatenated_df = pd.concat(df_list)
                mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
                std_df = concatenated_df.std(numeric_only=True).to_frame().T
                final_df = pd.DataFrame(columns=['ds_name', 
                                                 'mean_acc_l1_w', 'std_l1_w',
                                                'mean_acc_linf_w', 'std_linf_w'])
                final_df.loc[0] = [ds_name, 
                                   mean_df['solution l1 weights'][0], std_df['solution l1 weights'][0],
                                    mean_df['solution linf weights'][0], std_df['solution linf weights'][0]]
                final_df.to_csv('RESULTS_wL1_wLinf/'+ds_name+'/'+coal_type+'/row_mean_acc_std_retrain_'+model_string+'_'+f'n{nfeat}'+'.csv', index=False)

            subfolders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            v_folder_list = ['v', 'v_most_freq', 'v_stratified']
            for v_folder in v_folder_list:
                df_list = []
                for subfolder in subfolders:
                    df = pd.read_csv('RESULTS_wL1_wLinf/'+ds_name+'/'+coal_type+'/'+subfolder+'/'+v_folder+'/'+model_string+'/execution_time_cgt_'+model_string+'.csv')
                    df_list.append(df)
                concatenated_df = pd.concat(df_list)
                mean_df = concatenated_df.mean(numeric_only=True).to_frame().T
                std_df = concatenated_df.std(numeric_only=True).to_frame().T
                final_df = pd.DataFrame(columns=['ds_name', 
                                                 'mean_time_wl1_acc', 'std_wl1_acc', 'mean_time_wl1_f1', 'std_wl1_f1',
                                                'mean_time_wlinf_acc', 'std_wlinf_acc', 'mean_time_wlinf_f1', 'std_wlinf_f1'])
                final_df.loc[0] = [ds_name, 
                        mean_df['wl1_acc'][0], std_df['wl1_acc'][0],
                        mean_df['wl1_f1'][0], std_df['wl1_f1'][0],
                        mean_df['wlinf_acc'][0], std_df['wlinf_acc'][0],
                        mean_df['wlinf_f1'][0], std_df['wlinf_f1'][0]]
                final_df.to_csv('RESULTS_wL1_wLinf/'+ds_name+'/'+coal_type+'/row_mean_std_time_cgt_'+v_folder+'_'+model_string+'.csv', index=False)


