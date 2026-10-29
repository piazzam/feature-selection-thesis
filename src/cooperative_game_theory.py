import numpy as np
import gurobipy as gp
from scipy.special import comb
import itertools
import copy
import random
from scipy.special import binom
from math import factorial

### COMPUTE MATRIX ###

# coal_player_matrix_fast function

def coal_player_matrix_fast(nvar):
    nrows = 2 ** nvar
    matrix = np.zeros((nrows, nvar), dtype=int)
    for i in range(nvar):
        block_size = 2 ** (nvar - i - 1)
        repeat_count = nrows // (2 * block_size)
        matrix[:, i] = np.tile(np.repeat([0, 1], block_size).astype(int), repeat_count)
    
    return matrix


# check_binary_matrix function

def check_binary_matrix(nvar, matrix):
    vector = np.arange(0, 2**nvar)
    binary_matrix = np.array([list(format(x, f'0{nvar}b')) for x in vector], dtype=int)
    check = True
    for i in range(len(binary_matrix)):
        for j in range(len(binary_matrix[i])):
            if binary_matrix[i][j] != matrix[i][j]:
                check = False
                break

    return check


# coal2int function 

def coal2int(coal):
    N = len(coal)
    k = 0
    for i in range(N):
        k = k + coal[i]*2**(N-1-i)
    return k


# int2coal function 

def int2coal(nvar, k):
    coal = np.zeros(nvar, dtype=int)
    for i in range(nvar):
        if k >= 2**(nvar-1-i):
            k = k - 2**(nvar-1-i)
            coal[i] = 1
    return coal




### COMPUTE SHAPLEY VALUE ###

# uso la formula di Charnes et al. senza matrix

def shapley(v, n):
    
    matrix = coal_player_matrix_fast(n)
    
    # S contains the cardinality of each coalition
    S = np.array([0,1])
    for i in range(n-1):
        S = np.concatenate((S, S+1))
    
    weights = []
    for i in range(n-1):
        p = int(comb(n-2,i))
        weights.append(p)
    
    w = [0]
    for i in range(1,2**n-1):
        weight = 1/weights[S[i]-1]
        w.append(weight)
    w.append(0)
    w = np.array(w)
        
    mu = np.zeros(n)
    mu[0] = np.dot( w[2**(n-1):] , v[2**(n-1):]) 
    for p in range(1,n-1):
        mask = matrix[:,p] > 0
        mu[p] = np.dot(w[mask] , v[mask]) 
    mu[n-1] = np.dot(w[1: :2] , v[1: :2])
    M = np.sum(mu)
    sol = mu/(n-1) + v[-1]/n - M/(n*(n-1))
    return sol




### COMPUTE BANZHAF VALUE ###

def banzhaf(v,n):
    banzhaf = np.zeros(n)
    banzhaf[0] = np.sum(v[2**(n-1):]) - np.sum(v[0:2**(n-1)])

    for p in range(1, n-1):
        l = 2**(n-1-p)
        mask = (np.arange(2**n) % (2*l)) < l
        V0 = np.sum(v[mask])
        V1 = np.sum(v[~mask])
        banzhaf[p] = V1 - V0
    banzhaf[n-1] = np.sum(v[1::2]) - np.sum(v[0::2])
    banzhaf = banzhaf / (2**(n-1))

    # multiplicative normalization
    tot = np.sum(banzhaf)
    banzhaf_mult = banzhaf * v[-1] / tot

    # additive normalization
    banzhaf_add = banzhaf + (v[-1] - tot)/n

    return banzhaf, banzhaf_mult, banzhaf_add




### COMPUTE CORE ###

# compute_core_with_bounds function 

def compute_core_with_bounds(matrix, v, path='folder/'):
    nvar = matrix.shape[1]
    ncoals = matrix.shape[0]

    # problem: minimization of e 
    m = gp.Model()
    x = m.addVars(nvar, lb=-np.inf, name="x")
    e = m.addVar(lb=-np.inf, name="e")
    m.setObjective(e, gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == v[-1].item())
    for s in range(1, ncoals-1):         
        m.addConstr(gp.quicksum(matrix[s][i]*x[i] for i in range(nvar)) + e >= v[s].item()) 
    m.optimize() 

    if m.Status == gp.GRB.OPTIMAL:
        e_value = e.X
        x_values = [x[i].X for i in range(nvar)]
        if e_value <= 0:
            print(f"The core is non-empty with e = {e_value} and x = {x_values}")
        else:
            print(f"The core is empty with e = {e_value} and x = {x_values}")
    
    # least core: upper/lower bound x_i
    ublc = np.zeros(nvar)
    lblc = np.zeros(nvar)

    costr_e = m.addConstr(e==e_value)
    m.update()
    for i in range(nvar):
        m.setObjective(x[i], gp.GRB.MINIMIZE)
        m.optimize()
        lblc[i] = x[i].X
        m.setObjective(x[i], gp.GRB.MAXIMIZE)
        m.optimize()
        ublc[i] = x[i].X
    m.remove(costr_e)
    m.update()

    # core: upper/lower bound x_i
    ubc = np.zeros(nvar)
    lbc = np.zeros(nvar)

    if e_value <= 0: 
        costr_e = m.addConstr(e==0)
        m.update()
        for i in range(nvar):
            m.setObjective(x[i], gp.GRB.MINIMIZE)
            m.optimize()
            lbc[i] = x[i].X

            m.setObjective(x[i], gp.GRB.MAXIMIZE)
            m.optimize()
            ubc[i] = x[i].X
        m.remove(costr_e)
        m.update()

    # core = empty: infeasible constraints set
    if e_value > 0:
        m.addConstr(e==0)
        m.update()
        m.setObjective(0, gp.GRB.MINIMIZE)
        m.optimize()
        #if m.Status == gp.GRB.INFEASIBLE:
        #    m.computeIIS()
        #    m.write(path+"model.ilp")
    
    # egalitarian least core
    eg = gp.Model()
    x = eg.addVars(nvar, lb=-np.inf, name="x")
    eg.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE)
    eg.addConstr(gp.quicksum(x[i] for i in range(nvar)) == v[-1].item())
    for s in range(1, ncoals-1):         
        eg.addConstr(gp.quicksum(matrix[s][i]*x[i] for i in range(nvar)) + e_value >= v[s].item()) 
    eg.optimize() 
    egalit_value = [x[i].X for i in range(nvar)]
    
    return e_value, x_values, ublc, lblc, ubc, lbc, egalit_value


# binding_constraints function 

def binding_constraints(matrix, v, x_values, e_value):
    excess = v - np.dot(matrix, x_values) 
    bind_constr = []
    for i in range(len(excess)):
        if abs((e_value - excess[i]) / e_value) < 0.01:  #e_value == diff[i]
            bind_constr.append(i)

    return bind_constr


# shapley_in_leastcore function

def shapley_in_leastcore(matrix, shapley_values, e_value, v):
    shapley_in = True
    index_false = []
    prod = np.dot(matrix, shapley_values)
    
    for i in range(len(matrix)):
        if prod[i] + e_value < v[i]:
            index_false.append(i)
            shapley_in = False
            
    excesses = v - prod
    max_excess = np.max(excesses)
    shap_max_exc = [i for i in range(len(excesses)) if excesses[i] == max_excess]

    return shapley_in, index_false, max_excess, shap_max_exc



### VECTOR IN CORE ###

def vector_in_core(matrix, vector, v):
    vector_in = True
    index_false = []
    prod = np.dot(matrix, vector)

    for i in range(len(matrix)):
        if prod[i] < v[i]:
            index_false.append(i)
            vector_in = False
            
    return vector_in, index_false


### VECTOR IN LEAST CORE ###

def vector_in_leastcore(matrix, vector, e_value, v):
    vector_in = True
    index_false = []
    prod = np.dot(matrix, vector)

    for i in range(len(matrix)):
        if prod[i] + e_value < v[i]:
            index_false.append(i)
            vector_in = False
            
    return vector_in, index_false



### MONOTONICITY ###

def monotonicity(matrix, v):
    ncoals = matrix.shape[0]

    results = []
    is_monotonic = True
    max_diff = [0,0,0]
    NcoalST = 0
    Nmono = 0
    
    for i in range(ncoals):         # coalition S
        for j in range(ncoals):     # coalition T 
            if i != j and all(matrix[i] <= matrix[j]): # S subset T
                NcoalST = NcoalST + 1
                diff = v[i] - v[j]
                results.append([i,j,diff])
                if diff > max_diff[2]:
                    max_diff = [i, j, diff]

                if v[i] > v[j]:
                    is_monotonic = False
                else:
                    Nmono = Nmono + 1
                    
    results = np.array(results, dtype=object)
    results[:, :2] = results[:, :2].astype(int)
    mono_rate = Nmono/NcoalST
    
    return is_monotonic, results, max_diff, mono_rate




### SUPER/SUB ADDITIVITY ###

def super_sub_additive(matrix, v):
    ncoals = matrix.shape[0]
    is_superadd = True
    is_subadd = True
    Ndisjcoal = 0
    Nsuper = 0
    Nsub = 0

    for i in range(1, ncoals-1):         # coalition S
        for j in range(i+1, ncoals):     # coalition T 
            if all(matrix[i] <= 1-matrix[j]): # S, T disjoint
                Ndisjcoal = Ndisjcoal + 1
                union = np.maximum(matrix[i], matrix[j])
                indu = coal2int(union) 
                if v[indu] > v[i]+v[j]:
                    is_subadd = False
                    Nsuper = Nsuper + 1
                else: 
                    if v[indu] < v[i]+v[j]:
                        is_superadd = False
                        Nsub = Nsub + 1
                    else:
                        Nsuper = Nsuper + 1
                        Nsub = Nsub + 1
                        
    super_rate = Nsuper/Ndisjcoal
    sub_rate = Nsub/Ndisjcoal
                
    return is_superadd, is_subadd, super_rate, sub_rate




### VARIAZIONI PROBLEMA LEAST CORE: NORMA L1 E LINF ###

# problema: argmin ||theta(x)||_L1, x(N) = v(N)  
# --->  min \sum_{S in N} |e(S,x)|, x(N)=v(N)  
# --->  min \sum_{S in N} y_s, y_s >= v(S)-x(S), y_s >= x(S)-v(S), x(N)=v(N)

def compute_problem_l1(matrix, v):
    nvar = matrix.shape[1]
    ncoals = matrix.shape[0]

    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")
    y = m.addVars(ncoals, name="y")

    m.setObjective(gp.quicksum(y[s] for s in range(1,ncoals-1)), gp.GRB.MINIMIZE)

    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == v[-1].item())
    for s in range(1, ncoals-1):
        xs = gp.quicksum(matrix[s][i]*x[i] for i in range(nvar))         
        m.addConstr(y[s] >= v[s].item() - xs)
        m.addConstr(y[s] >= xs - v[s].item())
    
    m.optimize()

    if m.Status != gp.GRB.OPTIMAL:
        print(f"Unexpected status: {m.Status}")
        y_opt = np.inf
        ub_l1 = np.full(nvar, np.inf)
        lb_l1 = np.full(nvar, np.inf)
        return y_opt, lb_l1, ub_l1
    
    y_opt = m.ObjVal # y_opt è la somma delle y_s (valore ottimo della funzione obiettivo)
    
    ub_l1 = np.zeros(nvar)
    lb_l1 = np.zeros(nvar)

    bound_constr = m.addConstr(gp.quicksum(y[s] for s in range(1,ncoals-1)) == y_opt)
    m.update()

    # bound
    for i in range(nvar):
        m.setObjective(x[i], gp.GRB.MINIMIZE)
        m.optimize()
        lb_l1[i] = x[i].X
        m.setObjective(x[i], gp.GRB.MAXIMIZE)
        m.optimize()
        ub_l1[i] = x[i].X

    # individuo singolo punto con min \sum x_i
    m.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE)
    m.update()
    m.optimize()
    sol_vector_l1 = [x[i].X for i in range(nvar)]

    return y_opt, lb_l1, ub_l1, sol_vector_l1


# problema: argmin ||theta(x)||_Linf, x(N) = v(N)  
# --->  min max_{S in N} |e(S,x)|, x(N)=v(N)  
# --->  min y, y >= v(S)-x(S), y >= x(S)-v(S), x(N)=v(N)

def compute_problem_linf(matrix, v):
    nvar = matrix.shape[1]
    ncoals = matrix.shape[0]

    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")
    y = m.addVar(name="y")

    m.setObjective(y, gp.GRB.MINIMIZE)

    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == v[-1].item())
    for s in range(1, ncoals-1):
        xs = gp.quicksum(matrix[s][i]*x[i] for i in range(nvar))         
        m.addConstr(y >= v[s].item() - xs)
        m.addConstr(y >= xs - v[s].item())
    
    m.optimize()

    if m.Status != gp.GRB.OPTIMAL:
        print(f"Unexpected status: {m.Status}")
        y_opt = np.inf
        ub_linf = np.full(nvar, np.inf)
        lb_linf = np.full(nvar, np.inf)
        return y_opt, lb_linf, ub_linf

    y_opt = m.ObjVal # y_opt è il valore ottimo del max valore assoluto dell'eccesso
    
    ub_linf = np.zeros(nvar)
    lb_linf = np.zeros(nvar)

    bound_constr = m.addConstr(y == y_opt)
    m.update()

    # bound
    for i in range(nvar):
        m.setObjective(x[i], gp.GRB.MINIMIZE)
        m.optimize()
        lb_linf[i] = x[i].X
        m.setObjective(x[i], gp.GRB.MAXIMIZE)
        m.optimize()
        ub_linf[i] = x[i].X
    
    # individuo singolo punto con min \sum x_i
    m.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE)
    m.update()
    m.optimize()
    sol_vector_linf = [x[i].X for i in range(nvar)]

    return y_opt, lb_linf, ub_linf, sol_vector_linf




### TAU VALUE LEAST CORE ###

def compute_tau_leastcore(v, epsilon, matrix):
    
    nvar = matrix.shape[1]
    ncoals = matrix.shape[0]    

    # grand coalition
    vN = v[ncoals-1]

    # marginal contribution 
    marginal_contr = np.zeros(nvar)
    for j in range(nvar):
        coal_with_j = matrix[ncoals-1].copy()
        coal_with_j[j] = 0
        if ncoals == 2**nvar:
            marginal_contr[j] = vN - v[coal2int(coal_with_j)]   # matrice completa sfrutto indice in binario
        else:
            marginal_contr[j] = vN - v[ncoals - 1 - nvar + j]   # matrice ridotta sfrutto come è costruita

    # upper bound box
    upper_bounds = np.full(nvar, np.inf)
    for i in range(nvar):
        upper_bounds[i] = marginal_contr[i] + epsilon

    # lower bound box
    lower_bounds = np.full(nvar, -np.inf)
    if ncoals == 2**nvar:   # almeno il ciclo for parte da 0 ed è corretto quando matrix è ridotta e non ha la coalizione vuota
        matrix = matrix[1:]
        v = v[1:]
    for s in range(0, matrix.shape[0]-1):
        coal = matrix[s]
        cardinality = np.sum(coal)
        player_in_s = np.where(coal == 1)[0] 

        for i in player_in_s:
            players_not_i = [j for j in player_in_s if j != i]
            sum_Mj = sum(marginal_contr[j] for j in players_not_i)
            bound = v[s] - epsilon * cardinality - sum_Mj
            lower_bounds[i] = max(lower_bounds[i], bound)

    # tau vector: efficient point of the segment between upper/lower bound box
    # y = t * ub + (1-t) * lb
    sum_lb = np.sum(lower_bounds)
    sum_ub = np.sum(upper_bounds)
    t = (vN - sum_lb) / (sum_ub - sum_lb)
    tau_vector = lower_bounds + t * (upper_bounds - lower_bounds)

    return tau_vector











### CASO n GRANDE - APPROSSIMAZIONE ###

# calcolo matrix con coalizioni: tutti 0 + tutti 0 e uno 1 + tutti 1 e uno 0 + grande coalizione
def reduced_matrix(nvar):

    empty_coalition = np.zeros(nvar, dtype=int)
    single_player = np.eye(nvar, dtype=int)
    all_minus_one_player = 1 - single_player
    grand_coalition = np.ones(nvar, dtype=int)

    reduced_matrix = np.vstack((empty_coalition, single_player, all_minus_one_player, grand_coalition))

    return reduced_matrix


def reduced_random_matrix(nvar):
    np.random.seed(42)

    empty_coalition = np.zeros(nvar, dtype=int)
    single_player = np.eye(nvar, dtype=int)
    all_minus_one_player = 1 - single_player
    grand_coalition = np.ones(nvar, dtype=int)

    base_matrix = np.vstack((empty_coalition, single_player, all_minus_one_player, grand_coalition))
    base_set = {tuple(row) for row in base_matrix}

    random_uniform_coal = set()
    while len(random_uniform_coal) < nvar:  #aggiungo 2n coal -> controllo solo n (aggiungo complementare poi)
        coal = np.random.randint(0, 2, size=nvar)
        tuplecoal = tuple(coal)
        complement_tuplecoal = tuple(1-coal)

        if tuplecoal in base_set:
            continue
        if tuplecoal in random_uniform_coal:
            continue
        if complement_tuplecoal in random_uniform_coal:
            continue
        if all(coal == 0) or all(coal == 1):
            continue

        random_uniform_coal.add(tuplecoal)

    extra_matrix = []
    for coalition in random_uniform_coal:
        coal = np.array(coalition)
        complement_coal = 1 - coal
        extra_matrix.append(coal)
        extra_matrix.append(complement_coal)
    random_matrix = np.array(extra_matrix)

    reduced_random_matrix = np.vstack((empty_coalition, random_matrix, single_player, all_minus_one_player, grand_coalition))

    return reduced_random_matrix

# calcolo shapley approssimato come problema di ottimizzazione (no forma chiusa della soluzione)
def compute_shapley_approximation(reduced_matrix, reduced_v):

    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]

    weights = []
    for coal in reduced_matrix:
        cardinality = np.sum(coal)
        if cardinality > 0 and cardinality < nvar:
            weight = 1 / comb(nvar-2, cardinality-1)
        else:
            weight = 0
        weights.append(weight)
    weights = np.array(weights)

    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")

    m.setObjective(gp.quicksum(weights[s]*(gp.quicksum(reduced_matrix[s][i]*x[i] for i in range(nvar)) - reduced_v[s].item())**2 for s in range(1, ncoals-1)), gp.GRB.MINIMIZE)
    
    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == reduced_v[-1].item())

    m.optimize()
    
    shapley_approx = [x[i].X for i in range(nvar)]
    shapley_approx = np.array(shapley_approx)

    return shapley_approx


#  calcolo banzhaf add approssimato come problema di ottimizzazione
def compute_banzhaf_add_approximation(reduced_matrix, reduced_v):
    
    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]

    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")

    m.setObjective(gp.quicksum((gp.quicksum(reduced_matrix[s][i]*x[i] for i in range(nvar)) - reduced_v[s].item())**2 for s in range(1, ncoals-1)), gp.GRB.MINIMIZE)
    
    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == reduced_v[-1].item())

    m.optimize()
    
    banzhaf_add_approx = [x[i].X for i in range(nvar)]
    banzhaf_add_approx = np.array(banzhaf_add_approx)

    return banzhaf_add_approx


#  calcolo L1 / Linf approssimato come problema di ottimizzazione
def compute_solution_l1_approximation(reduced_matrix, reduced_v):
    
    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]

    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")
    y = m.addVars(ncoals, name="y")

    m.setObjective(gp.quicksum(y[s] for s in range(1,ncoals-1)), gp.GRB.MINIMIZE) # min \sum_S y_s = min \sum_S |v(S)-x(S)|

    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == reduced_v[-1].item()) # vincolo efficienza \sum_i x_i = v(N)
    for s in range(1, ncoals-1):
        xs = gp.quicksum(reduced_matrix[s][i]*x[i] for i in range(nvar)) # x(S) = \sum_{i in S} x_i
        m.addConstr(y[s] >= reduced_v[s].item() - xs) # y_s >= v(S) - x(S) serve per linearizzare valore assoluto nel min
        m.addConstr(y[s] >= xs - reduced_v[s].item()) # y_s >= x(S) - v(S) serve per linearizzare valore assoluto nel min
    
    m.optimize()
    
    y_opt = m.ObjVal # ottengo valore ottimo
    bound_constr = m.addConstr(gp.quicksum(y[s] for s in range(1,ncoals-1)) == y_opt)
    m.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE) # tra tutti i vettori che minimizzano la distanza L1 prendo quello con minima norma
    m.update()
    m.optimize()
    solution_l1_approx = [x[i].X for i in range(nvar)]
    solution_l1_approx = np.array(solution_l1_approx)

    return solution_l1_approx


def compute_solution_linf_approximation(reduced_matrix, reduced_v):
    
    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]

    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")
    y = m.addVar(name="y")

    m.setObjective(y, gp.GRB.MINIMIZE)

    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == reduced_v[-1].item())
    for s in range(1, ncoals-1):
        xs = gp.quicksum(reduced_matrix[s][i]*x[i] for i in range(nvar))         
        m.addConstr(y >= reduced_v[s].item() - xs)
        m.addConstr(y >= xs - reduced_v[s].item())
    
    m.optimize()

    y_opt = m.ObjVal
    bound_constr = m.addConstr(y == y_opt)
    m.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE)
    m.update()
    m.optimize()
    solution_linf_approx = [x[i].X for i in range(nvar)]
    solution_linf_approx = np.array(solution_linf_approx)

    return solution_linf_approx


#  calcolo tau value approssimato
def compute_tau_approximation(reduced_matrix, reduced_v, epsilon):
    
    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]    

    vN = reduced_v[ncoals-1]

    marginal_contr = np.zeros(nvar)
    for j in range(nvar):
        coal_with_j = reduced_matrix[ncoals-1].copy()
        coal_with_j[j] = 0
        marginal_contr[j] = vN - reduced_v[ncoals - 1 - nvar + j]

    upper_bounds = np.full(nvar, np.inf)
    for i in range(nvar):
        upper_bounds[i] = marginal_contr[i] + epsilon

    lower_bounds = np.full(nvar, -np.inf)
    for s in range(1, ncoals-1):
        coal = reduced_matrix[s]
        cardinality = np.sum(coal)
        player_in_s = np.where(coal == 1)[0] 

        for i in player_in_s:
            players_not_i = [j for j in player_in_s if j != i]
            sum_Mj = sum(marginal_contr[j] for j in players_not_i)
            bound = reduced_v[s] - epsilon * cardinality - sum_Mj
            lower_bounds[i] = max(lower_bounds[i], bound)

    sum_lb = np.sum(lower_bounds)
    sum_ub = np.sum(upper_bounds)
    t = (vN - sum_lb) / (sum_ub - sum_lb)
    tau_approx = lower_bounds + t * (upper_bounds - lower_bounds)
    tau_approx = np.array(tau_approx)

    return tau_approx








# matrice con coalizioni scelte in base a kernelshap

# n_features = numero di features del dataset
# nsamples = numero di coalizioni che voglio = budget (di default lo settano a 2*nfeatures+2048)
# k = dimensione della coalizione
# non vengono incluse la coalizione vuota e quella completa
# codice shap: https://github.com/shap/shap/blob/master/shap/explainers/_kernel.py riga 417

def sampling_coal_kernelshap(n_features, nsamples, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # definisco la cardinalità ammissibile (sfrutto simmetria considerando solo subset fino 
    # a dimensione (n-1)/2 e per quelli più piccoli genero anche il complementare)
    num_subset_sizes = int(np.ceil((n_features - 1) / 2.0))
    num_paired_subset_sizes = int(np.floor((n_features - 1) / 2.0))

    # pesi di kernelshap w(k)=(n-1)/(k*(n-k))
    weight_vector = np.array([(n_features - 1.0) / (i * (n_features - i)) for i in range(1, num_subset_sizes + 1)])
    weight_vector[:num_paired_subset_sizes] *= 2    # poichè considero sia una coal che la complementare il peso va raddoppiato
    weight_vector /= np.sum(weight_vector)          # normalizzo i pesi

    # costruzioni matrice coalizioni
    num_full_subsets = 0
    num_samples_left = nsamples
    group_inds = np.arange(n_features, dtype='int64')
    mask = np.zeros(n_features)
    remaining_weight_vector = copy.copy(weight_vector)
    masks = []
    n_sample_added = 0
    for subset_size in range(1, num_subset_sizes + 1):  # scorro sulle cardinalità
        nsubsets = binom(n_features, subset_size)
        if subset_size <= num_paired_subset_sizes:
            nsubsets *= 2
        if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:    # se posso permettermi di includere tutte le coalizioni di questa cardinalità senza violare la distribuzione kernel le enumero e riduco il budget residuo num_samples_left
            num_full_subsets += 1
            num_samples_left -= nsubsets
            if remaining_weight_vector[subset_size - 1] < 1.0:
                remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])
            w = weight_vector[subset_size - 1] / binom(n_features, subset_size)
            if subset_size <= num_paired_subset_sizes:
                w /= 2.0
            for inds in itertools.combinations(group_inds, subset_size):
                mask[:] = 0
                mask[np.array(inds, dtype='int64')] = 1
                m = copy.deepcopy(mask)
                masks.append(m)
                n_sample_added += 1
                if subset_size <= num_paired_subset_sizes:
                    mask[:] = np.abs(mask - 1)
                    m_due = copy.deepcopy(mask)
                    masks.append(m_due)
                    n_sample_added += 1
        else:
            break
    # add random samples from what is left of the subset space
    n_sample_left = nsamples - n_sample_added
    if num_full_subsets != num_subset_sizes:
        remaining_weight_vector = copy.copy(weight_vector)
        remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
        remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
        remaining_weight_vector /= np.sum(remaining_weight_vector)
        ind_set = np.random.choice(len(remaining_weight_vector), 4 * n_sample_left, p=remaining_weight_vector)
        ind_set_pos = 0
        while n_sample_left > 0 and ind_set_pos < len(ind_set):
            mask.fill(0.0)
            ind = ind_set[ind_set_pos]
            ind_set_pos += 1
            subset_size = ind + num_full_subsets + 1
            mask[np.random.permutation(n_features)[:subset_size]] = 1
            if np.sum(mask) == 0:
                r = random.randint(0, len(mask - 1))
                mask[r] = 1
            elif np.sum(mask) == n_features:
                r = random.randint(0, len(mask - 1))
                mask[r] = 0
            m = copy.deepcopy(mask)
            masks.append(m)
            n_sample_left -= 1
            if n_sample_left > 0 and subset_size <= num_paired_subset_sizes:
                mask[:] = np.abs(mask - 1)
                m_due = copy.deepcopy(mask)
                masks.append(m_due)
                n_sample_left -= 1
    masks = np.array(masks)

    # riordino coalizioni per coerenza con altre matrici
    cardinality = masks.sum(axis=1)
    idx_single = np.where(cardinality==1)[0]
    idx_allminusone = np.where(cardinality==n_features-1)[0]
    idx_other = np.where((cardinality>1) & (cardinality<n_features-1))[0]
    masks = np.vstack((masks[idx_other], masks[idx_single], masks[idx_allminusone]))

    # aggiungo coalizione vuota e completa
    empty = np.zeros((1, n_features), dtype=int)
    full = np.ones((1, n_features), dtype=int)
    matrix = np.vstack((empty, masks, full))
    
    return matrix




# matrice con coalizioni scelte in base a banzhaf - sampling uniforme
# per avere coerenza con kernelshap passare il numero di features del dataset 
# e nsamples = 2*nfeatures+2048 (numero di coalizioni nella matrice a cui aggiungo vuota e completa))
# def sampling_banzhaf(n_features, nsamples, seed):
#     total_coalitions = 2 ** n_features

#     if nsamples >= (total_coalitions - 2):
#         powers_of_two = 1 << np.arange(n_features - 1, -1, -1)
#         sampled_ints = np.arange(1, total_coalitions - 1)
#         return ((sampled_ints[:, None] & powers_of_two) > 0).astype(int)

#     n_pairs = nsamples // 2
    
#     half_space = 2 ** (n_features - 1)
    
#     valid_pool = np.arange(1, half_space)  #--------------
    
#     rng = np.random.default_rng(seed)
    
#     if n_pairs > len(valid_pool):
#         n_pairs = len(valid_pool)
        
#     sampled_ints = rng.choice(valid_pool, size=n_pairs, replace=False)
    
#     powers_of_two = 1 << np.arange(n_features - 1, -1, -1)
#     masks_base = ((sampled_ints[:, None] & powers_of_two) > 0).astype(int)
    
#     masks_complement = 1 - masks_base
    
#     masks = np.empty((n_pairs * 2, n_features), dtype=int)
#     masks[0::2] = masks_base
#     masks[1::2] = masks_complement
    
#     if nsamples % 2 != 0:
#         remaining_pool = np.setdiff1d(valid_pool, sampled_ints)
#         if len(remaining_pool) > 0:
#             extra_int = rng.choice(remaining_pool)
#             extra_mask = ((extra_int & powers_of_two) > 0).astype(int).reshape(1, -1)
#             masks = np.vstack([masks, extra_mask])
    
#     empty = np.zeros((1, n_features), dtype=int)
#     full = np.ones((1, n_features), dtype=int)
#     matrix = np.vstack((empty, masks, full))
            
#     return matrix
# def sampling_banzhaf_hybrid(n_features, nsamples, seed=42, memory_threshold=26):
#     """
#     Campionamento uniforme Banzhaf SENZA re-inserimento (Paired Sampling).

#     nsamples = numero di coalizioni campionate ESCLUDENDO:
#         - coalizione vuota
#         - coalizione completa

#     Output:
#         matrice di dimensione (nsamples + 2, n_features)
#         con:
#         - prima riga: coalizione vuota
#         - ultima riga: coalizione completa
#     """

#     total_coalitions = 2 ** n_features
#     half_space = 2 ** (n_features - 1)
#     max_valid = total_coalitions - 2  # esclude vuota e piena

#     rng = np.random.default_rng(seed)

#     # ================================================================
#     # 1. Fallback: se chiediamo più sample del possibile
#     # ================================================================
#     if nsamples >= max_valid:
#         powers_of_two = np.int64(1) << np.arange(n_features - 1, -1, -1, dtype=np.int64)
#         sampled_ints = np.arange(1, total_coalitions - 1, dtype=np.int64)

#         masks = ((sampled_ints[:, None] & powers_of_two) > 0).astype(int)

#         empty = np.zeros((1, n_features), dtype=int)
#         full = np.ones((1, n_features), dtype=int)

#         return np.vstack((empty, masks, full))

#     # ================================================================
#     # 2. Paired sampling
#     # ================================================================
#     n_pairs = nsamples // 2

#     if n_features <= memory_threshold:
#         # Metodo RAM (senza duplicati garantiti)
#         valid_pool = np.arange(1, half_space, dtype=np.int64)

#         if n_pairs > len(valid_pool):
#             n_pairs = len(valid_pool)

#         sampled_ints = rng.choice(valid_pool, size=n_pairs, replace=False)

#     else:
#         # Metodo rejection (RAM safe)
#         sampled_ints = rng.integers(1, half_space, size=n_pairs, dtype=np.int64)
#         sampled_ints = np.unique(sampled_ints)

#         while len(sampled_ints) < n_pairs:
#             missing = n_pairs - len(sampled_ints)
#             extra_ints = rng.integers(1, half_space, size=missing, dtype=np.int64)
#             sampled_ints = np.unique(np.concatenate([sampled_ints, extra_ints]))

#     # ================================================================
#     # 3. Conversione in maschere binarie
#     # ================================================================
#     powers_of_two = np.int64(1) << np.arange(n_features - 1, -1, -1, dtype=np.int64)
#     masks_base = ((sampled_ints[:, None] & powers_of_two) > 0).astype(int)

#     # Complementi (paired sampling)
#     masks_complement = 1 - masks_base

#     # Interlacciamento
#     masks = np.empty((n_pairs * 2, n_features), dtype=int)
#     masks[0::2] = masks_base
#     masks[1::2] = masks_complement

#     # ================================================================
#     # 4. Caso dispari
#     # ================================================================
#     if nsamples % 2 != 0:
#         if n_features <= memory_threshold:
#             remaining_pool = np.setdiff1d(valid_pool, sampled_ints)
#             extra_int = rng.choice(remaining_pool)
#         else:
#             while True:
#                 extra_int = rng.integers(1, half_space, dtype=np.int64)
#                 if extra_int not in sampled_ints:
#                     break

#         extra_mask = ((extra_int & powers_of_two) > 0).astype(int).reshape(1, -1)
#         masks = np.vstack([masks, extra_mask])

#     # ================================================================
#     # 5. Aggiunta coalizioni fisse
#     # ================================================================
#     empty = np.zeros((1, n_features), dtype=int)
#     full = np.ones((1, n_features), dtype=int)

#     matrix = np.vstack((empty, masks, full))

#     return matrix

def sampling_banzhaf_hybrid(n_features, nsamples, seed=42):
    rng = np.random.default_rng(seed)

    n_pairs = nsamples // 2

    # genera direttamente vettori binari
    masks_base = rng.integers(0, 2, size=(n_pairs, n_features), dtype=np.int8)

    # filtra vuota e completa
    valid = (masks_base.sum(axis=1) > 0) & (masks_base.sum(axis=1) < n_features)
    masks_base = masks_base[valid]

    # refill (raro)
    while len(masks_base) < n_pairs:
        extra = rng.integers(0, 2, size=(n_pairs, n_features), dtype=np.int8)
        valid = (extra.sum(axis=1) > 0) & (extra.sum(axis=1) < n_features)
        masks_base = np.vstack([masks_base, extra[valid]])

    masks_base = masks_base[:n_pairs]

    # complementi
    masks_complement = 1 - masks_base

    # interlacciamento
    masks = np.empty((n_pairs * 2, n_features), dtype=np.int8)
    masks[0::2] = masks_base
    masks[1::2] = masks_complement

    # caso dispari
    if nsamples % 2 != 0:
        while True:
            extra = rng.integers(0, 2, size=(1, n_features), dtype=np.int8)
            s = extra.sum()
            if 0 < s < n_features:
                masks = np.vstack([masks, extra])
                break

    # aggiunta vuota e completa
    empty = np.zeros((1, n_features), dtype=np.int8)
    full = np.ones((1, n_features), dtype=np.int8)
    matrix = np.vstack((empty, masks, full))

    return matrix




## CALCOLO LINF E L1 CON AGGIUNTA NEL PROBLEMA DI OTTIMIZZAZIONE DEI PESI DI SHAP SECONDO CHARNES

# funzione per Linf con pesi di shap moltiplicati nei vincoli (se pesi sono a 1 allora è equivalente a calcolare Linf normale)
def compute_solution_linf_approximation_weights(reduced_matrix, reduced_v):
    
    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]

    # pesi di charnes: 1 / comb(nvar-2, cardinality-1) per coalizioni di cardinalità k, 0 < k < nvar
    weights_shap = []
    for coal in reduced_matrix:
        cardinality = np.sum(coal)
        if cardinality > 0 and cardinality < nvar:
            weight = 1 / comb(nvar-2, cardinality-1)
        else:
            weight = 0
        weights_shap.append(weight)
    weights_shap = np.array(weights_shap)

    # problema Linf con pesi di shap: min max w(s) |v(S)-x(S)|, x(N) = v(N)
    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")
    y = m.addVar(name="y")

    m.setObjective(y, gp.GRB.MINIMIZE)

    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == reduced_v[-1].item())
    for s in range(1, ncoals-1):
        xs = gp.quicksum(reduced_matrix[s][i]*x[i] for i in range(nvar))         
        m.addConstr(y >= weights_shap[s] * (reduced_v[s].item() - xs) )
        m.addConstr(y >= weights_shap[s] * (xs - reduced_v[s].item()) )
    
    m.optimize()

    y_opt = m.ObjVal
    tol = m.Params.FeasibilityTol   # aggiunta tolleranza per evitare Status=12 (errore numerico e problema non risolvibile)
    bound_constr = m.addConstr(y <= y_opt + tol)
    m.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE)
    m.update()
    m.optimize()
    solution_linf_approx_weights = [x[i].X for i in range(nvar)]
    solution_linf_approx_weights = np.array(solution_linf_approx_weights)

    return solution_linf_approx_weights


# funzione per L1 con pesi di shap moltiplicati nei vincoli (se pesi sono a 1 allora è equivalente a calcolare L1 normale)
def compute_solution_l1_approximation_weights(reduced_matrix, reduced_v):
    
    nvar = reduced_matrix.shape[1]
    ncoals = reduced_matrix.shape[0]

    # pesi di charnes: 1 / comb(nvar-2, cardinality-1) per coalizioni di cardinalità k, 0 < k < nvar
    weights_shap = []
    for coal in reduced_matrix:
        cardinality = np.sum(coal)
        if cardinality > 0 and cardinality < nvar:
            weight = 1 / comb(nvar-2, cardinality-1)
        else:
            weight = 0
        weights_shap.append(weight)
    weights_shap = np.array(weights_shap)

    # problema L1 con pesi di shap: min sum w(s) |v(S)-x(S)|, x(N) = v(N)
    m = gp.Model()

    x = m.addVars(nvar, lb=-np.inf, name="x")
    y = m.addVars(ncoals, name="y")

    m.setObjective(gp.quicksum(y[s] for s in range(1,ncoals-1)), gp.GRB.MINIMIZE) 
    m.addConstr(gp.quicksum(x[i] for i in range(nvar)) == reduced_v[-1].item()) 
    for s in range(1, ncoals-1):
        xs = gp.quicksum(reduced_matrix[s][i]*x[i] for i in range(nvar)) 
        m.addConstr(y[s] >= weights_shap[s] * (reduced_v[s].item() - xs) )
        m.addConstr(y[s] >= weights_shap[s] * (xs - reduced_v[s].item()) )
    
    m.optimize()

    # ----------
    # print("Status =", m.Status)
    # print("SolCount =", m.SolCount)
    # print("ObjVal =", m.ObjVal if m.SolCount else None)
    # if m.Status != gp.GRB.OPTIMAL:
    #     print(reduced_v)
    #     print(reduced_matrix)
    #     m.write("debug.lp")
    #     m.computeIIS()
    #     m.write("debug.ilp")
    #     raise RuntimeError("L1 infeasible")
    # ----------
    
    y_opt = m.ObjVal
    tol = m.Params.FeasibilityTol   # aggiunta tolleranza per evitare Status=12 (errore numerico e problema non risolvibile)
    tol = 1e-2
    bound_constr = m.addConstr(gp.quicksum(y[s] for s in range(1,ncoals-1)) <= y_opt + tol)
    m.setObjective(gp.quicksum((x[i])**2 for i in range(nvar)), gp.GRB.MINIMIZE) 
    m.update()
    m.optimize()
    solution_l1_approx_weights = [x[i].X for i in range(nvar)]
    solution_l1_approx_weights = np.array(solution_l1_approx_weights)

    return solution_l1_approx_weights




