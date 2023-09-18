# %% Extract data for simulations 

# pylint: disable=reportUnboundVariable
import copy
import importlib
import os
import random
import sys
import time

import igraph as ig
import matplotlib
import networkx as nx
import numpy as np
import powerlaw as pl
from scipy.stats import nbinom
from tqdm import tqdm

matplotlib.rc('font', **{'family': 'sans-serif'})

# %%
#N= 100


#%%
start_time = time.time()

# FUNCTION DEFINITION
def Epidemic(attributes_sim, attributes_old, par, adjacency, it):
    """Stochastic, discrete-time compartmental model coupled to a network. 
    Represents the dynamics of an epidemic in which it is possible to quarantine individuals. 
        - Susceptible (S): Healthy state. They can become E with a probability dependent on the 
        number of infected neighbours. 
        - Exposed (E): Incubation state. Individuals who have been infected but are not yet infectious. 
        After a certain incubation period they transition towards P. 
        - Presymptomatic (P): Non-sympthomatic infecitous state. These individuals are infectious but do not show 
        sympthoms. Therefore, they can not be detected and quarantined by the healthcare system. They can only 
        be quarantined through the App strategy. 
        - Infected (I): Sympthomatic infected state. Infectious individuals with symptoms. They can be 
        detected and quarantined until they recover with a certain detection rate. They can recover
        with a certain recovery rate (mu). 
        - Recovered (R): Inmunne state. They have overcome the disease and can not be reinfected. 
        
    Quarantine condition: All states except R can become quarantined. In the quarantine state they can not infect
    or get infected by other nodes. 
    INPUTS: 
        - attributes_sim: list containing the arrays with the attributes for each node. 
        The order of the array must be [states_matrix, gen_time_matrix, R_time_matrix, quarantine_matrix, 
        Inew_matrix, app_matrix, refractory_matrix, threshold_matrix]] 
        - attributes_old: same list but for the previous timepoint. 
        - par: list containing all the parameters for the simulation
        - adjacency: adjacency matrix of layer1. 
        - it: simulation timestep
    OUTPUTS: 
        attributes_sim: Updated version of the attributes list. 
    """

    # Unpack attributes from attributes_old
    state_matrix = attributes_old[0]
    quarantine_matrix = attributes_old[2]
    Inew_matrix = attributes_old[3]

    # EPIDEMIC DYNAMICS 
    # Identify infected nodes
    i_nodes = state_matrix == 3
    p_nodes = state_matrix == 2
    ip_nodes = i_nodes | p_nodes
    not_quar_nodes = quarantine_matrix == 0
    ip_not_quar = ip_nodes & not_quar_nodes
    float_ip_not_quar = (1 * ip_not_quar).astype(np.float32)

    # Force of infection 
    inf_neigh = float_ip_not_quar @ adjacency
    del adjacency
    force_inf_matrix = 1 - ((1 - par[0]) ** inf_neigh)
    stochastic_E = np.random.rand(repetitions, num_nodes).astype(np.float32) <= force_inf_matrix
    neigh_sus_free = (state_matrix == 0) & not_quar_nodes & stochastic_E
    attributes_sim[0][neigh_sus_free] = 1  # state
    attributes_sim[1][neigh_sus_free] = it #I_time
    attributes_sim[8][neigh_sus_free] = -it #tg_matrix

    # Incubation period: Change From E to P
    e_nodes = state_matrix == 1
    stochastic_P = np.random.rand(repetitions, num_nodes).astype(np.float32) <= par[2]
    e_to_p = e_nodes & stochastic_P
    attributes_sim[0][e_to_p] = 2  # state

    # Symptoms onset From P to I (add elements to Inew) 
    attributes_sim[3] = np.zeros([repetitions, num_nodes], dtype=np.float32)  # prepare for next iteration
    stochastic_I = np.random.rand(repetitions, num_nodes).astype(np.float32) <= par[3]
    p_to_i = p_nodes & stochastic_I
    attributes_sim[0][p_to_i] = 3  # state
    attributes_sim[3][p_to_i] = 1  # Inew

    # Recovery: From I to R
    stochastic_R = np.random.rand(repetitions, num_nodes).astype(np.float32) <= par[1]
    i_to_r = i_nodes & stochastic_R
    attributes_sim[0][i_to_r] = 4  # state
    attributes_sim[2][i_to_r] = 0  # quar
    attributes_sim[8][neigh_sus_free] = attributes_sim[8][neigh_sus_free]+it #tg_matrix

    #  QUARANTINE DYNAMICS 
    # Detection of new infected: quarantine new cases
    Inew_nodes = Inew_matrix == 1
    stochastic_detected = np.random.rand(repetitions, num_nodes).astype(np.float32) <= par[5]
    detected_I = Inew_nodes & stochastic_detected
    attributes_sim[2][detected_I] = 1  # quar

    # Leave quarantine (S, E or P):
    s_e_p = state_matrix < 3
    quar_nodes = quarantine_matrix == 1
    stochastic_leave = np.random.rand(repetitions, num_nodes).astype(np.float32) <= par[4]
    leave_quar = s_e_p & quar_nodes & stochastic_leave
    attributes_sim[2][leave_quar] = 0  # quar

    return attributes_sim


def App(attributes_sim, attributes_old, scenario, leave_ref_prob, adjacency,incidence, it):
    """ Threshold model representing the behaviour of a contact tracing app. 
    Infected users report their infection to their neighbours with the app, who are put in a 10 day 
    quarantine. The app adoption process depends on the infected fractionof the population and 
    an intrinsic threshold of each user. When the infected averagesurpasses the intrinsic 
    threshold of a user this user adopts the app and enters a refractory period where it cannot remove
    the app. After this refractory period if the infected average < threshold the user will uninstall 
    the app. 
    INPUTS: 
        - attributes_sim: list containing the arrays with the attributes for each node in 
        this order: [states_matrix, gen_time_matrix, quarantine_matrix, Inew_matrix, app_matrix, 
        refractory_matrix, threshold_matrix]] 
        - attributes_old: same list but for the previous timepoint. 
        - scenario: Flag indicating the simulation that is being tested (0: baseline , 1: effective app)
        - leave_ref_prob: probability for leaving the refractory state. 
        - adjacency: adjacency matrix of layer2. 
    OUTPUTS: 
        attributes_sim: Updated version of the attribute matrix. 
    """

    # Unpack attributes layer 1
    state_matrix = attributes_old[0]
    Inew_matrix = attributes_old[3]

    # Unpack attributes layer 2
    app_matrix = attributes_old[4]
    refractory_matrix = attributes_old[5]
    threshold_matrix = attributes_old[6]
    comp_matrix = attributes_old[7]

    # Estimate 7-day incidence 
    if window_size <= it : 
        inf_average= np.average(incidence[-window_size:,:], axis=0)
    else:
        inf_average= np.average(incidence,axis=0)
    
    # APP EFFECT
    # Check infected nodes with app
    if scenario == 1:
        
        # Find I nodes detected with app
        Inew_nodes = Inew_matrix == 1
        app_nodes = app_matrix == 1
        quar_next = attributes_sim[2] == 1  # quar in next step
        app_detected = (Inew_nodes) & (app_nodes) & (quar_next)
        reporting = (app_detected) & (comp_matrix == 1)
        reported = (1 * reporting).astype(np.float32)

        # Quarantine app user neighbours
        neigh_mat = (reported @ adjacency).astype(bool)
        neigh_app = (app_nodes) & (neigh_mat)
        del adjacency
        attributes_sim[2][neigh_app] = 1  # quar

        # APP ADOPTION
        # Check nodes willing to adopt :  no ref and threshold >=0
        adopter_nodes = threshold_matrix >= 0  # willing to adopt
        no_ref_nodes = refractory_matrix == 0  # no ref

        # Check threshold:
        threshold_adopt = np.zeros([repetitions, num_nodes], dtype=bool)
        threshold_remove = np.zeros([repetitions, num_nodes], dtype=bool)

        for i in range(0, repetitions):
            threshold_adopt[i, :] = threshold_matrix[i, :] < inf_average[i]
            threshold_remove[i, :] = threshold_matrix[i, :] >= inf_average[i]

        # Choose to adopt
        adoption_choice = (threshold_adopt) & (adopter_nodes) & (no_ref_nodes)
        attributes_sim[4][adoption_choice] = 1  # app
        attributes_sim[5][adoption_choice] = 1  # ref

        # Choose (not to adopt/remove) the app:
        remove_choice = (threshold_remove) & (adopter_nodes) & (no_ref_nodes)
        attributes_sim[4][remove_choice] = 0  # app
        attributes_sim[5][remove_choice] = 2  # ref

        # Leave refractory state
        ref_nodes = refractory_matrix >= 1
        stochastic_ref = np.random.rand(repetitions, num_nodes).astype(np.float32) <= leave_ref_prob
        leave_ref = ref_nodes * stochastic_ref
        attributes_sim[5][leave_ref] = 0  # ref

    return attributes_sim

def net_create(num_nodes, av_neighbours_l1, error_percent, flag):
    """ Function to create the networks for both layers. 
    Allows to create Erdös-Rényi graphs, Scale Free networks or more realistic networks 
    following a negative binomial distribution. 
    Networks:
    Random network: Created using Erdös-Renyi graph generator from igraph.
    Scale Free network: Uncorrelated Scale Free network with a defined average degree
        created using the configurational model fit with a power law distribution. 
    Realistic network: Created with the configurational model using a negative 
        binomial distribution fit with the data from the POLYMOD study. The fiting procedure
        is performed in the fit_nb.r file. 

    After the networks are created the nodes are randomly coupled before saving the network. 
    INPUTS: 
        - num_nodes: Number of nodes
        - av_neighbours_l1 and  : Average number of neighbours desired for layer1 
        and layer2 respectively. 
        - flag: Type of network desired. "ER" for random graph,  and "SF" for scale-free network. 
    OUTPUT: 
        - graph: igraph object contiang generated graph
    """

    # Create epidemic layer
    if flag == 'ER':
        layer1 = ig.Graph.Erdos_Renyi(
            num_nodes, av_neighbours_l1 / num_nodes, directed=False, loops=False)

    elif flag == 'SF' or flag == 'NB':
        layer1 = configurat_model(num_nodes, av_neighbours_l1, flag)
    
    else:
        print('Unaccepted network topology')
        sys.exit()
    
    # Save layer 1
    output_dir = "Networks"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    layer1.write_graphml('Networks/layer1' + flag)

    # CT APP layer (add random pairs)
    layer1_bckup = layer1
    edges_to_add = round(error_percent*layer1.ecount())
    #random pairs
    random_pairs=(np.random.randint(0,layer1.vcount(),2*edges_to_add),np.random.randint(0,layer1.vcount(),2*edges_to_add))
    nodes_no_self= np.sort(np.transpose(random_pairs)[random_pairs[0]!=random_pairs[1],:],axis=1)
    
    #check edges
    existing_edges= [edge.tuple for edge in layer1.es]
    tuple_pairs = tuple([tuple(row) for row in nodes_no_self])
    not_repeated = set(tuple_pairs)-set(existing_edges)
    # update edges
    edges_added = random.sample(list(not_repeated), edges_to_add)
    layer1.add_edges(edges_added) 
    #check layer1 is fully cointained
    list2= [edge.tuple for edge in layer1.es]
    list1 = [edge.tuple for edge in layer1_bckup.es]
    print('OVERLAP LAYERS:',(len(set(list1).intersection(set(list2)))/len(list1))*100,'%')
    #update layer2:
    layer1.write_graphml('Networks/layer2' + flag)

def configurat_model(num_nodes, av_degree, flag):
    """ Configurational model
    Implementation of the configurational model to create networks with a certain degree distribution.
    INPUTS:
        - num_nodes: number of nodes in the network.
        - av_degree: Desired average degree. Only 5 and 9 are valid for a network of
        10000 nodes.
        -  sf_power: Scale-free parameter.
    OUTPUT: 
        -graph: Generated igraph network.
    """

    if flag == 'SF':
        # Define k_min to obtain av_degree desired
        sf_power = 2.5
        if av_degree == 11.92:
            x_min = 5.3 #5.2 #.5
        elif av_degree ==21.456:
            x_min = 10.45 #.89#11 #.5
        else:
            print('Wrong average degree')
            sys.exit()

        # Define maximal average degree 
        k_max = np.sqrt(num_nodes)
        # Define power law
        power = pl.Power_Law(xmin=x_min, parameters=[sf_power], discrete=True, discrete_approximation='round')
        degree_dist = power.generate_random(num_nodes)

        # Resample values above k_max
        bool_vec = degree_dist >= k_max
        while np.sum(bool_vec) > 0:
            degree_dist[bool_vec] = power.generate_random(np.sum(bool_vec))
            bool_vec = degree_dist >= k_max

    elif flag == 'NB':
        n= 2.426 #Extracted from R code (2.42587127628641)
        p= n/(n+(av_degree-2))
        
        degree_dist = nbinom.rvs(n,p,size=num_nodes)+2 
    
    # Impose an even number of edges
    if (np.sum(degree_dist) % 2) != 0:
        degree_dist[-1] += 1  # add an extra edge to the last element (random)

    # Generate degree sequence
    graph = ig.Graph.Degree_Sequence(degree_dist.tolist(), method='vl')
    return graph


def initialise_attributes(repetitions, num_nodes):
    """ Function to create the inital attributes of all nodes. 
    Define the properties of the nodes in both layers. 
    INPUTS: 
        - repetitions: Number of repetitions desired. 
        - num_nodes: Number of nodes. 
    
    OUTPUT: attribute list (layers1 and 2) for both scenarios. Elements from attributes:
        - states_matrix: Matrix of size repetitions x num_nodes containing the state of each node at
        each simulation. 0=S, 1=E, 2=P, 3=I, 4=R. The matrix is initialised with 1 node in state 3 
        for each row. 
        - gen_time_matrix: Matrix of size repetitions x num_nodes containing the time of infection of each node. 
        Initialised with nans at all position except for early infections 0. 
        - R_time_matrix: Matrix of size repetitions x num_nodes containing the time of infection of each node. 
        Initialised with nans.
        - quarantine_matrix: Matrix of size repetitions x num_nodes. 1 for nodes in quarantine 0 otherwise. 
        Initialised with all nodes at 0. 
        - Inew_matrix: Matrix of size repetitions x num_nodes singaling nodes entering I state. Initialised with 1 
        in infected nodes and 0 otherwise. 
        - app_matrix: Matrix of size repetitions x num_nodes containing the state of the app adoption dynamic. 1= user, 
        0=not user. Initialised with a 1 in a fraction of early adopters for each row. 
        - refractory_matrix: Matrix of size repetitions x num_nodes indicating nodes in the refractory period. 
        0= not refract,  1= refractory. Initialised with all nodes at 0. 
        - threshold_matrix: Matrix of size repetitions x num_nodes containing the threhsold for app adoption of each node. 
        threshold is drawn from a poisson distribution centered at the av_reluct_thr parameter. threshold = -1 
        indicates that nodes are not adopters and early adopters have threhsold = 0. The matrix is initialised with a
        defined fraction of no adopters and early adopters.
    """

    #create state matrices 
    I_time_matrix = np.zeros ([repetitions, num_nodes], dtype=np.float32)
    I_time_matrix[:,:] = np.nan

    states_matrix = np.zeros([repetitions, num_nodes], dtype=np.float32)
    seeds_I = np.random.randint(num_nodes - 1, size=[5, repetitions])
    states_matrix[list(range(0, repetitions)), seeds_I] = 3

    #gen_time_matrix[list(range(0, repetitions)), seeds_I] = 0
    
    quarantine_matrix = np.zeros([repetitions, num_nodes], dtype=np.float32)

    Inew_matrix = np.zeros([repetitions, num_nodes], dtype=np.float32)
    Inew_matrix[list(range(0, repetitions)), seeds_I] = 1
    I_time_matrix[list(range(0, repetitions)), seeds_I] = 0

    tg_matrix = np.zeros ([repetitions, num_nodes], dtype=np.float32)
    tg_matrix[:,:] = np.nan
    tg_matrix [list(range(0,repetitions)),seeds_I] = 0
    # layer 2 app:
    app_matrix = (np.random.choice([0, 1], size=(repetitions, num_nodes), p=[
        1 - early_adopt, early_adopt])).astype(np.float32)  # initialise app

    refractory_matrix = np.zeros([repetitions, num_nodes], dtype=np.float32)  # initialise refractory

    # initialise threshold
    threshold_def = np.random.choice([0, 1], size=(repetitions, num_nodes), p=[
        1 - (max_adopters/100),(max_adopters/100)])
    num_adopters = np.sum(threshold_def, axis=1)

    threshold_matrix = np.zeros([repetitions, num_nodes], dtype=np.float32)
    for adopter in range(0, len(num_adopters)):
        prob_poiss = np.random.poisson(
            av_reluct_thr*(num_nodes/100000), num_adopters[adopter])  # Poisson with mean 0.05

        # Set fraction of no adopters
        no_adopters = np.repeat(-1., num_nodes - num_adopters[adopter])
        thresholds = np.append(prob_poiss, no_adopters)
        np.random.shuffle(thresholds)  # randomise
        threshold_matrix[adopter] = thresholds

    threshold_matrix[app_matrix == 1] = 0  # set threshold to 0 for early adopters

    comp_matrix = (np.random.choice([0, 1], size=(repetitions, num_nodes), p=[
        1 - (compliance/100), (compliance/100)])).astype(np.float32)  # initialise app

    # Find neighbour list for both layers and all simulations
    attributes1 = [states_matrix, I_time_matrix, quarantine_matrix, Inew_matrix, app_matrix, refractory_matrix,
                   threshold_matrix, comp_matrix, tg_matrix]
    attributes2 = copy.deepcopy(attributes1)
    return attributes1, attributes2


def simulations(attributes_sim, parameters, scenario, adj_l1, adj_l2,incidence_old, it):
    """ Function containing the simulation for a certain scenario.
    Execute the epdidemic and CT app dynamic for a certain timepoint and scenario. 
    INPUT: 
        - attributes_sim: list containing the arrays with the attributes for each node. 
        The order of the array must be [states_matrix, gen_time_matrix, quarantine_matrix, 
        Inew_matrix, app_matrix, refractory_matrix, threshold_matrix]].
        - parameters: list containing all the parameters for the epidemic simulation
        - adj_l1 and adj_l2: adjacency matrices of layer1 and 2. 
        - it: current timestep of the simulation.
    OUTPUT: 
        - attributes_sim: Updated attributes.
        - states, app: state matrix of epidemic and app adoption proceesses. (states_matrix,app_matrix)
        - incidence: Individuals infected in a timepoint. 
    """

    attributes_old = copy.deepcopy(attributes_sim)

    # Layer 1: Epidemic (SEPIR)
    attributes_sim = Epidemic(attributes_sim, attributes_old, parameters, adj_l1, it)

    # Layer 2: Threshold model
    attributes_sim = App(attributes_sim, attributes_old, scenario, leave_ref_prob, adj_l2,incidence_old,it)

    # Save state matrices for both layers (states_matrix, app_matrix)
    states = np.transpose(np.average(attributes_sim[0] == 3, axis=1))
    app = np.transpose(np.average(attributes_sim[4] == 1, axis=1))
    incidence = np.transpose(np.count_nonzero(attributes_sim[1] == it, axis=1))
    return attributes_sim, states, app, incidence


def filter_allignment(I_frac, App_frac, incidence, prevalence):
    """ Funciton to filter and allign the simulations. 
    The repetitions will be discarded if the maximum infected fracion is smaller than 0.01. 
    The remaining simulations are allined to the point 0.01. 
    WARNING: If no simulations reach the 0.01 all the simulations without filtering simulations 
    will be displayed but a warning message will pop up. 
    INPUT: 
        - Variables extracted from postprocessing
    OUTPUT: 
        - I_frac, App_frac, incidence, prevalence: the same variables than before but filtered
        (less repetitions).
        - time_range: Matrix of size repetitions x time_steps, contains the alligned time range for all simulations.
    """
    # Allign curves and filter no outbreak cases:
    filtering = np.max(incidence, axis=1) >=100*(num_nodes/100000)
    if np.sum(filtering == True) > 0:
        I_frac = I_frac[filtering]
        App_frac = App_frac[filtering]
        incidence = incidence[filtering]
        prevalence = prevalence[filtering]
        allignment = np.argmax(incidence >= 100*(num_nodes/100000), axis=1)
        start_end = np.transpose([-allignment, len(I_frac[0]) - allignment])
        time_range = (np.asarray(start_end)[:, :1] + np.arange(start_end[0][1] - start_end[0][0])).tolist()
    else:
        time_range = [list(range(0, len(I_frac[0]))), ] * len(I_frac)
        print('WARNING: Impossible filtering. No simulations had maximum infected average above 0.01.')

    return I_frac, App_frac, incidence, prevalence, time_range

def padding(I_frac_new, App_frac_new, prevalence, incidence, time_range, limits):
    """ Function to allign the curves
    The allingment process is performed by padding the extremes 
    of all simulations unitil they all have the same length. 
    INPUT
        - I_frac_new, App_frac_new, prevalence, incidence: Simulations to pad, 
        obtained from filter_allignment.
        - time_range: Matrix with the time-trace after the allignment at I_av =0.01.
        - limits: [min,max] min= minimum timestep of all repetitions, max= minimum 
        timestep of all repetitions. All simulations will be padded to have minimum timestep =min
        and maximum timestep = max. 
    OUTPUT: 
        I_frac, App_frac, prevalence, incidence: Padded versions of I_frac_new, App_frac_new, 
        prevalence and incidence. 
    """

    I_frac = []
    App_frac = []
    prevalence_frac = []
    incidence_frac = []

    for a in range(0, len(I_frac_new)):
        # Number of timesteps to pad left and right
        pad_left = np.abs(limits[0]) - np.abs(np.amin(time_range[a]))
        pad_right = np.abs(limits[1]) - np.abs(np.amax(time_range[a]))

        # Padding all simulations with extreme values
        I_frac.append(np.pad(I_frac_new[a], (int(pad_left), int(pad_right)), 'constant',
                             constant_values=(I_frac_new[a][0], I_frac_new[a][-1])))
        App_frac.append(np.pad(App_frac_new[a], (int(pad_left), int(pad_right)), 'constant',
                               constant_values=(App_frac_new[a][0], App_frac_new[a][-1])))
        prevalence_frac.append(np.pad(prevalence[a], (int(pad_left), int(pad_right)), 'constant',
                                      constant_values=(prevalence[a][0], prevalence[a][-1])))
        incidence_frac.append(np.pad(incidence[a], (int(pad_left), int(pad_right)), 'constant',
                                     constant_values=(incidence[a][0], incidence[a][-1])))

    return I_frac, App_frac, prevalence_frac, incidence_frac

# %% DEFINE GLOBAL PARAMETERS
import parameters_def

importlib.reload(parameters_def)
from parameters_def import baseline_param

[new_network, flag, num_nodes, av_neighbours_l1, error_percent, early_adopt, time_steps, repetitions, beta, mu, epsilon, rho,
 leave_quar_prob, detection_rate, max_adopters, compliance,
 leave_ref_prob, av_reluct_thr,window_size,epi_scenario] = baseline_param()

from parameters_def import main_obtainGt_param

[title_parameter, test_param, test_list, xview, yview] = main_obtainGt_param()

# OVERVIEW SIMULATION
print('Network parameters: ' + flag + ' Num nodes:' + str(num_nodes) + ' ' + str(
    [av_neighbours_l1, error_percent]))
print('Simulation parameters: Rep:' + str(repetitions) + ' Time:' + str(time_steps))
print('Epidemic Scenario: ' + str(epi_scenario))
print('Epidemic parameters [beta, mu, epsilon, rho, leave_quar_prob, detection_rate]: ' + str(
    np.round([beta, mu, epsilon, rho, leave_quar_prob, detection_rate], 3)))
print('App parameters [early_adopters, compliance, leave_ref_prob, av_reluct_thr]: ' + str(
    [early_adopt, compliance, leave_ref_prob, av_reluct_thr]))

# %% NETWORK CREATION:
if new_network ==1:
    # Run network creation 
    net_create(num_nodes, av_neighbours_l1, error_percent, flag)
    layer1 = nx.read_graphml('Networks/layer1' +flag)
    layer2 = nx.read_graphml('Networks/layer2' + flag)
    print('Average degree layer1: ',np.average([d for n, d in layer1.degree()] ))
    print('Average degree layer2: ',np.average([d for n, d in layer2.degree()] ))
else: 
    print('Using a pre-created network')

# %% MAIN CODE:

# Initialise storing variables (repetitions)
repetitions_I = {}
repetitions_I2 = {}
repetitions_app = {}
repetitions_app2 = {}
time_rep = {}
incidence_base_rep = {}
incidence_int_rep = {}
prevalence_base_rep = {}
prevalence_int_rep = {}

# Explore different values of a parameter 
for ks in range(0, len(test_list)):
    # Define parameter to test: 
    if test_param == 'beta': # fit R0 param
        beta = test_list[ks]
        compliance=0
        detection_rate= 0
        flag_par = 'b'
    elif test_param == 'av_reluct_thr':
        av_reluct_thr = test_list[ks]
        compliance = 100
        flag_par = 'a'
    elif test_param == 'compliance':
        compliance = test_list[ks]
        av_reluct_thr = 0
        flag_par = 'c'
    elif test_param == 'leave_ref_prob':
        leave_ref_prob = test_list[ks]
        flag_par = 'e'
    else:
        print('Not accepted parameter')
        sys.exit()

    # Load network structure (Using networkx for efficiency reasons)
    layer1 = nx.read_graphml('Networks/layer1_' + flag)
    layer2 = nx.read_graphml('Networks/layer2_' + flag)

    # Rise error if the networks loaded do not match parameters_def
    if len(layer1.nodes) != num_nodes | len(layer2.nodes) != num_nodes:
        raise NameError('Number of nodes in layer1 or layer2 not equal to num_nodes. Check parameters_def.py.')

    # Extract adjacency matrix 
    adj_l1 = nx.to_scipy_sparse_matrix(layer1, dtype=np.float32, format='csr')
    adj_l2 = nx.to_scipy_sparse_matrix(layer2, dtype=np.float32, format='csr')
    del layer1, layer2

    # MAIN LOOP

    # Initial attributes for all simulations 
    [attributes_base, attributes_int] = initialise_attributes(repetitions, num_nodes)

    # Initialise storing variable for scenario 0
    I_base = np.zeros([time_steps, repetitions], dtype=np.float32)
    I_base[0] = np.transpose(np.average(attributes_base[0] == 3, axis=1))
    App_base = np.zeros([time_steps, repetitions], dtype=np.float32)
    App_base[0] = np.transpose(np.average(attributes_base[4] == 1, axis=1))

    # Initialise storing variable for scenario 1
    I_int = np.zeros([time_steps, repetitions], dtype=np.float32)
    I_int[0] = np.transpose(np.average(attributes_int[0] == 3, axis=1))
    App_int = np.zeros([time_steps, repetitions], dtype=np.float32)
    App_int[0] = np.transpose(np.average(attributes_int[4] == 1, axis=1))

    incidence_base = np.zeros([time_steps, repetitions], dtype=np.float32)
    incidence_base[0] = np.transpose(np.count_nonzero(attributes_base[1] == 0, axis=1))
    incidence_int = np.zeros([time_steps, repetitions], dtype=np.float32)
    incidence_int[0] = np.transpose(np.count_nonzero(attributes_int[1] == 0, axis=1))
    
    # Loop for simulate all timesteps:
    for it in tqdm(range(1, time_steps)):
        # Scenario 0: baseline simulation
        parameters = [beta, mu, epsilon, rho, leave_quar_prob, detection_rate]
        scenario = 0

        [attributes_base, I_base[it], App_base[it], incidence_base[it]] = simulations(attributes_base, parameters, scenario,
                                                                              adj_l1, adj_l2, incidence_base[:it,:], it)

        scenario = 1
        [attributes_int, I_int[it], App_int[it], incidence_int[it]] = simulations(attributes_int, parameters, scenario,
                                                                              adj_l1, adj_l2, incidence_int[:it,:], it)
    del adj_l1, adj_l2

    print('Postprocessing data obtained')

    prevalence_base = np.cumsum(incidence_base, axis=0)
    prevalence_int = np.cumsum(incidence_int, axis=0)

    # Allign all repetitions to 0.01 I_frac and remove unsuccessful iterations (max I<0.01%)
    [I_base, App_base, incidence_base, prevalence_base, time_range] = filter_allignment(np.transpose(I_base),
                                                                                np.transpose(App_base),
                                                                                np.transpose(incidence_base),
                                                                                np.transpose(prevalence_base))
    [I_int, App_int, incidence_int, prevalence_int, time_range2] = filter_allignment(np.transpose(I_int),
                                                                                 np.transpose(App_int),
                                                                                 np.transpose(incidence_int),
                                                                                 np.transpose(prevalence_int))
    # Find global timeline:
    glob_max = np.amax([np.amax(time_range), np.amax(time_range2)])
    glob_min = np.amin([np.amin(time_range), np.amin(time_range2)])
    limits = [glob_min, glob_max]
    time_rep[ks] = list(range(glob_min, glob_max + 1))

    # PADDING EXTREMES
    [I_base, App_base, prevalence_base, incidence_base] = padding(I_base, App_base, prevalence_base, incidence_base, time_range, limits)

    repetitions_I[ks] = np.array(I_base, dtype=np.float32)
    repetitions_app[ks] = np.array(App_base, dtype=np.float32)
    prevalence_base_rep[ks] = np.array(prevalence_base, dtype=np.float32)
    incidence_base_rep[ks] = np.array(incidence_base, dtype=np.float32)

    [I_int, App_int, prevalence_int, incidence_int] = padding(I_int, App_int, prevalence_int, incidence_int, time_range2,
                                                          limits)

    repetitions_I2[ks] = np.array(I_int, dtype=np.float32)
    repetitions_app2[ks] = np.array(App_int, dtype=np.float32)
    prevalence_int_rep[ks] = np.array(prevalence_int, dtype=np.float32)
    incidence_int_rep[ks] = np.array(incidence_int, dtype=np.float32)

    print('End of the data extraction process-Surviving repetitions: ' + str(len(incidence_base_rep[ks])) + ' ' + str(
        len(incidence_int_rep[ks])))

if not os.path.isdir("Simulations/"+ flag+'/main_simulation'):
        os.makedirs("Simulations/"+ flag+'/main_simulation')

# For multiplot
if len(test_list)==1:
    multi=''
else: 
    multi='m'

#np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_I_base_'+flag_par +'_'+ flag +multi, repetitions_I)
#np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_I_int_'+flag_par +'_'+ flag +multi, repetitions_I2)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_App_base_'+flag_par +'_'+flag +multi, repetitions_app)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_App_int_'+flag_par +'_'+flag +multi, repetitions_app2)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_prevalence_base_'+flag_par +'_'+ flag +multi, prevalence_base_rep)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_prevalence_int_' +flag_par +'_'+ flag +multi, prevalence_int_rep)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_incidence_base_' +flag_par +'_'+ flag +multi, incidence_base_rep)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_incidence_int_'+flag_par +'_'+ flag +multi, incidence_int_rep)
np.save('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_time_rep_' +flag_par +'_'+ flag +multi, time_rep)  

#%%