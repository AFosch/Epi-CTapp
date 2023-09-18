# %% RUN CODE FROM TERMINAL, NOT NOTEBOOK
# MODIFIED FOR OLD NETWORK 
#MAIN CODE: Fixed?
import copy
import importlib
import multiprocessing as mp
import os
import time
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style({'font.family':'sans-serif'})
#%%
start_time =time.time()
# FUNCTION DEFINITION 

def Epidemic(attributes_sim, attributes_old, par, adjacency, it):
    """ Stochastic, discrete-time compartmental model coupled to a network. 
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

def App(attributes_sim, attributes_old, scenario, leave_ref_prob, compliance2, adjacency, incidence, it):
    """ Threshold model representing the behaviour of a contact tracing app. 
    Infected users report their infection to their neighbours with the app, who are 
    then put in a 10 day quarantine. The app adoption process depends on the infected fraction
    of the population and an intrinsic threshold of each user. When the infected average
    surpasses the intrinsic threshold of a user this user adopts the app and enters a 
    refractory period where it can not remove the app. After this refractory period if the 
    infected average < threshold the user will uninstall the app. 
    INPUTS: 
        - attributes_sim: list containing the arrays with the attributes for each node in 
        this order: [states_matrix, gen_time_matrix, quarantine_matrix, Inew_matrix, app_matrix, 
        refractory_matrix, threshold_matrix]] 
        - attributes_old: same list but for the previous timepoint. 
        - scenario: Flag indicating hich scenario is being tested (0: baseline , 1: effective app)
        - leave_ref_prob: probability for leaving the refractory state. 
        - adjacency: adjacency matrix of layer2. 
    OUTPUTS: 
        attributes_sim: Updated version of the attribute matrix. 
    """

    # Unpack attributes layer 1
    state_matrix = attributes_old[0]
    Inew_matrix = attributes_old[3]

    # unpack attributes layer 2
    app_matrix = attributes_old[4]
    refractory_matrix = attributes_old[5]
    threshold_matrix = attributes_old[6]
    comp_matrix = attributes_old[7]

    #NEW: Average cumm incidence 
    window_size= 7 #(7day window_size)
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

def initialise_attributes(repetitions, num_nodes,max_adopters2, compliance2, av_reluct_thr2):
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
        1 - (max_adopters2/100), (max_adopters2/100)])
    num_adopters = np.sum(threshold_def, axis=1)

    threshold_matrix = np.zeros([repetitions, num_nodes], dtype=np.float32)
    for adopter in range(0, len(num_adopters)):
        prob_poiss = np.random.poisson(
            av_reluct_thr2*(num_nodes/100000), num_adopters[adopter])  # Poisson with mean 0.05

        # Set fraction of no adopters
        no_adopters = np.repeat(-1., num_nodes - num_adopters[adopter])
        thresholds = np.append(prob_poiss, no_adopters)
        np.random.shuffle(thresholds)  # randomise
        threshold_matrix[adopter] = thresholds

    threshold_matrix[app_matrix == 1] = 0  #set threshold to 0 for early adopters

    comp_matrix = (np.random.choice([0, 1], size=(repetitions, num_nodes), p=[
        1 - (compliance2/100), (compliance2/100)])).astype(np.float32)  # initialise app

    # Find neighbour list for both layers and all simulations
    attributes1 = [states_matrix, I_time_matrix, quarantine_matrix, Inew_matrix, app_matrix, refractory_matrix,
                   threshold_matrix, comp_matrix, tg_matrix]
    attributes2 = copy.deepcopy(attributes1)
    return attributes1, attributes2

def simulations(attributes_sim, parameters, scenario, adj_l1, adj_l2,it,compliance2,incidence_old): 
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
    attributes_sim = App(attributes_sim, attributes_old, scenario, leave_ref_prob, compliance2, adj_l2,incidence_old,it)

    # Save state matrices for both layers (states_matrix, app_matrix)
    incidence = np.transpose(np.count_nonzero(attributes_sim[1] == it, axis=1))
    return attributes_sim, incidence

def filter_allignment(incidence, prevalence):
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
    filtering = np.max(incidence, axis=1) >= 100*(num_nodes/100000)
    if np.sum(filtering == True) > 0:
        incidence = incidence[filtering]
        prevalence = prevalence[filtering]
        allignment = np.argmax(incidence >= 100*(num_nodes/100000), axis=1)
        start_end = np.transpose([-allignment, len(incidence[0]) - allignment])
        time_range = (np.asarray(start_end)[:, :1] + np.arange(start_end[0][1] - start_end[0][0])).tolist()
    else:
        time_range = [list(range(0, len(incidence[0]))), ] * len(incidence)
        print('WARNING: Impossible filtering. No simulations had maximum infected average above 0.01.')

    return incidence, prevalence, time_range

def padding(incidence,prevalence, time_range, limits):
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

    prevalence_frac = []
    incidence_frac = []

    for a in range(0, len(incidence)):
        # Number of timesteps to pad left and right
        pad_left = np.abs(limits[0]) - np.abs(np.amin(time_range[a]))
        pad_right = np.abs(limits[1]) - np.abs(np.amax(time_range[a]))

        # Padding all simulations with extreme values
        prevalence_frac.append(np.pad(prevalence[a], (int(pad_left), int(pad_right)), 'constant',
                                      constant_values=(prevalence[a][0], prevalence[a][-1])))
        incidence_frac.append(np.pad(incidence[a], (int(pad_left), int(pad_right)), 'constant',
                                     constant_values=(incidence[a][0], incidence[a][-1])))

    return incidence_frac, prevalence_frac

def main (input_par):
    ''' Full simulation for different combinations of parameters. 
    INPUT: 
    - input_par: dictionary with all the input parameters. 
    OUTPUT: 
    - pack: group of array with simulations for the epidemic and CT app adoption. 
    '''
    
    # Explore different values of a parameter 
    max_adopters2 = input_par[1]
    # Define parameter to test: 
    if test_param[0] == 'beta':
        compliance2 = 100
        av_reluct_thr2 = av_reluct_thr
        beta2 = input_par[0]
    elif test_param[0] == 'av_reluct_thr':
        av_reluct_thr2 = input_par[0]
        compliance2 = 100
        beta2 = beta
    elif test_param[0]== 'compliance': #for comp scenario impose threshold = 0 
        compliance2 = input_par[0]
        beta2 = beta
        av_reluct_thr2= 0
        if test_param[1] == 'av_reluct_thr': #adcomp scenario
            max_adopters2 = 70
            av_reluct_thr2 = input_par[1]
    else: 
        print('Not accepted parameter')
        exit()

    print(beta2, max_adopters2, compliance2, av_reluct_thr2)
    #Load network structure (Using networkx for efficiency reasons)
    
    layer1 = nx.read_graphml('Networks/layer1_'+str(flag))
    layer2 = nx.read_graphml('Networks/layer2_'+str(flag))

    # Extract adjacency matrix 
    adj_l1 = nx.to_scipy_sparse_matrix(layer1, dtype=np.float32, format='csr')
    adj_l2 = nx.to_scipy_sparse_matrix(layer2, dtype=np.float32, format='csr')

    del layer1, layer2

    # MAIN LOOP
    # Initial attributes for all simulations 
    [attributes_base, attributes_int] = initialise_attributes(repetitions, num_nodes,max_adopters2, compliance2,av_reluct_thr2)
    
    # Initialise storing variable for scenario 0
    incidence_base = np.zeros([time_steps, repetitions], dtype=np.float32)
    incidence_base[0] = np.transpose(np.count_nonzero(attributes_base[1] == 0, axis=1))
    incidence_int = np.zeros([time_steps, repetitions], dtype=np.float32)
    incidence_int[0] = np.transpose(np.count_nonzero(attributes_int[1] == 0, axis=1))
    
    # Loop for all simulations:
    for it in range(1, time_steps) :
        # Scenario 0: baseline simulation
        parameters = [beta2, mu, epsilon, rho, leave_quar_prob, detection_rate]
        # Baseline scenario
        scenario = 0
        [attributes_base, incidence_base[it]] = simulations (attributes_base, parameters, scenario, adj_l1, adj_l2, it, compliance2, incidence_base[:it,:]) 
        # CT app scenario 
        scenario = 1
        [attributes_int, incidence_int[it]] = simulations (attributes_int, parameters, scenario, adj_l1, adj_l2, it, compliance2, incidence_int[:it,:]) 

    del adj_l1, adj_l2
    # Allign all repetitions to 1% I_frac and remove unsuccessful iterations (max I<0.01%)
    prevalence_base = np.cumsum(incidence_base, axis=0)
    prevalence_int = np.cumsum(incidence_int, axis=0)

    [incidence_base,prevalence_base,time_range] = filter_allignment(np.transpose(incidence_base),np.transpose(prevalence_base))
    [incidence_int,prevalence_int, time_range2] = filter_allignment(np.transpose(incidence_int),np.transpose(prevalence_int))

    # Find global timeline:
    glob_max = np.amax([np.amax(time_range), np.amax(time_range2)])
    glob_min = np.amin([np.amin(time_range), np.amin(time_range2)])
    limits = [glob_min, glob_max]
    
    # PADDING EXTREMES
    [incidence_base,prevalence_base] = padding (incidence_base,prevalence_base, time_range, limits)
    [incidence_int, prevalence_int] = padding (incidence_int, prevalence_int, time_range2, limits)
    
    #7 day average (convolution)
    window_size=7
    I_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=incidence_base, axis=1))
    I_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=incidence_int, axis=1))

    prev_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=prevalence_base, axis=1))
    prev_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=prevalence_int, axis=1))
    
    mean_base=np.average(I_base_conv, axis=0)
    mean_int=np.average(I_int_conv, axis=0)
    mean_p_base=np.average(prev_base_conv, axis=0) 
    mean_p_int=np.average(prev_int_conv, axis=0)    

    print(time.time()-start_time, input_par,(1-(np.max(mean_int)/np.max(mean_base)))*100)
    pack = [(1-(np.max(mean_int)/np.max(mean_base)))*100, (1-(np.max(mean_p_int)/np.max(mean_p_base)))*100]
    return pack

#%%  PARAMETER DEFINITION 
import parameters_def

importlib.reload(parameters_def)
from parameters_def import baseline_param

[new_network, flag,num_nodes,av_neighbours_l1,error_percent,early_adopt,time_steps,repetitions,beta,mu,epsilon,rho,
leave_quar_prob,detection_rate,max_adopters,compliance,
leave_ref_prob,av_reluct_thr,window_size,epi_scenario] = baseline_param()

from parameters_def import heatmap_param

[range1,range2,test_list,test_list2,test_param,
title_parameter,title_parameter2,flag_comp,rounder,smooth_par] = heatmap_param()

# OVERVIEW SIMULATION
print('Network parameters: '+str(flag)+' Num nodes:'+str(num_nodes)+' '+str([av_neighbours_l1, error_percent]))
print('Simulation parameters: Rep:'+str(repetitions)+' Time:'+str(time_steps))
print('Epidemic Scenario: ' + str(epi_scenario))
print('Epidemic parameters [beta, mu, epsilon, rho, leave_quar_prob, detection_rate]: '+str(np.round([beta,mu, epsilon,rho,leave_quar_prob, detection_rate],3)))
print('App parameters [early_adopters, compliance, leave_ref_prob, av_reluct_thr]: '+str([early_adopt, compliance, leave_ref_prob, av_reluct_thr]))

#%% MAIN CODE: 

# Initialise storing variables (repetitions)

if __name__ == '__main__':

    count= time.time()
    mp.set_start_method('forkserver', force=True)
    
    # Define all parameter combinations 
    param_list, param_list2 = np.meshgrid(test_list, test_list2)
    parameter_space = np.transpose(np.array([param_list.flatten(),param_list2.flatten()]))
    # Define cores to use
    m_processes = mp.cpu_count()-3
    print(m_processes,'\n')

    # Generate save directory 
    if not os.path.isdir("Plots/"+ flag+'/heatmap'):
        os.makedirs("Plots/"+ flag+'/heatmap')

    # Multiprocessing
    with mp.Pool(processes= m_processes) as p: # set number of threads 
        pack = p.map(main, [parameter_space[ks,:] for ks in range(0,len(parameter_space))] ,chunksize=1)
    
    # Extract results multiprocessing
    pack=pd.DataFrame(pack)
    results = pd.DataFrame(pack).loc[:,0]
    results_p =pd.DataFrame(pack).loc[:,1]
    print(np.array(results))
    results = np.array(results).reshape(len(test_list2),len(test_list))
    results_p = np.array(results_p).reshape(len(test_list2),len(test_list))
    
    ellapsed = (time.time() - count)/60
    print(ellapsed)
    
    if not os.path.isdir("Simulations/"+ flag+'/heatmap_simulation'):
        os.makedirs("Simulations/"+ flag+'/heatmap_simulation')


    # Save data 
    if flag_comp == 1:
        np.savetxt('Simulations/'+str(flag)+'/heatmap_simulation/'+epi_scenario+'_delta_comp_'+str(flag)+'.csv', np.array(results), delimiter=',')
        np.savetxt('Simulations/'+str(flag)+'/heatmap_simulation/'+epi_scenario+'_delta_comp_p_'+str(flag)+'.csv', np.array(results_p), delimiter=',') 
    elif flag_comp == 0:
        np.savetxt('Simulations/'+str(flag)+'/heatmap_simulation/'+epi_scenario+'_delta_'+str(flag)+'.csv', np.array(results), delimiter=',')
        np.savetxt('Simulations/'+str(flag)+'/heatmap_simulation/'+epi_scenario+'_delta_p_'+str(flag)+'.csv', np.array(results_p), delimiter=',')
    else:
        np.savetxt('Simulations/'+str(flag)+'/heatmap_simulation/'+epi_scenario+'_delta_adcomp_'+str(flag)+'.csv', np.array(results), delimiter=',')
        np.savetxt('Simulations/'+str(flag)+'/heatmap_simulation/'+epi_scenario+'_delta_p_adcomp_'+str(flag)+'.csv', np.array(results_p), delimiter=',')

#%%