# %% Test R0 with different betas

# Import packages
import importlib
import os
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs

matplotlib.rc('font', **{'family': 'sans'})
matplotlib.rcParams['mathtext.default'] = 'regular'


# %% PARAMETER DEFINITION
start_time = time.time()

# PARAMETER DEFINITION

# import global parameters
import parameters_def

importlib.reload(parameters_def)
from parameters_def import baseline_param

[new_network, flag, num_nodes, av_neighbours_l1, error_percent, early_adopt, time_steps, repetitions, beta, mu, epsilon, rho,
 leave_quar_prob, detection_rate, max_adopters, compliance,
 leave_ref_prob, av_reluct_thr, window_size,epi_scenario] = baseline_param()

# import script specificparameters
from parameters_def import R0_fit_par

[test_list, title_parameter, test_param,
 compliance, detection_rate] = R0_fit_par()

print(compliance, detection_rate)

#%% FUNCTION DEFINITION

def max_prevalence (flag,epi_scenario= epi_scenario):
    # PREVALENCE PLOTS 
    if test_param == 'av_reluct_thr':
        flag_par='a'
    elif test_param =='compliance':
        flag_par='c'
    elif test_param == 'beta':
        flag_par = 'b'
    else: 
        flag_par = 'e'

    if len(test_list) ==1:
        multi = ''
    else:   
        multi='m'

    final_prev_base = np.zeros(len(test_list))
    final_prev_int = np.zeros(len(test_list))

    for ks in range(0, len(test_list)):
        # Select data to plot
        try: 
            prevalence_base = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_prevalence_base_'+flag_par+'_' + flag + multi+'.npy', allow_pickle=True).item()[ks] / num_nodes
            prevalence_int = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_prevalence_int_'+flag_par+'_' + flag + multi+'.npy', allow_pickle=True).item()[ks] / num_nodes
        except: 
            print('Missing simulations try to run main_simulation.py with:\n'\
            'test_list= np.round(np.arange(0.01, 0.065, 0.005),3)\n'\
            'test_param = beta and detection_rate = 0\n'\
            'Repeat for the 3 networks used:\n '
            "1)flag = 'NB', 2)flag = 'SF', 3) flag='ER'\n")
            sys.exit()

        # Mean and CI 
        prev_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=prevalence_base, axis=1))
        prev_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=prevalence_int, axis=1))

        prev_mean_rep = np.mean(prev_base_conv, axis=0)
        final_prev_base[ks] = np.max(prev_mean_rep)# find max prev

        prev_mean_rep2 = np.mean(prev_int_conv, axis=0)
        final_prev_int[ks] =np.max(prev_mean_rep2) #prev_mean_rep #find max av.prev
    
    return ([final_prev_base,final_prev_int])

def R0 (flag,epi_scenario= epi_scenario):
    #Estimate R0 
    #range_exp = [4, 7]
    mean_r = []

    # Other epidemic parameters
    if epi_scenario == 'S1':
        mu = 1/2 #prob recovery I-R (modified by generation time )
        rho = 1/1.5 #1 / 2  # P to I 
    elif epi_scenario == 'S2':
        mu = 1/3.5 #prob recovery I-R (modified by generation time )
        rho = 1/1.5  # P to I 
        #Generation time : 2+3+2= 7 days infectious 
    elif epi_scenario == 'AL':
        mu = 1/2 #prob recovery I-R (modified by generation time )
        rho = 1/ 2  # P to I 
        # Generation time : 2+3+2= 7 days infectious 
    else:
        print('Wrong epi_scenario')
        sys.exit()

    #Read epidemic layer 
    layer1 = nx.read_graphml('Networks/layer1_' + flag)
    adjacency= nx.to_scipy_sparse_array(layer1, dtype=np.float32, format='csr')

    #Eigenvalue adjacency matrix
    eigen_adj, _ = eigs(adjacency)

    #Estimate time spent between time presymp and recovered: 
    mean_r = [ (beta*np.max(np.real(eigen_adj)))*((1/mu)+(1/rho)) for beta in test_list]
    mu_ir = (mu*rho)/(mu + rho) 
    mean_r = [ (beta*np.max(np.real(eigen_adj)))/(mu_ir) for beta in test_list]

    print(mean_r)
    mean_r = np.array(mean_r)


    # Generate save directory and save R0 estimation
    if not os.path.isdir("Simulations/"+ flag+'/R0_fit'):
        os.makedirs("Simulations/"+ flag+'/R0_fit')

    np.savetxt('Simulations/'+str(flag)+'/R0_fit/'+epi_scenario +'_mean_r_' + str(flag) + '.csv', mean_r, delimiter=',')
    return()
    
# %% Show comparison scenarios
if epi_scenario == 'AL':
    # Estimate R0:
    R0('ER','AL')
    R0('SF','AL')
    R0('NB','AL')

    #Load R0 for the 3 populaitons: 
    mean_R0_SF = np.genfromtxt('Simulations/SF/R0_fit/'+epi_scenario+'_mean_r_SF.csv', delimiter=',')
    mean_R0_ER = np.genfromtxt('Simulations/ER/R0_fit/'+epi_scenario+'_mean_r_ER.csv', delimiter=',')
    mean_R0_NB = np.genfromtxt('Simulations/NB/R0_fit/'+epi_scenario+'_mean_r_NB.csv', delimiter=',')
    test_list = np.array(test_list)

    # Prepare plot
    plt.figure(figsize=[6.4, 3.8])

    ax1 = plt.subplot(1, 2, 2)
    plt.xticks(test_list, rotation='vertical')
    fit_ER = ax1.plot(test_list, mean_R0_ER, color = '#C291DC')
    ax1.axhline(3, color='silver', linestyle='dashed', zorder=1)
    ax1.axhline(2, color='silver', linestyle='dashed', zorder=1)

    data_ER = ax1.scatter(test_list, mean_R0_ER, label='ER', zorder=2,color = '#C291DC')
    data_SF = ax1.scatter(test_list, mean_R0_SF, label='SF',zorder=4,color = '#4B1D95')
    fit_SF = ax1.plot(test_list, mean_R0_SF,color = '#4B1D95')

    data_NB = ax1.scatter(test_list, mean_R0_NB, label='NB',zorder=4,color='#197180')
    fit_NB = ax1.plot(test_list, mean_R0_NB,color='#197180')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$Mean\ R_0$')
    plt.title(r"$\bf{b)}$ $R_0$ vs $\beta$")

    #Obtained from 
    max_prev0_ER , _  = max_prevalence('ER','AL')
    max_prev0_SF , _  = max_prevalence('SF','AL')
    max_prev0_NB , _  = max_prevalence('NB','AL') 
    #np.genfromtxt('Simulations/ER/main_plot/'+epi_scenario+'max_prev_base_ER.csv', delimiter=',')
    #max_prev0_SF = np.genfromtxt('Simulations/SF/main_plot/'+epi_scenario+'max_prev_base_SF.csv', delimiter=',')
    #max_prev0_NB = np.genfromtxt('Simulations/NB/main_plot/'+epi_scenario+'max_prev_base_NB.csv', delimiter=',')

    ax2 = plt.subplot(1, 2, 1)

    plt.xticks(test_list, rotation='vertical')
    ax2.ticklabel_format(axis='x', style='sci')
    plt.title(r"$\bf{a)}$ Prevalence vs $\beta$")
    plt.ylabel('Max prevalence (%)')
    plt.xlabel(r'$\beta$')
    plt.xticks(test_list)
    ax2.plot(test_list, max_prev0_ER, zorder=1, color = '#C291DC')
    ax2.scatter(test_list, max_prev0_ER, zorder=2, color = '#C291DC')
    ax2.plot(test_list, max_prev0_SF, zorder=3,color = '#4B1D95')
    ax2.scatter(test_list, max_prev0_SF, zorder=4,color = '#4B1D95')
    ax2.plot(test_list, max_prev0_NB, zorder=5,color='#197180')
    ax2.scatter(test_list, max_prev0_NB, zorder=6,color='#197180')
    plt.figlegend(loc='lower center',ncol=3,frameon='False')
    plt.tight_layout()

    if not os.path.isdir('Plots/'+str(flag)):
            os.makedirs('Plots/'+str(flag))
    plt.savefig('Plots/Mean_R0_compare.pdf')
else:
    # Estimate R0 for the different epidemic scenarios
    R0('NB','AL')
    max_prev_NB , _ = max_prevalence('NB','AL')
    mean_R0_NB = np.genfromtxt('Simulations/NB/R0_fit/AL_mean_r_NB.csv', delimiter=',')

    R0('NB','S1')
    S1_max_prev_NB , _  = max_prevalence('NB','S1')
    S1_mean_R0_NB = np.genfromtxt('Simulations/NB/R0_fit/S1_mean_r_NB.csv', delimiter=',')
    
    R0('NB','S2')
    S2_max_prev_NB , _  = max_prevalence('NB','S2')
    S2_mean_R0_NB = np.genfromtxt('Simulations/NB/R0_fit/S2_mean_r_NB.csv', delimiter=',')
    
    # PLOT R0 compare
    plt.figure(figsize=[6.4, 4])

    ax1 = plt.subplot(1, 2, 2)
    plt.xticks(test_list, rotation='vertical')
    ax1.axhline(3, color='silver', linestyle='dashed', zorder=1)
    ax1.axhline(2, color='silver', linestyle='dashed', zorder=1)

    data_alpha= ax1.scatter(test_list, mean_R0_NB, label='Alpha', zorder=2, color = '#4B1D95')
    fit_alpha = ax1.plot(test_list, mean_R0_NB, color = '#4B1D95')

    data_S1 = ax1.scatter(test_list, S1_mean_R0_NB, label='Scen. 1',zorder=4,color = '#C291DC')
    fit_S1 = ax1.plot(test_list, S1_mean_R0_NB,color = '#C291DC')

    data_S2 = ax1.scatter(test_list, S2_mean_R0_NB, label='Scen. 2',zorder=4,color='#197180')
    fit_S2 = ax1.plot(test_list, S2_mean_R0_NB,color='#197180')


    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$Mean\ R_0$')
    plt.title(r"$\bf{b)}$ $R_0$ vs $\beta$")
    ax2 = plt.subplot(1, 2, 1)

    plt.xticks(test_list, rotation='vertical')
    ax2.ticklabel_format(axis='x', style='sci')
    plt.title(r"$\bf{a)}$ Prevalence vs $\beta$")
    plt.ylabel('Max prevalence (%)')
    plt.xlabel(r'$\beta$')
    #plt.yticks(np.arange(0, 1.2, 0.2), (np.arange(0, 1.2, 0.2) * 100).astype(int))
    plt.xticks(test_list)
    ax2.plot(test_list, max_prev_NB, zorder=1, color = '#4B1D95')
    ax2.scatter(test_list, max_prev_NB, zorder=2, color = '#4B1D95')
    ax2.plot(test_list, S2_max_prev_NB, zorder=3,color='#197180')
    ax2.scatter(test_list, S2_max_prev_NB, zorder=4,color='#197180')
    ax2.plot(test_list, S1_max_prev_NB, zorder=5,color = '#C291DC')
    ax2.scatter(test_list, S1_max_prev_NB, zorder=6,color = '#C291DC')
    plt.figlegend(loc='lower center',ncol=3,frameon='False',bbox_to_anchor=(0.5,0))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    if not os.path.isdir('Plots/'+str(flag)):
            os.makedirs('Plots/'+str(flag))
    plt.savefig('Plots/Variants_R0_compare.pdf')
    
#%%
