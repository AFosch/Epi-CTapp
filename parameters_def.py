import sys
import numpy as np

#%% Baseline parameters (for all scripts)
def baseline_param():
    """ Defines the baseline parameters in the Epidemic-CT app model. 
    It is loaded by all the scipts, as it initialises the main variables.
    Simulation parameters
        - new_network: Create new network, 1= yes 0= use pre-created networks in the Network folder. 
        - flag= Defines the degree distribution of network to test. ER: Erdos-Renyi network, 
            SF: Scale Free network, NB: negative binomial from POLYMOD data. Also defines the average degrees of both layers.
        - num_nodes= Number of nodes in the network (default=10,000)
        - time_steps= Maximum number of timepoints of every simulation(default=500).
        - repetitions= Number of repetitions for each simultation (default = 1000). 
        - av_neighbours_l1 = Av. degree epidemic layer. 
        - error_percent = Spurious links added in Bluetood layer (25% av degree)
        - window_size = Aggregation window of the time series reported. (7-day incidence) 
    Epidemic parameters: 
        - beta= infection rate, mu= recovery rate, epsilon= exposed to presymptomatic infectious, 
        - rho= presymp infectious to symptomatic infectious.
        - leave_quar_prob = probability to leave quarantine.
        - detection_rate = healthcare detection rate.
        - epi_scenario = Selects the epidemic parameters for the scenarios shown in the paper. 
    App model parameters: 
        - early_adopt = Fraction of early adopters of the app, max_adopters= maximal fraction of adopters, 
        - compliance = Fraction of compliant users, leave_ref_prob = probability of leaving refractory state, 
        - av_reluct_thr = Average adoption threshold.
        - max_adopters = Upper bound percentage of adopters (normal 70%)
        - leave_ref_prob = Prob. to leave refractory state-> reevaluate the adoption status (10 days)
    """

    #SIMULATION PARAMETERS
    new_network = 0 # 0= no new network, 1= new network
    flag = 'NB'
    num_nodes = 10000

    time_steps = 500 
    repetitions = 1000

    #Network specifications
    av_neighbours_l1 = 11.92 # Extracted from age_contact matrix
    error_percent = 0.25 # Spurious links added in Bluetood layer (25% av degree)

    window_size= 7 #7-day window_size

    # Parameters of Epidemic model
    beta = 0.045 # Probability of infection S-E
    epi_scenario = 'AL'
    # 'S2' = scenario2 with mu= 1/3.5 
    # 'S1' = scenario 1 with mu= 1/2 ,
    # 'AL' = alpha

    # Other epidemic parameters
    if epi_scenario == 'S1':
        epsilon = 1/2  # E to P prob
        rho = 1/1.5 # P to I
        mu = 1/2 # prob recovery I-R 
    elif epi_scenario == 'S2':
        epsilon = 1/2 
        rho = 1/1.5  
        mu = 1/3.5 
    elif epi_scenario == 'AL':
        epsilon = 1/3 
        rho = 1/ 2  
        mu = 1/2
    else:
        print('Wrong Flag: Epidemic scenario')
        sys.exit()
    
    leave_quar_prob = 1 / 10
    detection_rate = 0.5 

    # APP MODEL PARAMETERS: 
    early_adopt = 0.01 
    max_adopters = 70 
    compliance = 100 
    av_reluct_thr = 230 
    leave_ref_prob = 1 / 10 

    par_list = [new_network, flag, num_nodes, av_neighbours_l1, error_percent,
                early_adopt, time_steps, repetitions, beta, mu, epsilon, rho,
                leave_quar_prob, detection_rate, max_adopters, compliance,
                leave_ref_prob, av_reluct_thr, window_size,epi_scenario]
    return par_list

# Parameters for main_simulation.py and obtain_gt.py
def main_simulation_param():
    """ Function defining the parameters for main_simulation.py and obtain_gt.py
    Compute incidence or generation interval with a defined set of parameters. 
    It can be used to test multiple values for a single parameter. 
    - test_param= Name of the parameter to modify. 
    - title_parameter: Name for the test_param variable. It is specified for the most common parameters but it can modified.
    - test_list= Range of values of the parameter test_param to test in the simulation. Must be a list even for 1 value.
    - xview= Specifies plot's Xlim: crop = [x_min,300], all: [x_min,x_max]
    - yview= Ylim of the fiugre: crop = [0,3500], all:[y_min,y_max]
    """
    
    test_param = 'compliance'#'av_reluct_thr' #'compliance' #beta
    
    if test_param == 'av_reluct_thr':
        title_parameter =r"$I_{thr}$"
    elif test_param == 'compliance':
        title_parameter ='Compliant users'
    else:
        title_parameter= 'Specified name'

    # Parameter to test in the simulation (should always be inside a list)
    test_list = [70] #np.round(np.arange(0.01, 0.065, 0.005),3) 

    # Parameters for visualization incidence (main_plot.py and multiplot.py): 
    xview= 'crop' # all or crop
    yview= 'crop' # all or crop
    par_main = [title_parameter, test_param, test_list, xview, yview]

    # Parameters to reproduce plots from paper.
    # WARNING: There is stochastic variability.
    #    Figure 3a:  test_param = 'av_reluct_thr', test_list=[230], 
    #    Figure 3b: test_param = 'compliance' test_list=[70] 
    #    Figure S2A: test_param = 'av_reluct_thr', list(np.linspace(100, 3000, 5)) 
    #    Figure: S2B: test_param = 'compliance' list(np.linspace(0, 100, 5))
    #    For R0_fit: test_param = 'beta' np.round(np.arange(0.01, 0.065, 0.005),3) 

    return par_main

# Parameters for R0_fit_par.py
def R0_fit_par():
    """ Function defining the parameters for R0_fit.py
    max_prev_betas computes the maximal prevalence for multiple transmissibility parameters. 
    Always computed in the case without any effective control measure (detection_rate=0).
    - test_param= Name of the parameter to modify. 
    - title_parameter: Name for the test_param variable. It defines the title in the output plots. 
    - test_list= Range of values of the parameter test_param to test in the simulation.     
    """
    test_param = 'beta'
    test_list = np.round(np.arange(0.01, 0.065, 0.005),3) 
    title_parameter = "Beta"
    # Scenario uncontrolled (no detection or app)
    compliance = 100
    detection_rate = 0
    R0_pars = [test_list, title_parameter, test_param, compliance, detection_rate]
    return R0_pars

# Parameters for heatmap_simulation.py
def heatmap_param():
    """ Function defining the parameters for heatmap_simulation.py
    phase_plane.py allows to create heatmaps exploring the impact of multiple combinations 
    of 2 parameters in the effectiveness of the CT app. The parameter in the inner loop can be defined 
    to be av_reluct_thr or compliance, to reflect the voluntary adoption and imposed adoption scenarios
    respectively. The parameter in the outer loop is defined always to max_adopters. The function can be modified 
    to define the desired range of values to test for both parameters and the name displayed in the output
    plots.
    test_param: Parameter to explore in range1 (either av_reluct_thr or compliance).
    range1: Range of values to explore for the variable defined in test_param. Must be a list with 3 elements: 
    min value, max value, step.
    title_parameter: Name for the test_param variable. It defines the xlabel in the output plots. 
    title_parameter2: Name for the max_adopters variable. It defines the ylabel in the output plots. 
    """
    
    # Prameter to test in the inner loop: 
    scenario= 'voluntary' #"voluntary", "imposed" or "adh_comp"

    if scenario== 'voluntary':
        # Average reluctancy threshold 
        flag_comp=0
        test_param ="av_reluct_thr"
        range1 = [0, 1400, 100]
        title_parameter = 'Av. reluctancy threshold (Incidence/100,000 inh.)' 
        # Max. level of adoption 
        test_param2= "max_adopters" 
        range2 = [0, 75, 5]
        title_parameter2 = 'Max fraction adopters'   
    elif scenario== 'imposed':
        # Fraction compliant users
        flag_comp=1
        test_param ="compliance" 
        range1 = [0,110,10]
        title_parameter = 'Fraction of compliant users (%)' 
        # Max. level of adoption 
        test_param2= "max_adopters" 
        range2 = [0, 75, 5]  
        title_parameter2 = 'Max fraction adopters'  
    elif scenario== 'adh_comp':
        # Fraction compliant users
        flag_comp=2
        test_param ="compliance" 
        range1 = [0,110,10]
        title_parameter = 'Fraction of compliant users (%)' 
        # Average reluctancy threshold 
        test_param2= "av_reluct_thr" 
        range2 =  [0, 1400, 100]
        title_parameter2 = 'Av. reluctancy threshold (Incidence/100,000 inh.)'
    else: 
        print('Unknown scenario')
        exit()

    # Define parameter space to explore 
    test_list = np.arange(range1[0], range1[1], range1[2]) 
    test_list2 = np.arange(range2[0], range2[1],range2[2]) 
    
    # Decimal precision delta 
    rounder=3
    smooth_par=20
    print('Tested range: test_list:' + str(range1) + ' test_list2: ' + str(range2) + ' flag_comp: ' + str(flag_comp))
    phase_param = [range1, range2, test_list, test_list2, [test_param, test_param2], title_parameter, title_parameter2, flag_comp,rounder,smooth_par]
    return phase_param