# Plot simulations made with main_simulation.py: if only 1 parameter is used the function single_plot()
# is applied. For multiple parameters multiplot() is used instead. 

import importlib
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

#%% LOAD GLOBAL VARIALBES
import parameters_def

importlib.reload(parameters_def)
from parameters_def import baseline_param

[new_network,flag, num_nodes, av_neighbours_l1, error_percent, early_adopt, time_steps, repetitions, beta, mu, epsilon, rho,
 leave_quar_prob, detection_rate, max_adopters, compliance,
 leave_ref_prob, av_reluct_thr,window_size,epi_scenario] = baseline_param()

from parameters_def import main_simulation_param

[title_parameter, test_param, test_list, xview, yview] = main_simulation_param()

# Define flags 
if test_param == 'av_reluct_thr':
    flag_par='a'
elif test_param =='compliance':
    flag_par='c'
elif test_param == 'beta':
    flag_par = 'b'
else: 
    flag_par = 'e'

# Define title labels
if flag == 'NB':
    flag_text='Negative Binomial'
elif test_param =='compliance':
    flag_text='Scale Free'

else: 
    flag_text = 'Erd\H{o}s-R\'{e}nyi'	


# %% Functions single and multiplot

def single_plot ():
        
    t_min_list = []
    t_max_list = []
    # Compute time range for all simulations
    for ks in range(0, len(test_list)):
        time_rep = np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_time_rep_'+flag_par+'_' + flag + '.npy', allow_pickle=True).item()[ks]
        t_min_list.append(time_rep[0])
        t_max_list.append(time_rep[-1])

    x_lim = [np.min(t_min_list), np.max(t_max_list)]

    # Plot incidence curve  
    fig = plt.figure(1, figsize=[15, 10])
    if flag_par == 'a':
        plt.suptitle(r"a) " + flag_text + ": Voluntary adoption", fontweight='bold',
                fontsize='28')
    elif flag_par == 'c':
        plt.suptitle(r"b) " + flag_text + ": Imposed adoption", fontweight='bold',
                fontsize='28')
    else: 
        plt.suptitle("Incidence", fontsize='32')


    for ks in range(0, len(test_list)):
        # Select data to plot
        I_base = (100000/num_nodes)*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_incidence_base_'+flag_par+'_' + flag + '.npy', allow_pickle=True).item()[ks]
        I_int = (100000/num_nodes)*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_incidence_int_'+flag_par+'_' + flag + '.npy', allow_pickle=True).item()[ks]

        App_base =100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_App_base_'+flag_par+'_' + flag+  '.npy', allow_pickle=True).item()[ks]
        App_int = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_App_int_'+flag_par+'_' + flag+  '.npy', allow_pickle=True).item()[ks]

        time_all = np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_time_rep_'+flag_par+'_' + flag+ '.npy', allow_pickle=True).item()[ks]

        # I repetitions mean and confidence interval
        window_size=7
        I_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=I_base, axis=1))
        App_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=App_base, axis=1))

        mean_rep=np.mean(I_base_conv, axis=0)
        inf_perc = np.quantile(I_base_conv, 0.025, axis=0)
        sup_perc = np.quantile(I_base_conv, 0.975, axis=0)
        
        I_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=I_int, axis=1))
        App_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=App_int, axis=1))

        mean_rep2 = np.mean(I_int_conv, axis=0)

        inf_perc2 = np.quantile(I_int_conv, 0.025, axis=0)
        sup_perc2 = np.quantile(I_int_conv, 0.975, axis=0)

        # App plot mean and confidence interval 
        app_mean_rep = np.mean(App_base_conv, axis=0)
        app_inf_perc = np.quantile(App_base_conv, 0.025, axis=0)
        app_sup_perc = np.quantile(App_base_conv, 0.975, axis=0)

        app_mean_rep2 = np.mean(App_int_conv, axis=0)
        app_inf_perc2 = np.quantile(App_int_conv, 0.025, axis=0)
        app_sup_perc2 = np.quantile(App_int_conv, 0.975, axis=0)

        # Define grid plots 
        outer = gridspec.GridSpec(int(np.ceil(np.sqrt(len(test_list)))),
                                int(np.round(np.sqrt(len(test_list)))))

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[ks], hspace=0.07)
        
        # Epidemic plot:
        ax0 = fig.add_subplot(inner[0])
        ax0.axhline(np.max(mean_rep), color='silver', linestyle='dashed', linewidth=2, zorder=1)
        ax0.axhline(np.max(mean_rep2), color='silver', linestyle='dashed', linewidth=2, zorder=1)

        ax0.text(np.min(x_lim[0]) +5, np.max(mean_rep) + 10000*0.01,
                '∆='"{:0.2f}".format((1 - (np.max(mean_rep2) / np.max(mean_rep))) * 100) + '%', size=28)
        ax0.plot(time_all, mean_rep, color='#4b1d95', label='Baseline', linewidth=3, zorder=2,linestyle='dashed')
        ax0.plot(time_all, mean_rep2, color='#197180', label='Effective app', linewidth=3, zorder=3)
        ax0.fill_between(time_all, inf_perc, sup_perc, alpha=.3, color='#4b1d95', zorder=2)
        ax0.fill_between(time_all, inf_perc2, sup_perc2, alpha=.3, color='#197180', zorder=3)
        plt.title(str(title_parameter) + '= ' + str(test_list[ks]), size=30)

        # Define panel title
        if test_param == 'av_reluct_thr':
            plt.title(str(title_parameter) + ' = ' + str(int(test_list[ks])), size=30)    
        else: 
            plt.title(str(title_parameter) + ' = ' + str(int(test_list[ks]))+'%', size=30)    #str(int(100*test_list[ks]))+'%', size=30)   
        
        # View y axis:
        if yview== 'crop':
            plt.ylim(top=100000*0.035)
        
        #View x axis
        if xview== 'all':
            plt.xlim(x_lim)
        else: 
            plt.xlim([x_lim[0], 300])

        plt.setp(ax0.get_yticklabels(), size=25)
        ax0.tick_params(direction='in')
        ax0.set_yticks(np.arange(0,5000,1000))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.ylabel('Incidence/100,000 inh.', size=25)
        plt.legend(handlelength=1, loc='upper right', prop={'size': 30})

        # APP PLOT: 
        ax1 = fig.add_subplot(inner[1])
        ax1.axhline(70, color='silver', linestyle='dashed')
        ax1.plot(time_all, app_mean_rep, color='#4b1d95', label='Baseline', linewidth=3, linestyle='dashed')
        ax1.plot(time_all, app_mean_rep2, color='#197180', label='Effective app', linewidth=3)
        ax1.fill_between(time_all, app_inf_perc, app_sup_perc, alpha=.3, color='#4b1d95')
        ax1.fill_between(time_all, app_inf_perc2, app_sup_perc2, alpha=.3, color='#197180')

        # View y axis:
        plt.ylim(top= 75)

        # View x axis
        if xview== 'all':
            plt.xlim(x_lim)
        else: 
            plt.xlim([x_lim[0], 300])

        plt.setp(ax1.get_xticklabels(), size=25)
        plt.yticks(np.arange(0, 80, 20), size=28) # plt.yticks(100*np.arange(0, 0.8, 0.2), size=28)
        plt.setp(ax1.get_yticklabels(), size=25)
        plt.ylabel('App adoption (%)', size=25, labelpad = 30)
        plt.xlabel('t (days)', size=28)

        print('Percentage of mean peak reduction' + ': ' +
            "{:0.2f}".format((1 - (np.max(mean_rep2) / np.max(mean_rep))) * 100) + '%')
    plt.tight_layout()

    if not os.path.isdir('Plots/'+flag+'/single_plot/'):
            os.makedirs('Plots/'+flag+'/single_plot/')
    # Save plot
    plt.savefig('Plots/'+flag+'/single_plot/'+epi_scenario+'_temporal_evolution_'+flag_par+'_' + flag + '.pdf')

    # Prevalence plot 
    plt.figure(2, figsize=[15, 10])
    if flag_par == 'a':
        plt.suptitle(r"a) " + flag_text + ": Voluntary adoption", fontweight='bold',
                fontsize='28')
    elif flag_par == 'c':
        plt.suptitle(r"b) " + flag_text + ": Imposed adoption", fontweight='bold',
                fontsize='28')
    else: 
        plt.suptitle(flag + ": Prevalence", fontweight='bold',
                fontsize='28')
    
    # PREVALENCE PLOTS 
    final_prev_base = np.zeros(len(test_list))
    final_prev_int = np.zeros(len(test_list))

    for ks in range(0, len(test_list)):
        # Select data to plot
        time_all = np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_time_rep_'+flag_par+'_' + flag+ '.npy', allow_pickle=True).item()[ks]
        prevalence_base = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_prevalence_base_'+flag_par+'_' + flag + '.npy', allow_pickle=True).item()[ks] / num_nodes
        prevalence_int = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_prevalence_int_'+flag_par+'_' + flag + '.npy', allow_pickle=True).item()[ks] / num_nodes

        # Mean and CI 
        prev_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=prevalence_base, axis=1))
        prev_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=prevalence_int, axis=1))

        prev_mean_rep = np.mean(prev_base_conv, axis=0)
        final_prev_base[ks] = prev_mean_rep[-1]
        prev_inf_perc = np.quantile(prev_base_conv, 0.025, axis=0)
        prev_sup_perc = np.quantile(prev_base_conv, 0.975, axis=0)

        prev_mean_rep2 = np.mean(prev_int_conv, axis=0)
        final_prev_int[ks] = prev_mean_rep2[-1]

        prev_inf_perc2 = np.quantile(prev_int_conv, 0.025, axis=0)
        prev_sup_perc2 = np.quantile(prev_int_conv, 0.975, axis=0)

        # Plot prevalence
        ax0 = plt.subplot(int(np.ceil(np.sqrt(len(test_list)))), int(np.round(np.sqrt(len(test_list)))), ks + 1)
        plt.axhline(np.max(prev_mean_rep), color='silver', linestyle='dashed', linewidth=2, zorder=1)
        plt.axhline(np.max(prev_mean_rep2), color='silver', linestyle='dashed', linewidth=2, zorder=1)

        plt.plot(time_all, prev_mean_rep, color='#4b1d95', label='Baseline', linewidth=3, linestyle='dashed',zorder=1)
        plt.plot(time_all, prev_mean_rep2, color='#197180', label='Effective app', linewidth=3, zorder=3)
        plt.fill_between(time_all, prev_inf_perc, prev_sup_perc, alpha=.3, color='#4b1d95', zorder=1)
        plt.fill_between(time_all, prev_inf_perc2, prev_sup_perc2, alpha=.3, color='#197180', zorder=2)
        plt.text(np.min(x_lim[0]) + 10, np.max(prev_mean_rep) + 100*0.015,
                '∆='"{:0.2f}".format((1 - (np.max(prev_mean_rep2) / np.max(prev_mean_rep))) * 100) + '%', size=28)

        if test_param == 'av_reluct_thr':
            plt.title(str(title_parameter) + ' = ' + str(int(test_list[ks])), size=30)    
        else: 
            plt.title(str(title_parameter) + ' = ' + str(int(test_list[ks]))+'%', size=30) #str(int(100*test_list[ks]))+'%', size=30)     
        plt.setp(ax0.get_yticklabels(), size=25)
        plt.setp(ax0.get_xticklabels(), size=25)
        plt.ylabel('Cummulative infected fraction (%) ', size=28)
        plt.xlabel('t (days)', size=28)
        
        # View y axis:
        plt.ylim(top=100)
        #View x axis

        if xview== 'crop':
            plt.xlim([x_lim[0], 300])
        
            
    plt.tight_layout()
    plt.legend(handlelength=1, loc='lower right', prop={'size': 28})

    # Save data for mean prevalence
    np.savetxt('Simulations/'+str(flag)+'/main_plot/'+epi_scenario+'_max_prev_base_'+flag_par+'_' + flag + '.csv', final_prev_base, delimiter=',')
    np.savetxt('Simulations/'+str(flag)+'/main_plot/'+epi_scenario+'_max_prev_int_' +flag_par+'_' + flag + '.csv', final_prev_int, delimiter=',')

    # Save plot prevalence
    plt.savefig('Plots/'+flag+'/single_plot/'+epi_scenario+'_prev_'+flag_par+'_' + flag + '.pdf')
    plt.show()

def multiplot ():
    
    if test_param == 'av_reluct_thr':
        flag_par='a'
    elif test_param =='compliance':
        flag_par='c'
    elif test_param =='beta':
        flag_par='b'
    else: 
        flag_par = 'e'

    t_min_list = []
    t_max_list = []
    # Compute time range for all simulations
    for ks in range(0, len(test_list)):
        time_rep = np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_time_rep_'+flag_par+'_' + flag + multi+'.npy', allow_pickle=True).item()[ks]
        t_min_list.append(time_rep[0])
        t_max_list.append(time_rep[-1])

    x_lim = [np.min(t_min_list), np.max(t_max_list)]

    # Plot incidence curve  
    fig = plt.figure(1,figsize=[15, 10])
    outer = gridspec.GridSpec(2,1)
    # Epidemic plot:
    ax0 = fig.add_subplot(outer[0])
    ax1 = fig.add_subplot(outer[1])

    ax1.axhline(70, color='silver', linestyle='dashed',linewidth=3)

    #titles
    if flag_par == 'a':
        plt.suptitle(r'$\bf{a)} $'+' '+ flag + ": Temporal evolution of incidence (voluntary adoption)",
                fontsize='28')
    elif flag_par == 'c':
        plt.suptitle(r'$\bf{b)} $'+' '+ flag + ": Temporal evolution of incidence (imposed adoption)",
                fontsize='28')
    else: 
        plt.suptitle("Temporal evolution of incidence", fontsize='28')


    for ks in range(0, len(test_list)):
        # Select data to plot
        I_base = 10*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_incidence_base_'+flag_par+'_' + flag+ multi+ '.npy', allow_pickle=True).item()[ks]
        I_int = 10*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_incidence_int_'+flag_par+'_' + flag+ multi+ '.npy', allow_pickle=True).item()[ks]

        App_base = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_App_base_'+flag_par+'_' + flag+ multi+ '.npy', allow_pickle=True).item()[ks]
        App_int = 100*np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_App_int_'+flag_par+'_' + flag+ multi+ '.npy', allow_pickle=True).item()[ks]

        time_all = np.load('Simulations/'+flag+'/main_simulation/'+epi_scenario+'_time_rep_'+flag_par+'_' + flag+ multi+ '.npy', allow_pickle=True).item()[ks]

        #NEW
        window_size=7
        I_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=I_base, axis=1))
        App_base_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=App_base, axis=1))

        mean_rep=np.mean(I_base_conv, axis=0)

        I_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=I_int, axis=1))
        App_int_conv=np.array(np.apply_along_axis(lambda m: np.convolve(m,np.ones(int(window_size))/float(window_size),'same'),arr=App_int, axis=1))

        mean_rep2 = np.mean(I_int_conv, axis=0)

        # App plot mean and confidence interval 
        app_mean_rep2 = np.mean(App_int_conv, axis=0)

        if flag_par == 'c':
            labels = (test_list[ks]).astype(int)  #labels = (100*test_list[ks]).astype(int)
            end = '%'
        else: 
            labels = (test_list[ks]).astype(int)
            end = ''
        ax0.plot(time_all, mean_rep2,label=title_parameter+' = %.0f' %(labels)+ end, linewidth=3, zorder=3)
        ax0.set_ylabel('Infected/100,000 inh.', size=25)

        #View x axis
        if xview== 'all':
            ax0.set_xlim(x_lim) 
            ax1.set_xlim(x_lim) 
        else: 
            ax0.set_xlim([x_lim[0], 250])  
            ax1.set_xlim([x_lim[0], 250]) 
        
        plt.setp(ax0.get_yticklabels(), size=25)
        ax0.tick_params(direction='in')
        ax0.set_yticks(np.arange(0,3500,1000))

        plt.setp(ax0.get_xticklabels(), visible=False)
    
        # App plot: 
        ax1.plot(time_all, app_mean_rep2 , label=labels, linewidth=3) #App_int

        plt.setp(ax1.get_xticklabels(), size=25)
        plt.yticks(100*np.arange(0, 1, 0.2), size=25)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
        plt.ylabel('App adoption (%)', size=25,labelpad = 30)
        plt.xlabel('t (days)', size=28)

    if not os.path.isdir('Plots/'+flag+'/multiplot'):
            os.makedirs('Plots/'+flag+'/multiplot')
    # Save plot
    plt.tight_layout()
    plt.savefig('Plots/'+flag+'/multiplot/'+epi_scenario+'_temporal_evolution_'+flag_par+'_' + flag+ multi+ '.pdf')
    return ()

# %% MAIN CODE 

#Check scenario and execute adequate plot function
if not os.path.isdir('Simulations/'+flag+'/main_plot'):
        os.makedirs('Simulations/'+flag+'/main_plot')
if len(test_list) ==1:
    multi = ''
    single_plot()
    plt.show()
else:   
    multi='m'
    multiplot()
    plt.show()

# %%