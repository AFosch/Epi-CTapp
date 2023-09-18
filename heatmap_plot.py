# %% RUN CODE FROM TERMINAL, NOT NOTEBOOK 
#MODIFIED FOR OLD NETWORK
import copy
import importlib
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import ndimage

sns.set_style({'font.family':'sans-serif'})

#%%  PARAMETER DEFINITION 
import parameters_def

importlib.reload(parameters_def)
from parameters_def import baseline_param

[new_network, flag,num_nodes,av_neighbours_l1,error_rate,early_adopt,time_steps,repetitions,beta,mu,epsilon,rho,
leave_quar_prob,detection_rate,max_adopters,compliance,
leave_ref_prob,av_reluct_thr,window_size,epi_scenario] = baseline_param()

from parameters_def import heatmap_param

[range1,range2,test_list,test_list2,test_param,
title_parameter,title_parameter2,flag_comp,rounder,smooth_par] = heatmap_param()
# OVERVIEW SIMULATION
print('Network parameters: '+str(flag) +' Num nodes:'+str(num_nodes)+' '+str([av_neighbours_l1,error_rate]))
print('Simulation parameters: Rep:'+str(repetitions)+' Time:'+str(time_steps))
print('Epidemic Scenario: ' + str(epi_scenario))
print('Epidemic parameters [beta, mu, epsilon, rho, leave_quar_prob, detection_rate]: '+str(np.round([beta,mu, epsilon,rho,leave_quar_prob, detection_rate],3)))
print('App parameters [early_adopters, compliance, leave_ref_prob, av_reluct_thr]: '+str([early_adopt, compliance, leave_ref_prob, av_reluct_thr]))

# %% PLOT with saved delta data 
# Create folder to save plots 

if not os.path.isdir("Plots/"+ flag+'/heatmap'):
        os.makedirs("Plots/"+ flag+'/heatmap')


if epi_scenario =='AL':
    flag_scenario = flag
else: 
    flag_scenario = epi_scenario
  
if flag_comp == 2: #for mixed
    # Load data and define parameters 
    fig3= plt.figure(figsize=(28,11))
    results= np.genfromtxt('Simulations/'+flag+'/heatmap_simulation/'+epi_scenario+'_delta_adcomp_'+flag+'.csv', delimiter=',')
        
    # Create plot grid
    gs=gridspec.GridSpec(ncols=2, nrows=1, figure=fig3)
    f3_ax1 = fig3.add_subplot(gs[0, 0])
    fmt = lambda x,pos: '{:.0f}'.format(x)
    # Heatmap voluntary peak incidence
    heat2= sns.heatmap(results,xticklabels=(np.array(test_list)).astype(int),yticklabels=(np.array(test_list2)).astype(int),
    cbar_kws={'format': FuncFormatter(fmt)}) # annot=True 
    heat2.invert_yaxis()

    #Labels and titles
    plt.ylabel(title_parameter2,fontsize=22)
    plt.xlabel(title_parameter,fontsize=22)
    plt.title (r'$\bf{a) } $'+' '+flag_scenario+': Peak incidence (adherence & compliance)',fontsize='28')

    #Define ticks
    plt.yticks(rotation=0,fontsize=22)
    plt.xticks(rotation=35,fontsize=22)
    ax = plt.gca()
    ax.tick_params()
    cbar = heat2.collections[0].colorbar
    cbar.set_label('Peak incidence reduction (%)',fontsize=22,labelpad=10)
    cbar.ax.tick_params(labelsize=22)

    # Draw isoclines
    results=ndimage.zoom(results,smooth_par)
    cont=heat2.contour(np.linspace(0, len(test_list), len(test_list) * smooth_par),
                    np.linspace(0, len(test_list2), len(test_list2) * smooth_par),
                    results, levels=(5,10,20), colors='lightseagreen')
    plt.clabel(cont,(5,10,20),fontsize=25, inline=True)

    #PREVALENCE 
    results= np.genfromtxt('Simulations/'+flag+'/heatmap_simulation/'+epi_scenario+'_delta_p_adcomp_'+flag+'.csv', delimiter=',')

    f3_ax2 = fig3.add_subplot(gs[0, 1])

    # Heatmap definition: mean peak reduction plot
    fmt = lambda x,pos: '{:.0f}'.format(x)
    heat2= sns.heatmap(results, xticklabels=(np.array(test_list)).astype(int),
            yticklabels=(np.array(test_list2)).astype(int), cbar_kws={'format': FuncFormatter(fmt)}) #annot=True
    heat2.invert_yaxis()

    #Labels and titles
    plt.ylabel(title_parameter2,fontsize=22)
    plt.xlabel(title_parameter,fontsize=22)
    plt.title (r'$\bf{b) } $'+' '+flag_scenario+': Prevalence (adherence & compliance)',fontsize='28')

    #Define ticks
    plt.yticks(rotation=0,fontsize=22)
    plt.xticks(rotation=35,fontsize=22)
    ax = plt.gca()
    ax.tick_params()
    cbar = heat2.collections[0].colorbar
    cbar.set_label('Prevalence reduction (%)',fontsize=22,labelpad=10)
    cbar.ax.tick_params(labelsize=22)

    # Draw isoclines
    results=ndimage.zoom(results,smooth_par)
    cont=heat2.contour(np.linspace(0, len(test_list), len(test_list) * smooth_par),
                    np.linspace(0, len(test_list2), len(test_list2) * smooth_par),
                    results, levels=(5,10,20), colors='lightseagreen')
    plt.clabel(cont,(5,10,20),fontsize=25,inline=True)

    #Adjust and save plots 
    plt.subplots_adjust(wspace=0.12)
    plt.savefig('Plots/'+flag+'/heatmap/'+epi_scenario+'_heatmap_adcomp_'+flag+'.pdf',bbox_inches='tight')
    plt.show()

# Plot voluntary and imposed 
else:  
    fig3= plt.figure(figsize=(26,24))
    smooth_par=20
    gs=gridspec.GridSpec(ncols=2, nrows=2, figure=fig3)
    results= np.genfromtxt('Simulations/'+flag+'/heatmap_simulation/'+epi_scenario+'_delta_comp_'+flag+'.csv', delimiter=',')
    results2= np.genfromtxt('Simulations/'+flag+'/heatmap_simulation/'+epi_scenario+'_delta_'+flag+'.csv', delimiter=',')
    
    max_cb= np.ceil(np.max(np.array([np.max(results), np.max(results2)])))
    min_cb = np.ceil(np.min(np.array([np.min(results), np.min(results2)])))
    
    # Generate figure heatmap
    f3_ax1 = fig3.add_subplot(gs[0, 0])
    plt.title (r'$\bf{a) } $'+' '+ flag_scenario +': Peak incidence (voluntary adoption)',fontsize='28')
    
    #Define parameters for voluntary adoption plot if imposed scenario is defined 
    test_list_og = copy.deepcopy(test_list)
    if flag_comp == 1:
        range1= [0, 1400, 100]
        test_list=  np.arange(range1[0],range1[1],range1[2])
    
    # Define grid 
    fmt = lambda x,pos: '{:.0f}'.format(x)
    heat= sns.heatmap(results2,xticklabels=(np.array(test_list)).astype(int), yticklabels=(np.array(test_list2)).astype(int),
    vmin = min_cb, vmax= max_cb,cbar_kws={'format': FuncFormatter(fmt)}) #annot=True, 
    heat.invert_yaxis()
    plt.ylabel('Max fraction of adopters (%)', fontsize=22)
    results2_2=ndimage.zoom(results2,smooth_par)
    cont=heat.contour(np.linspace(0, len(test_list), len(test_list) * smooth_par),
                    np.linspace(0, len(test_list2), len(test_list2) * smooth_par),
                    results2_2, levels=(5,10,20), colors='lightseagreen',linewidths=4)
    plt.clabel(cont,(5,10,20),fontsize=25,inline=True)

    plt.xlabel('Av. reluctancy threshold (Incidence/100,000 inh.)', fontsize=22)
    plt.xticks(rotation=35, fontsize=22)
    plt.yticks(rotation=0, fontsize=22)
    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label('Peak incidence reduction (%)', fontsize=22,labelpad=10)

    # PEAK INCIDENCE imposed 
    f3_ax2 = fig3.add_subplot(gs[0, 1])
    plt.title (r'$\bf{b) } $'+' '+ flag_scenario+': Peak incidence (imposed adoption)',fontsize='28')
    
    # Define parameters for imposed adoption if scenario is imposed 
    if flag_comp == 1:
        test_list=  test_list_og
    else: 
        range1= [0, 110, 10]  
        test_list=  np.arange(range1[0],range1[1],range1[2])

    # Plot heatmap
    ax = plt.gca()
    ax.tick_params(width=2)
    fmt = lambda x,pos: '{:.0f}'.format(x)
    heat2= sns.heatmap(results,xticklabels=(np.array(test_list)).astype(int),yticklabels=(np.array(test_list2)).astype(int),
    vmin = min_cb, vmax= max_cb, cbar_kws={'format': FuncFormatter(fmt)}) #,annot=True 
    heat2.invert_yaxis()
    # Define labels
    plt.yticks( rotation=0,fontsize=22)
    plt.xticks(rotation=35, fontsize=22)
    
    cbar = heat2.collections[0].colorbar
    cbar.set_label('Peak incidence reduction (%)', fontsize=22,labelpad=10)
    cbar.ax.tick_params(labelsize=22)
    plt.ylabel('Max fraction of adopters (%)', fontsize=22)
    plt.xlabel('Fraction of compliant users (%)', fontsize=22,labelpad=10)
    results=ndimage.zoom(results,smooth_par)
    cont=heat2.contour(np.linspace(0, len(test_list), len(test_list) * smooth_par),
                    np.linspace(0, len(test_list2), len(test_list2) * smooth_par),
                    results, levels=(5,10,20), colors='lightseagreen',linewidths=4)
    plt.clabel(cont,(5,10,20),fontsize=25,inline=True)
    plt.subplots_adjust(wspace=0.12)

    # PREVALENCE PLOT 
    f3_ax3= fig3.add_subplot(gs[1, 1])
    results= np.genfromtxt('Simulations/'+flag+'/heatmap_simulation/'+epi_scenario+'_delta_comp_p_'+flag+'.csv', delimiter=',')
    results2= np.genfromtxt('Simulations/'+flag+'/heatmap_simulation/'+epi_scenario+'_delta_p_'+flag+'.csv', delimiter=',')
    
    # Define common limits colorbar 
    max_cb= np.ceil(np.max(np.array([np.max(results), np.max(results2)])))
    min_cb = np.ceil(np.min(np.array([np.min(results), np.min(results2)])))
    
    # Define heatmap 
    ax = plt.gca()
    ax.tick_params(width=2)
    plt.title (r'$\bf{d) } $'+' '+ flag_scenario+': Prevalence (imposed adoption)',fontsize='28')
    fmt = lambda x,pos: '{:.0f}'.format(x)
    heat2= sns.heatmap(results,xticklabels=(np.array(test_list)).astype(int), yticklabels= (np.array(test_list2)).astype(int),
    vmin = min_cb, vmax= max_cb,cbar_kws={'format': FuncFormatter(fmt)}) #annot=True,
    heat2.invert_yaxis()

    # Define labels and ticks
    plt.yticks(rotation=0,fontsize=22)
    plt.xticks(rotation=35, fontsize=22)
    cbar = heat2.collections[0].colorbar
    cbar.set_label('Prevalence reduction (%)', fontsize=22,labelpad=10)
    cbar.ax.tick_params(labelsize=22)
    plt.ylabel('Max fraction of adopters (%)', fontsize=22)
    plt.xlabel('Fraction of compliant users (%)', fontsize=22,labelpad=10)

    #Define isoclines 
    results=ndimage.zoom(results,smooth_par)
    cont=heat2.contour(np.linspace(0, len(test_list), len(test_list) * smooth_par),
                    np.linspace(0, len(test_list2), len(test_list2) * smooth_par),
                    results, levels=(5,10,20),  colors='lightseagreen',linewidths=4)
    plt.clabel(cont,(5,10,20),fontsize=25,inline=True)
    plt.subplots_adjust(hspace=0.23)

    # PREVALENCE FOR VOLUNTARY ADOPTION
    f3_ax4= fig3.add_subplot(gs[1, 0])
    
    plt.title (r'$\bf{c) } $'+' '+ flag_scenario +': Prevalence (voluntary adoption)',fontsize='28')

    #Define parameters to match 
    if flag_comp == 1:
        range1= [0, 1400,100]
        test_list=  np.arange(range1[0],range1[1],range1[2])
    else: 
        test_list= test_list_og

    # Define heatmaps
    heat= sns.heatmap(results2,xticklabels=(np.array(test_list)).astype(int), yticklabels=(np.array(test_list2)).astype(int),
    vmin = min_cb, vmax= max_cb, cbar_kws={'format': FuncFormatter(fmt)}) #annot=True,  
    heat.invert_yaxis()
    plt.ylabel('Max fraction of adopters (%)', fontsize=22)
    results2_2=ndimage.zoom(results2,smooth_par)
    cont=heat.contour(np.linspace(0, len(test_list), len(test_list) * smooth_par),
                    np.linspace(0, len(test_list2), len(test_list2) * smooth_par),
                    results2_2, levels=(5,10,20),colors='lightseagreen',linewidths=4)
    plt.clabel(cont,(5,10,20),fontsize=25,inline=True)

    plt.xlabel('Av. reluctancy threshold (Incidence/100,000 inh.)', fontsize=22,)
    plt.xticks(rotation=35, fontsize=22)
    plt.yticks(rotation=0, fontsize=22)
    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label('Prevalence reduction (%)', fontsize=22,labelpad=10)

    plt.subplots_adjust(wspace=0.12,hspace=0.23)
    #Save final plot
    plt.savefig('Plots/'+flag+'/heatmap/'+epi_scenario+'_heatmap_' +flag+'.pdf',bbox_inches='tight')#, bbox_inches='tight'
    plt.show()

#%%
 