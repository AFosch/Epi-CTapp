# CTapp_model

This repository contains the code for the article “Characterising the role of human behaviour in the effectiveness of contact-tracing applications” ( [Fosch et al. (2023)](https://doi.org/10.48550/arXiv.2307.13157)).

## Introduction
Here we present a novel approach for modelling the co-evolution of an epidemic outbreak and the implementation of a digital contact tracing system, Contact-Tracing (CT) apps. In this model, app adoption follows a threshold dynamic depending on epidemic progression and heterogeneities in the reluctance threshold. Besides, we also include the effect of the level of compliance with infection reporting. Using this model we explored the interplay between app adoption and epidemic progression and characterized how three human behavioural heterogeneities alter the performance of CT apps. This was achieved by simulating three separate scenarios: the voluntary adoption (assuming complete compliance) the imposed adoption (assuming no reluctancy towards app adoption) and the ``adherence & compliance'' scenario, where the only constraint is the maximum number of people who can download the app (70% of the total population). In Fosch et al.(2023) we use these scenarios to extract a set of recommendations (good practices) that may interest policy-makers when planning to use digital contact tracing systems in future epidemic outbreaks. For more information see [Fosch et al. (2023)](https://doi.org/10.48550/arXiv.2307.13157).

![Image](/Fig1.jpg?raw=true)

## Installation

1. Clone the GitHub repository to your local device. 
    ```bash
    git clone git@github.com:Ariein2/CTapp_model.git
    ```
2. Create a conda environment from the ``environments.yml`` file. For more guidelines on installing conda refer to [conda](https://conda.io/docs/user-guide/install/). 
    ```bash
        conda env create --name epi_app --file environment.yml     
    ```
3. To execute the scripts activate the conda environment using:
    ```bash
        conda activate epi_app
    ```

## Scripts 
- ``parameter_def.py`` = Contains the parameter definition for all scripts. There are two types of parameters. 
    -  ``baseline_param()``: Are imported by all scripts. Please take a look at the definition of the Network structure and the parameters of the dynamics.
    - ``name_of_script_param()``: These parameters are only imported by the script ``name_of_script.py``.

- ``main_simulation.py`` = Runs one simulation of the epidemic-CT app dynamical model using the parameters specified in ``baseline_param()`` and ``main_simulation_param()``. It allows exploring the impact of a parameter of choice (``test_param``) on the effectiveness of the CT app. 
    - ``test_param``: Name of the parameter to explore (*str*). It can be chosen by the user. The most common ones needed are: ``av_reluct_thr``, ``compliance``, ``beta``.
    - ``test_list``: Values to explore for the parameter defined in ``test_param``. Note: It must be written inside a list, even if it only contains one value. 

- ``main_plot.py``: Uses the output Simulations from ``main_simulation.py`` to plot the temporal evolution of the epidemic and CT app adoption dynamics. It contains two types of plots. 
    - ``single_plot()``: Plots the temporal evolution of the epidemic and app adoption dynamics for an uncontrolled outbreak (``base``) and the same outbreak when the intervention is applied (`int`). 
    - ``multiplot()``: Plot the change in the effectiveness of the intervention (`int`) when different values of an epidemic parameter are provided (``test_list`` with multiple values). Compares the performance of CT apps with a range of values in a specific parameter. 

- ``check_network_properties.py``: Compare the properties of the Erdős-Rényi, Scale Free and Negative networks used to represent the population. 
    - Degree distribution plot 
    - Power-law plot. 

- ``R_0_fit``: Plot to estimate the basic reproductive number ($R_0$) for a ranging values of ``beta``. This script requires first to run the ``main_simulation.py``file with ``test_param= 'beta'`` and ``test_list= np.round(np.arange(0.01, 0.065, 0.005),3)`` to save the maximal prevalence estimations. 

- ``heatmap_simulation.py``: Script for simulating the exploratory analysis of 3 human behaviour parameters: ``av_reluct_thr``, ``max_adopters`` and ``compliance``. The script uses parallel computing, please modify the variable ``m_processes`` to select the number of cores to be used. 

    The analysis is performed by exploring 3 different scenarios:
    - ``scenario= 'voluntary'``: Explore the impact of ``max_adopters`` and ``av_reluct_thr`` (``compliance=100``).

    - ``scenario= 'imposed'``: Explore the impact of ``max_adopters`` and ``compliance`` (``av_reluct_thr =0``).

    - ``scenario= 'av_reluct_thr'``: Explore the impact of ``av_reluct_thr`` and ``compliance`` (``max_adopters =70``).

- ``heatmap_plot.py``: Plots the heatmaps that summarise the results form the exploratory analysis of the voluntary, imposed and adh_comp scenarios. Note: The voluntary and imposed adoption scenarios are plot as a single figure and the adh_comp as an independent plot. 

## Replicate Figure 2

1. Edit ``paramters_def.py``: Define the following parameters in ``main_simulation_param()``.
    ```python
    flag = 'NB'
    scenario = 'AL'
    test_param = 'beta'
    test_list = np.round(np.arange(0.01, 0.065, 0.005),3) 
    detection_rate = '0'
    ```

2. Run ``main_simulation.py`` to estimate the max prevalence value for each ``beta`` specified.
    ```bash
        python3 main_simulation.py
    ```
3. Repeat steps 1 and 2 for the Erdős-Rényi and the Scale-free networks  (``flag = 'ER'`` and ``flag = 'SF'``).

3. Run ``R_0_fit.py`` to estimate R0 and generate the plot. 
    ```bash
        python3 R_0_fit.py
    ```

## Replicate Figure 3 
1. Edit ``paramters_def.py``: Define the following parameters in ``main_simulation_param()``.
    ```python
    test_param = 'av_reluct_thr'
    test_list = [230]
    ```
2. Run ``main_simulation.py`` (~12min). Then, run ``main_plot.py`` to obtain Figure 3a. 
    ```bash
        python3 main_simulation.py
        python3 main_plot.py
    ```
3. Change the parametes in ``main_simulation_param()`` to simulate the imposed adoption scenario. Rerun ``main_simulation.py``and ``main_plot.py``.
    ```python
    test_param = 'compliance'
    test_list = [70]
    ```
> Please note the stochastic nature of the simulations may result in slightly different effectiveness of the interventions each time the simulations are executed. 

## Replicate Figure 4 
1. Edit ``paramters_def.py``: Define the ``voluntary``scenario in the function ``heatmap_param()``.  
    ```python
    scenario = 'voluntary'
    ```
2. Edit ``main_simulation.py`` and define the number of cores (``m_processes``) to use in parallel computing (~3h with 20 cores). 

3. Run the simulation 
    ```bash
    python3 heatmap_simulation.py
    ```

4. Repeat steps 1-3 but changing the scenario to the imposed adoption ones. 
    ```python
    scenario = 'imposed'
    ```
5. Execute ``heatmap_plot.py`` to obtain Figure 3. 
<Note, it is necessary to simulate both the voluntary and imposed adoption scenarios before trying to plot the results.>

## Replicate Figure 5 

1. Edit ``paramters_def.py``: Define the ``voluntary``scenario in the function ``heatmap_param()``.  
    ```python
    scenario = 'adh_comp'
    ``` 
2. Run the simulation and plot. 
    ```bash
    python3 heatmap_simulation.py
    python3 heatmap_plot.py
    ```

## Supplementary Results
- **Figure S2:** Run ``check_network_properties.py``. Obtain Figure S2.

- **Figure S3:** Follow instructions for Figure 3 using the following parameters

    **Voluntary adoption:**
    ```python   
        test_param ='av_reluct_thr', 
        test_list = list(np.linspace(100, 3000, 5))
    ```
    **Imposed adoption:**
    ```python
        test_param ='compliance', 
        test_list = list(np.linspace(0, 100, 5))
    ```

- **Figure S4:** Follow instructions for Figure 4 with ``flag = 'ER'``.

- **Figure S5:** Follow instructions for Figure 4 with ``flag = 'SF'``.
- **Figure S6:** Follow instructions for Figure 5 with ``flag = 'ER'``.
- **Figure S7:** Follow instructions for Figure 5 with ``flag = 'SF'``.

- **Figure S8:** Follow instructions for Figure 3 with ``flag ='NB'`` and modifying the epidemic scenarios instead ``epi_scenario = 'AL'``,``epi_scenario = 'S1'`` and ``epi_scenario = 'S2'``. 

- **Figure S7:** Follow instructions for Figure 4 with ``flag = 'NB'`` and ``epi_scenario = 'S1'``.

- **Figure S7:** Follow instructions for Figure 5 with ``flag = 'NB'`` and ``epi_scenario = 'S1'``.

## Cite as: 
Fosch A, Aleta A and Moreno Y (2023) Characterizing the role of human behavior in the effectiveness of contact-tracing applications. Front. Public Health 11:1266989. doi: 10.3389/fpubh.2023.1266989
