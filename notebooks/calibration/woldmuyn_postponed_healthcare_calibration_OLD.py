"""
This script contains a calibration of an PI constraint postponed healthcare model using data from 2020-2021.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import json
import argparse
import sys,os
import random
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import csv
# pySODM packages
from pySODM.models.base import BaseModel
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian
# COVID-19 package
from covid19model.data.sciensano import get_sciensano_COVID19_data

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-08-01')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate. Format 'YYYY-MM-DD'.",default='2021-02-01')
parser.add_argument("-n_nm", "--n_nm", help="Maximum number of Nelder Mead iterations.", default=20)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default =20)
parser.add_argument("-ID", "--identifier", help="Name in output files.", default = 'test')
parser.add_argument("-MDC", "--included_MDC", help="MDC classes to model, setting to 'all' equals all MDC keys. \ndefault=['01','05','06','08']",default=['01','05','06','08'])
parser.add_argument("-MDC_plot", "--MDC_plot", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['01','05','06','08']",default=['01','05','06','08'])
args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True
# HPC
if args.high_performance_computing == False:
    high_performance_computing = True
else:
    high_performance_computing = False
# Identifier (name)
if args.identifier:
    identifier = str(args.identifier)
else:
    raise Exception("The script must have a descriptive name for its output.")
# Maximum number of PSO iterations
n_nm = int(args.n_nm)
# Maximum number of MCMC iterations
n_mcmc = int(args.n_mcmc)
# Date at which script is started
run_date = str(datetime.date.today())
# Keep track of runtime
initial_time = datetime.datetime.now()
# Start and end of calibration
start_date = pd.to_datetime(args.start_calibration)
if args.end_calibration:
    end_date = pd.to_datetime(args.end_calibration)

abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/interim/QALY_model/postponement_non_covid_care/UZG/'
# MDC_dict to translate keys to disease group
MDC_dict={}
file_name = 'MDC_dict.csv'
with open(os.path.join(abs_dir,rel_dir,file_name), mode='r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        MDC_dict.update({row[0]:row[1]})

MDC_dict.pop('.')
MDC_keys = list(MDC_dict.keys())

# MDC classes to model
if args.included_MDC == 'all':
    included_MDC = np.array(list(MDC_keys))
else:
    if isinstance(args.included_MDC, str):
        included_MDC = np.array(args.included_MDC[1:-1].split(','))
    else:
        included_MDC = np.array(args.included_MDC)

# MDC classes to plot
if args.MDC_plot == 'all':
    MDC_plot  = np.array(list(MDC_keys))
else:
    if isinstance(args.MDC_plot , str):
        MDC_plot  = np.array(args.MDC_plot [1:-1].split(','))
    else:
        MDC_plot  = np.array(args.MDC_plot )

##################
## Define model ##
##################

class constrained_PI(BaseModel):
    """
    Test constrainted PI model
    """
    
    state_names = ['H','E']
    parameter_names = ['covid_H','covid_dH']
    parameters_stratified_names = ['Kp', 'Ki', 'alpha','gamma', 'covid_capacity']
    stratification_names = ['APR_MDC_key']
    
    @staticmethod
    def integrate(t, H, E, covid_H, covid_dH, Kp, Ki, alpha, gamma, covid_capacity):

        epsilon = 1-H
        dE = epsilon - gamma*E
        u = np.where(covid_H <= covid_capacity, Kp * np.where(E<=0,epsilon,0) + Ki * np.where(E>0,E,0), 0)
        dH = -alpha*covid_dH + u

        return dH, dE 

class PI(BaseModel):
    """
    Test PI model
    """
    
    state_names = ['H','E']
    parameter_names = ['covid_H']
    parameters_stratified_names = ['Kp', 'Ki', 'alpha', 'gamma']
    stratification_names = ['APR_MDC_key']
    
    @staticmethod
    def integrate(t, H, E, covid_H, Kp, Ki, alpha, gamma):

        epsilon = 1-H
        dE = epsilon - gamma*E
        u = Kp*epsilon + Ki*E
        dH = -alpha*covid_H + u

        return dH, dE 

#################
## Define TDPF ##
#################

from functools import lru_cache

class get_covid_H():

    def __init__(self, data):
        self.data = data

    def H_wrapper_func(self, t, states, param):
        return self.__call__(t)

    @lru_cache()
    def __call__(self, t):
        t = pd.to_datetime(t).round(freq='D')
        try:
            covid_H = self.data.loc[t]
        except:
            covid_H = 0
        return covid_H  

class get_covid_dH():

    def __init__(self, data):
        self.data = data

    def dH_wrapper_func(self, t, states, param):
        return self.__call__(t)

    @lru_cache
    def __call__(self, t):
        t = pd.to_datetime(t).round(freq='D')
        try:
            covid_dH = self.data.loc[t]
        except:
            covid_dH = 0

        return covid_dH

#################
## Setup model ##
#################

def init_model(model_class, included_MDC=included_MDC):
    n = len(included_MDC)
    # params
    if model_class.__name__ == "PI":
        # params
        alpha = (3.0e-05*np.ones(n))
        gamma = (0.02*np.ones(n))

        # PI params
        Kp = (5.91e-04*np.ones(n))
        Ki = (8.22*5.91e-04*np.ones(n))
    else:
        # params
        alpha = (5e-04*np.ones(n))
        gamma = (5e-02*np.ones(n))

        # PI params
        Kp = (7e-02*np.ones(n))
        Ki = (7e-03*np.ones(n))

    # initial values covid params
    covid_H = 0
    covid_dH = 0
    covid_capacity = (400*np.ones(n))

    global_params_dict={'alpha':alpha,
            'gamma':gamma,
            'Kp':Kp,
            'Ki':Ki,
            'covid_H':covid_H,
            'covid_dH':covid_dH,
            'covid_capacity':covid_capacity
            }

    #time dependent functions
    H_function = get_covid_H(df_covid_H_tot).H_wrapper_func
    dH_function = get_covid_dH(df_covid_dH).dH_wrapper_func
    global_time_dependent_parameters={'covid_H': H_function,'covid_dH':dH_function}

    # Initialize model
    init_states = {'H':np.ones(n),'E':np.zeros(n)}
    coordinates={'APR_MDC_key': included_MDC}

    # param and time dependent param dictionary specific for model class
    params_dict = {}
    time_dependent_parameters = {}

    for param in model_class.parameter_names + model_class.parameters_stratified_names:
        params_dict.update({param:global_params_dict[param]})
        if param in list(global_time_dependent_parameters.keys()):
            time_dependent_parameters.update({param:global_time_dependent_parameters[param]})

    # Initialize model
    model = model_class(init_states,params_dict,coordinates,time_dependent_parameters=time_dependent_parameters)
    return model

if __name__ == '__main__':

    ##############################
    ## Define results locations ##
    ##############################

    abs_dir = os.path.dirname(__file__)
    # Path where traceplot and autocorrelation figures should be stored.
    # This directory is split up further into autocorrelation, traceplots
    fig_path = os.path.join(abs_dir,'../../results/calibrations/Postponed_Healthcare/')
    # Path where MCMC samples should be saved
    samples_path = os.path.join(abs_dir,'../../data/interim/model_parameters/Postponed_Healthcare/calibrations/')
    # Path where samples backend should be stored
    backend_folder = os.path.join(abs_dir,'../../results/calibrations/Postponed_Healthcare/')
    # Verify that the paths exist and if not, generate them
    for model_subdirectory in ['constrained_PI/','PI/']:
        for directory in [fig_path+model_subdirectory, samples_path+model_subdirectory, backend_folder+model_subdirectory]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    # Verify that the fig_path subdirectories used in the code exist
    for model_subdirectory in ['constrained_PI/','PI/']:
        for directory in [fig_path+model_subdirectory+"autocorrelation/", fig_path+model_subdirectory+"traceplots/", fig_path+model_subdirectory+"fits/", fig_path+model_subdirectory+"variance/"]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    ###############
    ## Load data ##
    ###############

    # Postponed healthcare data
    abs_dir = os.path.dirname(__file__)
    rel_dir = '../../data/interim/QALY_model/postponement_non_covid_care/UZG/'
    file_name = '2020_2021_normalized.csv'
    types_dict = {'APR_MDC_key': str}
    # mean data
    hospitalizations_normalized = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)['mean']
    hospitalizations_normalized = hospitalizations_normalized.reorder_levels(['date','APR_MDC_key'])
    hospitalizations_normalized=hospitalizations_normalized.sort_index()
    MDC_keys = hospitalizations_normalized.index.get_level_values('APR_MDC_key').unique().values

    # lower and upper quantiles
    hospitalizations_normalized_quantiles = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True).loc[(slice(None), slice(None)), ('q0.025','q0.975')]
    hospitalizations_normalized_quantiles = hospitalizations_normalized_quantiles.reorder_levels(['date','APR_MDC_key'])
    hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.sort_index()

    # COVID-19 data
    covid_data, _ , _ , _ = get_sciensano_COVID19_data(update=False)
    new_index = pd.MultiIndex.from_product([pd.to_datetime(hospitalizations_normalized.index.get_level_values('date').unique()),covid_data.index.get_level_values('NIS').unique()])
    covid_data = covid_data.reindex(new_index,fill_value=0)
    df_covid_H_in = covid_data['H_in'].loc[:,40000]
    df_covid_H_tot = covid_data['H_tot'].loc[:,40000]
    df_covid_dH = df_covid_H_tot.diff().fillna(0)

    ##########################################################
    ## Compute the overdispersion parameters for our H data ##
    ##########################################################

    sigmas_dict = {}
    data=hospitalizations_normalized.loc[start_date:end_date]
    results, ax = variance_analysis(data, resample_frequency='W')
    for MDC_key in MDC_keys:
        sigmas_dict.update({MDC_key:results['theta'].loc[MDC_key,'gaussian']})
    plt.close()
    
    #############################
    ## plot initial conditions ##
    #############################

    model_classes = [constrained_PI,PI]

    # plot 
    fig,axs = plt.subplots(len(model_classes)+2,sharex=True,figsize=(8,6))
    plot_time = pd.date_range(start_date,end_date)
    idx = 0
    axs[idx].plot(df_covid_H_tot.loc[plot_time])
    axs[idx].set_title('covid_H')

    idx = 1
    axs[idx].plot(plot_time,hospitalizations_normalized.loc[plot_time,'05'].values)
    axs[idx].set_title('data MDC:05')
    axs[idx].xaxis.set_major_locator(plt.MaxNLocator(4))
    axs[idx].grid(False)
    axs[idx].tick_params(axis='both', which='major', labelsize=10)
    axs[idx].tick_params(axis='both', which='minor', labelsize=8)

    models = []
    for idx,model_class in enumerate(model_classes):
        model = init_model(model_class,included_MDC==['05'])
        models.append(model)
        out = model.sim([start_date, end_date])

        idx += 2
        axs[idx].plot(plot_time,out['H'][0])
        axs[idx].set_title(model_class.__name__)
        axs[idx].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[idx].grid(False)
        axs[idx].tick_params(axis='both', which='major', labelsize=10)
        axs[idx].tick_params(axis='both', which='minor', labelsize=8)

        for idx in range(len(axs)-1):
            axs[idx+1].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 

    plt.subplots_adjust(hspace=0.5)
    #plt.show()
    plt.savefig(os.path.join(fig_path,str(identifier)+'_init_condition.pdf'))
    plt.close()

    ######################
    ## Calibrate models ##
    ######################

    # MCMC settings
    multiplier_mcmc = 10
    max_n = n_mcmc
    print_n = 10
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))

    # parameters and bounds
    pars_global = ['Kp', 'Ki', 'alpha', 'gamma','covid_capacity']
    labels_global = ['$K_p$', '$K_i$', '$\\alpha$', '$\\gamma$','covid_capacity']
    bounds_global = [(1e-05,1000),(1e-05,1000),(1e-05,1),(1e-5,1),(1e-05,1000)]

    # calibration per model and per MDC key
    thetas_dict = {}
    samples_dict = {}

    for model_class in model_classes:
        model_name = model_class.__name__
        thetas_dict.update({model_name:{}})
        samples_dict.update({model_name:{}})
        theta = []

        # pars specific for model class
        pars = []
        labels = []
        bounds = []
        for param in model_class.parameter_names + model_class.parameters_stratified_names:
            if param in pars_global:
                id = pars_global.index(param)
                pars.append(param)
                labels.append(labels_global[id])
                bounds.append(bounds_global[id])

        for MDC_key in included_MDC:
            # Define dataset
            data=[hospitalizations_normalized.loc[start_date:end_date, MDC_key].to_frame(),]
            states = ["H",]
            weights = np.array([1,]) 
            # Setup likelihood functions and arguments
            log_likelihood_fnc = [ll_gaussian,]
            log_likelihood_fnc_args = [[sigmas_dict[MDC_key]],]
            # Setup prior functions and arguments
            log_prior_fnc = len(bounds)*[log_prior_uniform,]
            log_prior_fnc_args = bounds

            # init model (nodig, want coordinaten kloppen anders niet)
            model = init_model(model_class,included_MDC=[MDC_key])

            # Setup objective function without priors and with negative weights 
            objective_function = log_posterior_probability([],[],model,pars,bounds,data,states,
                                                log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)
 
            # wanneer eerste nelder_mead voor model: neem initiele parameters
            # anders neemt theta resultaat van vorige nelder mead
            # MAAR 01 heeft geen inhaalbeweging, dus worden ze allemaal plat..., tg beter init theta nemen
            # if not np.any(theta):
            theta = np.array([model.parameters[param] for param in pars]).squeeze()

            string = 'Calibrating {model} for MDC:{MDC}'.format(model=model_name,MDC=MDC_key)
            print("*"*len(string)+"\n"+string+"\n"+"*"*len(string))

            # Nelder-mead
            step = len(bounds)*[0.40,]
            theta,_ = nelder_mead.optimize(objective_function, theta, bounds, step, processes=processes, max_iter=n_nm)
            thetas_dict[model_name].update({MDC_key:theta})

            theta = np.where(theta==0,0.00001,theta)
            # MCMC
            # Perturbate previously obtained estimate
            ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.3,]*len(bounds), multiplier=multiplier_mcmc, bounds=log_prior_fnc_args, verbose=True)
            # initialize objective function
            objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
            # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
            settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                        'labels': labels, 'parameters': pars, 'starting_estimate': list(theta)}
            # Start calibration
            sys.stdout.flush()
            # Sample n_mcmc iterations
            sampler = run_EnsembleSampler(pos, max_n, identifier+'_'+model_name+'_'+MDC_key, objective_function, (), (),
                                            fig_path=fig_path+model_name, samples_path=samples_path+model_name, print_n=print_n, backend=None, processes=processes, progress=True,
                                            settings_dict=settings) 
            # Generate a sample dictionary
            # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
            thin = 1
            try:
                autocorr = sampler.get_autocorr_time(flat=True)
                thin = max(1,int(0.5 * np.min(autocorr)))
            except:
                pass
            chain_length = sampler.get_chain(discard=0, thin=thin, flat=True).shape[0]
            if sampler.get_chain(discard=int(chain_length*0.5), thin=thin, flat=True).shape[0] > 20:
                discard = int(chain_length*0.5)
            else:
                discard = 0
            samples_dict[model_name].update({MDC_key:emcee_sampler_to_dictionary(sampler, pars, discard=discard, thin=thin, settings=settings)})
            
    ####################
    ## saving results ##
    ####################
    # Save samples dictionary to json for long-term storage: _SETTINGS_ and _BACKEND_ can be removed at this point

    with open(os.path.join(samples_path, str(identifier)+'_SAMPLES_'+run_date+'.json'), 'w') as fp:
        json.dump(samples_dict, fp)

    # Extract formatted parameter_names, bounds and labels
    # pars_postprocessing = objective_function.parameter_names_postprocessing
    # labels = objective_function.labels 
    # bounds = objective_function.bounds

    ######################
    ## Visualize result ##
    ######################

    # Define dataset
    data=hospitalizations_normalized.loc[start_date:end_date, MDC_plot]

    for model_class in model_classes:
        model_name = model_class.__name__
        model = init_model(model_class,MDC_plot)

        # pars specific for model class
        pars = []
        labels = []
        bounds = []
        for param in model_class.parameter_names + model_class.parameters_stratified_names:
            if param in pars_global:
                id = pars_global.index(param)
                pars.append(param)
                labels.append(labels_global[id])
                bounds.append(bounds_global[id])

        # objective function 
        objective_function = log_posterior_probability([],[],model,pars,bounds,[data,],["H",],[ll_gaussian,],[[sigmas_dict[key] for key in MDC_plot],],-np.array([1,]))
        scores = []
                
        ###############
        # Nelder-Mead #
        ###############

        # Assign thetas to model
        param_dict = {}
        theta_per_MDC = thetas_dict[model_name]
        for idx,param in enumerate(pars):
            param_dict.update({param:[]})
            for MDC_key in MDC_plot:
                param_dict[param].append(theta_per_MDC[MDC_key][idx])
            param_dict.update({param:np.array(param_dict[param])})
        model.parameters.update(param_dict)

        # simulate model
        out_nm = model.sim([start_date, end_date])
        scores.append(objective_function.compute_log_likelihood(out_nm.copy(), objective_function.states, objective_function.data, objective_function.weights, objective_function.log_likelihood_fnc, objective_function.log_likelihood_fnc_args, objective_function.n_log_likelihood_extra_args))
        ########
        # MCMC #
        ########
        samples_per_MDC = samples_dict[model_name]

        # Define draw function
        def draw_fcn(param_dict, samples_dict):
            param_dict = {}
            for param in pars:
                param_dict.update({param:[]})
                for MDC_key in MDC_plot:
                    samples_for_MDC = samples_dict[MDC_key]
                    idx = random.randint(0,len(samples_for_MDC[list(samples_for_MDC.keys())[0]])-1)
                    param_dict[param].append(samples_for_MDC[param][idx])
                param_dict.update({param:np.array(param_dict[param])})
            return param_dict

        # Simulate model
        N=50
        out_mcmc = model.sim([start_date, end_date], N=N, samples=samples_per_MDC, draw_fcn=draw_fcn, processes=processes)
        scores.append(objective_function.compute_log_likelihood(out_mcmc.mean(dim='draws'), objective_function.states, objective_function.data, objective_function.weights, objective_function.log_likelihood_fnc, objective_function.log_likelihood_fnc_args, objective_function.n_log_likelihood_extra_args))
        
        ########
        # plot #
        ########
        algorithms = ['Nelder-Mead','MCMC']
        fig,axs = plt.subplots(len(MDC_plot),2,sharex=True,sharey=True,figsize=(8,6))
        for i,MDC_key in enumerate(MDC_plot):
            for j,out in enumerate([out_nm,out_mcmc]):
                #data
                axs[i,j].plot(plot_time,data.loc[plot_time,MDC_key], label='data', alpha=0.7)
                axs[i,j].fill_between(plot_time,hospitalizations_normalized_quantiles['q0.025'].loc[plot_time,MDC_key],
                                        hospitalizations_normalized_quantiles['q0.975'].loc[plot_time,MDC_key], alpha=0.2, label='uncertainty on data')
                #simulation
                if j==0:
                    axs[i,j].plot(plot_time,out.sel(date=plot_time,APR_MDC_key=MDC_key)['H'],label='simulated_H',color='black')
                else:    
                    axs[i,j].plot(out['date'].values,out['H'].sel(date=plot_time,APR_MDC_key=MDC_key).mean(dim='draws'), color='black', label='simulated_H')
                    axs[i,j].fill_between(out['date'].values,out['H'].sel(date=plot_time,APR_MDC_key=MDC_key).quantile(dim='draws', q=0.025),
                                        out['H'].sel(date=plot_time,APR_MDC_key=MDC_key).quantile(dim='draws', q=0.975), color='black', alpha=0.2, label='Simulation 95% CI')
                # make plot fancy
                if i == 0:
                    axs[i,j].set_title(algorithms[j]+' (score:{:.3e})'.format(scores[j]))
                if j == 0:
                    axs[i,j].set_ylabel(MDC_key)
                axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(2))
                axs[i,j].grid(False)
                axs[i,j].tick_params(axis='both', which='major', labelsize=10)
                axs[i,j].tick_params(axis='both', which='minor', labelsize=8)
                axs[i,j].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 
                #axs[idx].tick_params('x', labelbottom=False)

        handles, plot_labels = axs[0,0].get_legend_handles_labels()
        handles += [axs[0,1].get_legend_handles_labels()[0][-1]]
        plot_labels += [axs[0,1].get_legend_handles_labels()[1][-1]]
        fig.legend(handles, plot_labels, loc='right',fontsize=8) 
        fig.suptitle(model_name)

        plt.subplots_adjust(right=0.8)
        #plt.show()
        plt.savefig(os.path.join(fig_path+model_name,'fits/',str(identifier)+'_'+model_name + '_' +run_date+'.pdf'))
        plt.close()