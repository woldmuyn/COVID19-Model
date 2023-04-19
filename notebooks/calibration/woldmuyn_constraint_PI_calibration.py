"""
This script contains a calibration of an PI constraint postponed healthcare model using data from 2020-2021.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import csv
import json
import argparse
import netCDF4
import sys,os
import random
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
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
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-10-01')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate",default='2021-02-01')
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 50)
parser.add_argument("-ID", "--identifier", help="Name in output files.", default = 'constraint_PI_calibration_2ndWAVE')
parser.add_argument("-MDC_classes", "--MDC_classes", help="MDC classes to model, setting to 'all' equals all MDC keys",default=['01','05','06','08'])
parser.add_argument("-MDC_plot_classes", "--MDC_plot_classes", help="MDC classes to plot, setting to 'all' equals all MDC keys",default=['01','05','06','08'])
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
    identifier = 'postponed_healthcare_' + str(args.identifier)
else:
    raise Exception("The script must have a descriptive name for its output.")
# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Maximum number of MCMC iterations
n_mcmc = int(args.n_mcmc)
# Date at which script is started
run_date = str(datetime.date.today())
# Keep track of runtime
initial_time = datetime.datetime.now()
# Start and end of calibration
start_calibration = pd.to_datetime(args.start_calibration)
if args.end_calibration:
    end_calibration = pd.to_datetime(args.end_calibration)

##############################
## Define results locations ##
##############################

abs_dir = os.path.dirname(__file__)
# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = os.path.join(abs_dir,'../../results/calibrations/Postponed_Healthcare/constraint_PI/')
# Path where MCMC samples should be saved
samples_path = os.path.join(abs_dir,'../../data/interim/model_parameters/Postponed_Healthcare/calibrations/constraint_PI/')
# Path where samples backend should be stored
backend_folder = os.path.join(abs_dir,'../../results/calibrations/Postponed_Healthcare/constraint_PI/')
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"fits/",fig_path+"variance/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

###############
## Load data ##
###############

# Postponed healthcare data
abs_dir = os.path.dirname(__file__)
rel_dir = ''
file_name = '../../data/interim/QALY_model/postponement_non_covid_care/UZG/2020_2021_normalized.csv'
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

##################
## Define model ##
##################

class postponed_healthcare_model(BaseModel):
    """
    Test constrainted PI model
    """
    
    state_names = ['H','E']
    parameter_names = ['covid_dH','covid_H','covid_min', 'Kp']
    parameters_stratified_names = ['Ki','alpha','beta','gamma']
    stratification_names = ['APR_MDC_key']
    
    @staticmethod
    def integrate(t, H, E, covid_dH, covid_H, covid_min, Kp, Ki, alpha, beta, gamma):

        epsilon = H-1
        dE = epsilon - gamma*E
        if covid_H < covid_min:
            u = Kp * np.where(E>=0,epsilon,0) + Ki * np.where(E<0,E,0)
            dH = alpha*covid_dH - beta*u
        else:
            dH = alpha*covid_dH

        return dH, dE 

#################
## Define TDPF ##
#################

from functools import lru_cache

class get_covid_H():

    def __init__(self, data):
        self.data = data

    def H_wrapper_func(self, t, states, param, tau):
        return self.__call__(t, tau)

    @lru_cache()
    def __call__(self, t, tau):
        t = pd.to_datetime(t).round(freq='D')-pd.Timedelta(days=tau)
        try:
            covid_H = self.data.loc[t]
        except:
            covid_H = 0
        return covid_H  

class get_covid_dH():

    def __init__(self, data):
        self.data = data

    def dH_wrapper_func(self, t, states, param, tau):
        return self.__call__(t, tau)

    @lru_cache
    def __call__(self, t, tau):
        t = pd.to_datetime(t).round(freq='D')-pd.Timedelta(days=tau)
        try:
            covid_dH = self.data.loc[t]
        except:
            covid_dH = 0

        return covid_dH

#################
## Setup model ##
#################

#set start and end date of sim
start_date = start_calibration
end_date = end_calibration
sim_len = (end_date - start_date)/pd.Timedelta(days=1)

# Define MDC classes to model
included_MDC = MDC_plot_classes = np.array(['01','03','05','06']) 

# params
alpha = np.ones(len(included_MDC))*-0.0011
beta = np.ones(len(included_MDC))*0.01
gamma = np.ones(len(included_MDC))*0.1

# PI params
Kp = 1
Ki = np.ones(len(included_MDC))*0.05

# initial values covid params
covid_H = 0
covid_dH = 0
covid_min = 400

params_dict={'alpha':alpha,
             'beta':beta,
             'gamma':gamma,
             'Kp':Kp,
             'Ki':Ki,
             'covid_dH':covid_dH,
             'covid_H':covid_H,
             'covid_min':covid_min
            }
#time dependent parameters
params_dict.update({'tau': 0})
H_function = get_covid_H(df_covid_H_tot).H_wrapper_func
dH_function = get_covid_dH(df_covid_dH).dH_wrapper_func

# Initialize model
init_states = {'H':np.ones(len(included_MDC)),'E':np.zeros(len(included_MDC))}
coordinates={'APR_MDC_key': included_MDC}
# Initialize model
model = postponed_healthcare_model(init_states,params_dict,coordinates,time_dependent_parameters={'covid_dH': dH_function,'covid_H':H_function})

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    ##########################################################
    ## Compute the overdispersion parameters for our H data ##
    ##########################################################

    data=hospitalizations_normalized.loc[start_date:end_date, included_MDC]
    results, ax = variance_analysis(data, resample_frequency='W')
    sigmas = results['theta'].loc[slice(None),'gaussian'].values
    #plt.show()
    plt.savefig(os.path.join(fig_path,'variance/',str(identifier)+'_VARIANCE_'+run_date+'.pdf'))
    plt.close()

    #####################################
    ## Calibated parameters and bounds ##
    #####################################

    # Calibated parameters
    pars = ['Ki','alpha','beta','gamma']
    labels = ['Ki','$\\alpha$', '$\\beta$', '$\\gamma$']
    bounds = [(0.00001,10),(-1, -0.00001),(0.00001,10),(0.00001,10)]

    ####################################
    ## dataset and objective function ##
    ####################################

    # Define dataset
    data=[hospitalizations_normalized.loc[start_date:end_date, included_MDC],]
    states = ["H",]
    weights = np.array([1,]) 
    log_likelihood_fnc = [ll_gaussian,]
    log_likelihood_fnc_args = [sigmas,]

    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability([],[],model,pars,bounds,data,states,
                                               log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # Extract formatted parameter_names, bounds and labels
    pars_postprocessing = objective_function.parameter_names_postprocessing
    labels = objective_function.labels 
    bounds = objective_function.bounds

    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Maximum number of PSO iterations

    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 5
    maxiter = n_pso
    popsize = multiplier_pso*processes
    
    # copy init parameters to theta/good guess  
    theta = []
    for key in pars:
        if hasattr(params_dict.get(key),'__len__'):
            for item in params_dict.get(key):
                theta.append(item)
        else:
            theta.append(params_dict.get(key))

    # PSO
    #theta,_ = pso.optimize(objective_function, bounds,
    #                   swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)
    # Nelder-mead
    step = len(bounds)*[0.10,]
    theta,_ = nelder_mead.optimize(objective_function, theta, bounds, step, processes=processes, max_iter=n_pso)
    
    #theta = np.array([0.05, -5.26178149e-04, -5e-04, -4.96233434e-04, -3.79090846e-05,
    #                    1e-03, 1e-03, 1.59975284e-01, 1.00000000e+00,
    #                    1.00000000e+00, 1e-03, 6.08285658e-01, 4.02388060e-01])

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    idx = 0
    for param in pars:
        if hasattr(model.parameters[param],'__len__'):
            n = len(model.parameters[param])
        else:
            n=1
        model.parameters.update({param: theta[idx:idx+n]})
        idx = idx + n

    # Simulate
    out = model.sim([start_date, end_date])

    # plot 
    plot_time = out['date'].values
    fig,axs = plt.subplots(len(MDC_plot_classes),sharex=True,sharey=True,figsize=(8,6))
    if hasattr(axs,'shape'):
        axs = axs.reshape(-1)
        for idx,disease in enumerate(MDC_plot_classes):
            axs[idx].plot(plot_time,out.sel(date=plot_time,APR_MDC_key=disease)['H'],label='simulated_H')
            axs[idx].plot(plot_time,data[0].loc[plot_time,disease], label='data')
            axs[idx].fill_between(plot_time,hospitalizations_normalized_quantiles['q0.025'].loc[plot_time,disease],
                                    hospitalizations_normalized_quantiles['q0.975'].loc[plot_time,disease], alpha=0.2, label='uncertainty')
            axs[idx].set_ylabel(disease)
            axs[idx].xaxis.set_major_locator(plt.MaxNLocator(4))
            axs[idx].grid(False)
            axs[idx].tick_params(axis='both', which='major', labelsize=10)
            axs[idx].tick_params(axis='both', which='minor', labelsize=8)
            #axs[idx].tick_params('x', labelbottom=False)

        handles, plot_labels = axs[idx].get_legend_handles_labels()
        fig.legend(handles, plot_labels, loc='upper center',fontsize=8)

        for idx in range(len(axs)):
            axs[idx].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 
    else:
        for disease in MDC_plot_classes:
            axs.plot(plot_time,out.sel(date=plot_time,APR_MDC_key=disease)['H'],label='simulated_H')
            axs.plot(plot_time,data[0].loc[plot_time,disease], label='data')
            axs.fill_between(plot_time,hospitalizations_normalized_quantiles['q0.025'].loc[plot_time,disease],
                                hospitalizations_normalized_quantiles['q0.975'].loc[plot_time,disease], alpha=0.2, label='uncertainty')
            axs.set_ylabel(disease)
            axs.xaxis.set_major_locator(plt.MaxNLocator(4))
            axs.grid(False)
            axs.tick_params(axis='both', which='major', labelsize=10)
            axs.tick_params(axis='both', which='minor', labelsize=8)
            #axs[idx].tick_params('x', labelbottom=False)

        handles, plot_labels = axs.get_legend_handles_labels()
        fig.legend(handles, plot_labels, loc='upper center',fontsize=8)
        axs.axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5)

    #plt.show()
    plt.savefig(os.path.join(fig_path,'fits/',str(identifier)+'_PSO_fit_'+run_date+'.pdf'))
    plt.close()

    ##########
    ## MCMC ##
    ##########

    # Maximum number of MCMC iterations
    # MCMC settings
    multiplier_mcmc = 10
    max_n = n_mcmc
    print_n = 10

    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.5,]*len(bounds), multiplier=multiplier_mcmc, bounds=log_prior_fnc_args, verbose=True)
    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                'labels': labels, 'parameters': pars_postprocessing, 'starting_estimate': list(theta)}
    # Start calibration
    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()
    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, (), (),
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings) 
    # Generate a sample dictionary
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(sampler, pars_postprocessing, discard=0, settings=settings)
    # Save samples dictionary to json for long-term storage: _SETTINGS_ and _BACKEND_ can be removed at this point
    with open(os.path.join(samples_path, str(identifier)+'_SAMPLES_'+run_date+'.json'), 'w') as fp:
        json.dump(samples_dict, fp)
    # Look at the resulting distributions in a cornerplot
    import corner
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E") #dict(smooth=0.90,title_fmt=".2E",labelpad = 0.3,max_n_ticks=3,label_kwargs=dict(fontsize=18))
    thin = 1
    try:
        autocorr = sampler.get_autocorr_time(flat=True)
        thin = max(1,int(0.5 * np.min(autocorr)))
    except:
        pass
    
    discard=0
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    for i, disease in enumerate(included_MDC):
        idx = np.isin(pars_postprocessing,pars+[par + '_' + str(i) for par in pars])
        fig = corner.corner(flat_samples[:,idx], labels=np.array(labels)[idx], **CORNER_KWARGS)
        for idx,ax in enumerate(fig.get_axes()):
            ax.grid(False)
        plt.subplots_adjust(left=0.15,bottom=0.15)
        plt.savefig(os.path.join(fig_path,'cornerplots/',str(identifier)+'_MDC'+disease+'_CORNER_'+run_date+'.pdf'))
        #plt.show()
        plt.close()

    ######################
    ## Visualize result ##
    ######################

    # Define draw function
    def draw_fcn(param_dict, samples_dict):
        # @Wolf: Waarom zorg ik ervoor dat de samples voor 'beta' en 'f_a' van dezelfde positie 'idx' komen?
        idx = random.randint(0,len(samples_dict[list(samples_dict.keys())[0]]))
        for param in pars:
            param_with_indices = [param_post for param_post in pars_postprocessing if param in param_post]
            param_dict[param] = np.array([samples_dict[param_post][idx] for param_post in param_with_indices])
        return param_dict
    
    # Simulate model
    N=50
    out = model.sim([start_date, end_date], N=N, samples=samples_dict, draw_fcn=draw_fcn, processes=processes)

    # Make visualization
    fig, axs = plt.subplots(len(MDC_plot_classes),1,sharex=True, sharey=True, figsize=(8,6))
    
    if hasattr(axs,'shape'):
        axs = axs.reshape(-1)
        for idx, disease in enumerate(MDC_plot_classes):
            axs[idx].plot(out['date'].values,out['H'].sel(date=plot_time,APR_MDC_key=disease).mean(dim='draws'), color='black', label='Simulation mean') # Dimensie erbij gekomen: draws !
            axs[idx].fill_between(out['date'].values,out['H'].sel(date=plot_time,APR_MDC_key=disease).quantile(dim='draws', q=0.025),
                                out['H'].sel(date=plot_time,APR_MDC_key=disease).quantile(dim='draws', q=0.975), color='black', alpha=0.2, label='Simulation 95% CI')
            
            axs[idx].plot(plot_time,data[0].loc[plot_time,disease], label='Data')
            axs[idx].fill_between(plot_time,hospitalizations_normalized_quantiles['q0.025'].loc[plot_time,disease],
                                  hospitalizations_normalized_quantiles['q0.975'].loc[plot_time,disease], alpha=0.2, label='Data 95% CI')

            axs[idx].set_ylabel(disease)
            axs[idx].xaxis.set_major_locator(plt.MaxNLocator(4))
            axs[idx].grid(False)
            axs[idx].tick_params(axis='both', which='major', labelsize=10)
            axs[idx].tick_params(axis='both', which='minor', labelsize=8)
        handles, plot_labels = axs[idx].get_legend_handles_labels()

    else:
        for disease in MDC_plot_classes:
            axs.plot(out['time'].values,out['H'].sel(time=plot_time,APR_MDC_key=disease).mean(dim='draws'), color='black', label='Simulation mean') # Dimensie erbij gekomen: draws !
            axs.fill_between(out['time'].values,out['H'].sel(time=plot_time,APR_MDC_key=disease).quantile(dim='draws', q=0.025),
                                out['H'].sel(time=plot_time,APR_MDC_key=disease).quantile(dim='draws', q=0.975), color='black', alpha=0.2, label='Simulation 95% CI')
            axs.plot(plot_time,data[0].loc[plot_time,disease], label='Data')
            axs.fill_between(plot_time,hospitalizations_normalized_quantiles['q0.025'].loc[plot_time,disease],
                             hospitalizations_normalized_quantiles['q0.975'].loc[plot_time,disease], alpha=0.2, label='Data 95% CI')
            axs.set_ylabel(disease)
            axs.xaxis.set_major_locator(plt.MaxNLocator(4))
            axs.grid(False)
            axs.tick_params(axis='both', which='major', labelsize=10)
            axs.tick_params(axis='both', which='minor', labelsize=8)
        handles, plot_labels = axs.get_legend_handles_labels()

    fig.legend(handles, plot_labels, loc='upper center',fontsize=8)
    #plt.show()
    plt.savefig(os.path.join(fig_path,'fits/',str(identifier)+'_MCMC_fit_'+run_date+'.pdf'))
    plt.close()
    