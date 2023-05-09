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
from matplotlib import font_manager
import csv
# pySODM packages
from pySODM.models.base import ODEModel
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian, ll_poisson
# COVID-19 package
from covid19_DTM.data.sciensano import get_sciensano_COVID19_data

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
parser.add_argument("-e", "--end_calibration", help="Calibration enddate. Format 'YYYY-MM-DD'.",default='2021-03-01')
parser.add_argument("-n_nm", "--n_nm", help="Maximum number of Nelder Mead iterations.", default=1)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default =1)
parser.add_argument("-ID", "--identifier", help="Name in output files.", default = 'queuing_model_test')
parser.add_argument("-MDC_sim", "--MDC_sim", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['01','05','06','08']",default=['03','04','05','06'])
parser.add_argument("-MDC_plot", "--MDC_plot", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['01','05','06','08']",default=['03','04','05'])
parser.add_argument("-ewm", "--ewm", help="span of ewm to apply to baseline. \ndefault=0",default=14)
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
ewm = int(args.ewm)
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
rel_dir = '../../data/PHM/interim/UZG/'
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
MDC_sim = np.array(list(MDC_keys))

# MDC classes to plot
if args.MDC_sim == 'all':
    MDC_sim  = np.array(list(MDC_keys))
else:
    if isinstance(args.MDC_sim , str):
        MDC_sim  = np.array(args.MDC_sim [1:-1].split(','))
    else:
        MDC_sim = np.array(args.MDC_sim )

# MDC classes to plot
if args.MDC_plot == 'all':
    MDC_plot  = np.array(list(MDC_keys))
else:
    if isinstance(args.MDC_plot , str):
        MDC_plot  = np.array(args.MDC_plot [1:-1].split(','))
    else:
        MDC_plot  = np.array(args.MDC_plot )

MDC_plot = sorted(MDC_plot)
MDC_sim = sorted(MDC_sim)

##############################
## Define results locations ##
##############################

abs_dir = os.path.dirname(__file__)
# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = os.path.join(abs_dir,'../../results/PHM/calibrations/queuing_model/')
# Path where MCMC samples should be saved
samples_path = os.path.join(abs_dir,'../../data/PHM/interim/model_parameters/queuing_model/calibrations/')
# Path where samples backend should be stored
backend_folder = os.path.join(abs_dir,'../../results/PHM/calibrations/queuing_model/')
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
    for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"fits/", fig_path+"variance/"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

label_font = font_manager.FontProperties(family='CMU Sans Serif',
                            style='normal', 
                            size=10)
legend_font = font_manager.FontProperties(family='CMU Sans Serif',
                                style='normal', 
                                size=8)

###############
## Load data ##
###############

# Postponed healthcare data
abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/PHM/interim/UZG'
file_name = '2020_2021_normalized.csv'
types_dict = {'APR_MDC_key': str}
# mean data
hospitalizations_normalized = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)['mean']
hospitalizations_normalized = hospitalizations_normalized.reorder_levels(['date','APR_MDC_key'])
hospitalizations_normalized=hospitalizations_normalized.sort_index()
hospitalizations_normalized=hospitalizations_normalized.reindex(hospitalizations_normalized.index.rename('MDC',level=1))
MDC_keys = hospitalizations_normalized.index.get_level_values('MDC').unique().values

# lower and upper quantiles
hospitalizations_normalized_quantiles = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True).loc[(slice(None), slice(None)), ('q0.025','q0.975')]
hospitalizations_normalized_quantiles = hospitalizations_normalized_quantiles.reorder_levels(['date','APR_MDC_key'])
hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.sort_index()
hospitalizations_normalized_quantiles=hospitalizations_normalized_quantiles.reindex(hospitalizations_normalized_quantiles.index.rename('MDC',level=1))

file_name = 'MZG_2016_2021.csv'
types_dict = {'APR_MDC_key': str}
hospitalizations = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
hospitalizations = hospitalizations.groupby(['APR_MDC_key','date']).sum()
hospitalizations.index = hospitalizations.index.set_levels([hospitalizations.index.levels[0], pd.to_datetime(hospitalizations.index.levels[1])])
hospitalizations = hospitalizations.reorder_levels(['date','APR_MDC_key'])
hospitalizations=hospitalizations.sort_index()
hospitalizations=hospitalizations.reindex(hospitalizations.index.rename('MDC',level=1))

# COVID-19 data
covid_data, _ , _ , _ = get_sciensano_COVID19_data(update=False)
new_index = pd.MultiIndex.from_product([pd.to_datetime(hospitalizations_normalized.index.get_level_values('date').unique()),covid_data.index.get_level_values('NIS').unique()])
covid_data = covid_data.reindex(new_index,fill_value=0)
df_covid_H_in = covid_data['H_in'].loc[:,40000]
df_covid_H_tot = covid_data['H_tot'].loc[:,40000]
df_covid_dH = df_covid_H_tot.diff().fillna(0)

# hospitalisation baseline
file_name = 'MZG_baseline.csv'
types_dict = {'APR_MDC_key': str, 'week_number': int, 'day_number':int}
hospitalization_baseline = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
hospitalization_baseline = hospitalization_baseline.groupby(['APR_MDC_key','week_number','day_number']).mean()
if ewm > 0:
    hospitalization_baseline = hospitalization_baseline.ewm(ewm).mean()

# mean hospitalisation length
file_name = 'MZG_residence_times.csv'
types_dict = {'APR_MDC_key': str, 'age_group': str, 'stay_type':str}
residence_times = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2],dtype=types_dict).squeeze()
mean_residence_times = residence_times.groupby(by=['APR_MDC_key']).mean()

##################
## Define model ##
##################

class Queuing_model(ODEModel):
    """
    Test model for postponed health_care using a waiting queue before hospitalization
    """
    
    state_names = ['W','H','R','NR','X']
    parameter_names = ['X_tot','f_UZG','covid_H']
    parameter_stratified_names = ['A','gamma','epsilon','sigma']
    dimension_names = ['MDC']
    
    @staticmethod
    def integrate(t, W, H, R, NR, X, X_tot, f_UZG, covid_H, A, gamma, epsilon, sigma):
        X_new = (X_tot-f_UZG*covid_H-sum(H - (1/gamma*H))) * (sigma*(A+W))/sum(sigma*(A+W))

        W_to_H = np.where(W>X_new,X_new,W)
        W_to_NR = epsilon*(W-W_to_H)
        A_to_H = np.where(A>(X_new-W_to_H),(X_new-W_to_H),A)
        A_to_W = A-A_to_H
        
        dX = -X + X_new - W_to_H - A_to_H
        dW = A_to_W - W_to_H - W_to_NR
        dH = A_to_H + W_to_H - (1/gamma*H)
        dR = (1/gamma*H)
        dNR = W_to_NR
        
        return dW, dH, dR, dNR, dX

#################
## Define TDPF ##
#################

from functools import lru_cache

class get_A():
    def __init__(self, baseline,mean_residence_times):
        self.baseline = baseline
        self.mean_residence_times = mean_residence_times
    
    def A_wrapper_func(self, t, states, param, MDC):
        return self.__call__(t,MDC)
    
    #@lru_cache()
    def __call__(self, t, MDC):

        t_string = t.strftime('%Y-%m-%d')
        week = t.isocalendar().week
        day = t.isocalendar().weekday

        A = (self.baseline.loc[(MDC,week,day)]/self.mean_residence_times.loc[MDC]).values

        return A 

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
    
#################
## Setup model ##
#################

def init_queuing_model(start_date):
    # Define model parameters, initial states and coordinates
    gamma =  mean_residence_times.loc[MDC_sim].values
    epsilon = np.ones(len(MDC_sim))*0.1
    sigma = np.ones(len(MDC_sim))

    f_UZG = 0.2
    X_tot = 1049

    t_string = start_date.strftime('%Y-%m-%d')
    covid_H = df_covid_H_tot.loc[t_string]*f_UZG
    H_init = hospitalization_baseline.loc[(MDC_sim,start_date.isocalendar().week,start_date.isocalendar().weekday)].values
    A = H_init/mean_residence_times.loc[MDC_sim]

    params={'A':A,'covid_H':covid_H,'X_tot':X_tot, 'gamma':gamma, 'epsilon':epsilon, 'sigma':sigma,'MDC':MDC_sim,'f_UZG':f_UZG}

    init_states = {'H':H_init}
    coordinates={'MDC':MDC_sim}

    daily_hospitalizations_func = get_A(hospitalization_baseline,mean_residence_times).A_wrapper_func
    covid_H_func = get_covid_H(df_covid_H_tot).H_wrapper_func

    # Initialize model
    model = Queuing_model(init_states,params,coordinates,time_dependent_parameters={'A': daily_hospitalizations_func,'covid_H':covid_H_func})

    return model

def normalize_model_output(out):
    out = out.copy()
    multi_index = pd.MultiIndex.from_product([MDC_sim,out.date.values])
    baseline = pd.Series(index=multi_index)
    for idx,(disease,date) in enumerate(multi_index):
        baseline[idx] = hospitalization_baseline.loc[(disease,date.isocalendar().week,date.isocalendar().weekday)]

    if 'draws' in out.dims:
        for disease in MDC_sim:
            for draw in out.draws:
                out.sel(MDC=disease,draws=draw)['H'][:]=out.sel(MDC=disease,draws=draw)['H']/baseline.loc[disease]
    else:
        for disease in MDC_sim:
            out.sel(MDC=disease)['H'][:]=out.sel(MDC=disease)['H']/baseline.loc[disease]
    
    return out


##########################################################
## Compute the overdispersion parameters for our H data ##
##########################################################

sigmas_dict = {}
data=hospitalizations.loc[start_date:end_date]
results, ax = variance_analysis(data, resample_frequency='W')
for MDC_key in MDC_keys:
    sigmas_dict.update({MDC_key:results['theta'].loc[MDC_key,'quasi-poisson']})
plt.close()
    
if __name__ == '__main__':

    #############################
    ## plot initial conditions ##
    #############################

    # plot 
    fig,axs = plt.subplots(3,sharex=True,figsize=(6,6))
    plot_time = pd.date_range(start_date,end_date)
    axs[0].plot(df_covid_H_tot.loc[plot_time])
    axs[0].set_title('covid_H',font=label_font)

    axs[1].plot(plot_time,hospitalizations_normalized.loc[plot_time,'05'].values)
    axs[1].set_title('data MDC:05',font=label_font)
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(4))
    axs[1].grid(False)
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].tick_params(axis='both', which='minor', labelsize=8)

    model = init_queuing_model(start_date)
    out = model.sim([start_date, end_date],tau=1)
    out = normalize_model_output(out)

    axs[2].plot(plot_time,out['H'].sel(MDC='05'))
    axs[2].set_title('model init',font=label_font)
    axs[2].xaxis.set_major_locator(plt.MaxNLocator(4))
    axs[2].grid(False)
    axs[2].tick_params(axis='both', which='major', labelsize=8)
    axs[2].tick_params(axis='both', which='minor', labelsize=8)

    for idx in range(len(axs)-1):
        axs[idx+1].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 

    #plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,str(identifier)+'_init_condition.pdf'))
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
    pars = ['f_UZG','epsilon','sigma']
    labels = ['$f_{UZG}$', '$\\epsilon$', '$\\sigma$']
    bounds = [(1*10**-10,1),(1*10**-10,1),(1*10**-10,10)]

    # Define dataset
    data=[hospitalizations.loc[start_date:end_date,MDC_sim].to_frame(),]
    states = ["H",]
    # Setup likelihood functions and arguments
    log_likelihood_fnc = [ll_poisson,]
    #log_likelihood_fnc_args = [[sigmas_dict[MDC_key]],]
    log_likelihood_fnc_args = [None,]
    # Setup prior functions and arguments
    #log_prior_fnc = len(bounds)*[log_prior_uniform,]
    #log_prior_fnc_args = bounds

    model = init_queuing_model(start_date)

    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                        log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)

    theta = pso.optimize(objective_function, swarmsize=30, max_iter=n_nm, processes=processes, debug=True,kwargs={'simulation_kwargs':{'tau': 1}})[0]
    #theta = np.array([])
    #for param in pars:
    #    theta = np.append(theta,model.parameters[param])

    print('Nelder-mead')
    # Nelder-mead
    #step = len(theta)*[0.5,]
    #theta,_ = nelder_mead.optimize(objective_function, theta, step, processes=processes, max_iter=n_nm,kwargs={'simulation_kwargs':{'tau': 1}})

    theta = np.where(theta==0,0.00001,theta)

    # MCMC
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.5,]*len(theta), multiplier=multiplier_mcmc, verbose=True)
    #print(pos)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                'labels': labels, 'parameters': pars, 'starting_estimate': list(theta),"MDC":list(MDC_sim)}
    # Start calibration
    sys.stdout.flush()
    # Sample n_mcmc iterations
    #if high_performance_computing:
    #    progress = False
    #else:
    #    progress = True
    if n_mcmc > 0:
        sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, objective_function_kwargs={'simulation_kwargs':{'tau': 1}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings) 
        # Generate a sample dictionary
        # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
        thin = 1
        try:
            autocorr = sampler.get_autocorr_time(flat=True)
            thin = max(1,int(0.5 * np.min(autocorr)))
        except:
            pass
        if sampler.get_chain(discard=int(n_mcmc*3/4), thin=thin, flat=True).shape[0] > 20:
            discard = int(n_mcmc*3/4)
        else:
            discard = 0
        samples_dict = emcee_sampler_to_dictionary(discard=discard, thin=thin, samples_path=samples_path,identifier=identifier)
                
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
    data = hospitalizations.loc[start_date:end_date,MDC_plot]
    #data=hospitalizations_normalized.loc[start_date:end_date, MDC_plot]
                
    #-------------#
    # Nelder-Mead #
    # ----------- #

    # Assign thetas to model
    param_dict = assign_theta(model.parameters,pars,theta)
    model.parameters.update(param_dict)

    # simulate model
    out_nm = model.sim([start_date, end_date],tau=1)
    #out_nm = normalize_model_output(out_nm)
    
    #------#
    # MCMC #
    #------#

    if n_mcmc > 0:
        # Define draw function
        def draw_fcn(param_dict, samples_dict):
            pars = samples_dict['parameters']

            if hasattr(samples_dict[pars[0]][0],'__iter__'):
                idx = random.randint(0,len(samples_dict[pars[0]][0])-1)
            else:
                idx = random.randint(0,len(samples_dict[pars[0]])-1)
            
            for param in pars:
                if hasattr(samples_dict[param][0],'__iter__'):
                    par_array = np.array([])
                    for samples in samples_dict[param]:
                        par_array = np.append(par_array,samples[idx])
                    param_dict.update({param:par_array})
                else:
                    param_dict.update({param:samples_dict[param][idx]})

            return param_dict

        # Simulate model
        N=20
        out_mcmc = model.sim([start_date, end_date], N=N, samples=samples_dict, draw_function=draw_fcn, processes=processes,tau=1)
        #out_mcmc = normalize_model_output(out_mcmc)
    
    ########
    # plot #
    ########
    algorithms = ['Nelder-Mead','MCMC']
    fig,axs = plt.subplots(len(MDC_plot),2,sharex=True,figsize=(6,2*len(MDC_plot)))
    for i,MDC_key in enumerate(MDC_plot):
        for j,(out_name,out) in enumerate(zip(['Nelder Mead','MCMC'],[out_nm, out_mcmc])):
            #data
            axs[i,j].plot(plot_time,data.loc[plot_time,MDC_key], label='data', alpha=0.7)
            axs[i,j].fill_between(plot_time,hospitalizations_normalized_quantiles['q0.025'].loc[plot_time,MDC_key],
                                    hospitalizations_normalized_quantiles['q0.975'].loc[plot_time,MDC_key], alpha=0.2, label='uncertainty on data')
            #simulation
            if j==0:
                axs[i,j].plot(plot_time,out.sel(date=plot_time,MDC=MDC_key)['H'],label='simulated_H',color='black')
            else:    
                axs[i,j].plot(out['date'].values,out['H'].sel(date=plot_time,MDC=MDC_key).mean(dim='draws'), color='black', label='simulated_H')
                axs[i,j].fill_between(out['date'].values,out['H'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.025),
                                    out['H'].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.975), color='black', alpha=0.2, label='Simulation 95% CI')
            # make plot fancy
            if i == 0:
                axs[i,j].set_title(out_name,font=label_font)
            if j == 0:
                axs[i,j].set_ylabel(MDC_key,font=label_font)
            axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i,j].grid(False)
            axs[i,j].tick_params(axis='both', which='major', labelsize=8)
            axs[i,j].tick_params(axis='both', which='minor', labelsize=8)
            axs[i,j].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 
            #axs[idx].tick_params('x', labelbottom=False)

    handles, plot_labels = axs[0,0].get_legend_handles_labels()
    handles += [axs[0,1].get_legend_handles_labels()[0][-1]]
    plot_labels += [axs[0,1].get_legend_handles_labels()[1][-1]]
    fig.legend(handles, plot_labels, loc='right',fontsize=8,prop=legend_font) 

    plt.subplots_adjust(right=0.8)
    plt.show()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'fits/',str(identifier)+'_'+'queuing_model' + '_' +run_date+'.pdf'))
    plt.close()