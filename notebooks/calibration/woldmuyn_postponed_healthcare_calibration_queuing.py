"""
This script contains a calibration of an queuing postponed healthcare model using data from 2020-2021.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import h5py  
import json
import argparse
import sys,os
import random
import datetime
import pandas as pd
import numpy as np
import emcee
from math import factorial
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import font_manager
import csv
# pySODM packages
from pySODM.models.base import ODEModel
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_gaussian, ll_poisson, validate_calibrated_parameters, expand_bounds
# COVID-19 package
from covid19_DTM.data.sciensano import get_sciensano_COVID19_data
from functools import lru_cache

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", default=None)
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-08-01')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate. Format 'YYYY-MM-DD'.",default='2021-12-31')
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=0)
parser.add_argument("-n_nm", "--n_nm", help="Maximum number of Nelder Mead iterations.", default=0)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default =1)
parser.add_argument("-ID", "--identifier", help="Name in output files.", default = 'queuing_model_test_second_wave')
parser.add_argument("-MDC_sim", "--MDC_sim", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['03','04','05','06']",default="all")
parser.add_argument("-MDC_plot", "--MDC_plot", help="MDC classes to plot, setting to 'all' equals all MDC keys. \ndefault=['03','04','05','06']",default="all")
parser.add_argument("-norm", "--norm", help="use normalized data to fit", action="store_true")
parser.add_argument("-filter", "--filter_args", help="Arguments for Savitzky Golay Filtering, [window_size, order]",default=None)
parser.add_argument("-pars", "--pars", help="parameters to calibrate",default=['epsilon','sigma','alpha'])
parser.add_argument("-covid_data", "--covid_data", help="Use COVID-19 data to estimate covid patients in UZ Gent",action="store_true")
args = parser.parse_args()

# HPC
if args.high_performance_computing:
    high_performance_computing = True
else:
    high_performance_computing = False
# Identifier (name)
if args.identifier:
    identifier = str(args.identifier)
    if 'first' in identifier:
        wave = 'first'
    elif 'second' in identifier:
        wave = 'second'
    else:
        wave = None
else:
    raise Exception("The script must have a descriptive name for its output.")
# Maximum number of NM iterations
n_pso = int(args.n_pso)
# Maximum number of NM iterations
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
rel_dir = '../../data/QALY_model/interim/postponed_healthcare/UZG/'
# MDC_dict to translate keys to disease group
MDC_dict={}
file_name = 'MDC_dict.csv'
with open(os.path.join(abs_dir,rel_dir,file_name), mode='r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        MDC_dict.update({row[0]:row[1]})

MDC_dict.pop('.')
MDC_keys = list(MDC_dict.keys())

# MDC classes to plot
if args.MDC_sim == 'all':
    MDC_sim  = np.array(MDC_keys)
else:
    if isinstance(args.MDC_sim , str):
        MDC_sim  = np.array(args.MDC_sim [1:-1].split(','))
    else:
        MDC_sim = np.array(args.MDC_sim )

# MDC classes to plot
if args.MDC_plot == 'all':
    MDC_plot  = np.array(MDC_keys)
else:
    if isinstance(args.MDC_plot , str):
        MDC_plot  = np.array(args.MDC_plot [1:-1].split(','))
    else:
        MDC_plot  = np.array(args.MDC_plot )

if '04' not in MDC_sim:
    MDC_sim = np.append(MDC_sim,'04')
MDC_plot = np.array(sorted(MDC_plot))
MDC_sim = np.array(sorted(MDC_sim))

# state to fit
if args.norm:
    state_to_fit = 'H_norm'
else:
    state_to_fit = 'H_adjusted'

# smoother
if args.filter_args:
    if isinstance(args.filter_args, str):
        filter_args  = np.array(args.filter_args[1:-1].split(','))
    else:
        filter_args  = np.array(args.filter_args)
else:
    filter_args = None

#pars to calibrate
if isinstance(args.pars, str):
    pars = np.array(args.pars[1:-1].split(','))
else:
    pars = np.array(args.pars)

# use covid data to estimate UZ Gent covid patients
use_covid_data = args.covid_data

if args.backend == None:
    backend = None
else:
    rel_dir = '../../data/QALY_model/interim/postponed_healthcare/model_parameters/queuing_model/'
    backend = emcee.backends.HDFBackend(os.path.join(abs_dir,rel_dir,args.backend)) 

##############################
## Define results locations ##
##############################

abs_dir = os.path.dirname(__file__)
# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = os.path.join(abs_dir,'../../results/QALY_model/postponed_healthcare/calibrations/queuing_model/')
# Path where MCMC samples should be saved
samples_path = os.path.join(abs_dir,'../../data/QALY_model/interim/postponed_healthcare/model_parameters/queuing_model/')
# Path where samples backend should be stored
backend_folder = os.path.join(abs_dir,'../../results/QALY_model/postponed_healthcare/calibrations/queuing_model/')
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
    for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"fits/", fig_path+"variance/"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

###############
## Load data ##
###############

# Postponed healthcare data
abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/QALY_model/interim/postponed_healthcare/UZG'
file_name = '2020_2021_normalized.csv'
types_dict = {'APR_MDC_key': str}
# mean data
hospitalizations_normalized = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)['mean']
hospitalizations_normalized = hospitalizations_normalized.reorder_levels(['date','APR_MDC_key'])
hospitalizations_normalized=hospitalizations_normalized.sort_index()
hospitalizations_normalized=hospitalizations_normalized.reindex(hospitalizations_normalized.index.rename('MDC',level=1))

# MDC_keys = hospitalizations_normalized.index.get_level_values('MDC').unique().values
dates = hospitalizations_normalized.index.get_level_values('date').unique().values

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

# mean hospitalisation length
file_name = 'MZG_residence_times.csv'
types_dict = {'APR_MDC_key': str, 'age_group': str, 'stay_type':str}
residence_times = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2],dtype=types_dict).squeeze()
mean_residence_times = residence_times.groupby(by=['APR_MDC_key']).mean()

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

multi_index = pd.MultiIndex.from_product([MDC_keys,dates])
baseline_in_date_form = pd.Series(index=multi_index,dtype='float')
for idx,(disease,date) in enumerate(multi_index):
    date = pd.to_datetime(date)
    baseline_in_date_form[idx] = hospitalization_baseline.loc[(disease,date.isocalendar().week,date.isocalendar().weekday)]

if filter_args is not None:
    window = filter_args[0]
    order = filter_args[1]

    hospitalizations_normalized_smooth = hospitalizations_normalized.copy()
    for MDC_key in MDC_keys:
        hospitalizations_normalized_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations_normalized.loc[slice(None),MDC_key],window,order)

    hospitalizations_smooth = hospitalizations.copy()
    for MDC_key in MDC_keys:
        hospitalizations_smooth.loc[slice(None),MDC_key] = savitzky_golay(hospitalizations.loc[slice(None),MDC_key],window,order)

    hospitalizations_baseline_smooth = baseline_in_date_form.copy()
    for MDC_key in MDC_keys:
        hospitalizations_baseline_smooth[MDC_key] = savitzky_golay(baseline_in_date_form[MDC_key],window,order)

##################
## Define model ##
##################

class Queuing_model(ODEModel):
    """
    Test model for postponed health_care using a waiting queue before hospitalization
    """
    
    state_names = ['W','H','H_adjusted','H_norm','R','NR','X']
    parameter_names = ['X_tot','f_UZG','covid_H','alpha','post_processed_H']
    parameter_stratified_names = ['A','gamma','epsilon','sigma']
    dimension_names = ['MDC']
    
    @staticmethod
    def integrate(t, W, H, H_adjusted, H_norm, R, NR, X, X_tot, f_UZG, covid_H, alpha, post_processed_H, A, gamma, epsilon, sigma):
        free_beds = X_tot-alpha*covid_H-sum(H - (1/gamma*H))
        #if free_beds > sum(A+W):
        #    X_new = free_beds*(A+W)/sum(A+W)
        #else:
        X_new = free_beds*(sigma*(A+W))/sum(sigma*(A+W))
        X_new = np.where(X_new < 0,0,X_new)

        W_to_H = np.where(W>X_new,X_new,W)
        W_to_NR = epsilon*(W-W_to_H)
        A_to_H = np.where(A>(X_new-W_to_H),(X_new-W_to_H),A)
        A_to_W = A-A_to_H
        
        dX = -X + X_new - W_to_H - A_to_H
        dW = A_to_W - W_to_H - W_to_NR
        dH = A_to_H + W_to_H - (1/gamma*H) 
        dR = (1/gamma*H)
        dNR = W_to_NR

        dH_adjusted = -H_adjusted + post_processed_H[0]
        dH_norm = -H_norm + post_processed_H[1]
        
        return dW, dH, dH_adjusted, dH_norm, dR, dNR, dX

#################
## Define TDPF ##
#################

class get_A():
    def __init__(self, baseline, mean_residence_times):
        self.baseline = baseline
        self.mean_residence_times = mean_residence_times
        
    def A_wrapper_func(self, t, states, param):
        return self.__call__(t)
    
    @lru_cache()
    def __call__(self, t):

        A = (self.baseline.loc[(slice(None),t)]/self.mean_residence_times).values

        return A 

class get_covid_H():

    def __init__(self, covid_data, baseline, hospitalizations):
        self.covid_data = covid_data
        self.baseline_04 = baseline['04']
        self.hospitalizations_04 = hospitalizations.loc[(slice(None),'04')]

    def H_wrapper_func(self, t, states, param, f_UZG):
        return self.__call__(t, f_UZG)

    def __call__(self, t, f_UZG):
        if use_covid_data:
            try:
                covid_H = self.covid_data.loc[t]*f_UZG
            except:
                covid_H = 0

        else:
            covid_H = max(self.hospitalizations_04.loc[t] - self.baseline_04.loc[t],0)
        
        return covid_H 
    
class H_post_processing():

    def __init__(self, baseline, MDC_sim):
        self.MDC = np.array(MDC_sim)
        self.baseline = baseline

    def H_post_processing_wrapper_func(self, t, states, param, covid_H):
        H = states['H']
        return self.__call__(t, H, covid_H)

    def __call__(self, t, H, covid_H):
        H_adjusted = np.where(self.MDC=='04',H+covid_H,H)
        H_norm = H_adjusted/self.baseline.loc[slice(None),t]
        return (H_adjusted,H_norm)
    
#################
## Setup model ##
#################

def init_queuing_model(start_date,end_date,MDC_sim,wave='first'):
    # Define model parameters, initial states and coordinates
    if wave == 'first':
        epsilon = [0.157, 0.126, 0.100, 0.623,
                0.785, 0.339, 0.318, 0.561,
                0.266, 0.600, 0.610, 0.562,
                0.630, 0.534, 0.488, 0.469,
                0.570, 0.612, 0.276, 0.900,
                0.577, 0.163, 0.865, 0.708,
                0.494, 0.557, 0.463, 0.538]
        sigma = [0.229, 0.816, 6.719, 1.023,
                1.251, 1.089, 1.695, 8.127,
                1.281, 1.063, 1.425, 2.819,
                1.899, 1.240, 8.057, 2.657,
                6.685, 2.939, 2.674, 1.011,
                3.876, 7.467, 0.463, 1.175,
                1.037, 1.181, 3.002, 6.024]
        alpha = 5.131
    elif wave == 'second':
        epsilon = [0.176,0.966,0.000,0.498,
                   0.868,0.648,0.000,0.000,
                   0.674,0.593,0.458,0.728,
                   0.666,0.499,0.623,0.777,
                   0.076,0.152,0.200,0.793,
                   0.031,0.100,0.272,0.385,
                   0.551,0.288,0.464,0.518]              
        sigma = [0.344,3.652,6.948,2.831,
                 2.079,5.101,5.865,8.442,
                 6.215,4.911,7.272,4.729,
                 4.415,2.914,5.240,4.299,
                 4.013,5.091,2.000,3.628,
                 0.333,6.550,1.468,6.147,
                 3.184,1.728,6.982,8.739]
        alpha = 5.106
    else:
        epsilon = np.ones(len(MDC_sim))*0.1
        sigma = np.ones(len(MDC_sim))
        alpha = 5

    gamma =  mean_residence_times.loc[MDC_sim].values
    f_UZG = 0.13
    X_tot = 1049

    start_date_string = start_date.strftime('%Y-%m-%d')
    start_date_week = start_date.isocalendar().week
    start_date_day = start_date.isocalendar().weekday
    sim_time = pd.date_range(start_date,end_date)

    if filter_args is not None:
        baseline = hospitalizations_baseline_smooth
    else:
        baseline = baseline_in_date_form

    covid_H = df_covid_H_tot.loc[start_date_string]*f_UZG
    if filter_args is not None:
        H_init = hospitalizations_smooth.loc[(start_date,MDC_sim)].values
    else:
        H_init = hospitalizations.loc[(start_date,MDC_sim)].values
    H_init_normalized = (H_init/baseline.loc[((MDC_sim,start_date))]).values
    A = H_init/mean_residence_times.loc[MDC_sim]

    params={'A':A,'covid_H':covid_H,'alpha':alpha,'post_processed_H':(H_init,H_init_normalized),'X_tot':X_tot, 'gamma':gamma, 'epsilon':epsilon, 'sigma':sigma,'f_UZG':f_UZG}

    init_states = {'H':H_init,'H_norm':np.ones(len(MDC_sim)), 'H_adjusted':H_init}
    coordinates={'MDC':MDC_sim}

    daily_hospitalizations_func = get_A(baseline,mean_residence_times.loc[MDC_sim]).A_wrapper_func
    covid_H_func = get_covid_H(df_covid_H_tot,baseline,hospitalizations).H_wrapper_func
    post_processing_H_func = H_post_processing(baseline,MDC_sim).H_post_processing_wrapper_func

    # Initialize model
    model = Queuing_model(init_states,params,coordinates,
                          time_dependent_parameters={'A': daily_hospitalizations_func,'covid_H':covid_H_func,'post_processed_H':post_processing_H_func})

    return model

##########################################################
## Compute the overdispersion parameters for our H data ##
##########################################################

if state_to_fit == 'H_adjusted':
    ruw_data = hospitalizations.loc[start_date:end_date,MDC_sim]
    if filter_args is not None:
        smooth_data = hospitalizations_smooth.loc[start_date:end_date,MDC_sim]
    method = 'quasi-poisson'
else:
    ruw_data = hospitalizations_normalized.loc[start_date:end_date,MDC_sim]
    if filter_args is not None:
        smooth_data = hospitalizations_normalized_smooth[start_date:end_date,MDC_sim]
    method = 'gaussian'

if filter_args is not None:
    data = smooth_data
else:
    data = ruw_data

sigmas_ll = []
results, ax = variance_analysis(data, resample_frequency='W')
for MDC_key in MDC_sim:
    sigmas_ll.append(results['theta'].loc[MDC_key,method])
plt.close()
    
if __name__ == '__main__':

    #############################
    ## plot initial conditions ##
    #############################

    # plot 
    fig,axs = plt.subplots(len(MDC_plot),1,sharex=True,figsize=(9,2*len(MDC_plot)))
    plot_time = pd.date_range(start_date,end_date)
    
    model = init_queuing_model(start_date,end_date,MDC_sim,wave)
    out = model.sim([start_date, end_date],tau=1)
    
    for i,MDC_key in enumerate(MDC_sim):
        axs[i].plot(plot_time,ruw_data.loc[plot_time,MDC_key].values,color='blue')
        if filter_args is not None:
            axs[i].plot(plot_time,smooth_data.loc[plot_time,MDC_key].values,color='red')
            axs[i].plot(plot_time,hospitalizations_baseline_smooth.loc[MDC_key,plot_time].values,color='black',alpha=0.5)
        
        axs[i].plot(plot_time,out[state_to_fit].sel(MDC=MDC_key),color='black')

        axs[i].set_ylabel(MDC_key)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i].grid(False)
        axs[i].tick_params(axis='both', which='major', labelsize=8)
        axs[i].tick_params(axis='both', which='minor', labelsize=8)

        if state_to_fit == 'H_norm':
            axs[i].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 

    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,str(identifier)+'_init_condition.pdf'))
    #plt.show()
    plt.close()

    #############################
    ## set up calibration plot ##
    #############################

    fig,axs = plt.subplots(len(MDC_plot),3,sharex=True,figsize=(9,2*len(MDC_plot)))

    axs[0,0].set_title('PSO')
    axs[0,1].set_title('Nelder-Mead')
    axs[0,2].set_title('MCMC')
    for i,MDC_key in enumerate(MDC_plot):
        axs[i,0].set_ylabel(MDC_key)
        for j in range(3):
            axs[i,j].xaxis.set_major_locator(plt.MaxNLocator(2))
            axs[i,j].grid(False)
            axs[i,j].tick_params(axis='both', which='major', labelsize=8)
            axs[i,j].tick_params(axis='both', which='minor', labelsize=8)
            if state_to_fit == 'H_norm':
                axs[i,j].axhline(y = 1, color = 'r', linestyle = 'dashed', alpha=0.5) 

    ######################
    ## Calibrate models ##
    ######################
    model = init_queuing_model(start_date,end_date,MDC_sim,wave)
    
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))

    # parameters and bounds
    labels_dict = {'f_UZG':'$f_{UZG}$', 'epsilon':'$\\epsilon$', 'sigma':'$\\sigma$', 'alpha':'$\\sigma_{covid}$','X_tot':'$X_{tot}$'}
    bounds_dict = {'f_UZG':(1*10**-10,1),'epsilon':(1*10**-10,1),'sigma':(1*10**-10,10), 'alpha':(1*10**-10,10),'X_tot':(1*10**-10,2000)}
    labels = [labels_dict[par] for par in pars]
    bounds = [bounds_dict[par] for par in pars]

    parameter_sizes, _ = validate_calibrated_parameters(pars, model.parameters)
    expanded_bounds = expand_bounds(parameter_sizes, bounds)

    # Define dataset
    if state_to_fit == "H_adjusted":
        calibration_data=[data.to_frame(),]
    else:
        calibration_data=[data.to_frame(),]
    
    states = [state_to_fit,]
    # Setup likelihood functions and arguments
    if state_to_fit == "H_adjusted":
        log_likelihood_fnc = [ll_poisson,]
        log_likelihood_fnc_args = [None,]
    else:
        log_likelihood_fnc = [ll_gaussian,]
        log_likelihood_fnc_args = [np.array(sigmas_ll),]

    # Setup objective function (no priors --> uniform priors based on bounds) 
    objective_function = log_posterior_probability(model,pars,bounds,calibration_data,states,
                                        log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)
    # --- #
    # PSO #
    # --- #
    if n_pso > 0:
        print('PSO')
        theta = pso.optimize(objective_function, swarmsize=len(expanded_bounds)*30, max_iter=n_pso, debug=True, processes=processes,kwargs={'simulation_kwargs':{'tau': 1}})[0]

        param_dict = assign_theta(model.parameters,pars,theta)
        calibrated_param_dict = {}
        for param in pars:
            if isinstance(param_dict[param], np.ndarray):
                calibrated_param_dict.update({param:list(param_dict[param])})
            else:
                calibrated_param_dict.update({param:param_dict[param]})
        with open(os.path.join(samples_path, str(identifier)+'_THETA_PSO_'+run_date+'.json'), 'w') as fp:
            json.dump(calibrated_param_dict, fp)

    else:
        theta = np.array([])
        for param in pars:
            theta = np.append(theta,model.parameters[param])

        param_dict = assign_theta(model.parameters,pars,theta)
    
    # simulate model
    model.parameters.update(param_dict)
    out_pso = model.sim([start_date, end_date],tau=1)

    # visualization
    for i,MDC_key in enumerate(MDC_plot):
        axs[i,0].plot(plot_time,ruw_data.loc[plot_time,MDC_key], label='ruw data', alpha=0.7, color='blue')
        if filter_args is not None:
            axs[i,0].plot(plot_time,smooth_data.loc[plot_time,MDC_key], label='filtered data', alpha=0.7, color='red')
        axs[i,0].plot(plot_time,out_pso.sel(date=plot_time,MDC=MDC_key)[state_to_fit],label='model',color='black')
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'fits/',str(identifier)+ '_' +run_date+'.pdf'))
    plt.close()

    # ----------- #
    # Nelder-mead #
    # ----------- #
    if n_nm > 0:
        print('Nelder-mead')
        step = len(theta)*[0.2,]
        theta,_ = nelder_mead.optimize(objective_function, theta, step, processes=processes, max_iter=n_nm,kwargs={'simulation_kwargs':{'tau': 1}})

        param_dict = assign_theta(model.parameters,pars,theta)
        calibrated_param_dict = {}
        for param in pars:
            if isinstance(param_dict[param], np.ndarray):
                calibrated_param_dict.update({param:list(param_dict[param])})
            else:
                calibrated_param_dict.update({param:param_dict[param]})
        with open(os.path.join(samples_path, str(identifier)+'_THETA_NM_'+run_date+'.json'), 'w') as fp:
            json.dump(calibrated_param_dict, fp)

        # simulate model
        model.parameters.update(param_dict)
        out_nm = model.sim([start_date, end_date],tau=1)

        # visualization
        for i,MDC_key in enumerate(MDC_plot):
            axs[i,1].plot(plot_time,ruw_data.loc[plot_time,MDC_key], label='ruw data', alpha=0.7, color='blue')
            if filter_args is not None:
                axs[i,1].plot(plot_time,smooth_data.loc[plot_time,MDC_key], label='filtered data', alpha=0.7, color='red')
            axs[i,1].plot(plot_time,out_nm.sel(date=plot_time,MDC=MDC_key)[state_to_fit],label='model',color='black')

        fig.tight_layout()
        fig.savefig(os.path.join(fig_path,'fits/',str(identifier) + '_' +run_date+'.pdf'))
        plt.close()

    # ---- #
    # MCMC #
    # ---- #
    if n_mcmc > 0:
        print('MCMC')
        multiplier_mcmc = 10
        print_n = 10

        theta = np.where(theta==0,0.00001,theta)

        # Perturbate previously obtained estimate
        ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.1,]*len(theta), bounds=expanded_bounds, multiplier=multiplier_mcmc, verbose=True)

        # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
        settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                    'labels': labels, 'parameters': list(pars), 'starting_estimate': list(theta),"MDC":list(MDC_sim)}
        # Start calibration
        sys.stdout.flush()
        # Sample n_mcmc iterations
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, objective_function_kwargs={'simulation_kwargs':{'tau': 1}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, processes=processes, progress=True,
                                    settings_dict=settings,backend=backend) 
        # Generate a sample dictionary
        thin = 1
        try:
            autocorr = sampler.get_autocorr_time(flat=True)
            thin = max(1,int(0.5 * np.min(autocorr)))
        except:
            pass

        print(sampler.get_chain(discard=int(n_mcmc*98/100), thin=thin, flat=True).shape[0])
        if sampler.get_chain(discard=int(n_mcmc*98/100), thin=thin, flat=True).shape[0] > 300:
            discard = int(n_mcmc*98/100)
        elif sampler.get_chain(discard=int(n_mcmc*3/4), thin=thin, flat=True).shape[0] > 200:
            discard = int(n_mcmc*3/4)
        elif sampler.get_chain(discard=int(n_mcmc*1/2), thin=thin, flat=True).shape[0] > 200:
            discard = int(n_mcmc*1/2)
        else:
            discard = 0
        
        if backend is not None:
            backend_identifier = args.backend[:-24]
            backend_run_date = args.backend[-15:-5]
            samples_dict = emcee_sampler_to_dictionary(discard=discard, thin=thin, samples_path=samples_path,identifier=backend_identifier,run_date=backend_run_date)
        else:
            samples_dict = emcee_sampler_to_dictionary(discard=discard, thin=thin, samples_path=samples_path,identifier=identifier)
        with open(os.path.join(samples_path, str(identifier)+'_SAMPLES_'+run_date+'.json'), 'w') as fp:
            json.dump(samples_dict, fp)

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

        # visualization
        for i,MDC_key in enumerate(MDC_plot):
            axs[i,2].plot(plot_time,ruw_data.loc[plot_time,MDC_key], label='ruw data', alpha=0.7, color='blue')
            if filter_args is not None:
                axs[i,2].plot(plot_time,smooth_data.loc[plot_time,MDC_key], label='filtered data', alpha=0.7, color='red')
            axs[i,2].plot(plot_time,out_mcmc[state_to_fit].sel(date=plot_time,MDC=MDC_key).mean(dim='draws'), color='black', label='simulated_H')
            axs[i,2].fill_between(plot_time,out_mcmc[state_to_fit].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.025),
                                out_mcmc[state_to_fit].sel(date=plot_time,MDC=MDC_key).quantile(dim='draws', q=0.975), color='black', alpha=0.2, label='Simulation 95% CI')

        fig.tight_layout()
        fig.savefig(os.path.join(fig_path,'fits/',str(identifier) + '_' +run_date+'.pdf'))
        plt.close()