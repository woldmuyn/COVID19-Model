"""
This script computes and visualises the R0 for WAVE1

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/covid19_DTM/interim/model_parameters/COVID19_SEIRD/calibrations/national/

Example use:
------------
python compute_R0_WAVE1.py -f BE_4_prev_full_2020-12-15_WAVE2_GOOGLE.json

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import json
import corner
import argparse
import numpy as np
import matplotlib.pyplot as plt
from covid19_DTM.data import mobility, sciensano, model_parameters

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
args = parser.parse_args()

# -----------------------
# Load samples dictionary
# -----------------------

samples_dict = json.load(open('../../data/covid19_DTM/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))

# --------------------
# Load additional data
# --------------------

# Contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices()
levels = initN.size
# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()

# ----------
# Compute R0
# ----------

N = initN.size
sample_size = len(samples_dict['beta'])
R0 = np.zeros([N,sample_size])
R0_norm = np.zeros([N,sample_size])
for i in range(N):
    for j in range(sample_size):
        R0[i,j] = (params['a'][i] * samples_dict['da'][j] + params['omega']) * samples_dict['beta'][j] * np.sum(Nc_all['total'], axis=1)[i]
    R0_norm[i,:] = R0[i,:]*(initN[i]/sum(initN))
    
R0_age = np.mean(R0,axis=1)
R0_overall = np.mean(np.sum(R0_norm,axis=0))

# ------------
# Visualize R0
# ------------

print(np.quantile(np.sum(R0_norm,axis=0),q=1-0.05/2), R0_overall, np.quantile(np.sum(R0_norm,axis=0),q=0.05/2))
print(np.quantile(np.sum(R0_norm,axis=0),q=0.25), R0_overall, np.quantile(np.sum(R0_norm,axis=0),q=0.75))


plt.hist(np.sum(R0_norm,axis=0),bins=12)
plt.show()