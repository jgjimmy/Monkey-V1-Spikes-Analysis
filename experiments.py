"""
    
Calls the "run_em" function in "analysis.py" to fit an Ising model to data
coming from given neurons of a given monkey. Does this for all stimulus
orientations. Saves the resulting EM Data containers in pickle files in the 
"Data" folder.
    
---

This code uses approximate inference methods for State-Space Analysis of Spike
Correlations previously developped (Shimazaki et al. PLoS Comp Bio 2012) to 
analyze monkey V1 neurons spiking data (Smith and Kohn, Journal of Neuroscience
2008). We acknowledge Thomas Sharp and Christian Donner respectively for
providing the code for exact inference (from repository 
<https://github.com/tomxsharp/ssll> or
<http://github.com/shimazaki/dynamic_corr> for Matlab code) and approximation 
methods (from repository <https://github.com/christiando/ssll_lib>).

In this library we use the existing codes to analyze the contributions of
pairwise interactions to macroscopic properties of neural populations and
stimulus coding of monkey V1 neurons. For details see: 
<https://arxiv.org/abs/1807.08900>.

Copyright (C) 2018

Authors of the analysis methods: Jimmy Gaudreault (jimmy.gaudreault@polymtl.ca)
                                 Hideaki Shimazaki (h.shimazaki@kyoto-u.ac.jp)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
"""

import analysis
import pickle

tweak = True
spike_shuffle = False
trials = 'all'

"""
    :param boolean tweak:
        False: Perform EM algorithm on original data.
        True:  Perform EM algorithm on data inverted in time. Then, uses the
        optimal lambda hyperparameter and the one-step predictor at the last
        time bin to fit an Ising model on uninverted data without
        hyperparameter optimization. This should allow a better prediction
        for the first time bin.

    :param boolean spike_shuffle:
        If True, the experiment will be done using trial-shuffled data, which
        should eliminate correlations.

    :param str trials
        If 'all', all trials are considered.
        If 'odd', only odd-numbered trials are considered.
        If 'even', only even-numbered trials are considered.
"""

# Monkey to analyze (0, 1, 2)
monkey = 0

# Index of the population to analyze ('1', '2', '3')
pop = '1'

# Order of interactions to consider in the Ising model (1 or 2)
O = 2

# Number of neurons to consider in the EM analysis
N = 12

# Length of time bins to use
dt = 0.01

# First guess for lambda hyperparameter 
lmbda = 10

# Indicate whether exact likelihood ('exact') or pseudo likelihood ('pseudo')
# should be used
param_est = 'exact'
# Eta parameters are either calculated exactly ('exact'), by mean
# field TAP approximation ('TAP'), or Bethe approximation (belief
# propagation-'bethe_BP', CCCP-'bethe_CCCP', hybrid-'bethe_hybrid')
param_est_eta = 'exact'

# Import neuron indices
import os
directory = os.getcwd()
f = open(directory+'/Preprocess/neurons', 'rb')
neurons_index = pickle.load(f)
f.close()

neurons = neurons_index[monkey][N*(int(pop)-1):N*int(pop)]

# Run the EM algorithm for every stimulus orientation
if tweak == False:
    for orientation in range(0, 360, 30):
        # Spike train
        spikes = analysis.get_spike_train(orientation, monkey, neurons,\
                                          dt=dt, spike_shuffle = spike_shuffle,\
                                          trials = trials)
        # Fit and save
        emd = analysis.run_em(orientation, O, monkey, spikes, \
                              spike_shuffle=spike_shuffle, lmbda1=lmbda, \
                              lmbda2=lmbda, param_est=param_est, \
                              param_est_eta=param_est_eta, trials=trials,\
                              pop=pop, save=True)
else:
    for orientation in range(0, 360, 30):
        # spike train
        spikes = analysis.get_spike_train(orientation, monkey, neurons, dt=dt, \
                                          spike_reverse=True, \
                                          spike_shuffle=spike_shuffle, \
                                          trials=trials)
        # Fit to inverted data
        emd_i = analysis.run_em(orientation, O, monkey, spikes, spike_reverse=True, \
                                spike_shuffle=spike_shuffle, lmbda1 = lmbda, \
                                lmbda2=lmbda, param_est = param_est, \
                                param_est_eta=param_est_eta, trials=trials,\
                                pop=pop)
        # Extract the one-step predictor at the last time bin and the optimized
        # lambda hyperparameter
        theta_o = emd_i.theta_o[-1,:]
        sigma_o = emd_i.sigma_o[-1]
        lmbda_opt = 1/emd_i.Q[0,0]
        # Invert the spike train (un-invert it)
        spikes = analysis.invert_spikes(spikes)
        # Fit to uninverted data and save
        emd = analysis.run_em(orientation, O, monkey, spikes, spike_shuffle=spike_shuffle, \
                              lmbda1=lmbda_opt, lmbda2=lmbda_opt, max_iter=1, \
                              param_est=param_est, param_est_eta=param_est_eta, \
                              theta_o=theta_o, sigma_o=sigma_o, mstep=False, \
                              trials=trials, pop=pop, save=True)

        
