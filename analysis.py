"""

Various functions to build the spiking time series and fit the state-space 
Ising model using the ssll library.

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

import numpy
import pickle
import random
import os
import sys

directory = os.getcwd()
sys.path.insert(1,directory + '\ssll_lib')

import __init__


def run_em(orientation, O, monkey, spikes, spike_reverse=False, 
           spike_shuffle=False, lmbda1=100, lmbda2=100, max_iter=100, 
           param_est='exact', param_est_eta='exact', stationary='None', 
           theta_o=0, sigma_o=0.1, mstep=True, trials='all', pop='',
           save=False):
    """
    Performs the EM algorithm to fit an Ising model on a given spike train
    coming from a given population of neurons from a given monkey exposed to
    gratings at a given orientation. Then, is save is True, creates a pickle
    file with a name containing the index of the monkey, the orientation of
    the stimulus and the type of spike train analyzed (i.e. reversed in time
    and/or trial shuffled or not). The file is saved in the 'Data' folder 
    in the working directory. This folder is created if inexistant.

    :param int orientation:
        Orientation of the stimulus to consider (0, 30, 60, 90,..., 330)
    :param int O:
        Order of interactions to consider in the Ising model (1 or 2)
    :param int monkey:
        Index of the monkey to consider (0, 1, 2)
    :param numpy.ndarray spikes
        Spike train to which the Ising model should be fitted (T,R,N) 
    :param boolean spike_reverse:
        If True, the spike train to analyze will be reversed in time
    :param boolean spike_shuffle:
        If True, the spike train to analyze will be trial-shuffled to remove
        correlations
    :param float lmbda1:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the first order theta parameters
    :param float lmbda2:
        Inverse coefficient on the identity matrix of the initial
        state-transition covariance matrix for the second order theta parameters
    :param int max_iter:
        Maximum number of iterations for which to run the EM algorithm.
    :param str param_est:
        Parameter whether exact likelihood ('exact') or pseudo likelihood
        ('pseudo') should be used
    :param str param_est_eta:
        Eta parameters are either calculated exactly ('exact'), by mean
        field TAP approximation ('TAP'), or Bethe approximation (belief
        propagation-'bethe_BP', CCCP-'bethe_CCCP', hybrid-'bethe_hybrid')
    :param stationary:
        To fit stationary model. Set 'all' to have stationary thetas
        (Default='None')
    :param numpy.ndarray theta_o:
        Prior mean at the first time bin (one-step predictor)
    :param numpy.ndarray sigma_o:
        Prior covariance at the first time bin (one-step predictor)
    :param boolean mstep:
        The m-step of the EM algorithm is performed only if this parameter
        is true
    :param str trials
        If 'all', all trials are considered.
        If 'odd', only odd-numbered trials are considered
        If 'even', only even-numbered trials are considered
    :param str pop
        Index of the population to analyze. Give different index numbers to
        populations of the same monkey, otherwise the results of different
        populations will overwrite each other.
    :param boolean save
        If True, the emd container will be saved in a pickle file in the
        "Data" folder.

    :returns:
        EMD container. Contains the results of the EM algorithm
    """

    # Marker for the name of the created file
    b = pop
    if spike_reverse == True:
        b += '_i'
    if spike_shuffle == True:
        b += '_shuffle'
    if trials != 'all':
        b += '_'+trials
    # Run the algorithm
    emd = __init__.run(spikes, O, map_function='cg', lmbda1=lmbda1, \
                       lmbda2=lmbda2, max_iter=max_iter, param_est=param_est, \
                       param_est_eta=param_est_eta, stationary=stationary, \
                       theta_o=theta_o, sigma_o=sigma_o, mstep=mstep)
    if save == True:
        # Export emd container to a pickle file
        directory = os.getcwd()
        if not os.path.exists(directory+'/Data/'):
            os.makedirs(directory+'/Data/')
        f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(orientation)+b, 'wb')
        pickle.dump(emd, f)
        f.close()
    print('Lambda = ' + str(1/emd.Q[0,0]) + ' for orientation ' + str(orientation))

    return emd

def get_spike_train(orientation, monkey, neurons, dt=0.01, spike_reverse=False,
           spike_shuffle=False, trials='all'):
    """
    Extracts the spike times form 'gratings_events', then bins the spike train
    of a given monkey for a given set of neurons and a given stimulus orientation.
    
    :param int orientation:
        Orientation of the stimulus to consider (0, 30, 60,..., 330). Only used
        to name the result file.
    :param int monkey:
        Monkey to consider (0, 1, 2)
    :param tuple neurons
        Tuple containing the index numbers of the neurons to include in the
        population to analyze
    :param float dt:
        Length of time bins used (s)
    :param boolean spike_reverse:
        If True, the spike train to analyze will be reversed in time
    :param boolean spike_shuffle:
        If True, the spike train to analyze will be trial-shuffled to remove
        correlations
    :param str trials
        If 'all', all trials are considered.
        If 'odd', only odd-numbered trials are considered
        If 'even', only even-numbered trials are considered

    :returns:
        spike train (T,R,N) as a numpy.ndarray
    """
    
    # Import spiking data
    directory = os.getcwd()
    f = open(directory+'/Preprocess/gratings_events', 'rb')
    events = pickle.load(f) #Time of spikes:[monkey][Neuron, Orientation, Trial]
    f.close()
    # Number of trials
    R = numpy.size(events[monkey], 2) 
    # Total time of the experiment (s)
    Total = 1.28
    # Number of time bins
    T = int(Total / dt)
    # Timing of the events
    spike_timing = events[monkey]
    G = int(orientation / 30)
    # Number of neurons to analyze
    N = len(neurons)
    # Indices of trials to consider
    if trials == 'all':
        R_index = range(R)
    elif trials == 'odd':
        R_index = range(1,R,2)
    elif trials == 'even':
        R_index = range(0,R,2)
    # Initialise the spike matrix
    spikes = numpy.zeros((T, len(R_index), N))
    # Binary spike train
    for t in range(T):
        r_count = -1
        for r in R_index:
            r_count +=1
            count = -1
            for n in neurons:
                count +=1            
                if numpy.any((spike_timing[n][G][r] > t*dt) & (spike_timing[n][G][r] < t*dt + dt)):
                    spikes[t,r_count,count] = 1
                
    if spike_reverse == True:                
        spikes = invert_spikes(spikes)
    if spike_shuffle == True:
        spikes = shuffle_spikes(spikes)

    return spikes

def invert_spikes(spikes):
    """
    Reverses the spike train in time, i.e. the last time bin becomes the first etc...

    :param numpy.ndarray spikes:
        spike train to invert in time (T,R,N)

    :returns:
        spikes_i, spike train inverted in time as a numpy.ndarray (T,R,N)
        
    """

    # Get the number of time bins
    T = spikes.shape[0]
    # Initialize the inverted spike train
    spikes_i = numpy.zeros(spikes.shape)
    # Iterate in time
    for t in range(T):
            spikes_i[t,:,:] = spikes[-t-1,:,:]

    return spikes_i

def shuffle_spikes(spikes):
    """
    Shuffles the data trial-wise. In the shuffled data, for a given trial, 
    each neuron spike train will come from a different trial in the original
    spike train. This should eliminate correlations.

    :param numpy.ndarray spikes:
        spike train to shuffle (T,R,N)

    :returns:
        spikes_shuffle, shuffled spike train as a numpy.ndarray (T,R,N)
    """

    # Get the number of time bins, of trials and of neurons
    T, R, N = spikes.shape
    # Randomize the trial indexes
    rand_trial = numpy.zeros((N,R))
    for i in range(N):
        rand_trial[i,:] = random.sample(range(R),R)
    # Initialize the shuffled spike train
    spikes_shuffle = numpy.zeros((T,R,N))
    # Iterate over trials and neurons
    for r in range(R):
        for n in range(N):
            spikes_shuffle[:,r,n] = spikes[:,int(rand_trial[n,r]),n]

    return spikes_shuffle
