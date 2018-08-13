"""

Computes all the macroscopic quantities required to trace figure 1. Saves the
results in a pickle file in the Data folder. This file will be loaded by
"fig1.py" or "fig1_without_quantiles.py". "experiments.py" must be run before
this code, so that the Ising models are first fitted to the data.

---

This code uses State-space Model of Time-varying Neural Interactions previously
developed (Shimazaki et al. PLoS Comp Biol 2012; Donner et al. PLoS Comp Biol
2017) to analyze monkey V1 neurons spiking data (Smith and Kohn, Journal of
Neuroscience 2008). We acknowledge Thomas Sharp and Christian Donner
respectively for providing codes for the inference method (from repository
<https://github.com/tomxsharp/ssll> or
<http://github.com/shimazaki/dynamic_corr> for Matlab), and its approximation
methods (from repository <https://github.com/christiando/ssll_lib>).

In this library, we use the existing codes to analyze contributions of pairwise
interactions to macroscopic properties of neural populations and stimulus coding
of monkey V1 neurons. For details see:

Jimmy Gaudreault and Hideaki Shimazaki. (2018) State-space analysis of an Ising
model reveals contributions of pairwise interactions to sparseness, fluctuation,
and stimulus coding of monkey V1 neurons. arXiv:1807.08900.
<https://arxiv.org/abs/1807.08900>

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
import os
import sys

directory = os.getcwd()
sys.path.insert(1,directory + '\ssll_lib')

import bethe_approximation
import energies
import probability
import transforms

# Index of the monkey to analyze (0,1,2)
monkey = 0

# Population to analyze ('1', '2', '3')
pop = '1'

# Shuffled-data boolean. If 'True', computes quantiles for models fit to
# shuffled data.
Shuffled = False
b = ''
if Shuffled:
    b += '_shuffle'

############# Extraction of the data #############
orientation = 0
f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(orientation)+pop+b, 'rb')
emd = pickle.load(f)
f.close()

# Number of time bins, trials and neurons
T, R, N = emd.spikes.shape

# Number of dimensions
D = emd.D

# Order of interactions considered in the Ising models
O = emd.order

# Initialise the theta matrix and the sigma matrix
theta_m = numpy.zeros((12,T,D))
theta_m[0,:,:] = emd.theta_s
sigma_m = numpy.zeros((12,T,D))
if emd.marg_llk == probability.log_marginal: # i.e. if exact model is used
    transforms.initialise(N,O)
    for t in range(T):
        sigma_m[0,t] = numpy.diag(emd.sigma_s[t]) # Because emd.sigma_s is ((T,D,D)) if exact, only need the diagonal for fig1
else:
    sigma_m[0] = emd.sigma_s

# Initialize the spike train array
spikes_m = numpy.zeros((12,T,R,N))
spikes_m[0,:,:,:] = emd.spikes

# Iteration over all orientation
for orientation in range(30,360,30):
    f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(orientation)+pop+b, 'rb')
    emd = pickle.load(f)
    f.close()
    G = int(orientation / 30)
    theta_m[G,:,:] = emd.theta_s
    if emd.marg_llk == probability.log_marginal:
        for t in range(T):
            sigma_m[G,t] = numpy.diag(emd.sigma_s[t])
    else:
        sigma_m[G] = emd.sigma_s
    spikes_m[G,:,:,:] = emd.spikes

############# Computation of macroscopic properties #############
eta = numpy.zeros((12,T,D))
psi = numpy.zeros((12,T))
epsilon = 1e-3
C = numpy.zeros((12,T))
p_silence = numpy.zeros((12,T))
p_spike = numpy.zeros((12,T))
S2 = numpy.zeros((12,T))
theta_ind = numpy.zeros((12,T,N))
eta_ind = numpy.zeros((12,T,N))
psi_ind = numpy.zeros((12,T))
S1 = numpy.zeros((12,T))
for k in range(12):
    if emd.marg_llk == probability.log_marginal: # If exact:
        for t in range(T):
            p = transforms.compute_p(theta_m[k,t,:])
            psi[k,t] = transforms.compute_psi(theta_m[k,t,:])
            eta[k,t,:] = transforms.compute_eta(p)
            tmp1 = transforms.compute_psi(theta_m[k,t,:] * (1 + epsilon))
            tmp2 = transforms.compute_psi(theta_m[k,t,:] * (1 - epsilon))
            c = tmp1 - 2 * psi[k,t] + tmp2
            d = epsilon**2
            C[k,t] = c / d
    else: # If approximations
        for t in range(T):
            eta[k,t,:], psi[k,t] = bethe_approximation.compute_eta_hybrid(theta_m[k,t,:], N, return_psi=True)
            tmp1 = bethe_approximation.compute_eta_hybrid(theta_m[k,t,:] * (1 + epsilon), N, return_psi=True)[1]
            tmp2 = bethe_approximation.compute_eta_hybrid(theta_m[k,t,:] * (1 - epsilon), N, return_psi=True)[1]
            c = tmp1 - 2 * psi[k,t] + tmp2
            d = epsilon**2
            C[k,t] = c / d
    p_silence[k,:] = numpy.exp(-psi[k,:])
    p_spike[k,:] = numpy.sum(eta[k,:,:N],axis=1) / N
    S2[k,:] = energies.compute_entropy(theta_m[k,:,:], eta[k,:,:], psi[k,:], 2)
    eta_ind[k,:,:] = eta[k,:,:N]
    theta_ind[k,:,:] = energies.compute_ind_theta(eta_ind[k,:,:])
    psi_ind[k,:] = energies.compute_ind_psi(theta_ind[k,:,:])
    S1[k,:] = energies.compute_entropy(theta_ind[k,:,:], eta_ind[k,:,:], psi_ind[k,:], 1)


############# Export the results to a pickle file #############
results = list()
results.extend((theta_m, sigma_m, p_spike, p_silence, S1, S2, C, spikes_m))

f = open(directory+'/Data/data_fig1_m'+str(monkey+1)+pop+b, 'wb')
pickle.dump(results, f)
f.close()
