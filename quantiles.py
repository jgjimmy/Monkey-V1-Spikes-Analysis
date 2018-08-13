"""

For all stimulus orientations, samples theta 'n_samples' times and computes the
macroscopic properties for every sample. Finally, saves lower and upper 
quantiles set by 'qp' to retrieve a credible interval for the properties at 
every time step. The results will be needed to run fig1.py.

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
from scipy.stats.mstats import mquantiles
import os
import sys

directory = os.getcwd()
sys.path.insert(1,directory + '\ssll_lib')

import transforms
import probability
import energies
import bethe_approximation

# Monkey to analyse (0,1,2)
monkey = 0

# Population to analyze ('1', '2', '3')
pop = '1'

# Shuffled-data boolean. If 'True', computes quantiles for models fit to
# shuffled data.
Shuffled = True
b = ''
if Shuffled:
    b += '_shuffle'

# Number of samples for the computation of the quantiles
n_samples = 100

# Desired quantile (percent i.e. qp=1=1%, we exclude 1% of the samples)
qp = 10

# Extract the data
if not os.path.exists(directory+'/Data_quantiles/'):
    os.makedirs(directory+'/Data_quantiles/')

# Iteration over all orientations
for k in range(12):
    f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(int(k*30))+pop+b, 'rb')
    emd = pickle.load(f)
    f.close()
    # Exact boolean (True if exact solution was used)
    if emd.marg_llk == probability.log_marginal:
        exact = True
    else:
        exact = False
    N = emd.N
    if exact:
        transforms. initialise(N, 2)
    T = emd.T
    D = numpy.size(emd.theta_s,1)
    theta = emd.theta_s
    if exact:
        sigma = emd.sigma_s
    else:
        sigma = numpy.zeros((T, D, D))
        for t in range(T):
            sigma[t] = numpy.diag(emd.sigma_s[t])
    S0 = N * numpy.log(2)
    ############# Quantiles computations #############
    # Compute macroscopic quantites for all theta samples
    physics_matrix = numpy.zeros((n_samples,8,T))
    epsilon = 1e-3
    for n in range(n_samples):
        # Sample theta
        theta_sampled = numpy.zeros((T,D))
        for t in range(T):
            theta_sampled[t] = numpy.random.multivariate_normal(theta[t,:],sigma[t], 1).reshape(D)
        physics_matrix[n,0,:] = numpy.mean(theta_sampled[:,:N], axis=1).T
        physics_matrix[n,1,:] = numpy.mean(theta_sampled[:,N:], axis=1).T
        print(n)
        # Obtain eta and psi
        eta = numpy.zeros((T,D))
        psi = numpy.zeros((T,))
        if exact:
            for t in range(T):
                p = transforms.compute_p(theta_sampled[t,:])
                eta[t,:] = transforms.compute_eta(p)
                psi[t] = transforms.compute_psi(theta_sampled[t,:])
        else:
            for t in range(T):
                eta[t,:], psi[t] = bethe_approximation.compute_eta_hybrid\
                           (theta_sampled[t,:], N, return_psi=True)
        # Entropy (2nd order)
        S2 = energies.compute_entropy(theta_sampled[:,:], eta, psi, 2)
        # p_silence
        p_silence = numpy.exp(-psi)
        # p_spike
        p_spike = numpy.mean(eta[:,:N],axis=1)
        # Heat capacity
        tmp1 = numpy.zeros((T,))
        tmp2 = numpy.zeros((T,))
        if exact:
            for t in range(T):
                tmp1[t] = transforms.compute_psi(theta_sampled[t,:] * (1 + epsilon))
                tmp2[t] = transforms.compute_psi(theta_sampled[t,:] * (1 - epsilon))
        else:
            for t in range(T):
                tmp1[t] = bethe_approximation.compute_eta_hybrid(theta_sampled[t,:] * (1 + epsilon), N, return_psi=True)[1]
                tmp2[t] = bethe_approximation.compute_eta_hybrid(theta_sampled[t,:] * (1 - epsilon), N, return_psi=True)[1]
        c = tmp1 - 2 * psi + tmp2
        d = epsilon**2
        C = c / d
        # Entropy (1st order)
        eta_ind = eta[:,:N]
        theta_ind = energies.compute_ind_theta(eta_ind)
        psi_ind = energies.compute_ind_psi(theta_ind)
        S1 = energies.compute_entropy(theta_ind, eta_ind, psi_ind, 1)
        # Entropy ratio
        ratio = (S1 - S2) / (S0 - S2)
        # Record results
        physics_matrix[n,2,:] = p_spike
        physics_matrix[n,3,:] = p_silence
        physics_matrix[n,4,:] = S1
        physics_matrix[n,5,:] = S2
        physics_matrix[n,6,:] = ratio
        physics_matrix[n,7,:] = C
    quantiles = numpy.zeros((8,T,2))
    for i in range(8):
        for t in range(T):
            quantiles[i,t,:] = mquantiles(physics_matrix[:,i,t], prob=[qp/2/100, 1-qp/2/100], axis=0)
    # Export results to file
    f = open(directory+'/Data_quantiles/quantiles_m'+str(monkey+1)+'d'+str(int(k*30))+pop+b,'wb')
    pickle.dump(quantiles,f)
    f.close()

