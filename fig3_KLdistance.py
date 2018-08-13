"""

Traces figure 3 while considering the KL distance instead of the Bhattacharyya
distance. "experiments.py" must be run before this code, so that the
state-space Ising models are fitted and saved.

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
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import os
import sys

directory = os.getcwd()
sys.path.insert(1,directory + '\ssll_lib')

import probability
import energies
import bethe_approximation
import transforms

# Total number of populations per monkey
npop = 3

############# Extraction of the data #############
orientation = 0
monkey = 0
pop = '1'
f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(orientation)+pop, 'rb')
emd = pickle.load(f)
f.close()

# Number of time bins, trials and neurons
T, R, N = emd.spikes.shape

# Number of dimensions
D = emd.D

# Initialise the theta matrices
theta_m = numpy.zeros((3,npop,12,T,D)) # Monkey X Population X Orientation X T,D
theta_ind = numpy.zeros((3,npop,12,T,D))
# Iterations over all monkeys, populations, orientations
for monkey in range(3):
    for i in range(npop):
        for orientation in range(0,360,30):
            b = str(i+1)
            f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(orientation)+b, 'rb')
            emd = pickle.load(f)
            f.close()
            G = int(orientation / 30)
            theta_m[monkey,i,G,:,:] = emd.theta_s
           ############# SHUFFLED #############
            f = open(directory+'/Data/m'+str(monkey+1)+'d'+str(orientation)+b+'_shuffle', 'rb')
            emd = pickle.load(f)
            f.close()
            G = int(orientation / 30)
            theta_ind[monkey,i,G,:,:] = emd.theta_s

############# Computation of the KL distance #############
# KL distance computation function
if emd.marg_llk == probability.log_marginal: # i.e. if exact model is used
    transforms.initialise(N,2)
    def KL_distance(theta1, theta2, N):
        D = theta1.shape[0]
        p1 = transforms.compute_p(theta1)
        eta1 = transforms.compute_eta(p1)
        psi1 = transforms.compute_psi(theta1)
        p2 = transforms.compute_p(theta2)
        eta2 = transforms.compute_eta(p2)
        psi2 = transforms.compute_psi(theta2)
        S1 = energies.compute_entropy(theta1.reshape(1,D), eta1.reshape(1,D), psi1, 2)
        S2 = energies.compute_entropy(theta2.reshape(1,D), eta2.reshape(1,D), psi2, 2)
        KL1 = -S1 + psi2 - eta1.dot(theta2)
        KL2 = -S2 + psi1 - eta2.dot(theta1)
        KL = (KL1 + KL2) / 2
        return KL
else: # If approximations are used
    def KL_distance(theta1, theta2, N):
        D = theta1.shape[0]
        eta1, psi1 = bethe_approximation.compute_eta_hybrid(theta1, N, return_psi=True)
        eta2, psi2 = bethe_approximation.compute_eta_hybrid(theta1, N, return_psi=True)
        S1 = energies.compute_entropy(theta1.reshape(1,D), eta1.reshape(1,D), psi1, 2)
        S2 = energies.compute_entropy(theta2.reshape(1,D), eta2.reshape(1,D), psi2, 2)
        KL1 = -S1 + psi2 - eta1.dot(theta2)
        KL2 = -S2 + psi1 - eta2.dot(theta1)
        KL = (KL1 + KL2) / 2
        return KL

# Distance between thetas of orientation 'i' and orientation 'j'
distance_pair = numpy.zeros((3,npop,12,12)) # pairwise model
distance_ind = numpy.zeros((3,npop,12,12)) # independant model

# Distances between thetas of orientations with a 30*i degree difference
diff_pair = numpy.zeros((3,npop,12,12))
diff_ind = numpy.zeros((3,npop,12,12))

for monkey in range(3):
    for g in range(npop):
        # Counter for the 12 pairs of orientations with a 30*i degree difference
        count = numpy.zeros((12,))
        for i in range(12):
            for j in range(12):
                if i != j:
                    for t in range(T):
                        # Pairwise
                        distance_pair[monkey,g,i,j] += KL_distance(theta_m[monkey,g,i,t,:],\
                                                                  theta_m[monkey,g,j,t,:], N)
                        # Shuffled
                        distance_ind[monkey,g,i,j] += KL_distance(theta_ind[monkey,g,i,t,:],\
                                                                 theta_ind[monkey,g,j,t,:], N)
                index = int(numpy.mod( (j - i)*30, 360 )/30)
                diff_pair[monkey,g,index,int(count[index])] = distance_pair[monkey,g,i,j]
                diff_ind[monkey,g,index,int(count[index])] = distance_ind[monkey,g,i,j]
                count[index] += 1

diff_pair_avg = numpy.mean(diff_pair, axis=3)
diff_ind_avg = numpy.mean(diff_ind, axis=3)

############# Figure #############
fig = plt.figure(figsize=(10,5), dpi=200)
fontsize=10
# A
ax1 = fig.add_axes([0.09, 0.58, 0.35, 0.38])
ax1.set_frame_on(False)

for i in range(npop):
    ax1.plot(range(30,210,30), diff_pair_avg[0,i,1:7], 'b')
    ax1.plot(range(30,210,30), diff_pair_avg[1,i,1:7], 'c')
    ax1.plot(range(30,210,30), diff_pair_avg[2,i,1:7], 'k')

ax1.set_xticks(range(30,210,30))
ax1.legend(['Monkey 1', 'Monkey 2', 'Monkey 3'])
ax1.set_xlabel(r'$\Delta\phi$')
ax1.set_ylabel('KL distance')
ymin, ymax = ax1.get_yaxis().get_view_interval()
xmin, xmax = ax1.get_xaxis().get_view_interval()
ax1.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax1.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))

# B
ax2 = fig.add_axes([0.09, 0.1, 0.35, 0.38])
ax2.set_frame_on(False)

orientation = 3
color = ('b','c','k')
for monkey in range(3):
    ax2.scatter(diff_ind[monkey,:,orientation,:], diff_pair[monkey,:,orientation,:], color=color[monkey], s=5)

ax2.legend(['Monkey 1', 'Monkey 2', 'Monkey 3'])
ax2.set_xlabel('Distance with surrogate data')
ax2.set_ylabel('Distance with original data')
ax2.plot((0,numpy.max(diff_pair[:,:,orientation,:])),(0,numpy.max(diff_pair[:,:,orientation,:])), 'r', linewidth=0.5)
ymin, ymax = ax2.get_yaxis().get_view_interval()
xmin, xmax = ax2.get_xaxis().get_view_interval()
ax2.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax2.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(diff_pair[:,:,orientation,:].reshape(int(3*npop*12)), diff_ind[:,:,orientation,:].reshape(int(3*npop*12)))[1]
if pvalue < 1e-10:
    ax2.text(numpy.max(diff_pair[:,:,orientation,:])*0.4,.0, 'p-value < 1e-10', fontsize=10)
elif pvalue > 1e-2:
    ax2.text(numpy.max(diff_pair[:,:,orientation,:])*0.4,.0, 'p-value = '+'%.4f' % pvalue, fontsize=10)
else:
    ax2.text(numpy.max(diff_pair[:,:,orientation,:])*0.4,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=10)

# C
ax3 = fig.add_axes([0.53, 0.15, 0.45, 0.7])
ax3.set_frame_on(False)

for i in range(npop):
    ax3.plot(range(30,210,30), diff_pair_avg[0,i,1:7], 'b')
    ax3.plot(range(30,210,30), diff_pair_avg[1,i,1:7], 'c')
    ax3.plot(range(30,210,30), diff_pair_avg[2,i,1:7], 'k')
    ax3.plot(range(30,210,30), diff_ind_avg[0,i,1:7], 'b--')
    ax3.plot(range(30,210,30), diff_ind_avg[1,i,1:7], 'c--')
    ax3.plot(range(30,210,30), diff_ind_avg[2,i,1:7], 'k--')

ax3.set_xticks(range(30,210,30))
ax3.set_xlabel(r'$\Delta\phi$')
ax3.set_ylabel('KL distance')
ymin, ymax = ax3.get_yaxis().get_view_interval()
xmin, xmax = ax3.get_xaxis().get_view_interval()
ax3.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax3.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax3.text(0.7*xmax,0.9*ymax,'Solid: Original Data')
ax3.text(0.7*xmax,0.8*ymax,'Dashed: Surrogate Data')

# A, B, C
ax = fig.add_axes([0.01,0.97,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'A', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.01,0.5,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'B', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.5,0.88,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'C', fontsize=fontsize, fontweight='bold')

# Save the figure to a file in the 'Figures' folder (create it if inexistant)
if not os.path.exists(directory+'/Figures/'):
   os.makedirs(directory+'/Figures/')

fig.savefig(directory+'/Figures/fig3_KL.eps')
