"""

Traces figure 2. "experiments.py"  and "data_fig1.py" must be run before this
code, so that the macroscopic properties are computed and saved.

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
import scipy.stats
import matplotlib as mpl

# Number of neurons
N = 12

# Total number of populations
npop = 3

# Number of time bins
T = 128

############# Extraction of data #############
import os
directory = os.getcwd()

p_silence = numpy.zeros((3,npop,12,T)) # monkey X population X orientaton X time bin
S1 = numpy.zeros((3,npop,12,T))
S2 = numpy.zeros((3,npop,12,T))
C = numpy.zeros((3,npop,12,T))
ratio = numpy.zeros((3,npop,12,T))
p_silence_shuffle = numpy.zeros((3,npop,12,T))
S1_shuffle = numpy.zeros((3,npop,12,T))
S2_shuffle = numpy.zeros((3,npop,12,T))
C_shuffle = numpy.zeros((3,npop,12,T))
ratio_shuffle = numpy.zeros((3,npop,12,T))
S0 = N * numpy.log(2)

for monkey in range(3):
    for B in range(npop):
        b = str(B+1)
        f = open(directory+'/Data/data_fig1_m'+str(monkey+1)+b, 'rb')
        p_silence_tmp, S1_tmp, S2_tmp, C_tmp = pickle.load(f)[3:7]
        f.close()
        p_silence[monkey,B] = p_silence_tmp
        S1[monkey,B] = S1_tmp
        S2[monkey,B] = S2_tmp
        C[monkey,B] = C_tmp
        f = open(directory+'/Data/data_fig1_m'+str(monkey+1)+b+'_shuffle', 'rb')
        p_silence_shuffle_tmp, S1_shuffle_tmp, S2_shuffle_tmp, C_shuffle_tmp = pickle.load(f)[3:7]
        f.close()
        p_silence_shuffle[monkey,B] = p_silence_shuffle_tmp
        S1_shuffle[monkey,B] = S1_shuffle_tmp
        S2_shuffle[monkey,B] = S2_shuffle_tmp
        C_shuffle[monkey,B] = C_shuffle_tmp
        # Computation of the ratio
        ratio[monkey,B] = (S1_tmp - S2_tmp) / (S0 - S2_tmp) * 100
        ratio_shuffle[monkey,B] = (S1_shuffle_tmp - S2_shuffle_tmp) /\
                                  (S0 - S2_shuffle_tmp) * 100

############# Figure Creation #############
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.major.width'] = .3
mpl.rcParams['ytick.major.width'] = .3
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2

fig = plt.figure(figsize=(5,5))
color = ('b','c','k')
# Choice of orientations
G = [3,6] # *30 = orientations in degrees
size = 1
line = 0.5

# Axis Titles
ax0 = fig.add_axes([0.06,0.05,0.88,0.9], frameon=0)
ax0.set_yticks([])
ax0.set_xticks([])
ax0.set_ylabel('Original data', fontsize=15)
ax0.set_xlabel('Surrogate data', fontsize=15)

# Orientations
ax = fig.add_axes([0.265,0.97,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,r'$\phi$ = '+str(int(G[0]*30))+r'$\degree$', fontsize=8, fontweight='bold')

ax = fig.add_axes([0.66,0.97,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,r'$\phi$ = '+str(int(G[1]*30))+r'$\degree$', fontsize=8, fontweight='bold')

# Entropy ratio,orientation 1
ax1 = fig.add_axes([0.13,0.66,0.35,0.3])
ax1.set_frame_on(False)
ax1.set_xticks([])
ax1.set_yticks([0,0.8,1.6,2.4])
ax1.set_ylabel('$\gamma$')
for i in range(3):
    ax1.scatter(ratio_shuffle[i,:,G[0]], ratio[i,:,G[0]], color=color[i], s=size)
    ax1.plot((0, numpy.max(ratio[i,:,G[0]])), (0, numpy.max(ratio[i,:,G[0]])), 'r', linewidth=line)
ymin, ymax = ax1.get_yaxis().get_view_interval()
xmin, xmax = ax1.get_xaxis().get_view_interval()
ax1.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax1.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(ratio[:,:,G[0]].reshape(int(3*npop*T)), ratio_shuffle[:,:,G[0]].reshape(int(3*npop*T)))[1]
if pvalue < 1e-10:
    ax1.text(numpy.max(ratio[:,:,G[0]])*0.3,.0, 'p-value < 1e-10', fontsize=8)
elif pvalue > 1e-2:
    ax1.text(numpy.max(ratio[:,:,G[0]])*0.3,.0, 'p-value = '+'%.4f' % pvalue, fontsize=8)
else:
    ax1.text(numpy.max(ratio[:,:,G[0]])*0.3,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=8)

# Entropy ratio,orientation 2
ax2 = fig.add_axes([0.53,0.66,0.35,0.3])
ax2.set_frame_on(False)
ax2.set_xticks([])
ax2.set_yticks([0,0.5,1,1.5])
for i in range(3):
    ax2.scatter(ratio_shuffle[i,:,G[1]], ratio[i,:,G[1]], color=color[i], s=size)
    ax2.plot((0, numpy.max(ratio[i,:,G[1]])), (0, numpy.max(ratio[i,:,G[1]])), 'r', linewidth=line)
ymin, ymax = ax2.get_yaxis().get_view_interval()
xmin, xmax = ax2.get_xaxis().get_view_interval()
ax2.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax2.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(ratio[:,:,G[1]].reshape(int(3*npop*T)), ratio_shuffle[:,:,G[1]].reshape(int(3*npop*T)))[1]
if pvalue < 1e-10:
    ax2.text(numpy.max(ratio[:,:,G[1]])*0.3,.0, 'p-value < 1e-10', fontsize=8)
elif pvalue > 1e-2:
    ax2.text(numpy.max(ratio[:,:,G[1]])*0.3,.0, 'p-value = '+'%.4f' % pvalue, fontsize=8)
else:
    ax2.text(numpy.max(ratio[:,:,G[1]])*0.3,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=8)

# Sparseness, orientation 1
ax3 = fig.add_axes([0.13,0.36,0.35,0.3])
ax3.set_frame_on(False)
ax3.set_xticks([])
ax3.set_yticks([0,0.5,1])
ax3.set_ylabel('$p_{silence}$')
for i in range(3):
    ax3.scatter(p_silence_shuffle[i,:,G[0]], p_silence[i,:,G[0]], color=color[i], s=size)
ax3.legend(['Monkey 1', 'Monkey 2', 'Monkey 3'], fontsize=8, loc=2)
for i in range(3):
    ax3.plot((0, numpy.max(p_silence[i,:,G[0]])), (0, numpy.max(p_silence[i,:,G[0]])), 'r', linewidth=line)
ymin, ymax = ax3.get_yaxis().get_view_interval()
xmin, xmax = ax3.get_xaxis().get_view_interval()
ax3.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax3.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(p_silence[:,:,G[0]].reshape(int(3*npop*T)), p_silence_shuffle[:,:,G[0]].reshape(int(3*npop*T)))[1]
if pvalue < 1e-10:
    ax3.text(numpy.max(p_silence[:,:,G[0]])*0.3,.0, 'p-value < 1e-10', fontsize=8)
elif pvalue > 1e-2:
    ax3.text(numpy.max(p_silence[:,:,G[0]])*0.3,.0, 'p-value = '+'%.4f' % pvalue, fontsize=8)
else:
    ax3.text(numpy.max(p_silence[:,:,G[0]])*0.3,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=8)

# Sparseness, orientation 2
ax4 = fig.add_axes([0.53,0.36,0.35,0.3])
ax4.set_frame_on(False)
ax4.set_xticks([])
ax4.set_yticks([0,0.5,1])
for i in range(3):
    ax4.scatter(p_silence_shuffle[i,:,G[1]], p_silence[i,:,G[1]], color=color[i], s=size)
    ax4.plot((0, numpy.max(p_silence[i,:,G[1]])), (0, numpy.max(p_silence[i,:,G[1]])), 'r', linewidth=line)
ymin, ymax = ax4.get_yaxis().get_view_interval()
xmin, xmax = ax4.get_xaxis().get_view_interval()
ax4.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax4.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(p_silence[:,:,G[1]].reshape(int(3*npop*T)), p_silence_shuffle[:,:,G[1]].reshape(int(3*npop*T)))[1]
if pvalue < 1e-10:
    ax4.text(numpy.max(p_silence[:,:,G[1]])*0.3,.0, 'p-value < 1e-10', fontsize=8)
elif pvalue > 1e-2:
    ax4.text(numpy.max(p_silence[:,:,G[1]])*0.3,.0, 'p-value = '+'%.4f' % pvalue, fontsize=8)
else:
    ax4.text(numpy.max(p_silence[:,:,G[1]])*0.3,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=8)

# Heat capacity, orientatiion 1
ax5 = fig.add_axes([0.13,0.06,0.35,0.3])
ax5.set_frame_on(False)
ax5.set_xticks([])
ax5.set_yticks([0,2,4,6])
ax5.set_ylabel('$C$')
for i in range(3):
    ax5.scatter(C_shuffle[i,:,G[0]], C[i,:,G[0]], color=color[i], s=size)
    ax5.plot((0, numpy.max(C[i,:,G[0]])), (0, numpy.max(C[i,:,G[0]])), 'r', linewidth=line)
ymin, ymax = ax5.get_yaxis().get_view_interval()
xmin, xmax = ax5.get_xaxis().get_view_interval()
ax5.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax5.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(C[:,:,G[0]].reshape(int(3*npop*T)), C_shuffle[:,:,G[0]].reshape(int(3*npop*T)))[1]
if pvalue < 1e-10:
    ax5.text(numpy.max(C[:,:,G[0]])*0.3,.0, 'p-value < 1e-10', fontsize=8)
elif pvalue > 1e-2:
    ax5.text(numpy.max(C[:,:,G[0]])*0.3,.0, 'p-value = '+'%.4f' % pvalue, fontsize=8)
else:
    ax5.text(numpy.max(C[:,:,G[0]])*0.3,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=8)

# Heat capacity, orientatiion 2
ax6 = fig.add_axes([0.53,0.06,0.35,0.3])
ax6.set_frame_on(False)
ax6.set_xticks([])
ax6.set_yticks([0,2,4,6])
for i in range(3):
    ax6.scatter(C_shuffle[i,:,G[1]], C[i,:,G[1]], color=color[i], s=size)
    ax6.plot((0, numpy.max(C[i,:,G[1]])), (0, numpy.max(C[i,:,G[1]])), 'r', linewidth=line)
ymin, ymax = ax6.get_yaxis().get_view_interval()
xmin, xmax = ax6.get_xaxis().get_view_interval()
ax6.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax6.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
pvalue = scipy.stats.wilcoxon(C[:,:,G[1]].reshape(int(3*npop*T)), C_shuffle[:,:,G[1]].reshape(int(3*npop*T)))[1]
if pvalue < 1e-10:
    ax6.text(numpy.max(C[:,:,G[1]])*0.3,.0, 'p-value < 1e-10', fontsize=8)
elif pvalue > 1e-2:
    ax6.text(numpy.max(C[:,:,G[1]])*0.3,.0, 'p-value = '+'%.4f' % pvalue, fontsize=8)
else:
    ax6.text(numpy.max(C[:,:,G[1]])*0.3,.0, 'p-value < 1e'+str(int(numpy.log10(pvalue))), fontsize=8)

# Save the figure to a file in the 'Figures' folder (create it if inexistant)
if not os.path.exists(directory+'/Figures/'):
   os.makedirs(directory+'/Figures/')

fig.savefig(directory+'/Figures/fig2.eps')
