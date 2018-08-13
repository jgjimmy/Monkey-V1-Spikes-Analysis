"""

Traces figure 1 without credible intervals for the entropy, entropy ratio,
sparseness and heat capacity. "experiments.py"  and "data_fig1.py" must be run
before this code, so that the macroscopic properties are computed and saved.

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
                                 Christian Donner (christian.donner@bccn-berlin.de)
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
from matplotlib import pyplot
import matplotlib as mpl
import pickle
import networkx as nx
import itertools

monkey = 2 # (0, 1, 2)
orientation = 300
pop = '1' # Index of the population ('1', '2', '3')
G = int(orientation/30)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

psi_color = numpy.array([51, 153, 255]) / 256
eta_color = numpy.array([0, 204, 102]) / 256
S_color = numpy.array([255, 162, 0]) / 256
C_color = numpy.array([204, 60, 60]) / 256
S_ratio_color = numpy.array([1.,.3,0.])

# Import Data
import os
directory = os.getcwd()
f = open(directory+'/Data/data_fig1_m'+str(monkey+1)+pop, 'rb')
results = pickle.load(f)
f.close()
f = open(directory+'/Data/data_fig1_m'+str(monkey+1)+pop+'_shuffle', 'rb')
results_shuffle = pickle.load(f)
f.close()
theta = results[0][G,:,:]
theta_shuffle = results_shuffle[0][G,:,:]
sigma = results[1][G,:,:]
p_spike = results[2][G]
p_spike_shuffle = results_shuffle[2][G]
p_silence = results[3][G]
p_silence_shuffle = results_shuffle[3][G]
S1 = results[4][G]
S1_shuffle = results_shuffle[4][G]
S2 = results[5][G]
S2_shuffle = results_shuffle[5][G]
C = results[6][G]
C_shuffle = results_shuffle[6][G]
spikes = results[7][G]
spikes_shuffle = results_shuffle[7][G]
N = spikes.shape[2]
T = numpy.size(theta, 0)
D = numpy.size(theta, 1)
Total = 1.28
dt = Total / T
bin_size = dt
Time = numpy.linspace(0, Total-dt, Total/dt)

############# Figure creation #############
size_in_cm = 19.*numpy.array([1, 1./2.])
size_in_inch = cm2inch(size_in_cm)[0]
fontsize = 6

mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['xtick.major.width'] = .3
mpl.rcParams['ytick.major.width'] = .3
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2

fig = pyplot.figure(figsize=size_in_inch, dpi=200)

# spikes
ax = fig.add_axes([0.07,0.5,.25,.4])
ax.imshow(spikes[:,0,:].transpose(), 'Greys', aspect=2)
ax.set_xticks([])
ax.set_yticks([])
ax = fig.add_axes([0.06,0.47,.25,.4])
ax.imshow(spikes[:,1,:].transpose(), 'Greys', aspect=2)
ax.set_xticks([])
ax.set_yticks([])
ax = fig.add_axes([0.05,0.44,.25,.4])
ax.imshow(spikes[:,2,:].transpose(), 'Greys', aspect=2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Unit ID', fontsize=fontsize)

# p_spike
ax = fig.add_axes([.05,0.1,.25,.33])
ax.set_frame_on(False)
rate = numpy.mean(numpy.mean(spikes[:,:,:],axis=1),axis=1)
rate_shuffle = numpy.mean(numpy.mean(spikes_shuffle[:,:,:],axis=1),axis=1)
ax.plot(Time, rate, linewidth=1, color='k')
ax.plot(Time, p_spike, linewidth=1, color=eta_color)
ax.plot(Time, p_spike_shuffle, linewidth=1, color='r')
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks([.02,.05,.08])
ax.set_xticks([0,1.28])
ax.set_ylabel('$p_{\\mathrm{spike}}$', fontsize=fontsize)
ax.yaxis.labelpad = -.05
pyplot.legend(['Data','Fit','Surrogate-data fit'], frameon=0, fontsize=int(.9*fontsize))
ax.set_xlabel('Time [s]', fontsize=fontsize)

# Network
bounds = numpy.empty([theta.shape[0],theta.shape[1] - N,2])
bounds[:,:,0] = theta[:,N:] - 1.96*numpy.sqrt(sigma[:,N:])
bounds[:,:,1] = theta[:,N:] + 1.96*numpy.sqrt(sigma[:,N:])

graph_ax = [fig.add_axes([.315,0.55,.13,.33]),
            fig.add_axes([.42,0.55,.13,.33]),
            fig.add_axes([.525,0.55,.13,.33])]

T_choice = numpy.array([0, 0.1, 0.8]) / dt
for i, t in enumerate(T_choice):
    t = int(t)
    i = int(i)
    idx = numpy.where(numpy.logical_or(bounds[t,:,0] > 0, bounds[t,:,1] < 0))[0]
    conn_idx_all = numpy.arange(0,N*(N-1)/2)
    conn_idx = conn_idx_all[idx]
    all_conns = itertools.combinations(range(N),2)
    conn_idx = conn_idx.astype(int)
    conns = numpy.array(list(all_conns))[conn_idx]
    G1 = nx.Graph()
    G1.add_nodes_from(range(N))
    G1.add_edges_from(conns)
    pos1 = nx.circular_layout(G1, scale=0.015)
    net_nodes = nx.draw_networkx_nodes(G1, pos1, ax=graph_ax[i], node_color=theta[t,:N],
                                       cmap=pyplot.get_cmap('hot'), vmin=-5,vmax=-1., node_size=25, linewidths=.5)
    e1 = nx.draw_networkx_edges(G1, pos1, ax=graph_ax[i], edge_color=theta[t,conn_idx].tolist(),
                                edge_cmap=pyplot.get_cmap('seismic'),edge_vmin=-1.,edge_vmax=1., width=1)
    graph_ax[i].axis('off')
    graph_ax[i].set_title('t=%.2f s' %Time[int(T_choice[i])], fontsize=fontsize)
    x0,x1 = graph_ax[i].get_xlim()
    y0,y1 = graph_ax[i].get_ylim()
    graph_ax[i].set_aspect(abs(x1-x0)/abs(y1-y0))
cbar_ax = fig.add_axes([0.5, 0.57, 0.1, 0.01])
cbar = fig.colorbar(net_nodes, cax=cbar_ax, orientation='horizontal')
cbar.outline.set_linewidth(.5)
cbar.set_ticks([-5,-3,-1])
cbar_ax.set_title('$\\theta_{i}$', fontsize=fontsize)
cbar_ax = fig.add_axes([0.38, 0.57, 0.1, 0.01])
cbar = fig.colorbar(e1, cax=cbar_ax, orientation='horizontal')
cbar.outline.set_linewidth(.5)
cbar.set_ticks([-.5,0.,.5])
cbar_ax.set_title('$\\theta_{ij}$', fontsize=fontsize)


# theta_i
ax = fig.add_axes([.35,0.33,.28,.2], frameon=0)
theta_mean = numpy.mean(theta[:,:N], axis=1)
theta_mean_shuffle = numpy.mean(theta_shuffle[:,:N], axis=1)
ax.plot(Time, theta_mean, color=[.3,.3,.3], lw=1)
ax.plot(Time, theta_mean_shuffle, 'r', lw=1)
ax.set_ylabel('$\\langle \\theta_{i} \\rangle_{i}$', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.yaxis.set_ticks_position('left')
ax.set_xticks([])
ax.set_yticks([-5,-4.4,-3.8])
ax.add_artist(pyplot.Line2D((0, 0), (ymin, ymax), color='black', linewidth=1, linestyle='--'))
ax.add_artist(pyplot.Line2D((0.1, 0.1), (ymin, ymax), color='black', linewidth=1, linestyle='--'))
ax.add_artist(pyplot.Line2D((0.8, 0.8), (ymin, ymax), color='black', linewidth=1, linestyle='--'))

# theta_ij
ax = fig.add_axes([.35,0.1,.28,.2], frameon=0)
theta_mean = numpy.mean(theta[:,N:], axis=1)
theta_mean_shuffle = numpy.mean(theta_shuffle[:,N:], axis=1)
ax.plot(Time, theta_mean, color=[.3,.3,.3], lw=1)
ax.plot(Time, theta_mean_shuffle, 'r', lw=1)
ax.set_ylabel('$\\langle \\theta_{ij} \\rangle_{ij}$', fontsize=fontsize)
ax.yaxis.labelpad = -.6
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.add_artist(pyplot.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1))
ax.set_yticks([-0.2, 0, 0.2])
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_xticks([0,0.1,0.8,1.28])
labels = ['0', '0.10', '0.80', '1.28']
ax.set_xticklabels(labels)
ax.add_artist(pyplot.Line2D((0, 0), (ymin, ymax), color='black', linewidth=1, linestyle='--'))
ax.add_artist(pyplot.Line2D((0.1, 0.1), (ymin, ymax), color='black', linewidth=1, linestyle='--'))
ax.add_artist(pyplot.Line2D((0.8, 0.8), (ymin, ymax), color='black', linewidth=1, linestyle='--'))

# Entropy
ax = fig.add_axes([.68,0.7,.28,.15])
ax.set_frame_on(False)
ax.plot(Time, S2, color='k',lw=1)
ax.plot(Time, S2_shuffle, color='r', lw=1)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(pyplot.Line2D((xmin, xmin), (0.8, 7.2), color='black', linewidth=1))
ax.set_xticks([])
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel('$S$', fontsize=fontsize)
ax.set_yticks([2,4,6])

# Entropy ratio
S0  = N*numpy.log(2)
ax = fig.add_axes([.68,0.5,.28,.15])
ax.set_frame_on(False)
ax.plot(Time, (S1 - S2)/(S0 - S2)*100, color='k',lw=1)
ax.plot(Time, (S1_shuffle - S2_shuffle)/(S0 - S2_shuffle)*100, color='r', lw=1)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(pyplot.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=1))
ax.set_xticks([])
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks([.4,.8,1.2])
ax.set_ylabel('$\\frac{S_{\\mathrm{ind}}-S_{\\mathrm{pair}}}{S_{\\mathrm{0}}-S_{\\mathrm{pair}}}\ [\%]$', fontsize=fontsize)

# p_silence
ax = fig.add_axes([.68,0.3,.28,.15])
ax.set_frame_on(False)
ax.plot(Time, p_silence, color='k',lw=1)
ax.plot(Time, p_silence_shuffle, color='r', lw=1)
ax.set_ylabel('$p_{silence}$', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.add_artist(pyplot.Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
ax.set_xticks([])
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks([0.2,0.4,.6])

# Heat capacity
ax = fig.add_axes([.68,0.1,.28,.15])
ax.set_frame_on(False)
ax.plot(Time, C, color='k',lw=1)
ax.plot(Time, C_shuffle, color='r',lw=1)
ax.set_ylabel('$C$', fontsize=fontsize)
ax.set_xlabel('Time [s]', fontsize=fontsize)
ymin, ymax = ax.get_yaxis().get_view_interval()
xmin, xmax = ax.get_xaxis().get_view_interval()
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks([8,11,14])
ax.add_artist(pyplot.Line2D((xmin, xmin), (ax.get_yticks()[0], ymax), color='black', linewidth=1))
ax.add_artist(pyplot.Line2D((xmin, xmax), (ax.get_yticks()[0], ax.get_yticks()[0]), color='black', linewidth=1))
ax.set_xticks([0, 1.28])
labels = ['0', '1.28']
ax.set_xticklabels(labels)

# A, B, C
ax = fig.add_axes([0.03,0.88,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'A', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.33,0.88,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'B', fontsize=fontsize, fontweight='bold')
ax = fig.add_axes([0.66,0.88,.05,.05], frameon=0)
ax.set_yticks([])
ax.set_xticks([])
ax.text(.0,.0,'C', fontsize=fontsize, fontweight='bold')

# Save the figure to a file in the 'Figures' folder (create it if inexistant)
if not os.path.exists(directory+'/Figures/'):
   os.makedirs(directory+'/Figures/')

fig.savefig(directory+'/Figures/fig1_m'+str(monkey+1)+'_'+str(orientation)+pop+'.eps')
