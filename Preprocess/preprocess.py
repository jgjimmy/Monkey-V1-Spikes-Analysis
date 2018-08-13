"""

Selects only neurons with SNR above "snr_min" and with firing rate higher than
"rate_min" spikes/s for at least one orientation. Randomizes their order and
saves their indices in a pickle file.

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

# Total number of neurons for every monkey
total_neurons = [106,88,112]

# Total time of the experiment (s)
total_time = 1.28

# Minimal SNR
snr_min = 2.75

# Minimal firing rate (spikes/s)
rate_min = 2

# Binary rejection array for every monkey (If the value for a neuron is 1,
# the neuron is rejected from the analysis)
rej_m1 = numpy.zeros(total_neurons[0])
rej_m2 = numpy.zeros(total_neurons[1])
rej_m3 = numpy.zeros(total_neurons[2])

# SNR
# Import SNR data
f = open('gratings_snr', 'rb')
SNR = pickle.load(f)
f.close()

# Reject neurons with SNR too low
rej_m1[numpy.where(SNR[0]<snr_min)[0]] = 1
rej_m2[numpy.where(SNR[1]<snr_min)[0]] = 1
rej_m3[numpy.where(SNR[2]<snr_min)[0]] = 1


# Spiking rate
# Import spiking data
f = open('gratings_events', 'rb')
events = pickle.load(f) # Time of spikes:[monkey][Neuron, Orientation, Trial]
f.close()
# Number of trials
R = numpy.size(events[0], 2)

# Counter for the total number of spikes for all trials for every neuron
count_m1 = numpy.zeros((12,total_neurons[0]))
count_m2 = numpy.zeros((12,total_neurons[1]))
count_m3 = numpy.zeros((12,total_neurons[2]))

for monkey in range(3):
    n_max = total_neurons[monkey]
    for k in range(12):
        for n in range(n_max):
            for r in range(R):
                if monkey == 0:
                    count_m1[k,n] += len(events[0][n,k,r])
                elif monkey == 1:
                    count_m2[k,n] += len(events[1][n,k,r])
                else:
                    count_m3[k,n] += len(events[2][n,k,r])

# Compare the spiking rate to the treshold
for n in range(total_neurons[0]):
    if numpy.max(count_m1[:,n]) < rate_min*total_time*R:
        rej_m1[n] = 1

for n in range(total_neurons[1]):
    if numpy.max(count_m2[:,n]) < rate_min*total_time*R:
        rej_m2[n] = 1

for n in range(total_neurons[2]):
    if numpy.max(count_m3[:,n]) < rate_min*total_time*R:
        rej_m3[n] = 1

# List of neurons NOT rejected
m1 = numpy.where(rej_m1 == 0)[0]
m2 = numpy.where(rej_m2 == 0)[0]
m3 = numpy.where(rej_m3 == 0)[0]

# Shuffle the neuron indices for random selection
numpy.random.seed(0)
m1_rand = numpy.random.choice(m1, len(m1), replace=False)
m2_rand = numpy.random.choice(m2, len(m2), replace=False)
m3_rand = numpy.random.choice(m3, len(m3), replace=False)

neurons = list()
neurons.extend((m1_rand, m2_rand, m3_rand))

# Save the neuron indices to a file
f = open('neurons', 'wb')
pickle.dump(neurons, f)
f.close()
