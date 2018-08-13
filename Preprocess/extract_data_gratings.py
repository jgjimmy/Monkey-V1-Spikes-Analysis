"""

Extracts the data from MATLAB files to pickle Python files.

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

import scipy.io
import pickle

# Load the data from the Matlab files
mat1 = scipy.io.loadmat('data_monkey1_gratings.mat')
mat2 = scipy.io.loadmat('data_monkey2_gratings.mat')
mat3 = scipy.io.loadmat('data_monkey3_gratings.mat')


# Events file (Spike Timings)
events = list()
events.append(mat1['data']['EVENTS'].item(0))
events.append(mat2['data']['EVENTS'].item(0))
events.append(mat3['data']['EVENTS'].item(0))
f = open('gratings_events', 'wb')
pickle.dump(events ,f)
f.close()

# SNR file
snr = list()
snr.append(mat1['data']['SNR'].item(0))
snr.append(mat2['data']['SNR'].item(0))
snr.append(mat3['data']['SNR'].item(0))
f = open('gratings_snr', 'wb')
pickle.dump(snr ,f)
f.close()


