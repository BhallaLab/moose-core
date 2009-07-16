# plot.py --- 
# 
# Filename: plot.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sun Mar  1 11:32:47 2009 (+0530)
# Version: 
# Last-Updated: Mon Mar  2 16:35:25 2009 (+0530)
#           By: subhasis ray
#     Update #: 117
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:
from pylab import *

def check_PID(simulator_name):
    """Explicitly calculate the output of the PID from its input records and plot it."""
    gen_cmd = load(simulator_name + '_vclamp.plot')
    gen_sns = load(simulator_name + '_Vm.plot')
    gen_out = load(simulator_name + '_inject.plot')
    gen_e = load(simulator_name + '_error.plot')
    gen_ei = load(simulator_name + '_eintegral.plot')
    gen_ed = load(simulator_name + '_ederiv.plot')
	
    tau_i = 20e-6
    tau_d = 5e-6
    gain = 1e-6
    my_out = zeros(len(gen_cmd))
    my_e =  zeros(len(gen_cmd))
    my_eint =  zeros(len(gen_cmd))
    my_ederiv =  zeros(len(gen_cmd))
    e_prev = 0
    simdt = 1e-6
    e_int = 0.0
    for ii in range(len(gen_cmd)):
        my_e[ii] = gen_cmd[ii] - gen_sns[ii]
        e_int = e_int + 0.5 * (e_prev + my_e[ii]) * simdt
        my_eint[ii] = e_int
        my_ederiv[ii] = (my_e[ii] - e_prev) / simdt
        my_out[ii] = gain * (my_e[ii] + my_eint[ii] / tau_i + my_ederiv[ii] * tau_d)
        e_prev = my_e[ii]

        subplot(2,2,1)
        plot(gen_e-my_e, 'x', label='e')
        # plot(gen_e, label='e')
        # plot(my_e, label='my')
        legend()
        subplot(2,2,2)
        plot(gen_ei - my_eint, 'x', label='e_i')
        # plot(gen_ei, label='e_i')
        # plot(my_eint, label='my')
        legend()
        subplot(2,2,3)
        plot(gen_ed - my_ederiv, 'x', label='e_d')
        # plot(gen_ed, label='e_d')
        # plot(my_ederiv, label='my')
        legend()
        subplot(2,2,4)
        plot(gen_out - my_out, 'x', label='o')
        # plot(gen_out, label='o')
        # plot(my_out, label='my')
        legend()
        show()

def plot_all(sim_name):
    """Create all plots for the simulator sim_name (either genesis or moose)"""
    vm = load(sim_name + '_Vm.plot')
    cmd = load(sim_name + '_vclamp.plot')
    inject = load(sim_name + '_inject.plot')
    gk = load(sim_name + '_gk.plot')
    gna = load(sim_name + '_gna.plot')
    ik = load(sim_name + '_ik.plot')
    ina = load(sim_name + '_ina.plot')
    subplot(2,2,1, title='Vm')
    plot(cmd, 'r-', label='command')
    plot(vm, 'b-', label='Vm')
    legend()
    subplot(2,2,2, title='conductance')
    plot(gk, 'r-', label='gK')
    plot(gna, 'b-', label='gNa')
    legend()
    subplot(2,2,3, title='inject')
    plot(inject)
    subplot(2,2,4, title='channel current')
    plot(ik, 'r-', label='iK')
    plot(ina, 'b-', label='iNa')
    legend()
    show()

def check_channels(sim_name):
    ina = load(sim_name + '_ina.plot')
    ik = load(sim_name + '_ik.plot')
    gna = load(sim_name + '_gna.plot')
    gk = load(sim_name + '_gk.plot')
    vm = load(sim_name + '_Vm.plot')
    gbar_Na = 9.4248e-4
    gbar_K = 2.8274e-4
    e_Na = 0.045
    e_K = - 0.082
    plot(gna * (e_Na - vm),'b+', label='INa1')
    plot(ina, 'y-', label='INa')
    plot(gk * (e_K - vm),'rx',  label='IK1')
    plot(ik, 'g-', label='IK')
    legend()
    show()


if __name__ == "__main__":
    plot_all('genesis')
#    check_channels('genesis')

# 
# plot.py ends here
