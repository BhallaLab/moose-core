# compare_genesis.py --- 
# 
# Filename: compare_genesis.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Jun 21 15:31:01 2013 (+0530)
# Version: 
# Last-Updated: Fri Jun 21 15:32:36 2013 (+0530)
#           By: subha
#     Update #: 8
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

moose_soma = loadtxt('symcompartment.txt')
plot(moose_soma[:,0], moose_soma[:,1], 'r-', label='moose-soma')
plot(moose_soma[:,0], moose_soma[:,2], 'g-', label='moose-d1')
plot(moose_soma[:,0], moose_soma[:,3], 'b-', label='moose-d2')

gen_d1 = loadtxt('genesis_d1_Vm.txt')
gen_soma = loadtxt('genesis_soma_Vm.txt')
gen_d2 = loadtxt('genesis_d2_Vm.txt')
plot(gen_soma[:, 0], gen_soma[:, 1], 'm-.', label='gen-soma')
plot(gen_d1[:,0], gen_d1[:,1], 'c-.', label='gen-d1')
plot(gen_d2[:,0], gen_d2[:,1], 'k-.', label='gen-d2')

gen_d1 = loadtxt('genesis_readcell_d1_Vm.txt')
gen_soma = loadtxt('genesis_readcell_soma_Vm.txt')
gen_d2 = loadtxt('genesis_readcell_d2_Vm.txt')
plot(gen_soma[:, 0], gen_soma[:, 1], 'm--', label='gen-readcell-soma')
plot(gen_d1[:,0], gen_d1[:,1], 'c--', label='gen-readcell-d1')
plot(gen_d2[:,0], gen_d2[:,1], 'k--', label='gen-readcell-d2')

legend()
show()

# 
# compare_genesis.py ends here
