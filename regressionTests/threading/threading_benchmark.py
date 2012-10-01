# threading_benchmark.py --- 
# 
# Filename: threading_benchmark.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Oct  1 17:31:36 2012 (+0530)
# Version: 
# Last-Updated: Mon Oct  1 17:57:48 2012 (+0530)
#           By: subha
#     Update #: 25
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

import os
import subprocess
from datetime import datetime
max_threads = 8

def test_singlemessages():
    nthreads = 1
    times = []
    while nthreads <= max_threads:
	start = datetime.now()
	os.environ['NUMPTHREADS'] = str(nthreads)	
	proc = subprocess.Popen(['python', 'twocomp.py'], 
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
	out, err = proc.communicate()
	end = datetime.now()
	delay = end - start
	times.append(delay)
	print 'finished with %d threads in %g s' % (nthreads, delay.seconds + delay.microseconds*1e-6)
	nthreads += 1
    return times

if __name__ == '__main__':
    times = test_singlemessages()
    for dt in times:
	print dt.seconds + dt.microseconds * 1e-6

# 
# threading_benchmark.py ends here
