# pyrun.py --- 
# 
# Filename: pyrun.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Oct 15 10:14:15 2014 (+0530)
# Version: 
# Last-Updated: 
#           By: 
#     Update #: 0
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

"""Shows the use of PyRun class to run Python statements"""
import moose

def example():
    hello_runner = moose.PyRun('Hello')
    hello_runner.initString = """
print 'Init Hello'
hello_count = 0
"""
    hello_runner.runString = """
print 'Running Hello'
print 'Hello count =', hello_count
hello_count += 1
"""
    hello_runner.run('from datetime import datetime')
    hello_runner.run('print "Hello: current time:", datetime.now().isoformat()')
    moose.useClock(0, hello_runner.path, 'process')
    world_runner = moose.PyRun('World')
    world_runner.initString = """
print 'Init World'
world_count = 0
"""
    world_runner.runString = """
print 'Running World'
print 'World count =', world_count
world_count += 1
"""
    world_runner.run('from datetime import datetime')
    world_runner.run('print "World: current time:", datetime.now().isoformat()')
    
    moose.useClock(0, world_runner.path, 'process')
    moose.reinit()
    moose.start(0.1)


if __name__ == '__main__':
    example()



# 
# pyrun.py ends here
