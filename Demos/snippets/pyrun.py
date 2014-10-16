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

"""You can use the PyRun class to run Python statements from MOOSE at runtime."""
import moose

def run_sequence():
    """In this example we demonstrate the use of PyRun objects to execute
    Python statements from MOOSE. Here is a couple of fun things to
    indicate the power of MOOSE-Python integration.

    First we create a PyRun object called `Hello`. In its `initString`
    we put in Python statements that prints the element's string
    representation using pymoose-API. When ``moose.reinit()`` is
    called, this causes MOOSE to execute these Python statements which
    include Python calling a MOOSE function
    (Python->MOOSE->Python->MOOSE) - isn't that cool!

    We also initialize a counter called `hello_count` to 0.

    The statements in initString gets executed once, when we call
    ``moose.reinit()``.

    In the `runString` we put a couple of print statements to indicate
    the name fof the object which is running and the current
    count. Then we increase the count directly.
    
    When we call ``moose.start()``, the `runString` gets executed at
    each time step.

    The other PyRun object we create, is `/World`. In its `initString`
    apart from ordinary print statements and initialization, we define
    a Python function called ``incr_count``. This silly little
    function just increments the global `world_count` by 1. 

    The `runString` for `World` simply calls this function to
    increment the count and print it.

    We may notice that we assign tick 0 to `Hello` and tick 1 to
    `World`. Looking at the output, you will realize that the
    sequences of the ticks strictly maintain the sequence of
    execution.

    """
    hello_runner = moose.PyRun('/Hello')
    hello_runner.initString = """
print 'Init', moose.element('/Hello')
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
def incr_count():
    global world_count
    world_count += 1
"""
    world_runner.runString = """
print 'Running World'
print 'World count =', world_count
incr_count()
"""
    world_runner.run('from datetime import datetime')
    world_runner.run('print "World: current time:", datetime.now().isoformat()')
    
    moose.useClock(0, world_runner.path, 'process')
    moose.reinit()
    moose.start(0.001)

    
def input_output():
    """The PyRun class can take a double input through `trigger`
    field. Whenever another object sends an input to this field, the
    `runString` is executed.

    The fun part of this is that you can use the input value in your
    python statements in `runString`. This is stored in a local
    variable called `input`. You can rename this by setting `inputVar`
    field.

    Things become even more interesting when you can send out a value
    computed using Python. PyRun objects allow you to define a local
    variable called `output` and whatever value you assign to this,
    will be sent out through the source field `output` on successful
    execution of the `runString`.

    You can rename the output variable by setting `outputVar` field.

    In this example, we send the output of a pulsegen object sending
    out the values 1, 2, 3 during each pulse and compute the square of
    these numbers in Python and set output to this square.

    """
    pass

if __name__ == '__main__':
    run_sequence()



# 
# pyrun.py ends here
