"""
Small script to demonstrate how to connect the TryConn and Rcv objects in
order to achieve message passing between custom objects.
"""

import moose

# Create the objects - using default values now
s = moose.Sender('/t')
r = moose.Receiver('/r')

# Set up the connection
conn = moose.connect(s, 'XOut', r, 'handleX')

# Create a tick and set Sender's process function to occur on every tick
moose.setClock(0, 1)
moose.useClock(0, s.path, 'process')

# Reinit is not really doing anything other than showing you the value of X
# once before starting
moose.reinit()
# This will run the simulation for 5 seconds => 5 time steps.
moose.start(5)

# Expected output is:
# 10
# 20
# 40
# 80
# 160
# 320
