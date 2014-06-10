#!/usr/bin/env python

# Filename       : logger.py 
# Created on     : 2013
# Author         : Dilawar Singh
# Email          : dilawars@ncbs.res.in
#
# Description    : Its  a logger 
#
# Logs           :


import logging
import moose
import logging
import datetime
import time
import os

st = time.time()
st = datetime.datetime.fromtimestamp(st).strftime('%Y-%m-%d-%H%M')

logFile = 'logs/moose.log'
if os.path.exists(logFile):
     os.rename(logFile, 'logs/{0}'.format(st))

def logPathsToFille(pat):
    moose_paths = [x.getPath() for x in moose.wildcardFind(pat)]
    moose_paths = "\n".join(moose_paths)
    with open(logFile, "w") as f:
        f.write(moose_paths)

# Here is our logger.
logging.basicConfig(level=logging.DEBUG
       , format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
       , datefmt='%m-%d %H:%M'
       , filename='logs/mumble.log'
       , filemode='w'
       )
#define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

