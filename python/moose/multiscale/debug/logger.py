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

logger = logging.getLogger('multiscale')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('multiscale.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('[%(levelname)s] %(filename)s:%(lineno)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)
