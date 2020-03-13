# -*- coding: utf-8 -*-

# WARNING: This is for debuging purpose and should not be used in production.
# Quickly load neuroml2 into MOOSE.


__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2019-, Dilawar Singh"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"

import moose

import logging
logger_ = logging.getLogger('moose.nml2')

def _addStreamer():
    logger_.warning('TODO. Add streamer.')


def main(**kwargs):
    logger_.info("Reading file %s" % kwargs['nml2file'])
    moose.mooseReadNML2(kwargs['nml2file'])
    _addStreamer()
    moose.reinit()
    simtime = kwargs['runtime']
    moose.start(simtime)
    logger_.info("All done")

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Load neuroml2 model into moose.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('nml2file', help = 'Neuroml2 file.'
            , metavar="<NEUROML2 FILE>"
           )
    parser.add_argument('--output', '-o'
            , required = False
            , help = 'Output file'
           )
    parser.add_argument('--runtime', '-T'
             , required = False, default = 1.0, type=float
             , help = 'Simulation time (sec)'
            )
    
    parser.add_argument('--debug', '-d'
            , required = False
            , default = 0
            , type = int
            , help = 'Enable debug mode. Default 0, debug level'
           )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main(**vars(args))
