# trbcell.py --- 
# 
# Filename: trbcell.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Mar  9 23:17:17 2012 (+0530)
# Version: 
# Last-Updated: Fri May  4 14:47:57 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 205
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
# 2012-03-09 23:17:24 (+0530) Subha started porting the cell.py file
# from old moose to dh_branch.
# 

# Code:

import moose
import config
from config import logger

_channels = {}

def get_chanlib():
    """Return a dict of channel name, channel prototype pairs. If the
    channel prototypes have not been initialized, this functions
    initializes the same."""
    global _channels
    if _channels:
        return _channels
    for chname in config.channel_names:
        chclass = eval(chname)
        ch = chclass('%s/%s' % (config.libpath, chname))
        _channels[chname] = ch
    return _channels
        
def adjust_chanlib(cdict):
    """Update the revarsal potentials for channels. Set the initial X
    value for AR channel. Set the tau for Ca pool."""
    for ch in get_chanlib().values():
        if isinstance(ch, KChannel):
            ch.Ek = cdict['EK']
        elif isinstance(ch, NaChannel):
            ch.Ek = cdict['ENa']
        elif isinstance(ch, CaChannel):
            ch.Ek = cdict['ECa']
        elif isinstance(ch, AR):
            ch.Ek = cdict['EAR']
            ch.X = cdict['X_AR']        
        elif isinstance(ch, CaPool):
            ch.tau = cdict['TauCa']            
        
def read_prototype(cls, cdict):
    """Read the cell prototype file for the specified class. The
    channel properties are updated using values in cdict."""
    filename = 'proto/%s.p' % cls.__name__
    logger.debug('Reading prototype file %s' % (filename))
    adjust_chanlib(cdict)
    cellpath = '%s/%s' % (config.libpath, cls.__name__)
    if moose.exists(cellpath):
        return moose.element(cellpath)
    for handler in logger.handlers:
        handlers.flush()
    proto = moose.loadModel(filename, cellpath)
    # If prototype files do not have absolute compartment positions,
    # set the compartment postions to origin. This will avoid
    # incorrect assignemnt of position when the x/y/z values in
    # prototype file is just to for setting the compartment length.
    if not config.has_comp_pos:
        for comp in moose.wildCardFind('%s/##[TYPE=Compartment]' % (proto.path)):
            comp.x = 0.0
            comp.y = 0.0
            comp.z = 0.0        
    return proto

def assign_depths(cell, depth, level):
    """Assign depths to the compartments in the cell. The original
    model assigns sets of compartments to particular levels and a
    depth is specified for each level. This should not be required if
    we have the z value in prototype file.

    cell : (prototype) cell instance

    depth : dict mapping level no. to physical depth

    level : dict mapping level no. to compartment nos. belonging to
    that level.
    """
    if not depth:
        return
    for level, complist in level.items():
        z = depth[level]
        for comp_number in complist:
            comp = moose.element('%s/comp_%d' % (cell.path, comp_number))
            comp.z = z

class CellMeta(type):
    def __new__(cls, name, bases, cdict):
        proto = read_prototype(cls, cdict)
        assign_depths(proto, cdict['depth'], cdict['level'])
        cdict['proto'] = proto
        return type.__new__(cls, name, bases, cdict)

    
class CellBase(moose.Cell):
    __metaclass__ = CellMeta
    def __init__(self, *args):
        moose.Cell.__init__(self, *args)
        self.method = config.neurosolver
        
    def comp(self, number)
        return moose.element('%s/comp_%d' % (self.path, number))

    @property
    def get_soma(self):
        return self.comp(1)
        
                             
# 
# trbcell.py ends here
