# cell.py --- 
# 
# Filename: cell.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Jul 24 10:04:47 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 17:17:50 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 225
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# This is an extension of Cell class - to add some utility 
# functions for debugging. All cell types should derive from this.
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
import sys
from collections import defaultdict
import numpy as np
# from enthought.mayavi import mlab
import moose
import config
import pymoose

from nachans import *
from kchans import *
from cachans import *
from archan import *
from capool import *
from compartment import MyCompartment

def init_channel_lib():
    """Initialize the prototype channels in library"""
    if not config.channel_lib:
        config.LOGGER.debug('* Generating channel prototypes in /library')
        for channel_name in config.channel_name_list:
            channel_class = eval(channel_name)
            channel = channel_class(channel_name, config.lib)
            config.channel_lib[channel_name] = channel
            config.LOGGER.debug( '* Created %s' % (channel.path))
        config.channel_lib['SpikeGen'] = moose.SpikeGen('spike', config.lib)
    return config.channel_lib

def nameindex(comp):
    """Utility function to sort by index in the compartment name"""
    if comp is None:
        return -1
    pos = comp.name.rfind('_')
    if pos >= 0:
        index = int(comp.name[pos+1:])
        return index
    else:
        return -1

def get_comp(cell, index):
    """Return a wrapper over compartment specified by index. None if
    no such compartment exists."""
    if index <= 0: return None
    path = cell.path + '/comp_' + str(index)
    # print 'get_comp', path
    if config.context.exists(path):
        return MyCompartment(path)
    else:
        raise Exception('Cell: %s , index: %d - no such compartment.' % (cell.path, index))


class TraubCell(moose.Cell):
    channel_lib = init_channel_lib()

    def __init__(self, *args):
        # print 'TraubCell.__init__:', args
        moose.Cell.__init__(self, *args)
        self.method = config.solver # To override hsolve and use ee
        # print 'Cell.__init__ done'


    # Dynamic access to a compartment by index.  It mimics a python
    # list 'comp' via underlying function call to get_comp(cell,
    # index)
    comp = moose.listproperty(get_comp)

    @property
    def soma(self):
        return get_comp(self, 1)

    def pfile_name(self):
        """Each cell type subclass should implement this"""
        raise NotImplementedError, "function pfile_name not implemented"

    @classmethod
    def read_proto(cls, filename, cellname, level_dict=None, depth_dict=None, params=None):
        """Read a prototype cell from .p file into library.  

        Each cell type class should initialize its prototype with a
        call to this function. with something like this within the
        class declaration:

        prototype = TraubCell.read_proto("MyCellType.p", "MyClassName")

        filename -- path(relative/absolute) of the cell prototype file.

        cellname -- path of the cell to be Created

        params -- if specified, channels in /library are adjusted with
        the parameters specified in this (via a call to
        adjust_chanlib).

        """
        config.LOGGER.debug('Reading proto:%s' % (filename))
        if params is not None:
            TraubCell.adjust_chanlib(params)
        ret = None
        cellpath = config.lib.path + '/' + cellname
        if not config.context.exists(cellpath):
            config.LOGGER.debug(__name__ + ' reading cell: ' + cellpath)
            for handler in config.LOGGER.handlers:
                handler.flush()
            config.context.readCell(filename, cellpath)
        else:
            config.LOGGER.debug(__name__ + ' cell exists: ' + cellpath)
        ret = moose.Cell(cellpath)
        # TraubCell.generate_morphology(ret)
        if (depth_dict is not None) and (level_dict is not None):
            for level, comp_nos in level_dict.items():
                try:
                    depth = depth_dict[level]
                    for comp_no in comp_nos:
                        comp = get_comp(ret, comp_no)
                        comp.z = depth
                except KeyError:
                       print 'No depth info for level %s' % (level)

        config.LOGGER.debug('Returning cell %s' % (ret.path))
        for handler in config.LOGGER.handlers:
            handler.flush()
        return ret

    @classmethod
    def adjust_chanlib(cls, chan_params):
        """Set the properties of prototype channels in /library to fit
        the channel properties of this cell type.

        chan_params -- dict containing the channel parameters. The
        following string keys should be there with float values:

        ENa -- Na channle reversal potential
        EK -- K+ channel reversal potential
        EAR -- AR channel reversal potential
        ECa -- Ca+2 channel reversal potential
        TauCa -- CaPool decay time constant 
        X_AR -- AR channel's initial X value.
        """
        config.LOGGER.debug('Adjusting channel properties.')
        for key, channel in init_channel_lib().items():
            if isinstance(channel, KChannel):
                channel.Ek = chan_params['EK']
            elif isinstance(channel, NaChannel):
                channel.Ek = chan_params['ENa']
            elif isinstance(channel, CaChannel):
                channel.Ek = chan_params['ECa']
            elif isinstance(channel, AR):
                channel.Ek = chan_params['EAR']
                try:
                    channel.X = chan_params['X_AR']
                except KeyError:
                    channel.X = 0.25
            elif isinstance(channel, CaPool):
                channel.tau = chan_params['TauCa']

    @classmethod
    def readlevels(cls, filename):
        """Read the mapping between levels and compartment numbers and
        return a defaultdict with level no. as key and set of
        compartments in it as value.

        The file filename should have two columns:

        comp_no level_no

        """
        ret = defaultdict(set)
        with(open(filename, 'r')) as level_file:
            for line in level_file:
                tokens = line.split()
                if not tokens:
                    continue
                if len(tokens) != 2:
                    print filename, ' - Tokens: ', tokens, len(tokens)
                    sys.exit(0)
                ret[int(tokens[1])].add(int(tokens[0]))
        return ret



    def _ca_tau(self):
        raise NotImplementedError("You must set tau for [Ca2+] decay in the method _ca_tau() in subclass.")

    def _setup_passive(self):
        raise NotImplementedError("You must define _setup_passive to set the passive membrane properties and other post-readcell tweakings.")

    def _setup_channels(self):
        raise NotImplementedError("You must define setup_channels to set the channel reversal potential and other post-readcell tweakings.")

    def _topology(self):
        raise NotImplementedError("You must define cell topology in the method _topology() in subclass.")

    def has_cycle(self, comp=None):
        if comp is None:
            comp = self.soma
        comp._visited = True
        ret = False
        for item in comp.raxial_list:
            if hasattr(item, '_visited') and item._visited:
                config.LOGGER.warning('Cycle between: %s and %s.' % (comp.path, item.path))
                return True
            ret = ret or has_cycle(item)
        return ret


    @classmethod
    def generate_morphology(cls, cell, iterations=50):
        """Automatically generate morphology information for spatial
        layout. 

        An implementation of Fruchterman Reingold algorithm in 3D."""
        nodes = defaultdict(dict)
        for comp in cell.childList:
            if moose.Neutral(comp).className == 'Compartment':
                config.LOGGER.debug('Appending %s' % (comp))
                nodes[comp]['pos'] = 0.0
                nodes[comp]['disp'] = 0.0
        # populate the edge set
        edges = set()        
        for comp in nodes.keys():
            nid_list = moose.Neutral(comp).neighbours('raxial')
            for neighbour in nid_list:
                config.LOGGER.debug('Adding (%s, %s)' % (comp, neighbour))
                edges.add((comp, neighbour))

        # Generate random initial positions for all the compartments
        init_pos = np.ones((len(nodes), 3)) * 0.5 - np.random.rand(len(nodes), 3) 
        width = 1.0
        depth = 1.0
        height = 1.0
        ii = 0
        for key, value in nodes.items():
            value['pos'] =init_pos[ii]
            ii += 1
        volume = width * height * depth
        k = np.power(volume / len(nodes), 1.0/3)
        t = 0.1
        dt = t / iterations
        for ii in range(iterations):   
            print 'Iteration', ii
            # calculate repulsive forces         
            for comp, data in nodes.items():
                data['disp'] = np.zeros(3)
                for other, o_data in nodes.items():
                    if comp != other:
                        delta = data['pos'] - o_data['pos'] 
                        distance = np.linalg.norm(delta)
                        if distance < 1e-2:
                            distance = 1e-2
                        data['disp'] += delta * k * k / distance ** 2
                        print comp, other, delta, data['disp']
            for edge in edges: # calculate attractive forces
                delta = nodes[edge[0]]['pos'] - nodes[edge[1]]['pos']
                distance = np.linalg.norm(delta)
                if distance < 1e-2:
                    distance = 1e-2
                nodes[edge[0]]['disp'] -= delta * distance / k
                nodes[edge[1]]['disp'] += delta * distance / k
                print edge[0], edge[1], delta, nodes[edge[0]]['disp'], nodes[edge[1]]['disp']

            for key, data in nodes.items():
                data['pos'] += data['disp']/np.linalg.norm(data['disp']) * min(np.linalg.norm(data['disp']), t)
                data['pos'][0] = min(width/2, max(-width/2, data['pos'][0]))
                data['pos'][1] = min(height/2, max(-height/2, data['pos'][1]))
                data['pos'][2] = min(depth/2, max(-depth/2, data['pos'][2]))
            t -= dt
        pos = []
        for key, data in nodes.items():
            print key, data['pos']
            pos.append(data['pos'])
        pos = np.array(pos)
        points = mlab.points3d(pos[:,0], pos[:, 1], pos[:, 2])
        mlab.show()
        raise Exception('Stop here for testing')

# 
# cell.py ends here
