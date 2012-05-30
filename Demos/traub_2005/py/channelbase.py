# trbchan.py --- 
# 
# Filename: trbchan.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri May  4 14:55:52 2012 (+0530)
# Version: 
# Last-Updated: Thu May 31 00:16:02 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 282
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Base class for channels in Traub model.
# 
# 

# Change log:
# 
# 2012-05-04 14:55:56 (+0530) subha started porting code from
# channel.py in old moose version to dh_branch.
# 

# Code:

from warnings import warn
import numpy as np
import moose
import config

vmin = -120e-3
vmax = 40e-3
vdivs = 640
v_array = np.linspace(vmin, vmax, vdivs+1)

ca_min = 0.0
ca_max = 1000.0
ca_divs = 1000
ca_conc = np.linspace(ca_min, ca_max, ca_divs + 1)

prototypes = {}

def setup_gate_tables(gate, param_dict, bases):
    suffix = None
    if gate.name == 'gateX':
        suffix = 'x'
    elif gate.name == 'gateY':
        suffix = 'y'
    elif gate.name == 'gateZ':
        suffix = 'z'
    else:
        raise NameError('Gate in a channel must have names in [`gateX`, `gateY`, `gateZ`]')
    if suffix in ['x', 'y']:
        gate.min = vmin
        gate.max = vmax
        gate.divs = vdivs
        gate.useInterpolation = True
    else:
        gate.min = ca_min
        gate.max = ca_max
        gate.divs = ca_divs
        gate.useInterpolation = False
    keys = ['%s_%s' % (key, suffix) for key in ['tau', 'inf', 'alpha', 'beta', 'tableA', 'tableB']]
    if keys[0] in param_dict:
        gate.tableA = param_dict[keys[1]] / param_dict[keys[0]]
        gate.tableB = 1 / param_dict[keys[0]]
        return True
    elif keys[2] in param_dict:
        gate.tableA = param_dict[keys[2]]
        gate.tableB = param_dict[keys[2]] + param_dict[keys[3]]
        return True
    elif keys[4] in param_dict:
        gate.tableA = param_dict[keys[4]]
        gate.tableB = param_dict[keys[5]]
        return True
    else:
        for base in bases:
            new_bases = base.mro()
            new_param_dict = base.__dict__
            if new_bases:
                new_bases = new_bases[1:]
            if setup_gate_tables(gate, new_param_dict, new_bases):
                return True
    return False
            
def get_class_field(name, cdict, bases, fieldname, default=None):
    if fieldname in cdict:
        print name, fieldname, cdict[fieldname]
        return cdict[fieldname]
    else:
        for base in bases:
            if hasattr(base, fieldname):
                return getattr(base, fieldname)
    warn('field %s not in the hierarchy of %s class. Returning default value.' % (fieldname, name))
    return default
            
class ChannelMeta(type):
    def __new__(cls, name, bases, cdict):     
        global prototypes
        # classes that set absract=True will be
        # abstract classes. Others will have the prototype insatntiated.
        if  'abstract' in cdict and cdict['abstract'] == True:
            return type.__new__(cls, name, bases, cdict)
        proto = moose.HHChannel('%s/%s' % (config.modelSettings.libpath, name))
        xpower = get_class_field(name, cdict, bases, 'Xpower', default=0.0)
        if xpower > 0:
            proto.Xpower = xpower
            gate = moose.HHGate('%s/gateX' % (proto.path))
            setup_gate_tables(gate, cdict, bases)
            cdict['xGate'] = gate
        ypower = get_class_field(name, cdict, bases, 'Ypower', default=0.0)
        if ypower > 0:
            proto.Ypower = ypower
            gate = moose.HHGate('%s/gateY' % (proto.path))
            setup_gate_tables(gate, cdict, bases)
            cdict['yGate'] = gate
        zpower = get_class_field(name, cdict, bases, 'Zpower', default=0.0)
        if zpower > 0:
            proto.Zpower = zpower
            gate = moose.HHGate('%s/gateZ' % (proto.path))
            setup_gate_tables(gate, cdict, bases)
            cdict['zGate'] = gate
            ca_msg_field = moose.Mstring('%s/addmsg1' % (proto.path))
            ca_msg_field.value = '../CaPool	concOut	. concen'
            proto.instant = get_class_field(name, cdict, bases, 'instant', default=0)
            proto.useConcentration = True
        proto.Ek = get_class_field(name, cdict, bases, 'Ek', default=0.0)
        X = get_class_field(name, cdict, bases, 'X', -1)
        if X > 0:
            proto.X = X
        Y = get_class_field(name, cdict, bases, 'Y', -1)
        if Y > 0:
            proto.Y = Y
        Z = get_class_field(name, cdict, bases, 'Z', -1)
        if Z > 0:
            proto.Z = Z
        mstring_field = get_class_field(name, cdict, bases, 'mstring')
        if mstring_field is not None:
            mstring = moose.MString('%s/%s' % (proto.path, mstring_field[0]))
            mstring.value = mstring_field[1]
        cdict['prototype'] = proto
        prototypes[name] = proto
        print 'Created prototype:', proto.path
        return type.__new__(cls, name, bases, cdict)


class ChannelBase(moose.HHChannel):
    abstract = True
    __metaclass__ = ChannelMeta
    def __init__(self, path, xpower=1, ypower=0, Ek=0.0):
        moose.HHChannel.__init__(self, path)


# 
# trbchan.py ends here
