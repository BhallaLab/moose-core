# ninemlio.py --- 
# 
# Filename: ninemlio.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Tue May 31 11:24:53 2011 (+0530)
# Version: 
# Last-Updated: Wed Jun  1 17:52:04 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 178
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 2011-06-01 16:51:37 (+0530) TODO: It may be useful to make this a
# class: NineMLModel and let that store two way mapping:
# moose-object to nineml object and the reverse. All the methods then
# become part of that class.
# 
# 

# Change log:
# 
# 2011-05-31 11:25:07 (+0530) Initial version
#
# 2011-06-01 16:51:14 (+0530) Added translation for IntFire

# Code:

import nineml.user_layer as ninemlul
import moose

class NineMLModel(object):
    to_SI = {'mV': 1e-3,
             'ms': 1e-3,
             'mS': 1e-3
             }
    def __init__(self):
        self._moose_to_nineml = {}
        self._nineml_to_moose = {}
        self._nineml = None
        self._moose_root = None
        
    def _get_SI_value(self, component, param_name):
        """Return the SI value of the given parameter."""
        print '_get_SI_value', component, param_name
        value = 0.0
        param = component.parameters[param_name]
        print 'value', param.value, NineMLModel.to_SI[param.unit]
        if isinstance(param.value, float):
            value = param.value * NineMLModel.to_SI[param.unit]        
        return value
    
    def _create_IntFire(self, component):
        if not isinstance(component, ninemlul.SpikingNodeType):        
            raise TypeError('Component must be a SpikingNodeType object')
        node = moose.IntFire(component.name.replace(' ', '_')) # MOOSE does not allow spaces in element name
        node.thresh = self._get_SI_value(component, 'threshold')
        node.tau = self._get_SI_value(component, 'membraneTimeConstant')
        node.refractoryPeriod = self._get_SI_value(component, 'refractoryTime')
        # TODO: IntFire implementation of threadMsg branch does not have
        # resting membrane potential and reset potential available for the
        # user.
        #
        # If they are supported, we'll use the parameters
        #
        # 'restingPotential' and 'resetPotential'

    def readModel(self, filename, target):
        print type(target)
        if isinstance(target, str):
            if not moose.exists(target):
                target = moose.Neutral(target)
        elif isinstance(target, moose.NeutralArray):
            target = target._id
        elif isinstance(target, moose.Neutral):
            target = target._oid.getId()
        elif not (isinstance(target, moose.Id) or isinstance(target, moose.ObjId)):
            raise TypeError('Target must be a string or an Id or an ObjId or a moose object.')
        current = moose.getCwe()
        self._moose_root = target
        with open(filename) as model_file:
            model = ninemlul.parse(model_file)
            model.check()
            # These are preliminery investigations, will be superseeded by
            # actual object creation in moose.
            spiking_nodes = {}  
            projections = {}
            synapses = {}            
            for name, component in model.components.items():
                if isinstance(component, ninemlul.SpikingNodeType):
                    spiking_nodes[name] = component
            print spiking_nodes.keys()
            moose.setCwe(target)
            # We instantiate the model here. TODO: components are
            # actually prototypes, we need to make copies of them when
            # creating Groups
            for name, node in spiking_nodes.items():
                if node.definition.url == 'http://svn.incf.org/svn/nineml/trunk/catalog/neurons/IaF_tau.xml':
                    moose_object = self._create_IntFire(node)
                    self._moose_to_nineml[moose_object] = node
                    self._nineml_to_moose[node] = moose_object
            moose.setCwe(current)


import unittest

class TestNineMLModel(unittest.TestCase):
    def setUp(self):
        self.model_object = NineMLModel()
        self.model_object.readModel('simple_example.xml', '/')
        
    def testIntFire(self):
        inhibitory_neuron_path = self.model_object._moose_root.path + '/Inhibitory_neuron_type'
        self.assertTrue(moose.exists(inhibitory_neuron_path))
        inhibitory = moose.IntFire(inhibitory_neuron_path)
        self.assertAlmostEqual(inhibitory.tau, 20e-3)
        self.assertAlmostEqual(inhibitory.refractoryPeriod, 5e-3)
        self.assertAlmostEqual(inhibitory.thresh, -50e-3)

if __name__ == '__main__':
    unittest.main()
    print '9ml test main finished.'
# 
# ninemlio.py ends here
