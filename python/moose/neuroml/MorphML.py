# This file is part of MOOSE simulator: http://moose.ncbs.res.in.

# MOOSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MOOSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

# Description: class MorphML for loading MorphML from file or xml element into
# MOOSE

# Version 1.0 by Aditya Gilra, NCBS, Bangalore, India, 2011 for serial MOOSE

# Version 1.5 by Niraj Dudani, NCBS, Bangalore, India, 2012, ported to parallel
# MOOSE

# Version 1.6 by Aditya Gilra, NCBS, Bangalore, India, 2012, further changes for
# parallel MOOSE

# Version 1.7 by Aditya Gilra, NCBS, Bangalore, India, 2013, further support for
# NeuroML 1.8.1

"""

NeuroML.py is the preferred interface. Use this only if NeuroML L1,L2,L3 files
are misnamed/scattered.  Instantiate MorphML class, and thence use methods:

- readMorphMLFromFile(...) to load a standalone MorphML from file OR
- readMorphML(...) to load from an xml.etree xml element (could be part of a
larger NeuroML file).

It is assumed that any channels and synapses referred to by above MorphML have
already been loaded under that same name in /library in MOOSE (use ChannelML
loader).

"""


import string
import sys
import math
import inspect

from os import path

from .. import moose
from .. import utils 
from .. import moose_config 

from ChannelML import ChannelML
import utils as  neuroml_utils

from xml.etree import cElementTree as ET

class MorphML():

    def __init__(self,nml_params):
        self.neuroml = 'http://morphml.org/neuroml/schema'
        self.bio = 'http://morphml.org/biophysics/schema'
        self.mml = 'http://morphml.org/morphml/schema'
        self.nml = 'http://morphml.org/networkml/schema'
        self.meta = 'http://morphml.org/metadata/schema'
        self.cellDictBySegmentId = {}
        self.cellsInCable = {}
        self.nml_params = nml_params
        self.model_dir = nml_params['model_dir']
        self.temperature = nml_params['temperature']
        self.libpath = moose_config.libraryPath

        self.curCableId = ''
        self.curSegId = ''
        self.curComp = None
        self.curDiameter = 0.0
        self.curDiaNums = 0

        # All factors here.
        self.CMfactor = 1.0
        self.Cfactor = 1.0
        self.RAfactor = 1.0
        self.RMfactor = 1.0
        self.Rfactor = 1.0
        self.Efactor = 1.0
        self.Gfactor = 1.0
        self.Ifactor = 1.0
        self.Tfactor = 1.0


    def readMorphMLFromFile(self, filename, params={}):
        """
        specify global params as a dict (presently none implemented)
        returns { self.curCellName1 : segDict, ... }
        see readMorphML(...) for segDict 
        """
        utils.dump("DEBUG", "Reading morphology from {}".format(filename))
        tree = ET.parse(filename)
        neuroml_element = tree.getroot()

        cellsDict = {}
        for cell in neuroml_element.findall('.//{'+self.neuroml+'}cell'):
            params['lengthUnits'] = neuroml_element.attrib['lengthUnits']
            cellDict = self.readMorphML(cell, params)
            cellsDict.update(cellDict)
        return cellsDict

    def readMorphML(self, cell, params={}):
        """Read morphology of a cell.

        Returns a dictionary: {self.curCellName:segDict} where segDict = { 
            segid1 : [ segname,(proximalx,proximaly,proximalz),
            (distalx,distaly,distalz),diameter,length,[potential_syn1, ... ] ] 
            , ... 
            }

        segname is "<name>_<segid>" because 

        1) guarantees uniqueness,
        2) later scripts obtain segid from the compartment's name!

        """

        lengthUnits = params.get('lengthUnits', 'micron')
        if lengthUnits in ['micrometer','micron']:
            self.lScale = 1e-6
        else:
            self.lScale = 1.0

        self.curCellName = cell.attrib["name"]

        # creates /library in MOOSE tree; elif present, wraps
        moose.Neutral(self.libpath) 
        utils.dump("STEP"
                , "Loading cell morphology %s into moose library" % self.curCellName
                )

        #using moose Neuron class - in previous version 'Cell' class Chaitanya
        self.curCell = moose.Neuron(
                "{}/{}".format(self.libpath, self.curCellName)
                )
        
        self.cellDictBySegmentId[self.curCellName] = [self.curCell,{}]
        self.cellsInCable[self.curCellName] = [self.curCell,{}]
        self.segDict = {}
        
        # Load morphology and connections between compartments.
        
        # Many neurons exported from NEURON have multiple segments in a section.
        # Combine those segments into one Compartment / section assume segments
        # of a compartment/section are in increasing order and assume all
        # segments of a compartment/section have the same cableId.
        # Function findall() returns elements in document order:

        segments = cell.findall(".//{"+self.mml+"}segment")
        [ self.addSegment(s) for s in segments ]
        
        # load cablegroups into a dictionary
        self.cableGroup = {}

        # Two ways of specifying cablegroups in neuroml 1.x <cablegroup>s with
        # list of <cable>s
        cablegroups = cell.findall(".//{"+self.mml+"}cablegroup")
        [ self.addCableGroup(cg) for cg in cablegroups ]

        ## <cable>s with list of <meta:group>s
        cables = cell.findall(".//{"+self.mml+"}cable")
        [ self.addCable(cable) for cable in cables ]
                ###############################################
        #### load biophysics into the compartments
        biophysics = cell.find(".//{"+self.neuroml+"}biophysics")
        if biophysics is not None:
            self.addBiophysics(cell, biophysics)

        # Load connectivity / synapses into the compartments
        connectivity = cell.find(".//{"+self.neuroml+"}connectivity")
        if connectivity is not None:
            potential_syn_locs = cell.findall(".//{"+self.nml+"}potential_syn_loc")
            [ self.addSynaseLocation(cell, loc) for loc in potential_syn_locs ]

        utils.dump("MORPHML"
                , "Finished loading cell %s in library" % self.curCellName
                )
        return {self.curCellName:self.segDict}

    
    def addSynaseLocation(self, cell, potentialSynLoc):
        """ Add connectivity XML element of NML to moose """
        if 'synapse_direction' in potentialSynLoc.attrib.keys():
            if potentialSynLoc.attrib['synapse_direction'] == 'post':
                self.set_group_compartment_param(cell
                        , potentialSynLoc
                        , 'synapse_type'
                        , potentialSynLoc.attrib['synapse_type']
                        , self.nml
                        , mechName='synapse'
                        )
            if potentialSynLoc.attrib['synapse_direction'] == 'pre':
                self.set_group_compartment_param(cell
                        , potentialSynLoc
                        , 'spikegen_type'
                        , potentialSynLoc.attrib['synapse_type']
                        , self.nml
                        , mechName='spikegen'
                        )
        else:
            utils.dump("INFO"
                    , "No synapse_direction is found in potential_syn_loc "
                    "attributes. "
                    "Available attributes are : %s " % potentialSynLoc.attrib 
                    )


    def addBiophysics(self, cell, biophysics):
        """Add a biophysics """
        # see pg 219 (sec 13.2) of Book of Genesis
        if biophysics.attrib["units"] == 'Physiological Units':
            self.CMfactor = 1e-2    # F/m^2 from microF/cm^2
            self.Cfactor = 1e-6     # F from microF
            self.RAfactor = 1e1     # Ohm*m from KOhm*cm
            self.RMfactor = 1e-1    # Ohm*m^2 from KOhm*cm^2
            self.Rfactor = 1e-3     # Ohm from KOhm
            self.Efactor = 1e-3     # V from mV
            self.Gfactor = 1e1      # S/m^2 from mS/cm^2
            self.Ifactor = 1e-6     # A from microA
            self.Tfactor = 1e-3     # s from ms

        spec_capacitance = cell.find(".//{"+self.bio+"}spec_capacitance")
        for parameter in spec_capacitance.findall(".//{"+self.bio+"}parameter"):
            self.set_group_compartment_param(
                    cell
                    ,  parameter
                    , 'CM'
                    , float(parameter.attrib["value"]) * self.CMfactor
                    , self.bio
                    )

        spec_axial_resitance = cell.find(".//{"+self.bio+"}spec_axial_resistance")
        for parameter in spec_axial_resitance.findall(".//{"+self.bio+"}parameter"):
            self.set_group_compartment_param(
                    cell
                    , parameter
                    , 'RA'
                    , float(parameter.attrib["value"]) * self.RAfactor
                    , self.bio
                    )

        init_memb_potential = cell.find(".//{"+self.bio+"}init_memb_potential")
        for parameter in init_memb_potential.findall(".//{"+self.bio+"}parameter"):
            self.set_group_compartment_param(
                    cell
                    ,  parameter
                    , 'initVm'
                    , float(parameter.attrib["value"]) * self.Efactor
                    , self.bio
                    )

        for mechanism in cell.findall(".//{"+self.bio+"}mechanism"):
            mechName = mechanism.attrib["name"]
            passive = False
            if mechanism.attrib.has_key("passive_conductance"):
                if mechanism.attrib['passive_conductance'].lower() is "true":
                    passive = True

            utils.dump("INFO", "Loading mechanism %s " %  mechName)

            # ONLY creates channel if at least one parameter (like gmax) is
            # specified in the xml Neuroml does not allow you to specify all
            # default values.  However, granule cell example in neuroconstruct
            # has Ca ion pool without a parameter, applying default values to
            # all compartments!
            mech_params = mechanism.findall(".//{"+self.bio+"}parameter")

            # if no params, apply all default values to all compartments
            if len(mech_params) == 0:
                compartments = self.cellsInCable[self.curCellName][1].values()
                for compartment in compartments:
                    self.set_compartment_param(
                            compartment
                            , None
                            , 'default'
                            , mechName
                            )  
            else:
                [ self.addMechanicalParamter(param, mechName, cell, passive) 
                    for param in mech_params 
                    ]
            
        # Connect the Ca pools and channels
        # Am connecting these at the very end so that all channels and pools have been created
        # Note: this function is in moose.utils not moose.neuroml.utils !
        utils.connect_CaConc(
                self.cellsInCable[self.curCellName][1].values()
                , self.temperature+neuroml_utils.ZeroCKelvin
                ) 


    def addMechanicalParamter(self, parameter, mechName, cell, passive):
        """Add mechanical paramter """
        parametername = parameter.attrib['name']
        if passive:
            if parametername in ['gmax']:
                self.set_group_compartment_param(
                        cell
                        ,  parameter
                        , 'RM'
                        , self.RMfactor * 1.0/float(parameter.attrib["value"])
                        , self.bio
                        )

            elif parametername in ['e','erev']:
                self.set_group_compartment_param(
                        cell
                        ,  parameter
                        , 'Em'
                        , self.Efactor * float(parameter.attrib["value"])
                        , self.bio
                        )

            elif parametername in ['inject']:
                self.set_group_compartment_param(
                        cell
                        ,  parameter
                        , 'inject'
                        , self.Ifactor * float(parameter.attrib["value"])
                        , self.bio
                        )
            else:
                utils.dump("TODO"
                        , "Missing support for parameter "
                        "%s in mechanism %s" % (parametername, mechName)
                        )
        else:
            if parametername in ['gmax']:
                gmaxval = float(parameter.attrib.get("value", "0.0"))
                self.set_group_compartment_param(
                        cell
                        , parameter
                        , 'Gbar'
                        , self.Gfactor * gmaxval
                        , self.bio
                        , mechName
                        )
            elif parametername in ['e','erev']:
                self.set_group_compartment_param(
                        cell
                        , parameter
                        , 'Ek'
                        , self.Efactor * float(parameter.attrib["value"])
                        , self.bio
                        , mechName
                        )
            # has to be type Ion Concentration!
            elif parametername in ['depth']: 
                self.set_group_compartment_param(
                        cell
                        , parameter
                        , 'thick'
                        , self.lScale * float(parameter.attrib["value"])
                        , self.bio
                        , mechName
                        )
            elif parametername in ['v_reset']:
                self.set_group_compartment_param(
                        cell
                        , parameter
                        , 'v_reset'
                        , self.Efactor * float(parameter.attrib["value"])
                        , self.bio
                        , mechName
                        )
            elif parametername in ['threshold']:
                self.set_group_compartment_param(
                        cell
                        ,  parameter
                        , 'threshold'
                        , self.Efactor * float(parameter.attrib["value"])
                        , self.bio
                        , mechName
                        )
            else:
                utils.dump("TODO"
                        , "Missing support for parameter "
                        "%s in mechanism %s" % (parametername, mechName)
                        )


    def addCable(self, cable):
        """Add a cable """
        cableId = cable.attrib['id']
        cablegroups = cable.findall(".//{"+self.meta+"}group")
        for cablegroup in cablegroups:
            cableGroup = cablegroup.text
            if cableGroup in self.cableGroup.keys():
                self.cableGroup[cableGroup].append(cableId)
            else:
                self.cableGroup[cableGroup] = [cableId]

    def addCableGroup(self, cablegroup):
        """Add all cable groups in NML"""
        cableGroup = cablegroup.attrib['name']
        self.cableGroup[cableGroup] = []
        for cable in cablegroup.findall(".//{"+self.mml+"}cable"):
            cableId = cable.attrib['id']
            self.cableGroup[cableGroup].append(cableId)        

    def addSegment(self, segment):
        """Add a segment to cell """

        segmentName = segment.attrib['name']

        # Cable is an optional attribute. 
        # WARNING: Here I assume it is always present.
        cableId = segment.attrib['cable']
        segmentid = segment.attrib['id']

        # old cableId still running, hence don't start a new compartment, skip
        # to next segment
        if cableId == self.curCableId:
            self.cellDictBySegmentId[self.curCellName][1][segmentid] = self.curComp
            proximal = segment.find('./{'+self.mml+'}proximal')
            if proximal is not None:
                self.curDiameter += float(proximal.attrib["diameter"]) * self.lScale
                self.curDiaNums += 1
            distal = segment.find('./{'+self.mml+'}distal')
            if distal is not None:
                self.curDiameter += float(distal.attrib["diameter"]) * self.lScale
                self.curDiaNums += 1

        # new cableId starts, hence start a new compartment; also finish
        # previous / last compartment
        else:
            # Create a new compartment the moose "hsolve" method assumes
            # compartments to be asymmetric compartments and symmetrizes them
            # but that is not what we want when translating from Neuron which
            # has only symcompartments -- so be careful!

            # just segmentName is NOT unique - eg: mitral bbmit exported from NEURON
            mCompname = segmentName + '_' + segmentid 
            mComppath = self.curCell.path+'/'+mCompname
            mComp = moose.Compartment(mComppath)
            self.cellDictBySegmentId[self.curCellName][1][segmentid] = mComp

            # cables are grouped and densities set for cablegroups. Hence I need
            # to refer to segment according to which cable they belong to.
            self.cellsInCable[self.curCellName][1][cableId] = mComp 
            self.curCableId = cableId
            self.curSegId = segmentid
            self.curComp = mComp
            self.curDiameter = 0.0
            self.curDiaNums = 0
            if segment.attrib.has_key('parent'):
                # I assume the parent is created before the child so that I can
                # immediately connect the child.
                parentid = segment.attrib['parent'] 
                parent = self.cellDictBySegmentId[self.curCellName][1][parentid]

                # It is always assumed that axial of parent is connected to
                # raxial of moosesegment.

                # THIS IS ALSO IRRESPECTIVE OF fraction_along_parent SPECIFIED
                # IN CABLE!  THUS THERE WILL BE NUMERICAL DIFFERENCES BETWEEN
                # MOOSE/GENESIS and NEURON.  moosesegment sends Ra and Vm to
                # parent, parent sends only Vm actually for symmetric
                # compartment, both parent and moosesegment require each other's
                # Ra/2, but axial and raxial just serve to distinguish ends.
                moose.connect(parent, 'axial', mComp, 'raxial')
            else:
                parent = None

            proximal = segment.find('./{'+self.mml+'}proximal')

            # If proximal tag is not present,
            if proximal is None:         
                # then parent attribute MUST be present in the segment tag!  if
                # proximal is not present, then by default the distal end of the
                # parent is the proximal end of the child
                mComp.x0 = parent.x
                mComp.y0 = parent.y
                mComp.z0 = parent.z
            else:
                mComp.x0 = float(proximal.attrib["x"]) * self.lScale
                mComp.y0 = float(proximal.attrib["y"]) * self.lScale
                mComp.z0 = float(proximal.attrib["z"]) * self.lScale
                self.curDiameter += float(proximal.attrib["diameter"]) * self.lScale
                self.curDiaNums += 1

            distal = segment.find('./{'+self.mml+'}distal')
            if distal is not None:
                self.curDiameter += float(distal.attrib["diameter"]) * self.lScale
                self.curDiaNums += 1


        # Update the end position, diameter and length, and segDict of this
        # comp/cable/section with each segment that is part of this cable
        # (assumes contiguous segments in xml).  This ensures that we don't have
        # to do any 'closing ceremonies', if a new cable is encoutered in next
        # iteration.
        if distal is not None:
            self.curComp.x = float(distal.attrib["x"])*self.lScale
            self.curComp.y = float(distal.attrib["y"])*self.lScale
            self.curComp.z = float(distal.attrib["z"])*self.lScale

        # Set the compartment diameter as the average diameter of all the
        # segments in this section
        self.curComp.diameter = self.curDiameter / float(self.curDiaNums)

        # Set the compartment length
        self.curComp.length = math.sqrt(
                (self.curComp.x - self.curComp.x0)**2 + 
                (self.curComp.y - self.curComp.y0)**2 + 
                (self.curComp.z - self.curComp.z0)**2
                )

        # NeuroML specs say that if (x0,y0,z0)=(x,y,z), then round compartment
        # e.g. soma.  In Moose set length = dia to give same surface area as
        # sphere of dia.
        if self.curComp.length == 0.0:
            self.curComp.length = self.curComp.diameter

        # Set the segDict. The empty list at the end below will get populated
        # with the potential synapses on this segment, in function
        # set_compartment_param(..)
        self.segDict[self.curSegId] = [
                self.curComp.name
                , (self.curComp.x0, self.curComp.y0, self.curComp.z0)
                , (self.curComp.x, self.curComp.y, self.curComp.z)
                , self.curComp.diameter
                , self.curComp.length
                , []
                ]

        if neuroml_utils.neuroml_debug: 
            utils.dump("DEBUG"
                    , 'Set up compartment/section %s '% self.curComp.name
                    )
        
    def set_group_compartment_param(self, cell, parameter, name, value, groupType, mechName=None):
        """
        Find the compartments that belong to the cablegroups refered to
         for this parameter and set_compartment_param.
        """
        for group in parameter.findall(".//{"+groupType+"}group"):
            cableGroup = group.text
            if cableGroup == 'all':
                compartments = self.cellsInCable[self.curCellName][1].values()
                for c in compartments:
                    self.set_compartment_param(c, name, value, mechName)
            else:
                for cableId in self.cableGroup[cableGroup]:
                    compartment = self.cellsInCable[self.curCellName][1][cableId]
                    self.set_compartment_param(compartment,name,value,mechName)

    def set_compartment_param(self, compartment, name, value, mechName):
        """ Set the param for the compartment depending on name and mechName. """
        if name == 'CM':
            cm = value * math.pi * compartment.diameter * compartment.length
            compartment.Cm = cm
        elif name == 'RM':
            rm = value / (math.pi * compartment.diameter * compartment.length)
            compartment.Rm = rm
        elif name == 'RA':
            surfaceArea = math.pi * (compartment.diameter/2.0)**2
            compartment.Ra = value * compartment.length / surfaceArea
        elif name == 'Em':
            compartment.Em = value
        elif name == 'initVm':
            compartment.initVm = value
        elif name == 'inject':
            compartment.inject = value
        elif name in ['v_reset','threshold']:
            if moose.exists(compartment.path+'/IaF_spikegen'):
                # If these exist, they only get wrapped, not created again
                spikegen = moose.SpikeGen(compartment.path+'/IaF_spikegen')
                IaFthresholdfunc = moose.Func(compartment.path+'/IaF_thresholdfunc')
            else:
                # spikegen is on clock tick 2 and func is on tick 3.  If you use
                # moose.utils.resetSim() or moose.utils.assignDefaultTicks(),
                # hence spike is generated first, then voltage is reset
                spikegen = moose.SpikeGen(compartment.path+'/IaF_spikegen')

                # This ensures that spike is generated only on leading edge.
                spikegen.edgeTriggered = 1 
                IaFthresholdfunc = moose.Func(compartment.path+'/IaF_thresholdfunc')
                IaFthresholdfunc.expr = 'x > Vthreshold? Vreset: x'
                moose.connect(compartment, 'VmOut', IaFthresholdfunc, 'xIn')
                moose.connect(IaFthresholdfunc, 'valueOut', compartment, 'setVm')
                moose.connect(compartment, 'VmOut', spikegen, 'Vm')            
            if name == 'v_reset':
                IaFthresholdfunc.var['Vreset'] = value
            else:
                IaFthresholdfunc.var['Vthreshold'] = value
                spikegen.threshold = value
        # synapse being added to the compartment
        elif mechName is 'synapse': 
            # get segment id from compartment name
            segid = (string.split(compartment.name, '_')[-1]) 
            self.segDict[segid][5].append(value)
            return

        # spikegen being added to the compartment
        elif mechName is 'spikegen': 
            # these are potential locations, we do not actually make the spikegens.
            # spikegens for different synapses can have different thresholds,
            # hence include synapse_type in its name
            # value contains name of synapse i.e. synapse_type
            #spikegen = moose.SpikeGen(compartment.path+'/'+value+'_spikegen')
            #moose.connect(compartment,"VmSrc",spikegen,"Vm")
            pass

        # previous were mechanism that don't need a ChannelML definition
        # including integrate_and_fire (I ignore the ChannelML definition) thus
        # integrate_and_fire mechanism default values cannot be used i.e.
        # nothing needed in /library, but below mechanisms need.
        elif mechName is not None:
            # if mechanism is not present in compartment, deep copy from library
            if not moose.exists(compartment.path+'/'+mechName):
                # if channel does not exist in library load it from xml file
                if not moose.exists(self.libpath+"/"+mechName):
                    cmlR = ChannelML(self.nml_params)
                    model_filename = mechName+'.xml'
                    model_path = neuroml_utils.find_first_file(
                            model_filename
                            , self.model_dir
                            )
                    if model_path is not None:
                        cmlR.readChannelMLFromFile(model_path)
                    else:
                        raise IOError(
                            'For mechanism {0}: files {1} not found under {2}.'.format(
                                mechName, model_filename, self.model_dir
                            )
                        )

                neutralObj = moose.Neutral(self.libpath+"/"+mechName)
                if 'CaConc' == neutralObj.className: # Ion concentration pool
                    libcaconc = moose.CaConc(self.libpath+"/"+mechName)
                    ## deep copies the library caconc under the compartment
                    caconc = moose.copy(libcaconc,compartment,mechName)
                    caconc = moose.CaConc(caconc)

                    # CaConc connections are made later using connect_CaConc()
                    # Later, when calling connect_CaConc, B is set for caconc
                    # based on thickness of Ca shell and compartment l and dia
                    # OR based on the Mstring phi under CaConc path.
                    channel = None
                elif 'HHChannel2D' == neutralObj.className : ## HHChannel2D
                    libchannel = moose.HHChannel2D(self.libpath+"/"+mechName)
                    ## deep copies the library channel under the compartment
                    channel = moose.copy(libchannel,compartment,mechName)
                    channel = moose.HHChannel2D(channel)
                    moose.connect(channel,'channel',compartment,'channel')
                elif 'HHChannel' == neutralObj.className : ## HHChannel
                    libchannel = moose.HHChannel(self.libpath+"/"+mechName)
                    ## deep copies the library channel under the compartment
                    channel = moose.copy(libchannel,compartment,mechName)
                    channel = moose.HHChannel(channel)
                    moose.connect(channel,'channel',compartment,'channel')

            # If mechanism is present in compartment, just wrap it
            else:
                neutralObj = moose.Neutral(compartment.path+'/'+mechName)
                if 'CaConc' == neutralObj.className: # Ion concentration pool
                    caconc = moose.CaConc(compartment.path+'/'+mechName) # wraps existing channel
                    channel = None
                elif 'HHChannel2D' == neutralObj.className : ## HHChannel2D
                    channel = moose.HHChannel2D(compartment.path+'/'+mechName) # wraps existing channel
                elif 'HHChannel' == neutralObj.className : ## HHChannel
                    channel = moose.HHChannel(compartment.path+'/'+mechName) # wraps existing channel
            if name == 'Gbar':
                if channel is None: # if CaConc, neuroConstruct uses gbar for thickness or phi
                    ## If child Mstring 'phi' is present, set gbar as phi
                    ## BUT, value has been multiplied by Gfactor as a Gbar,
                    ## SI or physiological not known here,
                    ## ignoring Gbar for CaConc, instead of passing units here
                    child = utils.get_child_Mstring(caconc,'phi')
                    if child is not None:
                        #child.value = value
                        pass
                    else:
                        #caconc.thick = value
                        pass
                else: # if ion channel, usual Gbar
                    ar = value*math.pi*compartment.diameter*compartment.length
                    channel.Gbar = ar
            elif name == 'Ek':
                channel.Ek = value
            # thick seems to be NEURON's extension to NeuroML level 2.
            elif name == 'thick': 
                caconc.thick = value 
                # JUST THIS WILL NOT DO - HAVE TO SET B based on this thick!
                # Later, when calling connect_CaConc, B is set for caconc based
                # on thickness of Ca shell and compartment l and dia.  OR based
                # on the Mstring phi under CaConc path.
        if neuroml_utils.neuroml_debug: 
            utils.dump("DEBUG"
                    , "Setting %s for %s value %s " % (name, compartment.path, value)
                    )

