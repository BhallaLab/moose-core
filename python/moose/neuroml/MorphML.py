from xml.etree import ElementTree as ET
import string
import moose
import sys, math

from ChannelML import *
from moose.utils import *

class MorphML():

    def __init__(self):
        self.neuroml='http://morphml.org/neuroml/schema'
        self.bio='http://morphml.org/biophysics/schema'
        self.mml='http://morphml.org/morphml/schema'
        self.nml='http://morphml.org/networkml/schema'
        self.cellDictBySegmentId={}
        self.cellDictByCableId={}
        self.context = moose.PyMooseBase.getContext()

    def readMorphMLFromFile(self,filename,params={}):
        """
        specify global params as a dict (presently none implemented)
        returns { cellname1 : segDict, ... }
        see readMorphML(...) for segDict 
        """
        tree = ET.parse(filename)
        neuroml_element = tree.getroot()
        cellsDict = {}
        for cell in neuroml_element.findall('.//{'+self.neuroml+'}cell'):
            cellDict = self.readMorphML(cell,params,neuroml_element.attrib['lengthUnits'])
            cellsDict.update(cellDict)
        return cellsDict

    def readMorphML(self,cell,params={},lengthUnits="micrometer"):
        """
        returns {cellname:segDict}
        where segDict = { segid1 : [ segname,(proximalx,proximaly,proximalz),
            (distalx,distaly,distalz),diameter,length,[potential_syn1, ... ] ] , ... }
        segname is "<name>_<segid>" because 1) guarantees uniqueness,
        2) later scripts obtain segid from the compartment's name!
        """
        if lengthUnits in ['micrometer','micron']:
            self.length_factor = 1e-6
        else:
            self.length_factor = 1.0
        cellname = cell.attrib["name"]
        print "loading cell :", cellname
        moosecell = moose.Cell('/library/'+cellname)
        self.cellDictBySegmentId[cellname] = [moosecell,{}]
        self.cellDictByCableId[cellname] = [moosecell,{}]
        self.segDict = {}
        
        ############################################################
        #### load morphology and connections between compartments
        for segment in cell.findall(".//{"+self.mml+"}segment"):
            segmentname = segment.attrib['name']
            #print segmentname
            segmentid = segment.attrib['id']
            # the moose "hsolve" method assumes compartments to be asymmetric compartments and symmetrizes them
            # but that is not what we want when translating from Neuron which has only symcompartments -- so be careful!
            moosesegmentname = segmentname+'_'+segmentid
            self.segDict[segmentid]=[moosesegmentname]
            moosesegment = moose.Compartment(moosesegmentname, moosecell) # segmentname is NOT unique - eg: mitral bbmit exported from NEURON
            self.cellDictBySegmentId[cellname][1][segmentid] = moosesegment
            # cable is an optional attribute. Need to change the way things are done.
            cableid = segment.attrib['cable']
            ## ASSUME 1:1 CORRESPONDENCE BETWEEN Cable and Segment.
            ## THIS MAY NOT BE TRUE - ENSURE YOUR SEGMENTS HAVE UNIQUE CABLE IDs ALSO!!!!
            self.cellDictByCableId[cellname][1][cableid] = moosesegment # cables are grouped and densities set for cablegroups. Hence I need to refer to segment according to which cable they belong to.
            if segment.attrib.has_key('parent'):
                parentid = segment.attrib['parent'] # I assume the parent is created before the child so that I can immediately connect the child.
                parent = self.cellDictBySegmentId[cellname][1][parentid]
                # It is always assumed that axial of parent is connected to raxial of moosesegment
                # THIS IS WHAT GENESIS readcell() DOES!!! UNLIKE NEURON!
                # THIS IS IRRESPECTIVE OF WHETHER PROXIMAL x,y,z OF PARENT = PROXIMAL x,y,z OF CHILD.
                # THIS IS ALSO IRRESPECTIVE OF fraction_along_parent SPECIFIED IN CABLE!
                # THUS THERE WILL BE NUMERICAL DIFFERENCES BETWEEN MOOSE/GENESIS and NEURON.
                # moosesegment sends Ra and Vm to parent, parent sends only Vm
                # actually for symmetric compartment, both parent and moosesegment require each other's Ra/2,
                # but axial and raxial just serve to distinguish ends.
                parent.connect('axial',moosesegment,'raxial')
            else:
                parent = None
            proximal = segment.find('./{'+self.mml+'}proximal')
            if proximal==None:          # If proximal tag is not present,
                                        # then parent attribute MUST be present in the segment tag!
                ## if proximal is not present, then
                ## by default the distal end of the parent is the proximal end of the child
                moosesegment.x0 = parent.x
                moosesegment.y0 = parent.y
                moosesegment.z0 = parent.z
            else:
                moosesegment.x0 = float(proximal.attrib["x"])*self.length_factor
                moosesegment.y0 = float(proximal.attrib["y"])*self.length_factor
                moosesegment.z0 = float(proximal.attrib["z"])*self.length_factor
            distal = segment.find('./{'+self.mml+'}distal')
            moosesegment.x = float(distal.attrib["x"])*self.length_factor
            moosesegment.y = float(distal.attrib["y"])*self.length_factor
            moosesegment.z = float(distal.attrib["z"])*self.length_factor
            ## proximal tag may not be present, so take only distal diameter
            moosesegment.diameter = float(distal.attrib["diameter"]) * self.length_factor
            moosesegment.length = sqrt((moosesegment.x-moosesegment.x0)**2+\
                (moosesegment.y-moosesegment.y0)**2+(moosesegment.z-moosesegment.z0)**2)
            if moosesegment.length == 0.0:          # neuroconstruct seems to set length=0 for round soma!
                moosesegment.length = moosesegment.diameter
            ## the empty list at the end below will get populated 
            ## with the potential synapses on this segment inside set_compartment_param(..)
            self.segDict[segmentid].extend([(moosesegment.x0,moosesegment.y0,moosesegment.z0),\
                (moosesegment.x,moosesegment.y,moosesegment.z),moosesegment.diameter,moosesegment.length,[]])

        ###############################################
        #### load biophysics into the compartments
        biophysics = cell.find(".//{"+self.neuroml+"}biophysics")
        if biophysics is not None:
            if biophysics.attrib["units"] == 'Physiological Units': # see pg 219 (sec 13.2) of Book of Genesis
                CMfactor = 1e-2 # F/m^2 from microF/cm^2
                RAfactor = 1e1 # Ohm*m from KOhm*cm
                RMfactor = 1e-1 # Ohm*m^2 from KOhm*cm^2
                Efactor = 1e-3 # V from mV
                Gfactor = 1e1 # S/m^2 from mS/cm^2
            else:
                CMfactor = 1.0
                RAfactor = 1.0
                RMfactor = 1.0
                Efactor = 1.0
                Gfactor = 1.0
            spec_capacitance = cell.find(".//{"+self.bio+"}spec_capacitance")
            for parameter in spec_capacitance.findall(".//{"+self.bio+"}parameter"):
                self.set_group_compartment_param(cell, cellname, parameter,\
                 'CM', float(parameter.attrib["value"])*CMfactor, self.bio)
            spec_axial_resitance = cell.find(".//{"+self.bio+"}spec_axial_resistance")
            for parameter in spec_axial_resitance.findall(".//{"+self.bio+"}parameter"):
                self.set_group_compartment_param(cell, cellname, parameter,\
                 'RA', float(parameter.attrib["value"])*RAfactor, self.bio)
            init_memb_potential = cell.find(".//{"+self.bio+"}init_memb_potential")
            for parameter in init_memb_potential.findall(".//{"+self.bio+"}parameter"):
                self.set_group_compartment_param(cell, cellname, parameter,\
                 'initVm', float(parameter.attrib["value"])*Efactor, self.bio)
            for mechanism in cell.findall(".//{"+self.bio+"}mechanism"):
                mechanismname = mechanism.attrib["name"]
                passive = False
                if mechanism.attrib.has_key("passive_conductance"):
                    if mechanism.attrib['passive_conductance'] in ["true",'True','TRUE']:
                        passive = True
                #print "Loading mechanism ", mechanismname
                ## ONLY creates channel if at least one parameter (like gmax) is specified in the xml
                ## Neuroml does not allow you to specify all default values.
                ## However, granule cell example in neuroconstruct has Ca ion pool without
                ## a parameter, applying default values to all compartments!
                mech_params = mechanism.findall(".//{"+self.bio+"}parameter")
                ## if no params, apply all default values to all compartments
                if len(mech_params) == 0:
                    for compartment in self.cellDictByCableId[cellname][1].values():
                        self.set_compartment_param(compartment,None,'default',mechanismname)  
                ## if params are present, apply params to specified cable/compartment groups
                for parameter in mech_params:
                    parametername = parameter.attrib['name']
                    if passive==True:
                        if passive and parametername in ['gmax']:
                            self.set_group_compartment_param(cell, cellname, parameter,\
                             'RM', RMfactor*1.0/float(parameter.attrib["value"]), self.bio)
                        elif passive and parametername in ['e','erev']:
                            self.set_group_compartment_param(cell, cellname, parameter,\
                             'Em', Efactor*float(parameter.attrib["value"]), self.bio)
                        else:
                            print "Yo programmer! You left out parameter ",\
                             parametername, " in mechanism ",mechanismname
                    else:
                        if parametername in ['gmax']:
                            gmaxval = float(eval(parameter.attrib["value"],{"__builtins__":None},{}))
                            self.set_group_compartment_param(cell, cellname, parameter,\
                             'Gbar', Gfactor*gmaxval, self.bio, mechanismname)
                        elif parametername in ['e','erev']:
                            self.set_group_compartment_param(cell, cellname, parameter,\
                             'Ek', Efactor*float(parameter.attrib["value"]), self.bio, mechanismname)
                        elif parametername in ['depth']: # has to be type Ion Concentration!
                            self.set_group_compartment_param(cell, cellname, parameter,\
                             'thick', self.length_factor*float(parameter.attrib["value"]),\
                             self.bio, mechanismname)
                        else:
                            print "Yo programmer of morphml import! You left out parameter ",\
                             parametername, " in mechanism ",mechanismname
            #### Connect the Ca pools and channels
            #### Am connecting these at the very end so that all channels and pools have been created
            connect_CaConc(self.cellDictByCableId[cellname][1].values())
        
        ##########################################################
        #### load connectivity / synapses into the compartments
        connectivity = cell.find(".//{"+self.neuroml+"}connectivity")
        if connectivity is not None:
            for potential_syn_loc in cell.findall(".//{"+self.nml+"}potential_syn_loc"):
                if 'synapse_direction' in potential_syn_loc.attrib.keys():
                    if potential_syn_loc.attrib['synapse_direction'] in ['post']:
                        self.set_group_compartment_param(cell, cellname, potential_syn_loc,\
                         'synapse_type', potential_syn_loc.attrib['synapse_type'], self.nml, mechanismname='synapse')
                    if potential_syn_loc.attrib['synapse_direction'] in ['pre']:
                        self.set_group_compartment_param(cell, cellname, potential_syn_loc,\
                         'spikegen_type', potential_syn_loc.attrib['synapse_type'], self.nml, mechanismname='spikegen')

        #print "Finished loading into library cell: ",cellname
        return {cellname:self.segDict}

    def set_group_compartment_param(self, cell, cellname, parameter, name, value, grouptype, mechanismname=None):
        """
        Find the compartments that belong to the cablegroups refered to
         for this parameter and set_compartment_param.
        """
        for group in parameter.findall(".//{"+grouptype+"}group"):
            cablegroupname = group.text
            if cablegroupname == 'all':
                for compartment in self.cellDictByCableId[cellname][1].values():
                    self.set_compartment_param(compartment,name,value,mechanismname)
            else:
                for cablegroup in cell.findall(".//{"+self.mml+"}cablegroup"):
                    if cablegroup.attrib['name'] == cablegroupname:
                        for cable in cablegroup.findall(".//{"+self.mml+"}cable"):
                            cableid = cable.attrib['id']
                            compartment = self.cellDictByCableId[cellname][1][cableid]
                            self.set_compartment_param(compartment,name,value,mechanismname)

    def set_compartment_param(self, compartment, name, value, mechanismname):
        """ Set the param for the compartment depending on name and mechanismname. """
        if name == 'CM':
            compartment.Cm = value*math.pi*compartment.diameter*compartment.length
        elif name == 'RM':
            compartment.Rm = value/(math.pi*compartment.diameter*compartment.length)
        elif name == 'RA':
            compartment.Ra = value*compartment.length/(math.pi*(compartment.diameter/2.0)**2)
        elif name == 'Em':
            compartment.Em = value
        elif name == 'initVm':
            compartment.initVm = value
        elif mechanismname is 'synapse': # synapse being added to the compartment
            ## these are potential locations, we do not actually make synapses.
            #synapse = self.context.deepCopy(self.context.pathToId('/library/'+value),\
            #    self.context.pathToId(compartment.path),value) # value contains name of synapse i.e. synapse_type
            #compartment.connect("channel", synapse, "channel")
            ## I assume below that compartment name has _segid at its end
            segid = string.split(compartment.name,'_')[-1] # get segment id from compartment name
            self.segDict[segid][5].append(value)
        elif mechanismname is 'spikegen': # spikegen being added to the compartment
            ## these are potential locations, we do not actually make the spikegens.
            ## spikegens for different synapses can have different thresholds,
            ## hence include synapse_type in its name
            ## value contains name of synapse i.e. synapse_type
            #spikegen = moose.SpikeGen(compartment.path+'/'+value+'_spikegen')
            #compartment.connect("VmSrc",spikegen,"Vm")
            pass
        elif mechanismname is not None:
            ## if mechanism is not present in compartment, deep copy from library
            if not self.context.exists(compartment.path+'/'+mechanismname):
                ## if channel does not exist in library load it from xml file
                if not self.context.exists("/library/"+mechanismname):
                    cmlR = ChannelML()
                    cmlR.readChannelMLFromFile(mechanismname+'.xml')
                neutralObj = moose.Neutral("/library/"+mechanismname)
                if 'Conc' in neutralObj.className: # Ion concentration pool
                    libcaconc = moose.CaConc("/library/"+mechanismname)
                    ## deep copies the library caconc under the compartment
                    caconc = moose.CaConc(libcaconc,mechanismname,compartment)
                    ## CaConc connections are made later using connect_CaConc()
                    ## thickness of Ca shell is set below and
                    ## caconc.B is also set below based on thickness
                elif 'HHChannel2D' in neutralObj.className : ## HHChannel2D
                    libchannel = moose.HHChannel2D("/library/"+mechanismname)
                    ## deep copies the library channel under the compartment
                    channel = moose.HHChannel2D(libchannel,mechanismname,compartment)
                    channel.connect('channel',compartment,'channel')
                elif 'HHChannel' in neutralObj.className : ## HHChannel
                    libchannel = moose.HHChannel("/library/"+mechanismname)
                    ## deep copies the library channel under the compartment
                    channel = moose.HHChannel(libchannel,mechanismname,compartment)
                    channel.connect('channel',compartment,'channel')
            ## if mechanism is present in compartment, just wrap it
            else:
                neutralObj = moose.Neutral(compartment.path+'/'+mechanismname)
                if 'Conc' in neutralObj.className: # Ion concentration pool
                    caconc = moose.CaConc(compartment.path+'/'+mechanismname) # wraps existing channel
                elif 'HHChannel2D' in neutralObj.className : ## HHChannel2D
                    channel = moose.HHChannel2D(compartment.path+'/'+mechanismname) # wraps existing channel
                elif 'HHChannel' in neutralObj.className : ## HHChannel
                    channel = moose.HHChannel(compartment.path+'/'+mechanismname) # wraps existing channel
            if name == 'Gbar':
                channel.Gbar = value*math.pi*compartment.diameter*compartment.length
            elif name == 'Ek':
                channel.Ek = value
            elif name == 'thick':
                caconc.thick = value ## JUST THIS WILL NOT DO - HAVE TO SET B based on this thick!
                caconc.B = 1 / (2*FARADAY) / (math.pi*compartment.diameter*compartment.length * value)
                ## I am using a translation from Neuron, hence this method.
                ## In Genesis, gmax / (surfacearea*thick) is set as value of B!
        #print "Setting ",name," for ",compartment.path," value ",value
