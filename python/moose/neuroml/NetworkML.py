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

## Description: class NetworkML for loading NetworkML from file or xml element into MOOSE
## Version 1.0 by Aditya Gilra, NCBS, Bangalore, India, 2011 for serial MOOSE
## Version 1.5 by Niraj Dudani, NCBS, Bangalore, India, 2012, ported to parallel MOOSE
## Version 1.6 by Aditya Gilra, NCBS, Bangalore, India, 2012, further changes for parallel MOOSE
## Version 1.7 by Aditya Gilra, NCBS, Bangalore, India, 2013, further support for NeuroML 1.8.1


"""
NeuroML.py is the preferred interface. Use this only if NeuroML L1,L2,L3 files are misnamed/scattered.
Instantiate NetworlML class, and thence use method:
readNetworkMLFromFile(...) to load a standalone NetworkML file, OR
readNetworkML(...) to load from an xml.etree xml element (could be part of a larger NeuroML file).

"""

from xml.etree import cElementTree as ET
import string
import os
from math import cos, sin
from MorphML import MorphML
from ChannelML import ChannelML

import moose
import moose.utils as utils 

import utils as nml_utils



class NetworkML():

    def __init__(self, nml_params):
        self.cellDictBySegmentId={}
        self.cellDictByCableId={}
        self.nml_params = nml_params
        self.model_dir = nml_params['model_dir']

    def readNetworkMLFromFile(self,filename,cellSegmentDict,params={}):

        """ 
        specify tweak params = {'excludePopulations':[popname1,...]
            , 'excludeProjections':[projname1,...]
            , 'onlyInclude':{'includePopulation':(popname,[id1,...])
            ,'includeProjections':(projname1,...)} 
            }


        Notes by Aditya:

        If excludePopulations is present, then excludeProjections must also be
        present:

        Thus if you exclude some populations, ensure that you exclude
        projections that refer to those populations also!  Though for
        `onlyInclude`, you may specify only included cells and this reader will
        also keep cells connected to those in `onlyInclude`.  
        
        This reader first prunes the exclude-s, then keeps the onlyInclude-s and
        those that are connected.  Use 'includeProjections' if you want to keep
        some projections not connected to the primary 'includePopulation' cells
        but connected to secondary cells that connected to the primary ones:
        e.g.  baseline synapses on granule cells connected to
        'includePopulation' mitrals; these synapses receive file based
        pre-synaptic events, not presynaptically connected to a cell.  
        
        """

        utils.dump("NEUROML", "Reading file %s " % filename)
        tree = ET.parse(filename)
        root_element = tree.getroot()

        utils.dump("INFO", "Tweaking model")
        nml_utils.tweak_model(root_element, params)

        utils.dump("NEUROML", "Loading NeuroML model into MOOSE.")
        network = self.readNetworkML(root_element
                , cellSegmentDict
                , params
                )
        return network

    def readNetworkML(self, network, cellSegmentDict, params={}):
        """
        This returns populationDict = {
            'populationname1':(cellName,{int(instanceid1):moosecell, ... }) 
            , ... 
            }

        and projectionDict = { 
            'projectionname1':(source,target,[(syn_name1,pre_seg_path,post_seg_path)
            ,...]) 
            , ... 
            }
        """

        lengthUnits = network.attrib.get('lengthUnits', 'micron')
        if lengthUnits in ['micrometer','micron']:
            self.length_factor = 1e-6
        else:
            self.length_factor = 1.0

        self.network = network
        self.cellSegmentDict = cellSegmentDict
        self.params = params

        utils.dump("NEUROML", "Creating populations ... ")
        self.populationDict = {}
        populations = self.network.findall(".//{"+nml_utils.nml_ns+"}population")
        [self.insertPopulationIntoLibrary(p) for p in populations]

        utils.dump("NEUROML", "Creating connections ... ")
        self.createProjections() # create connections

        utils.dump("NEUROML", "Creating inputs in %s ... " % nml_utils.elecPath)
        self.createInputs() # create inputs (only current pulse supported)
        return (self.populationDict, self.projectionDict)

    def createInputs(self):
        for inputs in self.network.findall(".//{"+nml_utils.nml_ns+"}inputs"):
            units = inputs.attrib['units']
            if units == 'Physiological Units': # see pg 219 (sec 13.2) of Book of Genesis
                Vfactor = 1e-3 # V from mV
                Tfactor = 1e-3 # s from ms
                Ifactor = 1e-6 # A from microA
            else:
                Vfactor = 1.0
                Tfactor = 1.0
                Ifactor = 1.0
            for inputelem in inputs.findall(".//{"+nml_utils.nml_ns+"}input"):
                inputname = inputelem.attrib['name']
                pulseinput = inputelem.find(".//{"+nml_utils.nml_ns+"}pulse_input")
                if pulseinput is not None:
                    ## If nml_utils.elecPath doesn't exists then create it.
                    moose.Neutral(nml_utils.elecPath)
                    pulsegen = moose.PulseGen(nml_utils.elecPath+'/pulsegen_'+inputname)
                    iclamp = moose.DiffAmp(nml_utils.elecPath+'/iclamp_'+inputname)
                    iclamp.saturation = 1e6
                    iclamp.gain = 1.0
                    pulsegen.trigMode = 0 # free run
                    pulsegen.baseLevel = 0.0
                    pulsegen.firstDelay = float(pulseinput.attrib['delay'])*Tfactor
                    pulsegen.firstWidth = float(pulseinput.attrib['duration'])*Tfactor
                    pulsegen.firstLevel = float(pulseinput.attrib['amplitude'])*Ifactor
                    pulsegen.secondDelay = 1e6 # to avoid repeat
                    pulsegen.secondLevel = 0.0
                    pulsegen.secondWidth = 0.0
                    ## do not set count to 1, let it be at 2 by default
                    ## else it will set secondDelay to 0.0 and repeat the first pulse!
                    #pulsegen.count = 1
                    moose.connect(pulsegen,'output',iclamp,'plusIn')
                    target = inputelem.find(".//{"+nml_utils.nml_ns+"}target")
                    population = target.attrib['population']
                    for site in target.findall(".//{"+nml_utils.nml_ns+"}site"):
                        cell_id = site.attrib['cell_id']
                        if site.attrib.has_key('segment_id'): 
                            segment_id = site.attrib['segment_id']
                        else: 
                            segment_id = 0 # default segment_id is specified to be 0

                        # population is populationname,
                        # self.populationDict[population][0] is cellName
                        cell_name = self.populationDict[population][0]
                        segment_path = self.populationDict[population][1][int(cell_id)].path+'/'+\
                            self.cellSegmentDict[cell_name][segment_id][0]
                        compartment = moose.Compartment(segment_path)
                        utils.dump("DEBUG"
                                , "Connecting %s and %s" % (iclamp.path, segment_path)
                                )
                        moose.connect(iclamp,'output',compartment,'injectMsg')

    def insertPopulationIntoLibrary(self, population):
        """Insert a population entry into moose-library """

        cellName = population.attrib["cell_type"]
        popName = population.attrib["name"]
        utils.dump("INFO", "Inserting population `%s` into library" % popName)

        ## if cell does not exist in library load it from xml file
        if not moose.exists(nml_utils.libraryPath+'/'+cellName):
            mmlR = MorphML(self.nml_params)
            model_filenames = (cellName+'.xml', cellName+'.morph.xml')
            success = False
            for model_filename in model_filenames:
                model_path = nml_utils.find_first_file(model_filename,self.model_dir)
                if model_path is not None:
                    cellDict = mmlR.readMorphMLFromFile(model_path)
                    success = True
                    break
            if not success:
                raise IOError(
                    'For cell {0}: files {1} not found under {2}.'.format(
                        cellName, model_filenames, self.model_dir
                    )
                )
            self.cellSegmentDict.update(cellDict)

        #added cells as a Neuron class.
        libcell = moose.Neuron(nml_utils.libraryPath+'/'+cellName) 
        self.populationDict[popName] = (cellName,{})
        moose.Neutral('/cells')
        for instance in population.findall(".//{"+nml_utils.nml_ns+"}instance"):
            instanceid = instance.attrib['id']
            location = instance.find('./{'+nml_utils.nml_ns+'}location')
            rotationnote = instance.find('./{'+nml_utils.meta_ns+'}notes')
            if rotationnote is not None:
                ## the text in rotationnote is zrotation=xxxxxxx
                zrotation = float(string.split(rotationnote.text,'=')[1])
            else:
                zrotation = 0
            ## deep copies the library cell to an instance under '/cells' named as <arg3>
            ## /cells is useful for scheduling clocks as all sim elements are in /cells
            cellid = moose.copy(libcell,moose.Neutral('/cells'),popName+"_"+instanceid)
            cell = moose.Neuron(cellid)
            self.populationDict[popName][1][int(instanceid)]=cell
            x = float(location.attrib['x'])*self.length_factor
            y = float(location.attrib['y'])*self.length_factor
            z = float(location.attrib['z'])*self.length_factor
            nml_utils.translate_rotate(cell,x,y,z,zrotation)
            
    def createProjections(self):
        self.projectionDict={}
        projections = self.network.find(".//{"+nml_utils.nml_ns+"}projections")
        if projections is not None:
            if projections.attrib["units"] == 'Physiological Units': # see pg 219 (sec 13.2) of Book of Genesis
                Efactor = 1e-3 # V from mV
                Tfactor = 1e-3 # s from ms
            else:
                Efactor = 1.0
                Tfactor = 1.0

        for projection in self.network.findall(".//{"+nml_utils.nml_ns+"}projection"):
            projectionname = projection.attrib["name"]
            print "setting",projectionname
            source = projection.attrib["source"]
            target = projection.attrib["target"]
            self.projectionDict[projectionname] = (source,target,[])
            for syn_props in projection.findall(".//{"+nml_utils.nml_ns+"}synapse_props"):
                syn_name = syn_props.attrib['synapse_type']
                ## if synapse does not exist in library load it from xml file
                if not moose.exists("/library/"+syn_name):
                    cmlR = ChannelML(self.nml_params)
                    model_filename = syn_name+'.xml'
                    model_path = nml_utils.find_first_file(model_filename,self.model_dir)
                    if model_path is not None:
                        cmlR.readChannelMLFromFile(model_path)
                    else:
                        raise IOError(
                            'For mechanism {0}: files {1} not found under {2}.'.format(
                                "", model_filename, self.model_dir
                            )
                        )
                weight = float(syn_props.attrib['weight'])
                threshold = float(syn_props.attrib['threshold'])*Efactor
                if 'prop_delay' in syn_props.attrib:
                    prop_delay = float(syn_props.attrib['prop_delay'])*Tfactor
                elif 'internal_delay' in syn_props.attrib:
                    prop_delay = float(syn_props.attrib['internal_delay'])*Tfactor
                else: prop_delay = 0.0
                for connection in projection.findall(".//{"+nml_utils.nml_ns+"}connection"):
                    pre_cell_id = connection.attrib['pre_cell_id']
                    post_cell_id = connection.attrib['post_cell_id']
                    if 'file' not in pre_cell_id:
                        # source could be 'mitrals', self.populationDict[source][0] would be 'mitral'
                        pre_cell_name = self.populationDict[source][0]
                        if 'pre_segment_id' in connection.attrib:
                            pre_segment_id = connection.attrib['pre_segment_id']
                        else: pre_segment_id = "0" # assume default segment 0, usually soma
                        pre_segment_path = self.populationDict[source][1][int(pre_cell_id)].path+'/'+\
                            self.cellSegmentDict[pre_cell_name][pre_segment_id][0]
                    else:
                        # I've removed extra excitation provided via files, so below comment doesn't apply.
                        # 'file[+<glomnum>]_<filenumber>' # glomnum is
                        # for mitral_granule extra excitation from unmodelled sisters.
                        pre_segment_path = pre_cell_id+'_'+connection.attrib['pre_segment_id']
                    # target could be 'PGs', self.populationDict[target][0] would be 'PG'
                    post_cell_name = self.populationDict[target][0]
                    if 'post_segment_id' in connection.attrib:
                        post_segment_id = connection.attrib['post_segment_id']
                    else: post_segment_id = "0" # assume default segment 0, usually soma
                    post_segment_path = self.populationDict[target][1][int(post_cell_id)].path+'/'+\
                        self.cellSegmentDict[post_cell_name][post_segment_id][0]
                    self.projectionDict[projectionname][2].append((syn_name, pre_segment_path, post_segment_path))
                    properties = connection.findall('./{'+nml_utils.nml_ns+'}properties')
                    if len(properties)==0:
                        self.connect(syn_name, pre_segment_path, post_segment_path, weight, threshold, prop_delay)
                    else:
                        for props in properties:
                            synapse_type = props.attrib['synapse_type']
                            if syn_name in synapse_type:
                                weight_override = float(props.attrib['weight'])
                                if 'internal_delay' in props.attrib:
                                    delay_override = float(props.attrib['internal_delay'])
                                else: delay_override = prop_delay
                                if weight_override != 0.0:
                                    self.connect(syn_name, pre_segment_path, post_segment_path,\
                                        weight_override, threshold, delay_override)

    def connect(self, syn_name, pre_path, post_path, weight, threshold, delay):
        postcomp = moose.Compartment(post_path)
        ## We usually try to reuse an existing SynChan -
        ## event based SynChans have an array of weights and delays and can represent multiple synapses i.e.
        ## a new element of the weights and delays array is created
        ## every time a 'synapse' message connects to the SynChan (from 'event' of spikegen)
        ## BUT for a graded synapse with a lookup table output connected to 'activation' message,
        ## not to 'synapse' message, we make a new synapse everytime
        ## ALSO for a saturating synapse i.e. KinSynChan, we always make a new synapse
        ## as KinSynChan is not meant to represent multiple synapses
        libsyn = moose.SynChan(nml_utils.libraryPath+'/'+syn_name)
        gradedchild = utils.get_child_Mstring(libsyn,'graded')
        if libsyn.className == 'KinSynChan' or gradedchild.value == 'True': # create a new synapse
            syn_name_full = syn_name+'_'+utils.underscorize(pre_path)
            self.make_new_synapse(syn_name, postcomp, syn_name_full)
        else:
            ## if syn doesn't exist in this compartment, create it
            syn_name_full = syn_name
            if not moose.exists(post_path+'/'+syn_name_full):
                self.make_new_synapse(syn_name, postcomp, syn_name_full)
        ## moose.element is a function that checks if path exists,
        ## and returns the correct object, here SynChan
        syn = moose.element(post_path+'/'+syn_name_full) # wrap the synapse in this compartment
        ### SynChan would have created a new synapse if it didn't exist at the given path
        #syn = moose.SynChan(post_path+'/'+syn_name_full) # wrap the synapse in this compartment
        gradedchild = utils.get_child_Mstring(syn,'graded')
        #### weights are set at the end according to whether the synapse is graded or event-based

        #### connect pre-comp Vm (if graded) OR spikegen/timetable (if event-based) to the synapse
        ## I rely on second term below not being evaluated if first term is None; else None.value gives error.
        if gradedchild is not None and gradedchild.value=='True': # graded synapse
            table = moose.Table(syn.path+"/graded_table")
            #### always connect source to input - else 'cannot create message' error.
            precomp = moose.Compartment(pre_path)
            moose.connect(precomp,"VmOut",table,"msgInput")
            ## since there is no weight field for a graded synapse
            ## (no 'synapse' message connected),
            ## I set the Gbar to weight*Gbar
            syn.Gbar = weight*syn.Gbar
        else: # Event based synapse
            ## synapse could be connected to spikegen at pre-compartment OR a file!
            if 'file' not in pre_path:
                ## element() can return either Compartment() or IzhikevichNrn(),
                ## since it queries and wraps the actual object
                precomp = moose.element(pre_path)
                ## if spikegen for this synapse doesn't exist in this compartment, create it
                ## spikegens for different synapse_types can have different thresholds
                ## but an integrate and fire spikegen supercedes all other spikegens
                if not moose.exists(pre_path+'/IaF_spikegen'):
                    if not moose.exists(pre_path+'/'+syn_name+'_spikegen'):
                        ## create new spikegen
                        spikegen = moose.SpikeGen(pre_path+'/'+syn_name+'_spikegen')
                        ## connect the compartment Vm to the spikegen
                        moose.connect(precomp,"VmOut",spikegen,"Vm")
                        ## spikegens for different synapse_types can have different thresholds
                        spikegen.threshold = threshold
                        spikegen.edgeTriggered = 1 # This ensures that spike is generated only on leading edge.
                        ## usually events are raised at every time step that Vm > Threshold,
                        ## can set either edgeTriggered as above or refractT
                        #spikegen.refractT = 0.25e-3
                    ## wrap the existing or newly created spikegen in this compartment
                    spikegen = moose.SpikeGen(pre_path+'/'+syn_name+'_spikegen')
                else:
                    spikegen = moose.SpikeGen(pre_path+'/IaF_spikegen')
                ## connect the spikegen to the synapse
                ## note that you need to use Synapse (auto-created) under SynChan
                ## to get/set weights , addSpike-s etc.
                ## can get the Synapse element by moose.Synapse(syn.path+'/synapse') or syn.synapse
                ## Synpase is an array element, first add to it, to addSpike-s, get/set weights, etc.
                syn.numSynapses += 1
                ## above works, but below gives me an error sayin getNum_synapse not found
                ## but both work in Demos/snippets/lifcomp.py
                #syn.synapse.num += 1
                ## see Demos/snippets/synapse.py for an example of 
                ## how to connect multiple SpikeGens to the same SynChan
                m = moose.connect(spikegen, 'spikeOut',
                                    syn.synapse[-1], 'addSpike', 'Single')
            else:
                # if connected to a file, create a timetable,
                # put in a field specifying the connected filenumbers to this segment,
                # and leave it for simulation-time connection
                ## pre_path is 'file[+<glomnum>]_<filenum1>[_<filenum2>...]' i.e. glomnum could be present
                filesplit = pre_path.split('+')
                if len(filesplit) == 2:
                    glomsplit = filesplit[1].split('_',1)
                    glomstr = '_'+glomsplit[0]
                    filenums = glomsplit[1]
                else:
                    glomstr = ''
                    filenums = pre_path.split('_',1)[1]
                tt_path = postcomp.path+'/'+syn_name_full+glomstr+'_tt'
                if not moose.exists(tt_path):
                    ## if timetable for this synapse doesn't exist in this compartment, create it,
                    ## and add the field 'fileNumbers'
                    tt = moose.TimeTable(tt_path)
                    tt_filenums = moose.Mstring(tt_path+'/fileNumbers')
                    tt_filenums.value = filenums
                    ## Be careful to connect the timetable only once while creating it as below:
                    ## note that you need to use Synapse (auto-created) under SynChan
                    ## to get/set weights , addSpike-s etc.
                    ## can get the Synapse element by moose.Synapse(syn.path+'/synapse') or syn.synapse
                    ## Synpase is an array element, first add to it, to addSpike-s, get/set weights, etc.
                    syn.numSynapses += 1
                    ## above works, but below gives me an error sayin getNum_synapse not found
                    ## but both work in Demos/snippets/lifcomp.py
                    #syn.synapse.num += 1
                    m = moose.connect(tt,"eventOut",syn.synapse[-1],"addSpike","Single")
                else:
                    ## if it exists, append file number to the field 'fileNumbers'
                    ## append filenumbers from 'file[+<glomnum>]_<filenumber1>[_<filenumber2>...]'
                    tt_filenums = moose.Mstring(tt_path+'/fileNumbers')
                    tt_filenums.value += '_' + filenums
            #### syn.Gbar remains the same, but we play with the weight which is a factor to Gbar
            #### The delay and weight can be set only after connecting a spike event generator.
            #### delay and weight are arrays: multiple event messages can be connected to a single synapse
            ## first argument below is the array index, we connect to the latest synapse created above
            ## But KinSynChan ignores weight of the synapse, so set the Gbar for it
            if libsyn.className == 'KinSynChan':
                syn.Gbar = weight*syn.Gbar
            else:
                ## note that you need to use Synapse (auto-created) under SynChan
                ## to get/set weights , addSpike-s etc.
                ## can get the Synpase element by moose.Synapse(syn.path+'/synapse') or syn.synapse
                syn.synapse[-1].weight = weight
            syn.synapse[-1].delay = delay # seconds
            #print 'len = ',len(syn.synapse)
            #for i,syn_syn in enumerate(syn.synapse):
            #    print i,'th weight =',syn_syn.weight,'\n'

    def make_new_synapse(self, syn_name, postcomp, syn_name_full):
        ## if channel does not exist in library load it from xml file
        if not moose.exists(nml_utils.libraryPath+'/'+syn_name):
            cmlR = ChannelML(self.nml_params)
            cmlR.readChannelMLFromFile(syn_name+'.xml')
        ## deep copies the library synapse to an instance under postcomp named as <arg3>
        synid = moose.copy(moose.Neutral(nml_utils.libraryPath+'/'+syn_name),postcomp,syn_name_full)
        syn = moose.SynChan(synid)
        childmgblock = utils.get_child_Mstring(syn,'mgblockStr')
        #### connect the post compartment to the synapse
        if childmgblock.value=='True': # If NMDA synapse based on mgblock, connect to mgblock
            mgblock = moose.Mg_block(syn.path+'/mgblock')
            moose.connect(postcomp,"channel", mgblock, "channel")
        else: # if SynChan or even NMDAChan, connect normally
            moose.connect(postcomp,"channel", syn, "channel")
