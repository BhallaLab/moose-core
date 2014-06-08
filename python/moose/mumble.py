# This file is part of MOOSE simulator: http://_moose.ncbs.res.in.

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

"""mumbl.py: This file reads the mumbl file and load it onto _moose. 
This class is entry point of multiscale modelling.

Last modified: Fri Jan 24, 2014  05:27PM

"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import inspect
import os
import sys
import _moose
import inspect
import print_utils
import helper.moose_methods as moose_methods
import helper.types as types
import multiscale_config as config

class Mumble():
    """ Mumble: Class for loading mumble onto _moose.
    """
    def __init__(self, mumbl):
        self.mumblElem = mumbl
        self.rootElem = self.mumblElem.getroot()
        self.mumblPath = config.mumblePath
        self.mumblRootPath = os.path.dirname(self.mumblPath)
        self.global_ = self.rootElem.get('global')
        self.countElecModels = 0
        self.countChemModels = 0
        self.compartmentName = 'Compartment'
        self.adaptorPath = os.path.join(self.mumblPath, 'Adaptor')
        self.cellPath = config.cellPath
        self.nmlPath = config.nmlPath
        self.chemPath = os.path.join(self.mumblPath, 'Chemical')
        self.elecPath = os.path.join(self.mumblPath, 'Electrical')
        _moose.Neutral(self.mumblPath)
        _moose.Neutral(self.chemPath)
        _moose.Neutral(self.elecPath)
        _moose.Neutral(self.adaptorPath)

        # I might have to handle SBML
        self.adaptorCount = 0

        # clock
        self.mumbleClockId = 3
        self.dt = 1e-03
        _moose.setClock(self.mumbleClockId, self.dt)
        # Insert each model to this list, if model is already added then raise a
        # warning and don't insert the model.
        self.modelList = list()
        self.speciesDict = types.DoubleDict()

    def initPaths(self, paths, recursively=True):
        """
        Initialize all parents.
        """
        if recursively:
            ps = paths.split('/')
            p = ""
            for i in ps:
                p += ("/"+i)
                _moose.Neutral(p)
        else:
            try:
                _moose.Neutral(paths)
            except Exception as e:
                print_utils.dump("ERROR"
                , "Path {} does not exists".format(paths)
                )
                sys.exit()

    def prefixWithSet(self, var):
        assert len(var.strip()) > 0, "Empty variable name"
        var = var[0].upper() + var[1:]
        return 'set'+var

    def prefixWithGet(self, var):
        assert len(var.strip()) > 0, "Empty variable name"
        var = var[0].upper() + var[1:]
        return 'get'+var

    def load(self):
        """ Lead mumble element tree
        """
        print_utils.dump("INFO", "Loading mumble")
        [self.loadModel(model) for model in self.rootElem.findall('model') ]

        # Mappings from electrical to chemical and vice versa belongs to
        # "domain"
        domains = self.rootElem.findall('domain')
        [ self.mapDomainOntoDomain(d) for d in domains ]


    def loadModel(self, modelXml):
        """
        Load additional model to _moose.
        """
        if modelXml.get('load_model', 'true') != 'true':
            print_utils.dump("WARN", "No loading model %s" % modelXml)
            return None

        modelId = modelXml.get('id')
        if modelId in self.modelList:
            print_utils.dump("ERROR"
                    , "Two models have same id {0}! Ignoring...".format(modelId)
                    )
            return RuntimeError("Two models with same id")
        self.modelList.append(modelId)

        # Get the type of model and call appropriate function.
        modelType = modelXml.get('domain_type')
        if modelType == "electrical":
            self.loadElectricalModel(modelXml)
        elif modelType == 'chemical':
            self.loadChemicalModel(modelXml)
        else:
            print_utils.dump("TODO"
                    , "{0} : Un-supported Mumbl model".format(modelType)
                    , frame = inspect.currentframe()
                    )

    def createMoosePathForModel(self, modelNo, modelType, species=None):
        """
        Create _moose path for this chemical model.
        """
        if modelType == "chemical":
            modelPath = os.path.join(self.chemPath, '{0}')
            if species:
                return modelPath.format("chemical", self.countChemModels)
            else:
                return modelPath.format("chemical"
                        , species
                        , self.countChemModels
                        )
           
        elif modelType == "electrical":
            modelPath = "/models/electrical/{0}_{1}"
            return modelPath.format("e", self.countElecModels)
        else:
            print_utils.dump("TODO"
                    , "Unsupported model type : {0}".format(modelType)
                    )
            raise UserWarning, "Unsupported model type"
        return None 

    def loadElectricalModel(self, modelXml):
        """
        Load electrical model.
        """
        if modelXml.get('already_loaded') == "true":
            return 
        self.countElecModels += 1
        print_utils.dump("TODO"
                , "Elec model is not in NML. Ignoring for now..."
                , frame = inspect.currentframe()
                )

    def loadChemicalModel(self, modelXml):
        """
        This function load a chemical model described in mumble. Mumble can
        point to a model described in some file. Or the model may have been
        already loaded and one can specify the simulation specific details. We
        only spport Moose simulator.

        @param modelXml: This is xml elements.
        @type  param:  lxml.Elements

        @return:  None
        @rtype : None type.
        """

        if modelXml.get('already_loaded') == "true":
            print_utils.dump("USER"
                    , "This model is alreay loaded. Doing nothing..."
                    )
            return

        # Else load the model onto _moose.
        self.countChemModels += 1

        if modelXml.get('load_using_external_script') == "yes":
            print_utils.dump("TODO"
                    , "Loading user external script is not supported yet."
                    , frame = inspect.currentframe()
                    )
            raise UserWarning, "Unimplemented feature"

        # Otherwise load the model.
        modelFilePath = modelXml.get('file_path', None)
        if modelFilePath is not None:
            modelFilePath = os.path.join(self.mumblRootPath, modelFilePath)
            if not os.path.exists(modelFilePath):
                raise IOError("Failed to open a file %s" % modelFilePath)

        # Check if we are referencing to a sbml model
        if modelXml.get('model_type', '') == "sbml":
            print_utils.dump("DEBUG"
                    , "Got a SBML model describing a chemical compartment"
                    )
            try:
                _moose.readSBML(modelFilePath
                        , os.path.join(self.chemPath, self.compartmentName)
                        )

            except Exception as e:
                print_utils.dump("ERROR"
                        , [ "Failed to open or parse SBML %s " % modelFilePath
                            , "{}".format(e)
                            ]
                        , frame = inspect.currentframe()
                        )
                sys.exit(0)
            print_utils.dump("STEP", "Done parsing SBML")
                        
                    
        # get compartments and add species to these compartments.
        compsXml = modelXml.find('compartments')
        if compsXml is not None:
            comps = compsXml.findall('compartment')
            [ self.addCompartment(compsXml.attrib, c, "chemical") for c in comps ]

    def addCompartment(self, compsAttribs, xmlElem, chemType):
        """Add compartment if not exists and inject species into add.

        Ideally compartment must exist. 
        
        The id of compartment in xmlElement should be compatible with neuroml
        comparment ids.
        """
        if chemType != "chemical":
            raise UserWarning, "Only chemical models are supported"

        compPath = os.path.join(self.chemPath, self.compartmentName)
        _moose.Neutral(compPath)
        compPath = os.path.join(compPath, xmlElem.get('id'))
        _moose.Neutral(compPath)

        # Add pools to this compartment
        pools = xmlElem.findall('pool')
        for p in pools:
            speciesName = p.get('species')
            self.speciesDict.insertUniqueVal(speciesName)
            poolPath = os.path.join(compPath, speciesName)
            try:
                poolComp = _moose.Pool(poolPath)
            except Exception as e:
                print_utils.dump("WARN", "Perhaps the compartment_id is wrong!")
                raise KeyError("Missing parent path %s" % poolPath)

            poolComp.conc = moose_methods.stringToFloat(p.get('conc'))
            poolComp.speciesId = self.speciesDict.get(speciesName)
        
    def mapDomainOntoDomain(self, domain):
        """
        Each <domain> element is essentially a mapping from a compartment to
        another one. 

        """
        xmlType = domain.get('xml', 'neuroml')
        cellType = domain.get('cell_type', None)
        if not cellType:
            raise LookupError("In MuMBL, no cell_type is specified")
        segment = domain.get('segment')
        id = domain.get('instance_id')
        path = os.path.join(self.cellPath, cellType, segment)

        fullpath = moose_methods.moosePath(path, id)

        if domain.get('postfix', None) is not None:
            fullpath = os.path.join(fullpath, domain.get('postfix'))

        mappings = domain.findall('mapping')
        [self.mapping(a, fullpath) for a in mappings]

    def mapping(self, adaptor, moosePath):
        """
        Set up an adaptor for a given _moose path
        """
        direction = adaptor.get('direction')
        if direction is None:
            direction = 'out'
        else: pass
        if direction == "in":
            srcs = adaptor.findall('source')
            [self.inTarget(s, moosePath) for s in srcs]
        elif direction == "out":
            tgts = adaptor.findall('target')
            [self.outTarget(t, moosePath) for t in tgts]
        else:
            raise UserWarning, "Unsupported type or parameter", direction

    def inTarget(self, src, moosePath):
        """Set up incoming source and target.
        """
        try:
            _mooseSrc = _moose.Neutral(moosePath)
        except Exception as e:
            print_utils.dump("ERROR"
                    , [ "Source compartment %s is not found" % moosePath
                        , '%s' % e
                        ]
                    , frame = inspect.currentframe()
                    )
            print(moose_methods.dumpMatchingPaths(moosePath))
            sys.exit(-1)

        # Get the target.
        compType = src.get('type', 'chemical')
        if compType == "chemical":
            # in which compartment and which type of pool
            compId = src.get('compartment_id', None)
            if compId is None:
                raise UserWarning, "Missing parameter or value: compartment_id"
            species = src.get('species')
            assert species, "Must have species in <source>"
            poolId = os.path.join(self.compartmentName, compId)
            poolPath = os.path.join(self.chemPath, poolId, species)
            try:
                _mooseTgt = _moose.Pool(poolPath)
            except Exception as e:
                print_utils.dump("ERROR"
                        , [  "%s" % e
                            , "Given path is {}".format(poolPath) 
                            , " Doing nothing ... " 
                            ]
                        , frame = inspect.currentframe()
                        )
                sys.exit(0)
        else: 
            print_utils.dump("TODO", "Unsupported compartment type")
            raise UserWarning("Unsupported compartment type %s" % compType)
       
        # We need to send message now based on relation. This is done using
        # _moose.Adaptor class.
        relationXml = src.find('relation')
        # Here we modify the _mooseSrc according to _mooseTgt. MooseTgt is read
        # therefore it is the first arguement in function.
        self.setAdaptor(_mooseTgt, _mooseSrc, relationXml)
        _moose.reinit()

    
    def outTarget(self, tgt, moosePath):
        """Setup outgoing targets.
        """
        self.initPaths(moosePath, recursively=False)
        _mooseSrc = _moose.Neutral(moosePath)

        # Get the source
        compType = tgt.get('type', 'chemical')
        if compType != 'chemical':
            print_utils.dump("TODO"
                    , "Unsupported compartment type %s" % compType
                    )
            raise UserWarning("Unsupported feature")
        

        compId = tgt.get('compartment_id')
        if compId is None:
            raise UserWarning, "Missing parameter or value: compartment_id"

        assert int(compId) >= 0, "Not a valid compartment id: %s" % compId

        species = tgt.get('species')
        assert species, "Must have species in target"
        
        poolId = os.path.join(self.compartmentName, compId)
        poolPath = os.path.join(self.chemPath, poolId, species)
        try:
            _mooseTgt = _moose.Pool(poolPath)
        except Exception as e:
            print(moose_methods.dumpMatchingPaths(poolPath))
            print_utils.dump("ERROR"
                    , [ "Perhaps the compartment_id in mumbleML is wrong. "
                        , "Failed to create a pool with path `%s`" % poolPath 
                        , "Check the path list above "
                      ]
                    , frame = inspect.currentframe()
                    )
            sys.exit()

        relationXml = tgt.find('relation')
        self.setAdaptor(_mooseSrc, _mooseTgt, relationXml)
        _moose.useClock(0, '/mumble/##', 'process')

        _moose.reinit()


    def setAdaptor(self, src, tgt, relationXml):
        '''
        Construct a adaptor which sends message from src to tgt

        src: It is the _moose compartment which is read by _moose.
        tgt: It is the _moose compartment to which we write.
        '''

        assert src, "Source compartment is none"
        assert tgt, "Target compartment is none"
        self.adaptorCount += 1
        srcPath = src.path
        tgtPath = tgt.path
        adaptorPath = os.path.join(self.adaptorPath
                , 'adapt{}'.format(self.adaptorCount)
                )
        adaptor = _moose.Adaptor(adaptorPath)

        inputVar = relationXml.get('input')
        outputVar = relationXml.get('output')
        assert inputVar
        assert outputVar

        scale = relationXml.find('scale').text
        if not scale: scale = 1.0
        else: scale = float(scale)
        
        offset = relationXml.find('offset').text
        if not offset: offset = 0.0
        else: offset = float(offset)

        adaptor.setField('scale', scale)
        adaptor.setField('inputOffset', - offset)
        print_utils.dump("STEP"
                , 'Setting adaptor between {},`{}` and {},`{}`'.format(
                    src.path
                    , inputVar
                    , tgt.path
                    , outputVar
                    )
                )
        # Connect
        var = self.prefixWithGet(inputVar)
        try:
            _moose.connect(adaptor, 'requestField', src, var)
        except Exception as e:
            print_utils.dump("ERROR"
                    , [ 'Failed to connect var {} of {} with adaptor input'.format(
                        var, src.path)
                        , '%s' % e
                        ]
            
                    , frame = inspect.currentframe() 
                    )
            print_utils.dump("INFO"
                    , "Avalilable fields are"
                    )
            _moose.showfields(src)
            sys.exit()

        var = self.prefixWithSet(outputVar)
        try:
            _moose.connect(adaptor, 'output', tgt, var)
        except Exception as e:
            print_utils.dump("ERROR"
                    , 'Failed to connect var {} of {} with adaptor input'.format(
                        var, tgt.path
                        )
                    , frame = inspect.currentframe()
                    )
            print_utils.dump("INFO"
                    , "Avalilable fields are {}".format(_moose.showfield(tgt))
                    )
            sys.exit()
        _moose.useClock(self.mumbleClockId, adaptor.path, 'process')

    def exit(self, dump=True):
        sys.exit(0)

    def dumpMessages(self, root):
        """Dump the messages in _moose """
        msgs = _moose.wildcardFind('/Msgs/##')
