#########################################################################
## rdesigneur0_4.py ---
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2014 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU General Public License version 2 or later.
## See the file COPYING.LIB for the full notice.
#########################################################################

##########################################################################
## This class builds models of
## Reaction-Diffusion and Electrical SIGnaling in NEURons.
## It loads in neuronal and chemical signaling models and embeds the
## latter in the former, including mapping entities like calcium and
## channel conductances, between them.
##########################################################################
import imp
import os
import moose
import numpy as np
import math

from moose.neuroml.NeuroML import NeuroML
from moose.neuroml.ChannelML import ChannelML

#EREST_ACT = -70e-3
NA = 6.022e23
PI = 3.14159265359
FaradayConst = 96485.3365 # Coulomb/mol

class BuildError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#######################################################################

class rdesigneur:
    """The rdesigneur class is used to build models incorporating
    reaction-diffusion and electrical signaling in neurons.
    Params:
        useGssa: True/False               for GSSA in spine and PSD
        combineSegments: True/False       for NeuroML models
        meshLambda: default 2e-6
        adaptCa: [( Ca_wildcard_string, chem_wildcard_string, offset, scale ),...]
        adaptChem: [( Chem_wildcard_string, elec_wildcard_string, offset, scale ),...]

    I need to put the extra channels now into the NeuroML definition.
    """
    ################################################################
    def __init__(self, \
            modelPath = '/model', \
            useGssa = True, \
            combineSegments = True, \
            stealCellFromLibrary = False, \
            meshLambda= 2e-6, \
            temperature = 32, \
            chemDt= 0.001, \
            diffDt= 0.001, \
            elecDt= 50e-6, \
            cellProto = [], \
            spineProto = [], \
            chanProto = [], \
            chemProto = [], \
            passiveDistrib= [], \
            spineDistrib= [], \
            chanDistrib = [], \
            chemDistrib = [], \
            adaptorList= [] \
        ):
        """ Constructor of the rdesigner. This just sets up internal fields
            for the model building, it doesn't actually create any objects.
        """

        self.useGssa = useGssa
        self.combineSegments = combineSegments
        self.stealCellFromLibrary = stealCellFromLibrary
        self.meshLambda= meshLambda
        self.temperature = temperature
        self.chemDt= chemDt
        self.diffDt= diffDt
        self.elecDt= elecDt

        self.cellProtoList = cellProto
        self.spineProtoList = spineProto
        self.chanProtoList = chanProto
        self.chemProtoList = chemProto

        self.passiveDistrib = passiveDistrib
        self.spineDistrib = spineDistrib
        self.chanDistrib = chanDistrib
        self.chemDistrib = chemDistrib

        self.adaptorList = adaptorList
        self.cellPortionElist = []
        self.spineComptElist = []


    ################################################################
    def _printModelStats( self ):
        print "Rdesigneur: Elec model has", \
            self.elecid.numCompartments, "compartments and", \
            self.elecid.numSpines, "spines on", \
            len( self.cellPortionElist ), "compartments."
        if hasattr( self , 'chemid' ):
            dmstoich = moose.element( self.dendCompt.path + '/stoich' )
            smstoich = moose.element( self.spineCompt.path + '/stoich' )
            pmstoich = moose.element( self.psdCompt.path + '/stoich' )
            print "Chem part of model has ", \
                self.dendCompt.mesh.num, "dendrite voxels X", \
                dmstoich.numAllPools, "pools,\n    ", \
                self.spineCompt.mesh.num, "spine voxels X", \
                smstoich.numAllPools, "pools,", \
                self.psdCompt.mesh.num, "psd voxels X", \
                pmstoich.numAllPools, "pools."

    def buildModel( self, modelPath ):
        if not moose.exists( '/library' ):
            library = moose.Neutral( '/library' )
        if moose.exists( modelPath ):
            print "rdesigneur::buildModel: Build failed. Model '", \
                modelPath, "' already exists."
            return
        self.model = moose.Neutral( modelPath )
        try:
            self.buildCellProto()
            self.buildChanProto()
            self.buildSpineProto()
            self.buildChemProto()
            # Protos made. Now install the elec and chem protos on model.
            self.installCellFromProtos()
            # Now assign all the distributions
            self.buildPassiveDistrib()
            self.buildChanDistrib()
            self.buildSpineDistrib()
            self.buildChemDistrib()
            self._configureSolvers()
            self.buildAdaptors()
            self._configureClocks()
            self._printModelStats()

        except BuildError, msg:
            print "Error: rdesigneur: model build failed: ", msg
            moose.delete( self.model )

    def installCellFromProtos( self ):
        if self.stealCellFromLibrary:
            moose.move( self.elecid, self.model )
            if self.elecid.name != 'elec':
                self.elecid.name = 'elec'
        else:
            moose.copy( self.elecid, self.model, 'elec' )
            self.elecid = moose.element( self.model.path + '/elec' )
            self.elecid.buildSegmentTree() # rebuild: copy has happened.
        if hasattr( self, 'chemid' ):
            self.validateChem()
            if self.stealCellFromLibrary:
                moose.move( self.chemid, self.model )
                if self.chemid.name != 'chem':
                    self.chemid.name = 'chem'
            else:
                moose.copy( self.chemid, self.model, 'chem' )
                self.chemid = moose.element( self.model.path + '/chem' )

        ep = self.elecid.path
        somaList = moose.wildcardFind( ep + '/#oma#[ISA=CompartmentBase]' )
        if len( somaList ) == 0:
            somaList = moose.wildcardFind( ep + '/#[ISA=CompartmentBase]' )
        if len( somaList ) == 0:
            raise BuildError( "installCellFromProto: No soma found" )
        maxdia = 0.0
        for i in somaList:
            if ( i.diameter > maxdia ):
                self.soma = i

    ################################################################
    # Some utility functions for building prototypes.
    ################################################################
    # Return true if it is a function.
    def buildProtoFromFunction( self, func, protoName ):
        bracePos = func.find( '()' )
        if bracePos == -1:
            return False

        modPos = func.find( "." )
        if ( modPos != -1 ): # Function is in a file, load and check
            pathTokens = func[0:modPos].split('/')
            pathTokens = ['/'] + pathTokens
            modulePath = os.path.join(*pathTokens[:-1])
            moduleName = pathTokens[-1]
            funcName = func[modPos+1:bracePos]
            moduleFile, pathName, description = imp.find_module(moduleName, [modulePath])
            try:
                module = imp.load_module(moduleName, moduleFile, pathName, description)
                funcObj = getattr(module, funcName)
                funcObj(protoName)
                return True
            finally:
                moduleFile.close()
            return False
        if not func[0:bracePos] in globals():
            raise BuildError( \
                protoName + "Proto: global function '" +func+"' not known.")
        globals().get( func[0:bracePos] )( protoName )
        return True

    # Class or file options. True if extension is found in
    def isKnownClassOrFile( self, name, suffices ):
        for i in suffices:
            if name.rfind( '.'+i ) >= 0 :
                return True
        return False



    # Checks all protos, builds them and return true. If it was a file
    # then it has to return false and invite the calling function to build
    # If it fails then the exception takes over.
    def checkAndBuildProto( self, protoType, protoVec, knownClasses, knownFileTypes ):
        if len(protoVec) != 2:
            raise BuildError( \
                protoType + "Proto: nargs should be 2, is " + \
                    str( len(protoVec)  ))
        if moose.exists( '/library/' + protoVec[1] ):
            # Assume the job is already done, just skip it.
            return True
            '''
            raise BuildError( \
                protoType + "Proto: object /library/" + \
                    protoVec[1] + " already exists." )
            '''
        # Check and build the proto from a class name
        if protoVec[0][:5] == 'moose':
            protoName = protoVec[0][6:]
            if self.isKnownClassOrFile( protoName, knownClasses ):
                try:
                    getAttr( moose, protoName )( '/library/' + protoVec[1] )
                except AttributeError:
                    raise BuildError( protoType + "Proto: Moose class '" \
                            + protoVec[0] + "' not found." )
                return True

        if self.buildProtoFromFunction( protoVec[0], protoVec[1] ):
            return True
        # Maybe the proto is already in memory
        # Avoid relative file paths going toward root
        if protoVec[0][:3] != "../":
            if moose.exists( protoVec[0] ):
                moose.copy( protoVec[0], '/library/' + protoVec[1] )
                return True
            if moose.exists( '/library/' + protoVec[0] ):
                #moose.copy('/library/' + protoVec[0], '/library/', protoVec[1])
                print 'renaming /library/' + protoVec[0] + ' to ' + protoVec[1]
                moose.element( '/library/' + protoVec[0]).name = protoVec[1]
                #moose.le( '/library' )
                return True
        # Check if there is a matching suffix for file type.
        if self.isKnownClassOrFile( protoVec[0], knownFileTypes ):
            return False
        else:
            raise BuildError( \
                protoType + "Proto: File type '" + protoVec[0] + \
                "' not known." )
        return True

    ################################################################
    # Here are the functions to build the type-specific prototypes.
    ################################################################
    def buildCellProto( self ):
        for i in self.cellProtoList:
            if self.checkAndBuildProto( "cell", i, \
                ["Compartment", "SymCompartment"], ["swc", "p", "nml", "xml"] ):
                self.elecid = moose.element( '/library/' + i[1] )
            else:
                self._loadElec( i[0], i[1] )
            self.elecid.buildSegmentTree()

    def buildSpineProto( self ):
        for i in self.spineProtoList:
            if not self.checkAndBuildProto( "spine", i, \
                ["Compartment", "SymCompartment"], ["swc", "p", "nml", "xml"] ):
                self._loadElec( i[0], i[1] )

    def parseChanName( self, name ):
        if name[-4:] == ".xml":
            period = name.rfind( '.' )
            slash = name.rfind( '/' )
            if ( slash >= period ):
                raise BuildError( "chanProto: bad filename:" + i[0] )
            if ( slash < 0 ):
                return name[:period]
            else:
                return name[slash+1:period]

    def buildChanProto( self ):
        for i in self.chanProtoList:
            if len(i) == 1:
                chanName = self.parseChanName( i[0] )
            else:
                chanName = i[1]
            j = [i[0], chanName]
            if not self.checkAndBuildProto( "chan", j, [], ["xml"] ):
                cm = ChannelML( {'temperature': self.temperature} )
                cm.readChannelMLFromFile( i[0] )
                if ( len( i ) == 2 ):
                    chan = moose.element( '/library/' + chanName )
                    chan.name = i[1]

    def buildChemProto( self ):
        for i in self.chemProtoList:
            if not self.checkAndBuildProto( "chem", i, \
                ["Pool"], ["g", "sbml", "xml" ] ):
                self._loadChem( i[0], i[1] )
            self.chemid = moose.element( '/library/' + i[1] )

    ################################################################
    # Here we set up the distributions
    ################################################################
    def buildPassiveDistrib( self ):
        temp = []
        for i in self.passiveDistrib:
            temp.extend( i )
            temp.extend( [""] )
        self.elecid.passiveDistribution = temp

    def buildChanDistrib( self ):
        temp = []
        for i in self.chanDistrib:
            temp.extend( i )
            temp.extend( [""] )
        self.elecid.channelDistribution = temp

    def buildSpineDistrib( self ):
        temp = []
        for i in self.spineDistrib:
            temp.extend( i )
            temp.extend( [""] )
        self.elecid.spineDistribution = temp

    def buildChemDistrib( self ):
        for i in self.chemDistrib:
            pair = i[1] + " " + i[3]
            # Assign any other params. Possibly the first param should
            # be a global scaling factor.
            self.cellPortionElist = self.elecid.compartmentsFromExpression[ pair ]
            self.spineComptElist = self.elecid.spinesFromExpression[ pair ]
            if len( self.cellPortionElist ) == 0:
                raise BuildError( \
                    "buildChemDistrib: No elec compartments found in path: '" \
                        + pair + "'" )
            if len( self.spineComptElist ) == 0:
                raise BuildError( \
                    "buildChemDistrib: No spine compartments found in path: '" \
                        + pair + "'" )
            # Build the neuroMesh
            # Check if it is good. Need to catch the ValueError here.
            self._buildNeuroMesh()
            # Assign the solvers

    ################################################################
    # Here we set up the adaptors
    ################################################################
    def findMeshOnName( self, name ):
        pos = name.find( '/' )
        if ( pos != -1 ):
            temp = name[:pos]
            if temp == 'psd' or temp == 'spine' or temp == 'dend':
                return ( temp, name[pos+1:] )
        return ("","")


    def buildAdaptors( self ):
        for i in self.adaptorList:
            mesh, name = self.findMeshOnName( i[0] )
            if mesh == "":
                mesh, name = self.findMeshOnName( i[2] )
                if  mesh == "":
                    raise BuildError( "buildAdaptors: Failed for " + i[2] )
                self._buildAdaptor( mesh, i[0], i[1], name, i[3], True, i[4], i[5] )
            else:
                self._buildAdaptor( mesh, i[2], i[3], name, i[1], False, i[4], i[5] )


    ################################################################
    # Utility function for setting up clocks.
    def _configureClocks( self ):
        for i in range( 0, 8 ):
            moose.setClock( i, self.elecDt )
        moose.setClock( 10, self.diffDt )
        for i in range( 11, 18 ):
            moose.setClock( i, self.chemDt )
        moose.setClock( 18, self.chemDt * 5.0 )
        hsolve = moose.HSolve( self.elecid.path + '/hsolve' )
        hsolve.dt = self.elecDt
        hsolve.target = self.soma.path
    ################################################################
    ################################################################
    ################################################################

    def validateFromMemory( self, epath, cpath ):
        ret = self.validateChem()
        return ret

    #################################################################
    # assumes ePath is the parent element of the electrical model,
    # and cPath the parent element of the compts in the chem model
    def buildFromMemory( self, ePath, cPath, doCopy = False ):
        if not self.validateFromMemory( ePath, cPath ):
            return
        if doCopy:
            x = moose.copy( cPath, self.model )
            self.chemid = moose.element( x )
            self.chemid.name = 'chem'
            x = moose.copy( ePath, self.model )
            self.elecid = moose.element( x )
            self.elecid.name = 'elec'
        else:
            self.elecid = moose.element( ePath )
            self.chemid = moose.element( cPath )
            if self.elecid.path != self.model.path + '/elec':
                if ( self.elecid.parent != self.model ):
                    moose.move( self.elecid, self.model )
                self.elecid.name = 'elec'
            if self.chemid.path != self.model.path + '/chem':
                if ( self.chemid.parent != self.model ):
                    moose.move( self.chemid, self.model )
                self.chemid.name = 'chem'


        ep = self.elecid.path
        somaList = moose.wildcardFind( ep + '/#oma#[ISA=CompartmentBase]' )
        if len( somaList ) == 0:
            somaList = moose.wildcardFind( ep + '/#[ISA=CompartmentBase]' )
        assert( len( somaList ) > 0 )
        maxdia = 0.0
        for i in somaList:
            if ( i.diameter > maxdia ):
                self.soma = i
        #self.soma = self.comptList[0]
        self._decorateWithSpines()
        self.spineList = moose.wildcardFind( ep + '/#spine#[ISA=CompartmentBase],' + ep + '/#head#[ISA=CompartmentBase]' )
        if len( self.spineList ) == 0:
            self.spineList = moose.wildcardFind( ep + '/#head#[ISA=CompartmentBase]' )
        nmdarList = moose.wildcardFind( ep + '/##[ISA=NMDAChan]' )

        self.comptList = moose.wildcardFind( ep + '/#[ISA=CompartmentBase]')
        print "Rdesigneur: Elec model has ", len( self.comptList ), \
            " compartments and ", len( self.spineList ), \
            " spines with ", len( nmdarList ), " NMDARs"


        self._buildNeuroMesh()


        self._configureSolvers()
        for i in self.adaptorList:
            print i
            self._buildAdaptor( i[0],i[1],i[2],i[3],i[4],i[5],i[6] )

    ################################################################

    def buildFromFile( self, efile, cfile ):
        self.efile = efile
        self.cfile = cfile
        self._loadElec( efile, 'tempelec' )
        if len( self.chanDistrib ) > 0:
            self.elecid.channelDistribution = self.chanDistrib
            self.elecid.parseChanDistrib()
        self._loadChem( cfile, 'tempchem' )
        self.buildFromMemory( self.model.path + '/tempelec', self.model.path + '/tempchem' )

    ################################################################
    # Utility function to add a single spine to the given parent.

    # parent is parent compartment for this spine.
    # spineProto is just that.
    # pos is position (in metres ) along parent compartment
    # angle is angle (in radians) to rotate spine wrt x in plane xy.
    # Size is size scaling factor, 1 leaves as is.
    # x, y, z are unit vectors. Z is along the parent compt.
    # We first shift the spine over so that it is offset by the parent compt
    # diameter.
    # We then need to reorient the spine which lies along (i,0,0) to
    #   lie along x. X is a unit vector so this is done simply by
    #   multiplying each coord of the spine by x.
    # Finally we rotate the spine around the z axis by the specified angle
    # k is index of this spine.
    def _addSpine( self, parent, spineProto, pos, angle, x, y, z, size, k ):
        spine = moose.copy( spineProto, parent.parent, 'spine' + str(k) )
        kids = spine[0].children
        coords = []
        ppos = np.array( [parent.x0, parent.y0, parent.z0] )
        for i in kids:
            #print i.name, k
            j = i[0]
            j.name += str(k)
            #print 'j = ', j
            coords.append( [j.x0, j.y0, j.z0] )
            coords.append( [j.x, j.y, j.z] )
            self._scaleSpineCompt( j, size )
            moose.move( i, self.elecid )
        origin = coords[0]
        #print 'coords = ', coords
        # Offset it so shaft starts from surface of parent cylinder
        origin[0] -= parent.diameter / 2.0
        coords = np.array( coords )
        coords -= origin # place spine shaft base at origin.
        rot = np.array( [x, [0,0,0], [0,0,0]] )
        coords = np.dot( coords, rot )
        moose.delete( spine )
        moose.connect( parent, "raxial", kids[0], "axial" )
        self._reorientSpine( kids, coords, ppos, pos, size, angle, x, y, z )

    ################################################################
    ## The spineid is the parent object of the prototype spine. The
    ## spine prototype can include any number of compartments, and each
    ## can have any number of voltage and ligand-gated channels, as well
    ## as CaConc and other mechanisms.
    ## The parentList is a list of Object Ids for parent compartments for
    ## the new spines
    ## The spacingDistrib is the width of a normal distribution around
    ## the spacing. Both are in metre units.
    ## The reference angle of 0 radians is facing away from the soma.
    ## In all cases we assume that the spine will be rotated so that its
    ## axis is perpendicular to the axis of the dendrite.
    ## The simplest way to put the spine in any random position is to have
    ## an angleDistrib of 2 pi. The algorithm selects any angle in the
    ## linear range of the angle distrib to add to the specified angle.
    ## With each position along the dendrite the algorithm computes a new
    ## spine direction, using rotation to increment the angle.
    ################################################################
    def _decorateWithSpines( self ):
        args = []
        for i in self.addSpineList:
            if not moose.exists( '/library/' + i[0] ):
                print 'Warning: _decorateWithSpines: spine proto ', i[0], ' not found.'
                continue
            s = ""
            for j in range( 9 ):
                s = s + str(i[j]) + ' '
            args.append( s )
        self.elecid.spineSpecification = args
        self.elecid.parseSpines()

    ################################################################

    def _loadElec( self, efile, elecname ):
        if ( efile[ len( efile ) - 2:] == ".p" ):
            self.elecid = moose.loadModel( efile, '/library/' + elecname)[0]
            print self.elecid
        elif ( efile[ len( efile ) - 4:] == ".swc" ):
            self.elecid = moose.loadModel( efile, '/library/' + elecname)[0]
        else:
            nm = NeuroML()
            print "in _loadElec, combineSegments = ", self.combineSegments
            nm.readNeuroMLFromFile( efile, \
                    params = {'combineSegments': self.combineSegments, \
                    'createPotentialSynapses': True } )
            if moose.exists( '/cells' ):
                kids = moose.wildcardFind( '/cells/#' )
            else:
                kids = moose.wildcardFind( '/library/#[ISA=Neuron],/library/#[TYPE=Neutral]' )
                if ( kids[0].name == 'spine' ):
                    kids = kids[1:]

            assert( len( kids ) > 0 )
            self.elecid = kids[0]
            temp = moose.wildcardFind( self.elecid.path + '/#[ISA=CompartmentBase]' )

        transformNMDAR( self.elecid.path )
        kids = moose.wildcardFind( '/library/##[0]' )
        for i in kids:
            i.tick = -1


    #################################################################

    # This assumes that the chemid is located in self.parent.path+/chem
    # It moves the existing chem compartments into a NeuroMesh
    # For now this requires that we have a dend, a spine and a PSD,
    # with those names and volumes in decreasing order.
    def validateChem( self  ):
        cpath = self.chemid.path
        comptlist = moose.wildcardFind( cpath + '/#[ISA=ChemCompt]' )
        if len( comptlist ) == 0:
            raise BuildError( "validateChem: no compartment on: " + cpath )

        # Sort comptlist in decreasing order of volume
        sortedComptlist = sorted( comptlist, key=lambda x: -x.volume )
        if ( len( sortedComptlist ) != 3 ):
            print cpath, sortedComptlist
            raise BuildError( "validateChem: Require 3 chem compartments, have: " + str( len( sortedComptlist ) ) )
        if not( sortedComptlist[0].name.lower() == 'dend' and \
            sortedComptlist[1].name.lower() == 'spine' and \
            sortedComptlist[2].name.lower() == 'psd' ):
            raise BuildError( "validateChem: Invalid compt names: require dend, spine and PSD.\nActual names = " \
                    + sortedComptList[0].name + ", " \
                    + sortedComptList[1].name + ", " \
                    + sortedComptList[2].name )

    #################################################################

    def _buildNeuroMesh( self ):
        comptlist = moose.wildcardFind( self.chemid.path + '/#[ISA=ChemCompt]' )
        sortedComptList = sorted( comptlist, key=lambda x: -x.volume )
        # A little juggling here to put the chem pathways onto new meshes.
        self.chemid.name = 'temp_chem'
        newChemid = moose.Neutral( self.model.path + '/chem' )
        self.dendCompt = moose.NeuroMesh( newChemid.path + '/dend' )
        self.dendCompt.separateSpines = 1
        self.dendCompt.geometryPolicy = 'cylinder'
        self.spineCompt = moose.SpineMesh( newChemid.path + '/spine' )
        moose.connect( self.dendCompt, 'spineListOut', self.spineCompt, 'spineList' )
        self.psdCompt = moose.PsdMesh( newChemid.path + '/psd' )
        moose.connect( self.dendCompt, 'psdListOut', self.psdCompt, 'psdList','OneToOne')
        #Move the old reac systems onto the new compartments.
        self._moveCompt( sortedComptList[0], self.dendCompt )
        self._moveCompt( sortedComptList[1], self.spineCompt )
        self._moveCompt( sortedComptList[2], self.psdCompt )
        self.dendCompt.diffLength = self.meshLambda
        self.dendCompt.subTree = self.cellPortionElist
        moose.delete( self.chemid )
        self.chemid = newChemid

    #################################################################
    def _configureSolvers( self ) :
        if not hasattr( self, 'chemid' ):
            return
        if not hasattr( self, 'dendCompt' ):
            raise BuildError( "configureSolvers: no chem meshes defined." )
        dmksolve = moose.Ksolve( self.dendCompt.path + '/ksolve' )
        dmdsolve = moose.Dsolve( self.dendCompt.path + '/dsolve' )
        dmstoich = moose.Stoich( self.dendCompt.path + '/stoich' )
        dmstoich.compartment = self.dendCompt
        dmstoich.ksolve = dmksolve
        dmstoich.dsolve = dmdsolve
        dmstoich.path = self.dendCompt.path + "/##"
        # Put in spine solvers. Note that these get info from the dendCompt
        if self.useGssa:
            smksolve = moose.Gsolve( self.spineCompt.path + '/ksolve' )
        else:
            smksolve = moose.Ksolve( self.spineCompt.path + '/ksolve' )
        smdsolve = moose.Dsolve( self.spineCompt.path + '/dsolve' )
        smstoich = moose.Stoich( self.spineCompt.path + '/stoich' )
        smstoich.compartment = self.spineCompt
        smstoich.ksolve = smksolve
        smstoich.dsolve = smdsolve
        smstoich.path = self.spineCompt.path + "/##"
        # Put in PSD solvers. Note that these get info from the dendCompt
        if self.useGssa:
            pmksolve = moose.Gsolve( self.psdCompt.path + '/ksolve' )
        else:
            pmksolve = moose.Ksolve( self.psdCompt.path + '/ksolve' )
        pmdsolve = moose.Dsolve( self.psdCompt.path + '/dsolve' )
        pmstoich = moose.Stoich( self.psdCompt.path + '/stoich' )
        pmstoich.compartment = self.psdCompt
        pmstoich.ksolve = pmksolve
        pmstoich.dsolve = pmdsolve
        pmstoich.path = self.psdCompt.path + "/##"

        # Put in cross-compartment diffusion between ksolvers
        dmdsolve.buildNeuroMeshJunctions( smdsolve, pmdsolve )
        # Put in cross-compartment reactions between ksolvers
        smstoich.buildXreacs( pmstoich )
        #pmstoich.buildXreacs( smstoich )
        smstoich.buildXreacs( dmstoich )
        dmstoich.filterXreacs()
        smstoich.filterXreacs()
        pmstoich.filterXreacs()

        # set up the connections so that the spine volume scaling can happen
        self.elecid.setSpineAndPsdMesh( self.spineCompt, self.psdCompt )
        self.elecid.setSpineAndPsdDsolve( smdsolve, pmdsolve )
    ################################################################

    def _loadChem( self, fname, chemName ):
        chem = moose.Neutral( '/library/' + chemName )
        modelId = moose.loadModel( fname, chem.path, 'ee' )
        comptlist = moose.wildcardFind( chem.path + '/#[ISA=ChemCompt]' )
        if len( comptlist ) == 0:
            print "loadChem: No compartment found in file: ", fname
            return
        # Sort comptlist in decreasing order of volume
        sortedComptlist = sorted( comptlist, key=lambda x: -x.volume )
        if ( len( sortedComptlist ) != 3 ):
            print "loadChem: Require 3 chem compartments, have: ",\
                len( sortedComptlist )
            return False
        sortedComptlist[0].name = 'dend'
        sortedComptlist[1].name = 'spine'
        sortedComptlist[2].name = 'psd'

    ################################################################

    def _moveCompt( self, a, b ):
        b.setVolumeNotRates( a.volume )
        for i in moose.wildcardFind( a.path + '/#' ):
            if ( i.name != 'mesh' ):
                moose.move( i, b )
        moose.delete( a )
    ################################################################
    def _buildAdaptor( self, meshName, elecRelPath, elecField, \
            chemRelPath, chemField, isElecToChem, offset, scale ):
        mesh = moose.element( '/model/chem/' + meshName )
        #elecComptList = mesh.elecComptList
        if elecRelPath == 'spine':
            elecComptList = moose.vec( mesh.elecComptList[0].path + '/../spine' )
        else:
            elecComptList = mesh.elecComptList

        '''
        for i in elecComptList:
            print i.diameter
        print len( elecComptList[0] )
        print elecComptList[0][0].parent.path
        print "--------------------------------------"
        spine = moose.vec( elecComptList[0].path + '/../spine' )
        for i in spine:
            print i.headDiameter

        moose.le( elecComptList[0][0].parent )
        '''
        if len( elecComptList ) == 0:
            raise BuildError( \
                "buildAdaptor: no elec compts in elecComptList on: " + \
                mesh.path )
        startVoxelInCompt = mesh.startVoxelInCompt
        endVoxelInCompt = mesh.endVoxelInCompt
        capField = elecField[0].capitalize() + elecField[1:]
        capChemField = chemField[0].capitalize() + chemField[1:]
        chemPath = mesh.path + '/' + chemRelPath
        if not( moose.exists( chemPath ) ):
            raise BuildError( \
                "Error: buildAdaptor: no chem obj in " + chemPath )
        chemObj = moose.element( chemPath )
        assert( chemObj.numData >= len( elecComptList ) )
        adName = '/adapt'
        for i in range( 1, len( elecRelPath ) ):
            if ( elecRelPath[-i] == '/' ):
                adName += elecRelPath[1-i]
                break
        ad = moose.Adaptor( chemObj.path + adName, len( elecComptList ) )
        print 'building ', len( elecComptList ), 'adaptors ', adName, \
               ' for: ', mesh.name, elecRelPath, elecField, chemRelPath
        av = ad.vec
        chemVec = moose.element( mesh.path + '/' + chemRelPath ).vec

        for i in zip( elecComptList, startVoxelInCompt, endVoxelInCompt, av ):
            i[3].inputOffset = 0.0
            i[3].outputOffset = offset
            i[3].scale = scale
            if elecRelPath == 'spine':
                elObj = i[0]
            else:
                ePath = i[0].path + '/' + elecRelPath
                if not( moose.exists( ePath ) ):
                    raise BuildError( \
                        "Error: buildAdaptor: no elec obj in " + ePath )
                elObj = moose.element( i[0].path + '/' + elecRelPath )
            if ( isElecToChem ):
                elecFieldSrc = 'get' + capField
                chemFieldDest = 'set' + capChemField
                #print ePath, elecFieldSrc, scale
                moose.connect( i[3], 'requestOut', elObj, elecFieldSrc )
                for j in range( i[1], i[2] ):
                    moose.connect( i[3], 'output', chemVec[j],chemFieldDest)
            else:
                chemFieldSrc = 'get' + capChemField
                elecFieldDest = 'set' + capField
                for j in range( i[1], i[2] ):
                    moose.connect( i[3], 'requestOut', chemVec[j], chemFieldSrc)
                msg = moose.connect( i[3], 'output', elObj, elecFieldDest )

    #################################################################
    # Here we have a series of utility functions for building cell
    # prototypes.
    #################################################################
def transformNMDAR( path ):
    for i in moose.wildcardFind( path + "/##/#NMDA#[ISA!=NMDAChan]" ):
        chanpath = i.path
        pa = i.parent
        i.name = '_temp'
        if ( chanpath[-3:] == "[0]" ):
            chanpath = chanpath[:-3]
        nmdar = moose.NMDAChan( chanpath )
        sh = moose.SimpleSynHandler( chanpath + '/sh' )
        moose.connect( sh, 'activationOut', nmdar, 'activation' )
        sh.numSynapses = 1
        sh.synapse[0].weight = 1
        nmdar.Ek = i.Ek
        nmdar.tau1 = i.tau1
        nmdar.tau2 = i.tau2
        nmdar.Gbar = i.Gbar
        nmdar.CMg = 12
        nmdar.KMg_A = 1.0 / 0.28
        nmdar.KMg_B = 1.0 / 62
        nmdar.temperature = 300
        nmdar.extCa = 1.5
        nmdar.intCa = 0.00008
        nmdar.intCaScale = 1
        nmdar.intCaOffset = 0.00008
        nmdar.condFraction = 0.02
        moose.delete( i )
        moose.connect( pa, 'channel', nmdar, 'channel' )
        caconc = moose.wildcardFind( pa.path + '/#[ISA=CaConcBase]' )
        if ( len( caconc ) < 1 ):
            print 'no caconcs found on ', pa.path
        else:
            moose.connect( nmdar, 'ICaOut', caconc[0], 'current' )
            moose.connect( caconc[0], 'concOut', nmdar, 'assignIntCa' )
    ################################################################
    # Utility function for building a compartment, used for spines.
def buildCompt( pa, name, length, dia, xoffset, RM, RA, CM ):
    compt = moose.Compartment( pa.path + '/' + name )
    compt.x0 = xoffset
    compt.y0 = 0
    compt.z0 = 0
    compt.x = length + xoffset
    compt.y = 0
    compt.z = 0
    compt.diameter = dia
    compt.length = length
    xa = dia * dia * PI / 4.0
    sa = length * dia * PI
    compt.Ra = length * RA / xa
    compt.Rm = RM / sa
    compt.Cm = CM * sa
    return compt

    ################################################################
    # Utility function for building a synapse, used for spines.
def buildSyn( name, compt, Ek, tau1, tau2, Gbar, CM ):
    syn = moose.SynChan( compt.path + '/' + name )
    syn.Ek = Ek
    syn.tau1 = tau1
    syn.tau2 = tau2
    syn.Gbar = Gbar * compt.Cm / CM
    #print "BUILD SYN: ", name, Gbar, syn.Gbar, CM
    moose.connect( compt, 'channel', syn, 'channel' )
    sh = moose.SimpleSynHandler( syn.path + '/sh' )
    moose.connect( sh, 'activationOut', syn, 'activation' )
    sh.numSynapses = 1
    sh.synapse[0].weight = 1
    return syn

######################################################################
# Utility function, borrowed from proto18.py, for making an LCa channel.
# Based on Traub's 91 model, I believe.
def make_LCa():
        EREST_ACT = -0.060 #/* hippocampal cell resting potl */
        ECA = 0.140 + EREST_ACT #// 0.080
	if moose.exists( 'LCa' ):
		return
	Ca = moose.HHChannel( 'LCa' )
	Ca.Ek = ECA
	Ca.Gbar = 0
	Ca.Gk = 0
	Ca.Xpower = 2
	Ca.Ypower = 1
	Ca.Zpower = 0

	xgate = moose.element( 'LCa/gateX' )
	xA = np.array( [ 1.6e3, 0, 1.0, -1.0 * (0.065 + EREST_ACT), -0.01389, -20e3 * (0.0511 + EREST_ACT), 20e3, -1.0, -1.0 * (0.0511 + EREST_ACT), 5.0e-3, 3000, -0.1, 0.05 ] )
        xgate.alphaParms = xA
	ygate = moose.element( 'LCa/gateY' )
	ygate.min = -0.1
	ygate.max = 0.05
	ygate.divs = 3000
	yA = np.zeros( (ygate.divs + 1), dtype=float)
	yB = np.zeros( (ygate.divs + 1), dtype=float)


#Fill the Y_A table with alpha values and the Y_B table with (alpha+beta)
	dx = (ygate.max - ygate.min)/ygate.divs
	x = ygate.min
	for i in range( ygate.divs + 1 ):
		if ( x > EREST_ACT):
			yA[i] = 5.0 * math.exp( -50 * (x - EREST_ACT) )
		else:
			yA[i] = 5.0
		yB[i] = 5.0
		x += dx
	ygate.tableA = yA
	ygate.tableB = yB
        return Ca

    ################################################################
    # API function for building spine prototypes. Here we put in the
    # spine dimensions, and options for standard channel types.
    # The synList tells it to create dual alpha function synchans:
    # [name, Erev, tau1, tau2, conductance_density, connectToCa]
    # The chanList tells it to copy over channels defined in /library
    # and assign the specified conductance density.
    # If caTau <= zero then there is no caConc created, otherwise it
    # creates one and assigns the desired tau in seconds.
    # With the default arguments here it will create a glu, NMDA and LCa,
    # and add a Ca_conc.
def addSpineProto( name = 'spine', \
        RM = 1.0, RA = 1.0, CM = 0.01, \
        shaftLen = 1.e-6 , shaftDia = 0.2e-6, \
        headLen = 0.5e-6, headDia = 0.5e-6, \
        synList = ( ['glu', 0.0, 2e-3, 9e-3, 200.0, False],
                    ['NMDA', 0.0, 20e-3, 20e-3, 80.0, True] ),
        chanList = ( ['Ca', 1.0, True ], ),
        caTau = 13.333e-3
        ):
    if not moose.exists( '/library' ):
        library = moose.Neutral( '/library' )
    spine = moose.Neutral( '/library/spine' )
    shaft = buildCompt( spine, 'shaft', shaftLen, shaftDia, 0.0, RM, RA, CM )
    head = buildCompt( spine, 'head', headLen, headDia, shaftLen, RM, RA, CM )
    moose.connect( shaft, 'axial', head, 'raxial' )

    if caTau > 0.0:
        conc = moose.CaConc( head.path + '/Ca_conc' )
        conc.tau = caTau
        conc.length = head.length
        conc.diameter = head.diameter
        conc.thick = 0.0
        # The 'B' field is deprecated.
        # B = 1/(ion_charge * Faraday * volume)
        #vol = head.length * head.diameter * head.diameter * PI / 4.0
        #conc.B = 1.0 / ( 2.0 * FaradayConst * vol )
        conc.Ca_base = 0.0
    for i in synList:
        syn = buildSyn( i[0], head, i[1], i[2], i[3], i[4], CM )
        if i[5] and caTau > 0.0:
            moose.connect( syn, 'IkOut', conc, 'current' )
    for i in chanList:
        if ( moose.exists( '/library/' + i[0] ) ):
            chan = moose.copy( '/library/' + i[0], head )
        else:
            moose.setCwe( head )
            chan = make_LCa()
            chan.name = i[0]
            moose.setCwe( '/' )
        chan.Gbar = i[1] * head.Cm / CM
        #print "CHAN = ", chan, chan.tick, chan.Gbar
        moose.connect( head, 'channel', chan, 'channel' )
        if i[2] and caTau > 0.0:
            moose.connect( chan, 'IkOut', conc, 'current' )
    transformNMDAR( '/library/spine' )
    return spine

# Wrapper function. This is used by the proto builder from rdesigneur
def makeSpineProto( name ):
    addSpineProto( name = name, chanList = () )


