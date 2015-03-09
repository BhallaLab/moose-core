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

import moose
import numpy as np
from moose.neuroml.NeuroML import NeuroML

#EREST_ACT = -70e-3
NA = 6.022e23
PI = 3.14159265359
FaradayConst = 96485.3365 # Coulomb/mol


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
            useGssa =True, \
            combineSegments= True, \
            cellPortion = "/model/elec/#", \
            meshLambda= 2e-6, \
            chemDt= 0.005, \
            diffDt= 0.005, \
            elecDt= 50e-6, \
            adaptorList= [ \
                ( 'psd', 'Ca_conc', 'Ca', 'Ca_input', \
                True, 8e-5, 1.0 ), \
                ( 'psd', 'glu', 'Gbar', 'tot_PSD_R', \
                False, 0, 0.01 )], \
            addSpineList= [] \
            ):
        """ Constructor of the class """
        if moose.exists( modelPath ):
            print "rdesigneur init failed. Model '", \
                modelPath, "' already exists."
            return;
        self.model = moose.Neutral( modelPath )
        self.useGssa = useGssa
        self.combineSegments = combineSegments
        self.cellPortion = cellPortion
        self.meshLambda= meshLambda
        self.chemDt= chemDt
        self.diffDt= diffDt
        self.elecDt= elecDt
        self.adaptorList= adaptorList
        self.addSpineList = addSpineList

    ################################################################

    def validateFromMemory( self, epath, cpath ):
        ret = self.validateChem( cpath )
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

        #moose.le( self.elecid )

        self._buildNeuroMesh()


        self._configureSolvers()
        for i in self.adaptorList:
            self._buildAdaptor( i[0],i[1],i[2],i[3],i[4],i[5],i[6] )
        for i in range( 0, 8 ):
            moose.setClock( i, self.elecDt )
        moose.setClock( 10, self.diffDt )
        for i in range( 11, 18 ):
            moose.setClock( i, self.chemDt )
        moose.setClock( 18, self.chemDt * 5.0 )
        hsolve = moose.HSolve( ep + '/hsolve' )
        hsolve.dt = self.elecDt
        hsolve.target = self.soma.path

    ################################################################

    def buildFromFile( self, efile, cfile ):
        self.efile = efile
        self.cfile = cfile
        self._loadElec( efile, 'tempelec', self.combineSegments )
        self._loadChem( cfile, 'tempchem' )
        self.buildFromMemory( self.model.path + '/tempelec', self.model.path + '/tempchem' )

    ################################################################

    # Utility function to return a coordinate system where 
    # z is the direction of a dendritic segment, 
    # x is the direction of spines outward from soma and perpendicular to z
    # and y is the perpendicular to x and z.
    def _coordSystem( self, soma, dend ):
        EPSILON = 1e-20
        z = np.array( [dend.x - dend.x0, dend.y - dend.y0, dend.z - dend.z0 ] )
        dendLength = np.sqrt( np.dot( z, z ) )
        z = z / dendLength
        y = np.array( [dend.x0 - soma.x0, dend.y0 - soma.y0, dend.z0 - soma.z0 ] )
        y = np.cross( y, z )
        ylen = np.dot( y, y )
        if ylen < EPSILON:
            y[0] = np.sqrt( 2.0 )
            y[1] = np.sqrt( 2.0 )
            y[2] = 0.0
            y = np.cross( y, z )
            ylen = np.dot( y, y )
            assert( ylen > EPSILON )
        y = y / np.sqrt( ylen )
        x = np.cross( z, y )
        xlen = np.dot( x, x )
        assert( np.fabs( xlen - 1 ) < 1e-15 )
        return ( dendLength, x,y,z )

    ################################################################
    # Utility function to resize electrical compt electrical properties,
    # including those of its child channels and calcium conc.
    def _scaleSpineCompt( self, compt, size ):
        chans = moose.wildcardFind( compt.path + '/##[ISA=ChanBase]' )
        a = size * size
        for i in chans:
            i.Gbar *= a
        concs = moose.wildcardFind( compt.path + '/##[ISA=CaConcBase]' )
        for i in concs:
            i.B *= size * size * size
        compt.Rm /= a
        compt.Cm *= a
        compt.Ra /= size

    ################################################################
    # Utility function to change coords of spine so as to reorient it.
    def _reorientSpine( self, spineCompts, coords, parentPos, pos, size, angle, x, y, z ):
        rotationMatrix = []
        c = np.cos( angle )
        s = np.sin( angle )
        omc = 1.0 - c
        rotationMatrix.append( [\
            z[0]*z[0]*omc + c, z[1]*z[0]*omc - z[2]*s, \
            z[2]*z[0]*omc + z[1]*s ] )
        rotationMatrix.append( [\
            z[0]*z[1]*omc + z[2]*s, z[1]*z[1]*omc + c, \
            z[2]*z[1]*omc - z[0]*s ] )
        rotationMatrix.append( [\
            z[0]*z[2]*omc - z[1]*s, z[1]*z[2]*omc + z[0]*s, \
            z[2]*z[2]*omc + c ] )

        rotationMatrix = np.array( rotationMatrix ) * size
        translation = z * pos + parentPos
        coords = np.transpose( coords )
        ret = np.dot( rotationMatrix, coords ) 
        ret = np.transpose( ret ) + translation
        #print 'ret = ', np.shape( ret ), ret
        assert( len( spineCompts ) * 2 == len( ret ) )
        for i in range( len( spineCompts ) ):
            j = i * 2
            spineCompts[i].x0 = ret[j][0]
            spineCompts[i].y0 = ret[j][1]
            spineCompts[i].z0 = ret[j][2]
            j = j + 1
            spineCompts[i].x = ret[j][0]
            spineCompts[i].y = ret[j][1]
            spineCompts[i].z = ret[j][2]
        #print 'r: ', ret[0], '\n', parentPos

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
    ## API function to add a series of spines.

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
    ## Returns list of spines.


    def insertSpines( self, spineProto, parentList, \
            spacing, spacingDistrib = 0.0, \
            sizeDistrib = 0.0, \
            angle = 0.0, angleDistrib = 0.0, \
            rotation = 0.0, rotationDistrib = 0.0 ):
        k = 0
        for i in parentList:
            dendLength, x,y,z = self._coordSystem( self.soma, i )
            num = int( dendLength / spacing ) + 2
            if ( spacingDistrib > 0.0 ):
                pos = np.random.normal( spacing, spacingDistrib, num )
            else:
                pos = np.array( [spacing] * num )
            if ( angleDistrib > 0.0 ):
                angle += np.random.random() * angleDistrib
            if ( rotationDistrib > 0.0 ):
                theta = np.random.normal( rotation, rotationDistrib, num )
            else:
                theta = np.array( [rotation] * num )
            if ( sizeDistrib > 0.0 ):
                size = np.random.normal( 1.0, sizeDistrib, num )
            else:
                size = np.array( [1.0] * num )

            #print "insertSpines on ", i.name
            p = pos[-1] / 2.0
            for j in zip( pos, theta, size ):
                #print p, j[0], dendLength
                self._addSpine( i, spineProto, p, angle, x, y, z, j[2], k )
                k += 1
                p += j[0]
                angle += j[1]
                #print 'angle = ', angle
                if ( p > dendLength ):
                    break

    ################################################################
    def _decorateWithSpines( self ):
        for i in self.addSpineList:
            if not moose.exists( '/library/' + i[0] ):
                print 'Warning: _decorateWithSpines: spine proto ', i[0], ' not found.'
                continue
            spineProto = moose.element( '/library/' + i[0] )
            parentList = moose.wildcardFind( self.elecid.path + '/' + i[1] )
            self.insertSpines( spineProto, parentList, \
                    i[2], i[3], i[4], i[5], i[6], i[7], i[8] )

    ################################################################

    def _loadElec( self, efile, elecname, combineSegments ):
        library = moose.Neutral( '/library' )
        if ( efile[ len( efile ) - 2:] == ".p" ):
            self.elecid = moose.loadModel( efile, self.model.path + '/' + elecname )
        else:
            nm = NeuroML()
            nm.readNeuroMLFromFile( efile, \
                    params = {'combineSegments': combineSegments, \
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
            moose.move( self.elecid, self.model )
            self.elecid.name = elecname

        self._transformNMDAR( self.elecid.path )
        kids = moose.wildcardFind( '/library/##[0]' )
        for i in kids:
            i.tick = -1

    #################################################################
    def _transformNMDAR( self, path ):
        for i in moose.wildcardFind( path + "/##/#NMDA#[ISA!=NMDAChan" ):
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

    #################################################################

    # This assumes that the chemid is located in self.parent.path+/chem
    # It moves the existing chem compartments into a NeuroMesh
    # For now this requires that we have a dend, a spine and a PSD,
    # with those names and volumes in decreasing order.
    def validateChem( self, cpath  ):
        comptlist = moose.wildcardFind( cpath + '/#[ISA=ChemCompt]' )
        if len( comptlist ) == 0:
            print "ValidateChem: Invalid chemistry: No compartment on: ", cpath
            return False

        # Sort comptlist in decreasing order of volume
        sortedComptlist = sorted( comptlist, key=lambda x: -x.volume )
        if ( len( sortedComptlist ) != 3 ):
            print "ValidateChem: Invalid chemistry: Require 3 chem compartments, have: ",\
                len( sortedComptlist )
            return False
        if ( sortedComptlist[0].name.lower() == 'dend' and \
            sortedComptlist[1].name.lower() == 'spine' and \
            sortedComptlist[2].name.lower() == 'psd' ):
            return True

        print "ValidateChem: Invalid names: require dend, spine and PSD."
        print " Actual names = ", 
        for i in sortedComptlist:
            print i.name,
        print
        return False
            
    #################################################################

    def _buildNeuroMesh( self ):
        comptlist = moose.wildcardFind( self.chemid.path + '/#[ISA=ChemCompt]' )
        sortedComptList = sorted( comptlist, key=lambda x: -x.volume )
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
        #print self.elecid
        #print self.cellPortion
        self.dendCompt.cellPortion( self.elecid, self.cellPortion )

    #################################################################
    def _configureSolvers( self ) :
        dmksolve = moose.Ksolve( self.dendCompt.path + '/ksolve' )
        dmdsolve = moose.Dsolve( self.dendCompt.path + '/dsolve' )
        dmstoich = moose.Stoich( self.dendCompt.path + '/stoich' )
        dmstoich.compartment = self.dendCompt
        dmstoich.ksolve = dmksolve
        dmstoich.dsolve = dmdsolve
        dmstoich.path = self.dendCompt.path + "/##"
        print 'Dend solver: numPools = ', dmdsolve.numPools, \
            ', nvox= ', self.dendCompt.mesh.num, dmksolve.numAllVoxels
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
        print 'spine num Pools = ', smstoich.numAllPools, \
                ', nvox= ',  self.spineCompt.mesh.num, smksolve.numAllVoxels
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
        print 'psd num Pools = ', pmstoich.numAllPools, \
                ', voxels=', self.psdCompt.mesh.num, pmksolve.numAllVoxels
    
        # Put in cross-compartment diffusion between ksolvers
        dmdsolve.buildNeuroMeshJunctions( smdsolve, pmdsolve )
        # Put in cross-compartment reactions between ksolvers
        smstoich.buildXreacs( pmstoich )
        smstoich.buildXreacs( dmstoich )
        #smstoich.buildXreacs( pmstoich )
        dmstoich.filterXreacs()
        smstoich.filterXreacs()
        pmstoich.filterXreacs()
    ################################################################
    
    def _loadChem( self, fname, chemName ):
        chem = moose.Neutral( self.model.path + '/' + chemName )
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
            chemRelPath, isElecToChem, offset, scale ):
        mesh = moose.element( '/model/chem/' + meshName )
        elecComptList = mesh.elecComptList
        startVoxelInCompt = mesh.startVoxelInCompt
        endVoxelInCompt = mesh.endVoxelInCompt
        capField = elecField[0].capitalize() + elecField[1:]
        chemPath = mesh.path + '/' + chemRelPath
        if not( moose.exists( chemPath ) ):
            print "Error: buildAdaptor: no chem obj in ", chemPath
            return
        chemObj = moose.element( chemPath )
        assert( chemObj.numData >= len( elecComptList ) )
        adName = '/adapt'
        for i in range( 1, len( elecRelPath ) ):
            if ( elecRelPath[-i] == '/' ):
                adName += elecRelPath[1-i]
                break
        ad = moose.Adaptor( chemObj.path + adName, len( elecComptList ) )
        av = ad.vec
        # print 'building ', len( elecComptList ), 'adaptors ', adName, \
        #        ' for: ', mesh.name, elecRelPath, elecField, chemRelPath
        chemVec = moose.element( mesh.path + '/' + chemRelPath ).vec
    
        for i in zip( elecComptList, startVoxelInCompt, endVoxelInCompt, av ):
            i[3].inputOffset = 0.0
            i[3].outputOffset = offset
            i[3].scale = scale
            ePath = i[0].path + '/' + elecRelPath
            if not( moose.exists( ePath ) ):
                print "Error: buildAdaptor: no elec obj in ", ePath
                #moose.le( '/model[0]/elec[0]/Seg0_spine0_2001445' )
                return
            elObj = moose.element( i[0].path + '/' + elecRelPath )
            if ( isElecToChem ):
                elecFieldSrc = 'get' + capField
                moose.connect( i[3], 'requestOut', elObj, elecFieldSrc )
                for j in range( i[1], i[2] ):
                    moose.connect( i[3], 'output', chemVec[j], 'setConc')
            else:
                elecFieldDest = 'set' + capField
                for j in range( i[1], i[2] ):
                    moose.connect( i[3], 'requestOut', chemVec[j], 'getConc')
                moose.connect( i[3], 'output', elObj, elecFieldDest )
    
    ################################################################
    # Utility function for building a compartment, used for spines.
    def _buildCompt( self_, pa, name, length, dia, xoffset, RM, RA, CM ):
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
    def _buildSyn( self, name, compt, Ek, tau1, tau2, Gbar, CM ):
        syn = moose.SynChan( compt.path + '/' + name )
        syn.Ek = Ek
        syn.tau1 = tau1
        syn.tau2 = tau2
        syn.Gbar = Gbar * compt.Cm / CM
        moose.connect( compt, 'channel', syn, 'channel' )
        sh = moose.SimpleSynHandler( syn.path + '/sh' )
        moose.connect( sh, 'activationOut', syn, 'activation' )
        sh.numSynapses = 1
        sh.synapse[0].weight = 1
        return syn
    
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
    def addSpineProto( self, name, RM, RA, CM, \
            shaftLen = 1.e-6 , shaftDia = 0.2e-6, \
            headLen = 0.5e-6, headDia = 0.5e-6, \
            synList = ( ['glu', 0.0, 2e-3, 9e-3, 200.0, False],
                        ['NMDA', 0.0, 20e-3, 20e-3, 40.0, True] ),
            chanList = ( ['LCa', 40.0, True ], ),
            caTau = 13.333e-3
            ):
        if not moose.exists( '/library' ):
            library = moose.Neutral( '/library' )
        spine = moose.Neutral( '/library/spine' )
        shaft = self._buildCompt( spine, 'shaft', shaftLen, shaftDia, 0.0, RM, RA, CM )
        head = self._buildCompt( spine, 'head', headLen, headDia, shaftLen, RM, RA, CM )
        moose.connect( shaft, 'raxial', head, 'axial' )

        if caTau > 0.0:
            conc = moose.CaConc( head.path + '/Ca_conc' )
            conc.tau = caTau
            # B = 1/(ion_charge * Faraday * volume)
            vol = head.length * head.diameter * head.diameter * PI / 4.0
            conc.B = 1.0 / ( 2.0 * FaradayConst * vol )
            conc.Ca_base = 0.0
        for i in synList:
            syn = self._buildSyn( i[0], head, i[1], i[2], i[3], i[4], CM )
            if i[5] and caTau > 0.0:
                moose.connect( syn, 'IkOut', conc, 'current' )
        for i in chanList:
            if ( moose.exists( '/library/' + i[0] ) ):
                chan = moose.copy( '/library/' + i[0], head )
                chan.Gbar = i[1] * head.Cm / CM
                print "CHAN = ", chan, chan.tick
                moose.connect( head, 'channel', chan, 'channel' )
                if i[2] and caTau > 0.0:
                    moose.connect( chan, 'IkOut', conc, 'current' )
            else:
                print "Warning: addSpineProto: channel '", i[0], \
                    "' not found on /library."
                moose.le( '/library' )
        self._transformNMDAR( '/library/spine' )
        return spine
