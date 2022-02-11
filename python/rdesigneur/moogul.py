# Moogul.py: MOOSE Graphics Using Lines
# This is a fallback graphics interface for displaying neurons using
# regular matplotlib routines.
# Put in because the GL versions like moogli need all sorts of difficult 
# libraries and dependencies.
# Copyright (C) Upinder S. Bhalla NCBS 2018
# This program is licensed under the GNU Public License version 3.
#

import moose
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import vpython as vp
import time
from packaging import version
#from mpl_toolkits.mplot3d.art3d import Line3DCollection
NUM_CMAP = 64
SCALE_SCENE = 64
bgvector = vp.vector(0.7, 0.8, 0.9)  # RGB
bgDict = {'default': bgvector, 'black': vp.color.black, 'white': vp.color.white, 'cyan': vp.color.cyan, 'grey': vp.vector( 0.5, 0.5, 0.5 ) }

def bgLookup( bg ):
    col = bgDict.get( bg )
    if not col:
        return bgvector
    return col

class MoogulError( Exception ):
    def __init__( self, value ):
        self.value = value
    def __str__( self ):
        return repr( self.value )


class MooView:
    ''' The MooView class is a window in which to display one or more 
    moose cells, using the MooNeuron class.'''
    viewIdx = 0
    origScene = None
    rgb = []
    viewList = []
    consolidatedTitle = ""

    def __init__( self, swx = 10, swy = 10, hideAxis = True, title = "view", colormap = 'jet'
    ):
        self.viewIdx = MooView.viewIdx
        MooView.viewIdx += 1
        MooView.viewList.append( self )
        MooView.consolidatedTitle += title + "      "
        self.title = title
        self.swx = swx
        self.swy = swy
        self.drawables_ = []
        self.sensitivity = 0.05 # radians rotation, and other adjustments
        self.sleep = 0.005  # Seconds to sleep per frame
        self.colormap = colormap
        self.colorbar = None
        self.valMin = 0.0
        self.valMmax = 1.0

    @staticmethod
    def replayLoop():
        if len( MooView.viewList ) == 0:
            return
        numFrames = MooView.viewList[0].numFrames()
        while MooView.viewList[0].replayButton.text == "Stop Replay":
            for idx in range( numFrames ):
                for view in MooView.viewList:
                    view.replaySnapshot( idx )
                vp.sleep( MooView.viewList[0].sleep )
            vp.sleep( 0.5 ) # Pause 0.5 sec between replays

    def notifySimulationEnd( self ):
        if self.viewIdx == 0:
            self.replayButton.disabled = False
    
    def numFrames( self ):
        if len( self.drawables_ ) == 0:
            return 0
        return len( self.drawables_[0].snapshot )

    def addDrawable( self, n ):
        self.drawables_.append( n )
        if len( self.drawables_ ) == 1:
            self.valMin = n.valMin
            self.valMax = n.valMax
        # self.scene.objects also maintains list.

    def toggleReplay( self ):
        if self.replayButton.text == "Start Replay":
            self.replayButton.text = "Stop Replay"
            self.replayButton.background = vp.color.red
            MooView.replayLoop()
        else:
            self.replayButton.text = "Start Replay"
            self.replayButton.background = vp.color.white

    def setSleepTime( self ):
        self.sleep = self.sleepSlider.value
        self.sleepLabel.text = "    Frame dt = {:1.3f} sec".format( self.sleep )

    def updateAxis( self ):
        if not self.colorbar:
            return
        forward = vp.norm( self.scene.forward )
        screenUp = vp.norm( self.scene.up )
        right = vp.norm( vp.cross( forward, screenUp ) )
        up = vp.norm( vp.cross( right, forward ) )

        dx = 0.8

        x = vp.vector( dx, 0.0, 0.0 )
        y = vp.vector( 0.0, dx, 0.0 )
        z = vp.vector( 0.0, 0.0, dx )

        self.xAx.axis = vp.vector( x.dot( right ), x.dot( up ), 0.0 )
        self.yAx.axis = vp.vector( y.dot( right ), y.dot( up ), 0.0 )
        self.zAx.axis = vp.vector( z.dot( right ), z.dot( up ), 0.0 )
        self.axisLength.text = "{:.2f} <i>u</i>m".format( dx * 1e6*self.scene.range * self.colorbar.width / self.scene.width )

    def makeColorbar( self, doOrnaments = True, colorscale = 'jet', bg = 'default' ):
        title = None
        if doOrnaments:
            title = MooView.consolidatedTitle + "\n"
        barWidth = SCALE_SCENE * 1.5
        self.colorbar = vp.canvas( title = title, width = barWidth, height = self.swy * SCALE_SCENE, background = bgLookup(bg), align = 'left', range = 1, autoscale = False )
        #self.colorbar = vp.canvas( title = title, width = barWidth, height = self.swy * SCALE_SCENE, background = vp.color.cyan, align = 'left', range = 1, autoscale = False )
        self.colorbar.userzoom = False
        self.colorbar.userspin = False
        self.colorbar.userpan = False
        height = 0.10
        width = 5
        axOrigin = vp.vector( 0, -5.5, 0 )
        for idx, rgb in enumerate( self.rgb ):
            cbox = vp.box( canvas = self.colorbar, pos = vp.vector( 0, height * (idx - 26), 0), width = width, height = height, color = rgb )
        barName = self.title.replace( ' ', '\n' )
        self.barName = vp.label( canvas = self.colorbar, align = 'left', pixel_pos = True, pos = vp.vector( 2, (self.swy - 0.32) * SCALE_SCENE, 0), text = barName, height = 15, color = vp.color.black, box = False, opacity = 0 )
        self.barMin = vp.label( canvas = self.colorbar, align = 'center', pixel_pos = True, pos = vp.vector( barWidth/2, self.swy * SCALE_SCENE * 0.22, 0), text = "{:.3f}".format(self.valMin), height = 12, color = vp.color.black, box = False, opacity = 0 )
        self.barMax = vp.label( canvas = self.colorbar, align = 'center', pixel_pos = True, pos = vp.vector( barWidth/2, (self.swy - 1.2) * SCALE_SCENE, 0), text = "{:.3f}".format(self.valMax), height = 12, color = vp.color.black, box = False, opacity = 0 )
        self.xAx = vp.cylinder( canvas = self.colorbar, pos = axOrigin, axis = vp.vector( 0.8, 0, 0 ), radius = 0.04, color = vp.color.red )
        self.yAx = vp.cylinder( canvas = self.colorbar, pos = axOrigin, axis = vp.vector( 0, 0.8, 0 ), radius = 0.04, color = vp.color.green )
        self.zAx = vp.cylinder( canvas = self.colorbar, pos = axOrigin, axis = vp.vector( 0, 0, 0 ), radius = 0.04, color = vp.color.blue )
        self.axisLength = vp.label( pos = axOrigin + vp.vector(0, 1, 0), text = "1.00 <i>u</i>m", color = vp.color.black, box = False )
        if doOrnaments:
            self.timeLabel = vp.wtext( text = "Time =  0.000 sec", pos = self.colorbar.title_anchor )
            self.sleepLabel = vp.wtext( text = "    Frame dt = 0.005 sec", pos = self.colorbar.title_anchor )
            self.sleepSlider = vp.slider( pos = self.colorbar.title_anchor, length = 200, bind = self.setSleepTime, min = 0.0, max = 0.2, value = self.sleep )
            self.replayButton = vp.button( text = "Start Replay", pos = self.colorbar.title_anchor, bind=self.toggleReplay, disabled = True )
            self.colorbar.append_to_title("\n")

    def pickObj( self ):
        obj = self.scene.mouse.pick
        if obj == None:
            return
        elm = self.innerPickObj( obj )
        if elm:
            print( elm.path, elm.dataIndex )
            return
        elif self.viewIdx == 0: 
            for view in MooView.viewList[1:]:
                if view.colorbar == None:
                    elm = view.innerPickObj( obj )
                    if elm:
                        print( elm.path, elm.dataIndex )
                        return
        print( "Object {} not found on view {}".format( obj, self.title ) )

    def innerPickObj( self, obj ):
        for dr in self.drawables_:
            elm = dr.findDisplayObject( obj )
            if elm:
                return elm
        return None



    def makeScene( self, mergeDisplays, bg = 'default' ):
        if self.viewIdx == 0:
            MooView.origScene = vp.canvas( width = self.swx * SCALE_SCENE, height = self.swy * SCALE_SCENE, background = bgLookup( bg ), align = 'left', autoscale = True )
            self.scene = MooView.origScene
            self.scene.bind( 'keydown', self.moveView )
            self.scene.bind( 'keydown', self.updateAxis )
            self.scene.bind( 'mousedown', self.pickObj )
            #self.flatbox = vp.box( width = 10, height = 6 )
        elif mergeDisplays:
            self.scene = MooView.origScene
        else: 
            self.scene = vp.canvas( width = self.swx * SCALE_SCENE, height = self.swy * SCALE_SCENE, background = bgvector, align = 'left', autoscale = True )
            self.scene.bind( 'keydown', self.moveView )
            self.scene.bind( 'keydown', self.updateAxis )
            self.scene.bind( 'mousedown', self.pickObj )
        '''
        self.xAx2 = vp.cylinder( canvas = self.scene, pos = vp.vector( 0, 0, 0), axis = vp.vector( 1e-5, 0, 0 ), radius = 0.2e-6, color = vp.color.red )
        self.yAx2 = vp.cylinder( canvas = self.scene, pos = vp.vector( 0, 0, 0), axis = vp.vector( 0, 1e-5, 0 ), radius = 0.2e-6, color = vp.color.green )
        self.zAx2 = vp.cylinder( canvas = self.scene, pos = vp.vector( 0, 0, 0), axis = vp.vector( 0, 0, 1e-5 ), radius = 0.2e-6, color = vp.color.blue )
        '''
        self.scene.bind( 'mousedown mousemove mouseup', self.updateAxis )

    def firstDraw( self, mergeDisplays, rotation=0.0, elev=0.0, azim=0.0, center = [0.0, 0,0, 0.0], colormap = 'jet', bg = 'default' ):
        self.colormap = colormap
        cmap = plt.get_cmap( self.colormap, lut = NUM_CMAP )
        self.rgb = [ list2vec(cmap(i)[0:3]) for i in range( NUM_CMAP ) ]
        doOrnaments = (self.viewIdx == 0)
        if doOrnaments or not mergeDisplays:
            self.makeColorbar( doOrnaments = doOrnaments, bg = bg )
        self.makeScene( mergeDisplays, bg = bg )
        if rotation == 0.0:
            self.doRotation = False
            self.rotation = 0.1 # default rotation per frame, in radians.
        else:
            self.doRotation = True
            self.rotation = rotation # arg units: radians/frame
        
        for i in self.drawables_:
            i.rgb = self.rgb
            i.drawForTheFirstTime( self.scene )
        if doOrnaments or not mergeDisplays:
            if len( center ) == 3:
                self.scene.center = list2vec( center )
            else:
                self.doAutoscale()
            self.updateAxis()

    def updateValues( self ):
        simTime = moose.element( '/clock' ).currentTime
        #self.timeStr.set_text( "Time= {:.3f}".format( time ) )
        for i in self.drawables_:
            i.updateValues( simTime )
        if self.doRotation and abs( self.rotation ) < 2.0 * 3.14 / 3.0:
            self.scene.forward = vp.rotate( self.scene.forward, angle = self.rotation, axis = self.scene.up )
            self.updateAxis()
        if self.viewIdx == 0:
            self.timeLabel.text = "Time = {:7.3f} sec".format( simTime )
            vp.sleep( self.sleep )

    def replaySnapshot( self, idx ):
        for i in self.drawables_:
            simTime = i.replaySnapshot( idx )
        if self.viewIdx == 0:
            self.timeLabel.text = "Time = {:7.3f} sec".format( simTime )
            self.updateAxis()

    def doAutoscale( self ):
        if len( self.drawables_[0].activeDia ) == 0:
            print( "Warning: No values to display in Moogli view ", self.title )
            return
        cmin = self.drawables_[0].coordMin
        cmax = self.drawables_[0].coordMax
        diamax = max( self.drawables_[0].activeDia )
        v0 = vp.vector( cmin[0], cmin[1], cmin[2] )
        v1 = vp.vector( cmax[0], cmax[1], cmax[2] )
        #self.scene.camera.axis = self.scene.forward * vp.mag(v1 - v0) * 4
        self.scene.center = (v0 + v1 ) / 2.0
        self.scene.range = (diamax + vp.mag(v0 - v1 ) ) / 1.5

    def moveView(self, event):
        camAxis = self.scene.camera.axis
        camDist = vp.mag(self.scene.center - self.scene.camera.pos)
        dtheta = self.sensitivity
        up = self.scene.up

        if event.key in ["up", "k", "K"]:
            self.scene.camera.pos -= up.norm() * dtheta * camDist
            return
        if event.key in ["down", "j", "J"]:
            self.scene.camera.pos += up.norm() * dtheta * camDist 
            return
        if event.key in ["right", "l", "L"]:
            self.scene.camera.pos += vp.norm(up.cross(camAxis)) * dtheta * camDist 
            return
        if event.key in ["left", "h", "H"]:
            self.scene.camera.pos -= vp.norm(up.cross(camAxis)) * dtheta * camDist 
            return
        if event.key in [".", ">"]: # Get closer, by ratio
            ctr = self.scene.center
            self.scene.camera.pos = ctr - camAxis/( 1+dtheta )
            self.scene.camera.axis = ctr - self.scene.camera.pos
            return
        if event.key in [",", "<"]: # Get further
            ctr = self.scene.center
            self.scene.camera.pos = ctr - camAxis*( 1+dtheta )
            self.scene.camera.axis = ctr - self.scene.camera.pos
            return
        if event.key == "p": # pitch: Rotate camera around ctr-horiz axis
            self.scene.forward = vp.rotate( self.scene.forward, angle = dtheta, axis = vp.cross( self.scene.forward, self.scene.up ) )
            return
        if event.key == "P":
            self.scene.forward = vp.rotate( self.scene.forward, angle = -dtheta, axis = vp.cross( self.scene.forward, self.scene.up ) )
            return
        if event.key == "y": # yaw: Rotate camera around ctr - up axis.
            self.scene.forward = vp.rotate( self.scene.forward, angle = dtheta, axis = self.scene.up )
            return
            return
        if event.key == "Y":
            self.scene.forward = vp.rotate( self.scene.forward, angle = -dtheta, axis = self.scene.up )
            return
        if event.key == "r": # Roll, that is, change the 'up' vector
            self.scene.camera.rotate( angle = dtheta, axis = camAxis, origin = self.scene.camera.pos )
            return
        if event.key == "R":
            self.scene.camera.rotate( angle = -dtheta, axis = camAxis, origin = self.scene.camera.pos )
            return
        if event.key == "d": # Diameter scaling down
            for dbl in self.drawables_:
                dbl.diaScale *= 1.0 - self.sensitivity * 4
                dbl.updateDiameter()
            return
        if event.key == "D":
            for dbl in self.drawables_:
                dbl.diaScale *= 1.0 + self.sensitivity * 4
                dbl.updateDiameter()
            return
        if event.key == "s": # Scale down sleep time, make it faster.
            self.sleep *= 1 - self.sensitivity
            return
        if event.key == "S": # Scale up sleep time, make it slower.
            self.sleep *= 1 + self.sensitivity
            return
        if event.key == "a": # autoscale to fill view.
            self.doAutoscale()
            return
        if event.key == "g":
            self.hideAxis = not self.hideAxis
            # show/hide the axis here.
        if event.key == "t": # Turn on/off twisting/autorotate
            self.doRotation = not self.doRotation
        if event.key == "?": # Print out help for these commands
            self.printMoogulHelp()

    def printMoogulHelp( self ):
        print( '''
            Key bindings for Moogul:
            Up or k:    pan object up
            Down or j:  pan object down
            left or h:  pan object left. 
            right or l:  pan object right
            . or >:     Zoom in: make object appear bigger
            , or <:     Zoom out: make object appear smaller
            a:          Autoscale to fill view
            p:          Pitch down
            P:          Pitch up
            y:          Yaw counterclockwise
            Y:          Yaw counterclockwise
            d:          diminish diameter
            D:          Distend diameter.
            g:          Toggle visibility of grid
            t:          Toggle turn (rotation along long axis of cell)
            ?:          Print this help page.
        ''')

#####################################################################

def list2vec( arg ):
    return vp.vector( arg[0], arg[1], arg[2] )

class MooDrawable:
    ''' Base class for drawing things'''
    def __init__( self,
        fieldInfo, field, relativeObj,
        colormap,
        lenScale, diaScale, autoscale,
        valMin, valMax
    ):
        self.field = field
        self.relativeObj = relativeObj
        self.lenScale = lenScale
        self.diaScale = diaScale
        self.colormap = colormap
        self.autoscale = autoscale
        self.valMin = valMin
        self.valMax = valMax
        self.fieldInfo = fieldInfo
        self.fieldScale = fieldInfo[2]
        self.segments = []
        self.snapshot = []
        self.coordMin = np.zeros( 3 )
        self.coordMax = np.zeros( 3 )
        #cmap = plt.get_cmap( self.colormap, lut = NUM_CMAP )
        #self.rgb = [ list2vec(cmap(i)[0:3]) for i in range( NUM_CMAP ) ]

    def updateValues( self, simTime ):
        ''' Obtains values from the associated cell'''
        self.val = np.array([moose.getField(i, self.field) for i in self.activeObjs]) * self.fieldScale
        if self.autoscale:
            valMin = min( self.val )
            valMax = max( self.val )
        else:
            valMin = self.valMin
            valMax = self.valMax
        scaleVal = NUM_CMAP * (self.val - valMin) / (valMax - valMin)
        #indices = scaleVal.ndarray.astype( int )
        indices = np.maximum( np.minimum( scaleVal, NUM_CMAP-0.5), 0.0).astype(int)
        self.snapshot.append( [simTime, indices] )
        self.displayValues( indices )

    def displayValues( self, indices ):
        for idx, seg in zip( indices, self.segments ): 
            seg.color = self.rgb[ idx]
            #seg.radius = self.diaScale  * self.activeDia[idx]

    def replaySnapshot( self, idx ):
        if idx >= len( self.snapshot ):
            return 0.0
        self.displayValues( self.snapshot[idx][1] )
        return self.snapshot[idx][0]    # return frame time

    def updateDiameter( self ):
        for s, w in zip( self.segments, self.activeDia ):
            s.radius = self.diaScale * w / 2.0

    def cylinderDraw( self, _scene ):
        for idx, coord in enumerate( self.activeCoords ):
            v0 = list2vec( coord[0] )
            v1 = list2vec( coord[1] )
            self.coordMin = np.minimum( self.coordMin, coord[0][0:3] )
            self.coordMin = np.minimum( self.coordMin, coord[1][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[0][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[1][0:3] )
            radius = self.diaScale * self.activeDia[idx] / 2.0
            opacity = self.opacity[idx]
            rod = vp.cylinder( canvas = _scene, pos = v0, axis = v1 - v0, radius = radius, opacity = opacity )
            self.segments.append( rod )

    def findDisplayObject( self, obj ):
        try:
            idx = self.segments.index( obj )
        except ValueError:
            return None
        if idx >= len( self.activeObjs ):
            return None
        return self.activeObjs[idx]

#####################################################################

class MooNeuron( MooDrawable ):
    ''' Draws collection of line segments of defined dia and color'''
    def __init__( self,
        neuronId,
        fieldInfo,
        field = 'Vm', 
        relativeObj = '.', 
        colormap = 'jet', 
        lenScale = 1.0, diaScale = 1.0, autoscale = False, 
        valMin = -0.1, valMax = 0.05,
    ):
        #self.isFieldOnCompt = 
            #field in ( 'Vm', 'Im', 'Rm', 'Cm', 'Ra', 'inject', 'diameter' )
        
        MooDrawable.__init__( self, fieldInfo, field = field, 
                relativeObj = relativeObj,
                colormap = colormap, lenScale = lenScale, 
                diaScale = diaScale, autoscale = autoscale, 
                valMin = valMin, valMax = valMax )
        self.neuronId = neuronId
        self.updateCoords()

    def updateCoords( self ):
        ''' Obtains coords from the associated cell'''
        self.compts_ = moose.wildcardFind( self.neuronId.path + "/#[ISA=CompartmentBase]" )
        coords = np.array([[[i.x0,i.y0,i.z0],[i.x,i.y,i.z]] 
            for i in self.compts_])
        dia = np.array([i.diameter for i in self.compts_])
        if self.relativeObj == '.':
            self.activeCoords = coords
            self.activeDia = dia
            self.activeObjs = self.compts_
        else:
            self.activeObjs = []
            self.activeCoords = []
            self.activeDia = []
            for i,j,k in zip( self.compts_, coords, dia ):
                if moose.exists( i.path + '/' + self.relativeObj ):
                    elm = moose.element( i.path + '/' + self.relativeObj )
                    self.activeObjs.append( elm )
                    self.activeCoords.append( j )
                    self.activeDia.append( k )

        self.activeCoords = np.array( self.activeCoords ) * self.lenScale
        self.opacity = np.ones( len( self.activeDia ) ) * 0.5
        super().updateDiameter()

        return

    def drawForTheFirstTime( self, _scene ):
        self.cylinderDraw( _scene )

#####################################################################
class MooReacSystem( MooDrawable ):
    ''' Draws collection of line segments of defined dia and color'''
    def __init__( self,
        mooObj, fieldInfo,
        field = 'conc', 
        relativeObj = '.', 
        colormap = 'jet', 
        lenScale = 1e0, diaScale = 1.0, autoscale = False, 
        valMin = 0.0, valMax = 1.0
    ):
        
        MooDrawable.__init__( self, fieldInfo, field = field, 
                relativeObj = relativeObj,
                colormap = colormap, lenScale = lenScale, 
                diaScale = diaScale, autoscale = autoscale, 
                valMin = valMin, valMax = valMax )
        self.mooObj = mooObj
        self.updateCoords()

    def updateCoords( self ):
        activeCoords = []
        self.activeDia = []
        for pool in self.mooObj:
            coords = pool.coords
            meshType = pool.compartment.className
            if meshType in ["NeuroMesh", "CylMesh", "PsdMesh"]:
                # Unfortunately at present these return radius rather than
                # diameter in argument 6. To fix.
                # Make a cylinder
                activeCoords.append( [coords[0:3], coords[3:6]] )
                self.activeDia.append( coords[6] * 2 )
            elif meshType == "SpineMesh":
                # Spine entry has head[3], shaft[3], root[3], dia.
                activeCoords.append( [coords[0:3], coords[3:6]] )
                self.activeDia.append( coords[9] )
            elif meshType == "PresynMesh":
                # This returns diameter in argument 6.
                # first vec is centre of base, second axis pointing 
                # toward postsyn
                # Hack: make each bouton as a cone with length == dia.
                activeCoords.append( [coords[0:3], coords[6]*coords[3:6] + coords[0:3]] )
                self.activeDia.append( coords[6] )
                # Returns centre as args 0,1,2, diameter as argument 3.
                # Make a hemisphere
            elif meshType == "EndoMesh":
                # Make a sphere.
                activeCoords.append( [ coords[0:3], coords[0:3] ] )
                self.activeDia.append( coords[3] )
        self.activeCoords = np.array( activeCoords ) * self.lenScale
        self.activeDia = np.array( self.activeDia ) * self.diaScale
        self.opacity = np.ones( len( self.activeDia ) )
        self.activeObjs = self.mooObj
        return

    def drawForTheFirstTime( self, _scene ):
        if len( self.mooObj ) == 0:
            return
        meshType = self.mooObj[0].compartment.className
        if meshType in ["NeuroMesh", "CylMesh", "SpineMesh", "PsdMesh"]:
            self.cylinderDraw( _scene )
        elif meshType == "SpineMesh":
            self.spineDraw( _scene )
        elif meshType == "PresynMesh":
            self.presynDraw( _scene )
        elif meshType == "EndoMesh":
            self.endoDraw( _scene )

    def spineDraw( self, _scene ):
        # Spine entry has head[3], shaft[3], root[3], dia.
        for idx, coord in enumerate( self.activeCoords ):
            v0 = list2vec( coord[0] )
            v1 = list2vec( coord[1] )
            self.coordMin = np.minimum( self.coordMin, coord[0][0:3] )
            self.coordMin = np.minimum( self.coordMin, coord[1][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[0][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[1][0:3] )
            radius = self.diaScale * self.activeDia[idx] / 2.0
            opacity = self.opacity[idx]
            rod = vp.cylinder( canvas = _scene, pos = v0, axis = v1 - v0, radius = radius, opacity = opacity )
            self.segments.append( rod )

    def presynDraw( self, _scene ):
        for idx, coord in enumerate( self.activeCoords ):
            v0 = list2vec( coord[0] )
            v1 = list2vec( coord[1] )
            self.coordMin = np.minimum( self.coordMin, coord[0][0:3] )
            self.coordMin = np.minimum( self.coordMin, coord[1][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[0][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[1][0:3] )
            radius = self.diaScale * self.activeDia[idx] / 2.0
            opacity = self.opacity[idx]
            cone = vp.cone( canvas = _scene, pos = v0, axis = v0 - v1, radius = radius, opacity = opacity )
            self.segments.append( cone )

    def endoDraw( self, _scene ):
        for idx, coord in enumerate( self.activeCoords ):
            v0 = list2vec( coord[0] )
            v1 = list2vec( coord[1] )
            self.coordMin = np.minimum( self.coordMin, coord[0][0:3] )
            self.coordMax = np.maximum( self.coordMax, coord[0][0:3] )
            radius = self.diaScale * self.activeDia[idx] / 2.0
            opacity = self.opacity[idx]
            sphere = vp.sphere( canvas = _scene, pos = (v0 + v1)/2.0, radius = radius, opacity = opacity )
            self.segments.append( sphere )
