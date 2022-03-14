# Moogul.py: MOOSE Graphics 3D. 
# Copyright (C) Upinder S. Bhalla NCBS 2022
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

sleepTimes = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

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
    neurons, using the MooNeuron and MooReacSystemclass.'''
    viewIdx = 0
    origScene = None
    rgb = []
    viewList = []
    consolidatedTitle = ""
    colorbarViewIdx = 0    # Which View does the colorbar serve?

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
        self.simTime = 0.0
        self.plotFlag_ = True
        self.cbox = []

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
            t = time.time() + 0.5
            while time.time() < t:
                for view in MooView.viewList:
                    view.rotateFunc()
                vp.sleep( MooView.viewList[0].sleep )
            #vp.sleep( 0.5 ) # Pause 0.5 sec between replays

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
        idx = int( round( self.sleepSlider.value ) )
        self.sleep = sleepTimes[idx]
        self.sleepLabel.text = "Frame dt = {:1.4f} s".format( self.sleep )

    def setRotation( self, slider ):
        self.doRotation = ( abs( self.rotation ) > 0.005 )
        self.rotation = slider.value

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

    def raiseMax( self, isDouble = False ):
        moov = MooView.viewList[MooView.colorbarViewIdx]
        valRange = moov.valMax - moov.valMin
        if isDouble:
            moov.valMax = moov.valMin + 2 * valRange
        else:
            moov.valMax = moov.valMin + valRange * 1.1111111111111111111111
        return moov.valMax

    def lowerMax( self, isDouble = False ):
        moov = MooView.viewList[MooView.colorbarViewIdx]
        valRange = moov.valMax - moov.valMin
        if isDouble:
            moov.valMax = moov.valMin + 0.5 * valRange
        else:
            moov.valMax = moov.valMin + 0.9 * valRange
        return moov.valMax

    def raiseMin( self, isDouble = False ):
        moov = MooView.viewList[MooView.colorbarViewIdx]
        valRange = moov.valMax - moov.valMin
        if isDouble:
            moov.valMin = moov.valMmax - 0.5 * valRange
        else:
            moov.valMin = moov.valMax - valRange * 0.9
        return moov.valMin

    def lowerMin( self, isDouble = False ):
        moov = MooView.viewList[MooView.colorbarViewIdx]
        valRange = moov.valMax - moov.valMin
        if isDouble:
            moov.valMin = moov.valMax - 2 * valRange
        else:
            moov.valMin = moov.valMax - valRange * 1.1111111111111111111111
        return moov.valMin

    def scaleColorbar( self, event ):
        loc = event.pos
        if loc.y > 6:
            idx = MooView.colorbarViewIdx + 1
            if idx >= len( MooView.viewList ):
                idx = 0
            self.selectCbar( idx )
            return
        if loc.y > 4 and loc.y < 4.7:
            ret = self.lowerMax( self.scene.mouse.shift )
            self.barMax.text = "{:.3e}".format(ret)
        elif loc.y > 4.85 and loc.y < 5.5:
            ret = self.raiseMax( self.scene.mouse.shift )
            self.barMax.text = "{:.3e}".format(ret)
        elif loc.y > -3.25 and loc.y < -2.6:
            ret = self.raiseMin( self.scene.mouse.shift )
            self.barMin.text = "{:.3e}".format(ret)
        elif loc.y > -3.95 and loc.y < -3.45:
            ret = self.lowerMin( self.scene.mouse.shift )
            self.barMin.text = "{:.3e}".format(ret)
        for moov in MooView.viewList:
            moov.drawables_[0].updateLimits( moov.valMin, moov.valMax )

    def innerColorbar( self, title, bg ):
        barWidth = SCALE_SCENE * 1.5
        if ( bgLookup(bg).mag < 1 ):
            barTextColor = vp.color.white
        else:
            barTextColor = vp.color.black
        self.colorbar = vp.canvas( title = "Datasets =  ", width = barWidth, height = self.swy * SCALE_SCENE, background = bgLookup(bg), align = 'left', range = 1, autoscale = False )
        self.colorbar.userzoom = False
        self.colorbar.userspin = False
        self.colorbar.userpan = False
        #self.colorbar.bind( 'keydown', self.scaleColorbar )
        self.colorbar.bind( 'click', self.scaleColorbar )
        height = 0.10
        width = 5
        axOrigin = vp.vector( 0, -5.5, 0 )
        for idx, rgb in enumerate( self.rgb ):
            cbox = vp.box( canvas = self.colorbar, pos = vp.vector( 0, height * (idx - 26), 0), width = width, height = height, color = rgb )
        barName = self.title.replace( '.', '\n' )
        barName = barName.replace( '/', '\n', 1 )
        self.barName = vp.label( canvas = self.colorbar, align = 'left', pixel_pos = True, pos = vp.vector( 2, (self.swy - 0.32) * SCALE_SCENE, 0), text = barName, height = 15, color = barTextColor, box = False, opacity = 0 )
        self.barMin = vp.label( canvas = self.colorbar, align = 'center', pixel_pos = True, pos = vp.vector( barWidth/2, self.swy * SCALE_SCENE * 0.22, 0), text = "{:.3e}".format(self.valMin), height = 12, color = barTextColor, box = False, opacity = 0 )
        self.barMax = vp.label( canvas = self.colorbar, align = 'center', pixel_pos = True, pos = vp.vector( barWidth/2, (self.swy - 1.2) * SCALE_SCENE, 0), text = "{:.3e}".format(self.valMax), height = 12, color = barTextColor, box = False, opacity = 0 )
        self.xAx = vp.cylinder( canvas = self.colorbar, pos = axOrigin, axis = vp.vector( 0.8, 0, 0 ), radius = 0.04, color = vp.color.red )
        self.yAx = vp.cylinder( canvas = self.colorbar, pos = axOrigin, axis = vp.vector( 0, 0.8, 0 ), radius = 0.04, color = vp.color.green )
        self.zAx = vp.cylinder( canvas = self.colorbar, pos = axOrigin, axis = vp.vector( 0, 0, 0 ), radius = 0.04, color = vp.color.blue )
        self.axisLength = vp.label( pos = axOrigin + vp.vector(0, 1, 0), text = "1.00 <i>u</i>m", color = barTextColor, box = False )

    def makeColorbar( self, doOrnaments = True, colorscale = 'jet', bg = 'default' ):
        title = None
        if doOrnaments:
            title = MooView.consolidatedTitle + "\n"
        self.innerColorbar( title, bg )
        if doOrnaments:
            for idx, mv in enumerate( MooView.viewList ):
                chk = vp.checkbox( bind = mv.toggleView, checked = True, text = mv.title + "    ",  pos = self.colorbar.title_anchor )
                chk.idx = idx
                self.cbox.append( chk )
            self.colorbar.append_to_title("\n")
            self.timeLabel = vp.wtext( text = "Time =  0.000 s\n", pos = self.colorbar.title_anchor )
            self.sleepLabel = vp.wtext( text = "Frame dt = 0.0050 s", pos = self.colorbar.title_anchor )
            self.sleepSlider = vp.slider( pos = self.colorbar.title_anchor, length = 200, bind = self.setSleepTime, min = 0, max = len( sleepTimes ) -1, value = min( len( sleepTimes ), 2  ) )
            self.replayButton = vp.button( text = "Start Replay", pos = self.colorbar.title_anchor, bind=self.toggleReplay, disabled = True )
            self.rotateLabel = vp.wtext( text = "      Rotation", pos = self.colorbar.title_anchor )
            self.rotateSlider = vp.slider( pos = self.colorbar.title_anchor, length = 200,  bind=self.setRotation, min = -0.1, max = 0.1, value = 0.0 )
            self.colorbar.append_to_title("\n")

    def selectCbar( self, idx ):
        MooView.colorbarViewIdx = idx
        moov = MooView.viewList[idx]
        view0 = MooView.viewList[0]
        view0.barName.text = view0.cbox[idx].text.replace( '.', '\n' )
        view0.barName.text = view0.barName.text.replace( '/', '\n', 1 )
        view0.barMin.text = "{:.3e}".format(moov.valMin)
        view0.barMax.text = "{:.3e}".format(moov.valMax)

    def toggleView( self, cbox ):
        for d in self.drawables_:
            d.setVisible( cbox.checked )
        if cbox.checked:    # The colorbar is assigned to selected view:
            self.selectCbar( cbox.idx )

    def pickObj( self ):
        obj = self.scene.mouse.pick
        if obj == None:
            return
        elmPath = self.innerPickObj( obj )
        if elmPath:
            self.handlePick( elmPath )
            return
        elif self.viewIdx == 0: 
            for view in MooView.viewList[1:]:
                if view.colorbar == None:
                    elmPath = view.innerPickObj( obj )
                    if elmPath:
                        self.handlePick( elmPath )
                        return
        print( "Object {} not found on view {}".format( obj, self.title ) )

    def innerPickObj( self, obj ):
        for dr in self.drawables_:
            elmPath = dr.findDisplayObject( obj )
            if elmPath:
                return (elmPath[0], elmPath[1], dr)
        return None

    def handlePick( self, elmPath ):
        path, field, drawable = elmPath
        if self.plotFlag_:
            drawable.plotHistory( path, field, self.graph, self.graphPlot1 )
        else:
            print( path, field )



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
        if self.viewIdx == (MooView.viewIdx-1):
            MooView.viewList[0].graph = vp.graph( title = "Graph", xtitle = "Time (s)", ytitle = " Units here", width = 700, fast=False, align = "left" )
            MooView.viewList[0].graphPlot1 = vp.gcurve( color = vp.color.blue, interval=-1)
            
    def rotateFunc(self ):
        if self.doRotation and abs( self.rotation ) < 2.0 * 3.14 / 3.0:
            self.scene.forward = vp.rotate( self.scene.forward, angle = self.rotation, axis = self.scene.up )
            self.updateAxis()

    def updateValues( self, simTime ):
        self.simTime = simTime
        for i in self.drawables_:
            i.updateValues( simTime )
        self.rotateFunc()
        if self.viewIdx == 0:
            self.timeLabel.text = "Time = {:7.3f} s\n".format( simTime )
            vp.sleep( self.sleep )

    def replaySnapshot( self, idx ):
        for i in self.drawables_:
            simTime = i.replaySnapshot( idx )
        self.rotateFunc()
        if self.viewIdx == 0:
            self.timeLabel.text = "Time = {:7.3f} s\n".format( simTime )
            self.updateAxis()

    def doAutoscale( self ):
        if self.drawables_[0].dataWrapper_.numObj() == 0:
            print( "Warning: No values to display in Moogli view ", self.title )
            return
        cmin = self.drawables_[0].dataWrapper_.coordMin_
        cmax = self.drawables_[0].dataWrapper_.coordMax_
        diamax = max( self.drawables_[0].dataWrapper_.getCoords()[:,6] )
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
            moov = MooView.viewList[MooView.colorbarViewIdx]
            for dbl in moov.drawables_:
                dbl.diaScale *= 1.0 - moov.sensitivity * 4
                dbl.updateDiameter()
            return
        if event.key == "D":
            moov = MooView.viewList[MooView.colorbarViewIdx]
            for dbl in moov.drawables_:
                dbl.diaScale *= 1.0 + moov.sensitivity * 4
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

class DataWrapper:
    ''' Class for interfacing between moogli and the data source. Currently
    implemented for MOOSE and for nsdf reader.
    '''
    def __init__( self, field ):
        self.coordMin_ = np.zeros( 3 )
        self.coordMax_ = np.ones( 3 )
        self.field_ = field
        self.objList_ = []

    def getValues( self ):
        return np.zeros( 1 )

    def numObj( self ):
        return len( self.objList_ )

    def getCoords( self ):
        return np.array( [] )

    def getMinMax( self ):
        nmin = np.amin(self.coords_, axis = 0)
        self.coordMin_ = np.amin( np.array( [nmin[0:3], nmin[3:6]] ), axis = 0 )
        nmax = np.amax(self.coords_, axis = 0)
        self.coordMax_ = np.amax( np.array( [nmax[0:3], nmax[3:6]] ), axis = 0 )
    def objPathFromIndex( self, idx ):
        if idx < len( self.objList_ ):
            return self.objList_[idx].path
        return None

    def advance( self, simTime ):
        # Checks that the simTime has crossed upcomingTime
        return True # used for multi timestep cases.

    def getHistory( self, path, field ):
        # stub function. Derived classes fill it in and return useful values
        return [0, 1, 2, 3], [ 1, 4, 9, 16]

class MooDrawable:
    ''' Base class for drawing things'''
    def __init__( self,
        dataWrapper,
        colormap,
        lenScale, 
        diaScale, 
        fieldScale, 
        autoscale,
        valMin, valMax
    ):
        self.dataWrapper_ = dataWrapper
        self.lenScale = lenScale
        self.diaScale = diaScale
        self.fieldScale = fieldScale
        self.colormap = colormap
        self.autoscale = autoscale
        self.valMin = valMin
        self.valMax = valMax
        self.segments = []
        self.snapshot = []
        self.visible = True
        #cmap = plt.get_cmap( self.colormap, lut = NUM_CMAP )
        #self.rgb = [ list2vec(cmap(i)[0:3]) for i in range( NUM_CMAP ) ]

    def updateValues( self, simTime ):
        if self.dataWrapper_.advance( simTime ):
            self.val = self.dataWrapper_.getValues() * self.fieldScale
        else:
            return

        if self.autoscale:
            valMin = min( self.val )
            valMax = max( self.val )
        else:
            valMin = self.valMin
            valMax = self.valMax
        scaleVal = NUM_CMAP * (self.val - valMin) / (valMax - valMin)
        #indices = scaleVal.ndarray.astype( int )
        indices = np.maximum( np.minimum( scaleVal, NUM_CMAP-0.5), 0.0).astype(int)

        # Have to figure how this will work with multiple update rates.
        self.snapshot.append( [simTime, self.val] )

        self.displayValues( indices )

    def updateLimits( self, vmin, vmax ):
        if self.autoscale:
            valMin = min( self.val )
            valMax = max( self.val )
        else:
            valMin = self.valMin = vmin
            valMax = self.valMax = vmax
        scaleVal = NUM_CMAP * (self.val - valMin) / (valMax - valMin)
        indices = np.maximum( np.minimum( scaleVal, NUM_CMAP-0.5), 0.0).astype(int)
        self.displayValues( indices )


    def displayValues( self, indices ):
        for idx, seg in zip( indices, self.segments ): 
            seg.color = self.rgb[ idx]
            #seg.radius = self.diaScale  * self.activeDia[idx]

    def replaySnapshot( self, idx ):
        if idx >= len( self.snapshot ):
            return 0.0
        scaleVal = NUM_CMAP * (self.snapshot[idx][1] - self.valMin) / (self.valMax - self.valMin)
        indices = np.maximum( np.minimum( scaleVal, NUM_CMAP-0.5), 0.0).astype(int)
        self.displayValues( indices )
        return self.snapshot[idx][0]    # return frame time

    def updateDiameter( self ):
        dia = self.dataWrapper_.getCoords()[:,6]
        for s, w in zip( self.segments, dia ):
            s.radius = self.diaScale * w / 2.0

    def cylinderDraw( self, _scene ):
        for idx, coord in enumerate( self.dataWrapper_.getCoords() ):
            v0 = list2vec( coord[0:3] )
            v1 = list2vec( coord[3:6] )
            radius = self.diaScale * coord[6] / 2.0
            opacity = self.opacity[idx]
            rod = vp.cylinder( canvas = _scene, pos = v0, axis = v1 - v0, radius = radius, opacity = opacity )
            self.segments.append( rod )

    def findDisplayObject( self, obj ):
        try:
            idx = self.segments.index( obj )
            return self.dataWrapper_.objPathFromIndex( idx ), self.dataWrapper_.field_
        except ValueError:
            return None

    def plotHistory( self, path, field, graph, plot ):
        t, v = self.dataWrapper_.getHistory( path, field )
        if len( t ) == 0:
            print( "No data history for '", path, ".", field )
            return
        #self.graph = vp.graph( title = path + "." + field, xtitle = "Time (s)", ytitle = field + " Units here", width = 800, fast=False, pos=self.colorbar.caption_anchor )
        graph.title = path + "." + field
        dat = [[x,y] for x, y in zip( t, v ) ]
        plot.data = dat

    def setVisible( self, state ):
        if self.visible == state:
            return
        self.visible = state
        for s in self.segments:
            s.visible = state


#####################################################################

class MooNeuron( MooDrawable ):
    ''' Draws collection of line segments of defined dia and color'''
    def __init__( self, 
        dataWrapper,
        field = 'Vm', 
        colormap = 'jet', 
        lenScale = 1.0, diaScale = 1.0, fieldScale = 1.0,
        autoscale = False, 
        valMin = -0.1, valMax = 0.05,
    ):
        #self.isFieldOnCompt = 
            #field in ( 'Vm', 'Im', 'Rm', 'Cm', 'Ra', 'inject', 'diameter' )
        
        MooDrawable.__init__( self, dataWrapper,
                colormap = colormap, lenScale = lenScale, 
                diaScale = diaScale, fieldScale = fieldScale,
                autoscale = autoscale, 
                valMin = valMin, valMax = valMax )
        self.opacity = np.ones( dataWrapper.numObj() ) * 0.5

    def drawForTheFirstTime( self, _scene ):
        self.cylinderDraw( _scene )

#####################################################################
class MooReacSystem( MooDrawable ):
    ''' Draws collection of line segments of defined dia and color'''
    def __init__( self,
        dataWrapper,
        colormap = 'jet', 
        lenScale = 1e0, diaScale = 1.0, fieldScale = 1.0, 
        autoscale = False, 
        valMin = 0.0, valMax = 1.0
    ):
        
        MooDrawable.__init__( self, dataWrapper,
                colormap = colormap, lenScale = lenScale, 
                diaScale = diaScale, fieldScale = fieldScale, 
                autoscale = autoscale, 
                valMin = valMin, valMax = valMax )
        self.opacity = np.ones( dataWrapper.numObj() )


    def drawForTheFirstTime( self, _scene ):
        if self.dataWrapper_.numObj() == 0:
            return
        mt = self.dataWrapper_.meshType()
        if mt in ["NeuroMesh", "CylMesh", "SpineMesh", "PsdMesh"]:
            self.cylinderDraw( _scene )
        elif mt == "SpineMesh":
            self.spineDraw( _scene )
        elif mt == "PresynMesh":
            self.presynDraw( _scene )
        elif mt == "EndoMesh":
            self.endoDraw( _scene )

    def spineDraw( self, _scene ):
        # Spine entry has head[3], shaft[3], root[3], dia.
        for idx, coord in enumerate( self.dataWrapper_.getCoords() ):
            v0 = list2vec( coord[0:3] )
            v1 = list2vec( coord[3:6] )
            radius = self.diaScale * coord[6] / 2.0
            opacity = self.opacity[idx]
            rod = vp.cylinder( canvas = _scene, pos = v0, axis = v1 - v0, radius = radius, opacity = opacity )
            self.segments.append( rod )

    def presynDraw( self, _scene ):
        for idx, coord in enumerate( self.dataWrapper_.getCoords() ):
            v0 = list2vec( coord[0:3] )
            v1 = list2vec( coord[3:6] )
            radius = self.diaScale * coord[6] / 2.0
            opacity = self.opacity[idx]
            cone = vp.cone( canvas = _scene, pos = v0, axis = v0 - v1, radius = radius, opacity = opacity )
            self.segments.append( cone )

    def endoDraw( self, _scene ):
        for idx, coord in enumerate( self.dataWrapper_.getCoords() ):
            v0 = list2vec( coord[0:3] )
            v1 = list2vec( coord[3:6] )
            radius = self.diaScale * coord[6] / 2.0
            opacity = self.opacity[idx]
            sphere = vp.sphere( canvas = _scene, pos = (v0 + v1)/2.0, radius = radius, opacity = opacity )
            self.segments.append( sphere )
