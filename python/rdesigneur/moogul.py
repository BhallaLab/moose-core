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

class MoogulError( Exception ):
    def __init__( self, value ):
        self.value = value
    def __str__( self ):
        return repr( self.value )

class MooView:
    ''' The MooView class is a window in which to display one or more 
    moose cells, using the MooNeuron class.'''
    def __init__( self, swx = 10, swy = 12, hideAxis = True, title = "view", showColorbar = True
    ):
        self.scene = vp.canvas( title = title, width = swx * 50, height = swy * 50, background = vp.color.white )
        #        b = vp.box(pos=vp.vector( 3.0, 0, 0), size=vp.vector(0.3, 3, 4),  color = vp.color.red)
        self.drawables_ = []
        #self.ax.margins( tight = True )
        self.sensitivity = 0.05 # radians rotation, and other adjustments
        self.sleep = 0.005  # Seconds to sleep per frame
        self.showColorbar = showColorbar
        self.scene.bind( 'keydown', self.moveView )
        self.axisButton = vp.button( text = "Show Axis", pos = self.scene.title_anchor, bind=self.toggleAxis )
        self.colorbarButton = vp.button( text = "Show Color Scale", pos = self.scene.title_anchor, bind=self.toggleColorbar )
        self.hideAxis = hideAxis
        if self.hideAxis:
            self.axisButton.text = "Hide Axis"
            self.toggleAxis()

    def addDrawable( self, n ):
        self.drawables_.append( n )
        # self.scene.objects also maintains list.

    def toggleAxis( self ):
        if self.axisButton.text == "Show Axis":
            # show axis here
            self.axisButton.text = "Hide Axis"
            self.axisButton.background = vp.color.yellow
        else:
            # hide axis here
            self.axisButton.text = "Show Axis"
            self.axisButton.background = vp.color.white


    def toggleColorbar( self ):
        if self.colorbarButton.text == "Show Color Scale":
            # show colorbar here
            self.colorbarButton.text = "Hide Color Scale"
            self.colorbarButton.background = vp.color.cyan
        else:
            # hide colorbar here
            self.colorbarButton.text = "Show Color Scale"
            self.colorbarButton.background = vp.color.white


    def firstDraw( self, rotation = 0.0, elev = 0.0, azim = 0.0 ):
        if rotation == 0.0:
            self.doRotation = False
            self.rotation = 7 # default rotation per frame, in degrees.
        else:
            self.doRotation = True
            self.rotation = rotation * 180/np.pi # arg units: radians/frame
        
        for i in self.drawables_:
            i.drawForTheFirstTime()
        print( "Exiting MooView::firstDraw" )


        #self.ax.view_init( elev = -80.0, azim = 90.0 )
        #self.colorbar = plt.colorbar( self.drawables_[0].segments )
        #self.colorbar = self.fig_.colorbar( self.drawables_[0].segments )
        #self.colorbar.set_label( self.drawables_[0].fieldInfo[3])
        #self.timeStr = self.ax.text2D( 0.05, 0.05, "Time= 0.0", transform=self.ax.transAxes)

    def updateValues( self ):
        time = moose.element( '/clock' ).currentTime
        #self.timeStr.set_text( "Time= {:.3f}".format( time ) )
        for i in self.drawables_:
            i.updateValues()
        if self.doRotation and abs( self.rotation ) < 120:
            oldCenter = self.scene.center
            oldAxis = self.scene.camera.axis
        simTime = moose.element( '/clock' ).currentTime
        self.scene.caption = "Time = {:.3f} sec".format( simTime )
        vp.sleep( self.sleep )

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
        if event.key in ["left", "l", "L"]:
            self.scene.camera.pos += vp.norm(up.cross(camAxis)) * dtheta * camDist 
            return
        if event.key in ["right", "h", "H"]:
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
            return
        if event.key == "P":
            return
        if event.key == "y": # yaw: Rotate camera around ctr - up axis.
            return
        if event.key == "Y":
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
        print( "INITING MOO DRAWABLE" )
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
        cmap = plt.get_cmap( self.colormap, lut = NUM_CMAP )
        self.rgb = [ list2vec(cmap(i)[0:3]) for i in range( NUM_CMAP ) ]
        #FieldInfo = [baseclass, fieldGetFunc, scale, axisText, min, max]

    def updateValues( self ):
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
        for idx, seg in zip( indices, self.segments ): 
            #print( "IN segments, ", idx, scaleVal[idx], len( self.rgb ) )
            seg.color = self.rgb[ idx]
            #seg.radius = self.diaScale  * self.activeDia[idx]
        return

    def updateDiameter( self ):
        print( "IN UPDATE Diameter", len( self.segments), self.diaScale)
        for s, w in zip( self.segments, self.activeDia ):
            s.radius = self.diaScale * w / 2.0

    def drawForTheFirstTime( self ):
        #print( "Coords = ",  self.activeCoords)
        #print( "Dia = ",  self.activeDia)
        for idx, coord in enumerate( self.activeCoords ):
            v0 = list2vec( coord[0] )
            v1 = list2vec( coord[1] )
            radius = self.diaScale * self.activeDia[idx] / 2.0
            opacity = self.opacity[idx]
            rod = vp.cylinder( pos = v0, axis = v1 - v0, radius = radius, opacity = opacity )
            #print( "ROD = ", rod.pos, rod.axis, rod.radius )
            self.segments.append( rod )


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
        # Matplotlib3d isn't able to do full rotations about an y axis,
        # which is what the NeuroMorpho models use, so
        # here we shuffle the axes around. Should be an option.
        #coords = np.array([[[i.x0,i.y0,i.z0],[i.x,i.y,i.z]] 
            #for i in self.compts_])
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
        print( "GING REAC SYSTEM", lenScale, diaScale )
        
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
            if meshType in ["NeuroMesh", "CylMesh", "SpineMesh", "PsdMesh"]:
                # Unfortunately at present these return radius rather than
                # diameter in argument 6. To fix.
                # Make a cylinder
                activeCoords.append( [coords[0:3], coords[3:6]] )
                self.activeDia.append( coords[6] * 2 )
            elif meshType == "PresynMesh":
                # This returns diameter in argumetn 6.
                # Hack: make each voxel as a cylinder with length == dia.
                activeCoords.append( [coords[0:3], coords[6]*coords[3:6] + coords[0:3]] )
                self.activeDia.append( coords[6] * 0.8 )
                # Returns centre as args 0,1,2, diameter as argument 3.
                # Make a hemisphere
            elif meshType == "EndoMesh":
                print( "Don't yet know EndoMesh" )
                # Make a sphere.
        self.activeCoords = np.array( activeCoords ) * self.lenScale
        self.activeDia = np.array( self.activeDia ) * self.diaScale
        self.opacity = np.ones( len( self.activeDia ) )
        self.activeObjs = self.mooObj
        return
