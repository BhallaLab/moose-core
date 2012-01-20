import os

PATH_MAIN = os.path.abspath(__file__).rpartition('defaults.py')[0]
PATH_COLORMAPS = os.path.join(PATH_MAIN,'openGLHeaders','colormaps')
PATH_SAMPLES = os.path.join(PATH_MAIN,'samples')
PATH_ICONS = os.path.join(PATH_MAIN,'icons')

#1.Canvas Properties
#1.1 camera
DEFAULT_USE_CAMERA = False
#True- implement the gluLookAt function with DEFAULT_CAMERA_AT,DEFAULT_EYES_AT,DEFAULT_UPVECTOR
#True, NO navigation withing the canvas (lights move however) best suits when saving as .avi
#False - Navigation, freedom mostly
DEFAULT_CAMERA_AT = (1.0,1.0,1.0) #xyz 
DEFAULT_EYES_AT = (0.0,0.0,0.0) #xyz
DEFAULT_UPVECTOR = (1.0,0.0,0.0) #unitUPVector

#1.2 canvas background color
DEFAULT_BGCOLOR = (1.0,1.0,1.0,1.0) #rbga

#1.3 lights
#light position
DEFAULT_LIGHT_POSITION = (200.0, 200.0, 300.0) #xyz
#light colors
DEFAULT_DIFFUSE_COLOR = (0.8, 0.8, 0.8, 1.0) #rbga
DEFAULT_AMBIENT_COLOR = (0.2, 0.2, 0.2, 1.0) #rbga
DEFAULT_SPECULAR_COLOR = (0.5, 0.5, 0.5, 1.0) #rbga

#2.Drawing Properties
DEFAULT_DRAW = True
#True - AutoDraws canvas, draws all compartments in DEFAULT_DRAW_STYLE
#False - Polite, promts user, to select specific compartments to be drawn in specific styles
DEFAULT_DRAW_STYLE = 3 #1-Disks,2-Ball&Sticks,3-Cylinders,4-Capsules,5-Pyramids

#3.Visualization Properties
DEFAULT_VISUALIZE = False
#True auto visualizes everything that is available to visualize, with below properties
#False, Polite version, asks you what compartments are visualized with what colorMaps
DEFAULT_COLORMAP_SELECTION = 'jet'
#the resulting, colormap selection is PATH_COLORMAPS/DEFAULT_COLORMAP_SELECTION
DEFAULT_COLORMAP_MINVAL = -0.1
DEFAULT_COLORMAP_MAXVAL = 0.07
DEFAULT_COLORMAP_LABEL = 'Vm-Jet'

#4.Binning Properties
DEFAULT_BIN = False
DEFAULT_BIN_SIZE = 50 
DEFAULT_BIN_MODE = 3 #1-No bin,2-skipFrames,3-binMean,4-binMax

#5.Player Properties
DEFAULT_FRAMEUPDATEt = 90 #msec

#6.Save As Properties
DEFAULT_SAVE = False
DEFAULT_FILESAVE_LOCATION = PATH_MAIN
DEFAULT_SNAPSHOT_FILENAME = 'snapshot.png'
DEFAULT_MOVIE_FILENAME = 'movie.avi'

DEFAULT_MOVIE_WIDTH = 1280
DEFAULT_MOVIE_HEIGHT = 800
DEFAULT_MOVIE_FPS = 10 
