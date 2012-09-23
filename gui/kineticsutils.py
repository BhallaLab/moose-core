from collections import namedtuple
from PyQt4 import QtCore, QtGui
import moose

def get_color(colorspec, cmap=None):
    """Takes a color specification and a colormap and tries to build a
    QColor out of it.

    Parameters
    ----------
    colorspec: int, str or some sequence with 3-4 elements 
    if int or string representation of an int, it is used as an index
    into the colormap `cmap`. Otherwise it can be a named color or an
    triplet containing r,g, b values.

    cmap: a colormap that can be looked up using integers (a sequence
    or a dict)

    Returns 
    ------- 
    if colorspec is int, or a string representation of an int, 2 *
    colorspec-th entry from cmap is taken as rgb triplet and the
    corresponding QColor returned. If cmap is not specified this
    throws an error. If colorspec is a color name or an rgb triplet,
    corresponding QColor is returned.
    """
    color = None
    cindex = None
    if isinstance(colorspec, basestring) or \
            isinstance(colorspec, int):
        try:
            cindex = 2 * int(colorspec)
        except ValueError, e_value:
            # This is a funny situation. Not only does Qt accept
            # standard color names, but also arbitrary strings.
            color = QtGui.QColor(colorspec)
            if not color.isValid():
                raise Exception('Invalid color name: %s' % (colorspec))
    elif isinstance(colorspec, collections.Sequence):
        if len(colorspec) < 3:
            raise Exception('Invalid color specified. Need at least (r, g, b)')
        color = QtGui.QColor(*colorspec)
    if cindex:        
        rgb = cmap[cindex]
        color = QtGui.QColor(*rgb)            
    return color
                
            
                              
def sanitize_color(facecolor, bgcolor, cmap):
    """Sanitize the colors of font face and background using colormap
    if required
    
    Parameters
    ----------
    facecolor: string or tuple or list
    The font face color. If this is a string, it is taken as the color
    name. If it is a tuple or list, it is taken as r, g, b values.

    bgcolor: string or tuple or list
    Background color for the text item.

    cmap: a map from integer values to (r,g,b) triplets.

    """
    fc = None
    bc = None
    rgb_fc = []
    rgb_bc = []
    if not facecolor:
        rgb_fc = [0, 255, 0]
    if not bgcolor:
        rgb_bc = [0, 0, 255]
    # If for some reason face and background colors are same, make the
    # face color random. I do not know how this situation can arise!
    # -Subha (2012-09-22)
    if facecolor == bgcolor:
        rgb_fc = np.random.randint(low=0, high=255, size=3)
    if rgb_fc:
        fc = get_color(rgb_fc)
    else:
        fc = get_color(facecolor, cmap)
    if rgb_bc:
        bc = get_color(rgb_bc)
    else:
        bc = get_color(bgcolor, cmap)
    return fc, bc

# This creates a named tuple for display info for kinetic objects
# x and y are coordinates
# fc = face color, bc = background color
displayinfo = namedtuple('displayinfo', 
                         ['x', 'y', 'fc', 'bc'], 
                         verbose=True)

def extract_display_info(el, cmap):
    """Extract display information from element.
    
    The assumption is there will be an Annotator object named `info`
    under `el`. `info` will contain all the relevant display fields.
    """
    miteminfo = el.path+'/info'
    annotator = moose.Annotator(miteminfo)
    textcolor = annotator.getField('textColor')
    bgcolor  =  annotator.getField('color')
    textcolor,bgcolor = sanitize_color(textcolor,bgcolor,cmap)
    return displayinfo(x=float(annotator.getField('x')), 
                       y=float(annotator.getField('y')), 
                       fc=textcolor, 
                       bc=bgcolor)

def getReactionDisplayData(mentry, cmap):
    """Extract display information for all reactions under mesh
    entry `mentry`, Use `cmap` colormap for numeric color index
    when required
    """
    ret = {}
    for el in mentry:
        for reaction in el.getNeighbors('remeshReacs'):
            dinfo = extract_display_info(reaction[0], cmap)
            for r in reaction:
                ret[r] = dinfo
    return ret

def getPoolDisplayData(mentry, cmap):
    """Extract display information for all elements connected to
    `remesh` message of the MeshEntry `mentry`.  Normally Pools
    are connected to `remesh`. For enzymatic reactions, the pool
    is a child of an Enzyme object. Whereas substrates and
    products live directly under the mesh entry.

    Returns a 2-tuple containing two dicts. The first maps enzymes
    to display info. The second maps substrate/product pools to
    display info.
    """
    enzymes = {}
    others = {}
    ret = ()
    for el in mentry:
        for pool in el.getNeighbors('remesh'):                
            parent = moose.element(pool[0].parent)
            if isinstance(parent, moose.EnzBase):
                dinfo = extract_display_info(parent, cmap)
                for p in pool:
                    enzymes[p] = dinfo
            else:
                dinfo = extract_display_info(pool, cmap)
                for p in pool:
                    others[p] = dinfo            
    return (enzymes, others)
