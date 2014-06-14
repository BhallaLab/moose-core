
"""config.py: Configuration file for libnml.

Last modified: Sat Jan 18, 2014  05:01PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

from xml.etree import ElementTree as slowET

neuroml_debug = False
elecPath = '/elec'
libraryPath = '/library'

neuroml_ns='http://morphml.org/neuroml/schema'
nml_ns='http://morphml.org/networkml/schema'
mml_ns='http://morphml.org/morphml/schema'
bio_ns='http://morphml.org/biophysics/schema'
cml_ns='http://morphml.org/channelml/schema'
meta_ns='http://morphml.org/metadata/schema'
xsi_ns='http://www.w3.org/2001/XMLSchema-instance'

### ElementTree parse works an order of magnitude or more faster than minidom
### BUT it doesn't keep the original namespaces,
## from http://effbot.org/zone/element-namespaces.htm , I got _namespace_map
## neuroml_ns, bio_ns, mml_ns, etc are defined above
slowET._namespace_map[neuroml_ns] = 'neuroml'
slowET._namespace_map[nml_ns] = 'nml'
slowET._namespace_map[mml_ns] = 'mml'
slowET._namespace_map[bio_ns] = 'bio'
slowET._namespace_map[cml_ns] = 'cml'
slowET._namespace_map[meta_ns] = 'meta'
slowET._namespace_map[xsi_ns] = 'xsi'

### cElementTree is much faster than ElementTree and is API compatible with the latter,
### but instead of _namespace_map above, use register_namespace below ...
### but this works only with python2.7 onwards, so stick to above,
### with import elementtree.ElementTree alongwith importing cElementTree as at
### http://dev.blogs.nuxeo.com/2006/02/elementtree-serialization-namespace-prefixes.html
#ET.register_namespace('neuroml',neuroml_ns)
#ET.register_namespace('nml',nml_ns)
#ET.register_namespace('mml',mml_ns)
#ET.register_namespace('bio',bio_ns)
#ET.register_namespace('cml',cml_ns)
#ET.register_namespace('meta',meta_ns)
#ET.register_namespace('xsi',xsi_ns)

CELSIUS_default = 32.0 # deg C # default temperature if meta:property tag for temperature is not present
ZeroCKelvin = 273.15 # zero dec C in Kelvin
VMIN = -0.1 # Volts
VMAX = 0.1 # Volts
NDIVS = 200 # number
dv = ( VMAX - VMIN ) / NDIVS # Volts

