from .NeuroML import NeuroML, loadNeuroML_L123
from .NetworkML import NetworkML
from .MorphML import MorphML
from .ChannelML import ChannelML

import tempfile 
import logging

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename='__moose.nml__.log'
    )

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('moose.nml').addHandler(console)
_logger = logging.getLogger('moose.nml')
_logger.debug("Loading NML library")
