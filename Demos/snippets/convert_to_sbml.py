#convert Chemical Models to SBML 
#Moose needs to be compiled with libsbml: USE_SBML=1
import moose
moose.loadModel('../Genesis_files/Kholodenko.g','/Kholodenko')
moose.writeSBML('../Genesis_files/Kholodenko.xml','/Kholodenko')
