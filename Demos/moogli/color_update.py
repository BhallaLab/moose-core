import moogli
import moose
from moose import neuroml
from PyQt4 import Qt, QtCore, QtGui
import sys
import os
import random
import numpy as np

app = QtGui.QApplication(sys.argv)
filename = os.path.join( os.path.split(os.path.realpath(__file__))[0]
                       , "../neuroml/CA1/CA1.morph.pop.xml"
                       )
moose.neuroml.loadNeuroML_L123(filename)
morphology = moogli.read_morphology_from_moose(name = "", path = "")
morphology.create_group( "group-1"
                       , [ "/library[0]/CA1[0]/Seg13_user5_36_2126[0]"
                         , "/library[0]/CA1[0]/Seg14_user5_36_2127[0]"
                         , "/library[0]/CA1[0]/Seg15_user5_36_2128[0]"
                         , "/library[0]/CA1[0]/Seg16_user5_36_2129[0]"
                         , "/library[0]/CA1[0]/Seg17_user5_36_2130[0]"
                         , "/library[0]/CA1[0]/Seg18_user5_36_2131[0]"
                         , "/library[0]/CA1[0]/Seg0_user5_35_2104[0]"
                         , "/library[0]/CA1[0]/Seg1_user5_35_2105[0]"
                         , "/library[0]/CA1[0]/Seg2_user5_35_2106[0]"
                         , "/library[0]/CA1[0]/Seg3_user5_35_2107[0]"
                         , "/library[0]/CA1[0]/Seg4_user5_35_2108[0]"
                         , "/library[0]/CA1[0]/Seg5_user5_35_2109[0]"
                         , "/library[0]/CA1[0]/Seg6_user5_35_2110[0]"
                         , "/library[0]/CA1[0]/Seg7_user5_35_2111[0]"
                         , "/library[0]/CA1[0]/Seg8_user5_35_2112[0]"
                         , "/library[0]/CA1[0]/Seg0_user5_46_2206[0]"
                         , "/library[0]/CA1[0]/Seg1_user5_46_2207[0]"
                         , "/library[0]/CA1[0]/Seg2_user5_46_2208[0]"
                         , "/library[0]/CA1[0]/Seg3_user5_46_2209[0]"
                         , "/library[0]/CA1[0]/Seg4_user5_46_2210[0]"
                         , "/library[0]/CA1[0]/Seg5_user5_46_2211[0]"
                         , "/library[0]/CA1[0]/Seg0_user5_45_2195[0]"
                         , "/library[0]/CA1[0]/Seg1_user5_45_2196[0]"
                         , "/library[0]/CA1[0]/Seg2_user5_45_2197[0]"
                         , "/library[0]/CA1[0]/Seg3_user5_45_2198[0]"
                         , "/library[0]/CA1[0]/Seg4_user5_45_2199[0]"
                         , "/library[0]/CA1[0]/Seg5_user5_45_2200[0]"
                         , "/library[0]/CA1[0]/Seg6_user5_45_2201[0]"
                         , "/library[0]/CA1[0]/Seg7_user5_45_2202[0]"
                         , "/library[0]/CA1[0]/Seg8_user5_45_2203[0]"
                         , "/library[0]/CA1[0]/Seg9_user5_45_2204[0]"
                         , "/library[0]/CA1[0]/Seg10_user5_45_2205[0]"
                         , "/library[0]/CA1[0]/Seg0_soma_0_0[0]"
                         ]
                       , 10.0
                       , 200.0
                       , [1.0, 0.0, 0.0, 1.0]
                       , [0.0, 1.0, 0.0, 1.0]
                       )
morphology.create_group( "group-2"
                       , [ "/library[0]/CA1[0]/Seg3_apical_dendrite_5_929[0]"
                         , "/library[0]/CA1[0]/Seg4_apical_dendrite_5_930[0]"
                         , "/library[0]/CA1[0]/Seg5_apical_dendrite_5_931[0]"
                         , "/library[0]/CA1[0]/Seg6_apical_dendrite_5_932[0]"
                         , "/library[0]/CA1[0]/Seg7_apical_dendrite_5_933[0]"
                         , "/library[0]/CA1[0]/Seg8_apical_dendrite_5_934[0]"
                         , "/library[0]/CA1[0]/Seg9_apical_dendrite_5_935[0]"
                         , "/library[0]/CA1[0]/Seg10_apical_dendrite_5_936[0]"
                         , "/library[0]/CA1[0]/Seg11_apical_dendrite_5_937[0]"
                         , "/library[0]/CA1[0]/Seg0_user5_13_1888[0]"
                         , "/library[0]/CA1[0]/Seg1_user5_13_1889[0]"
                         , "/library[0]/CA1[0]/Seg2_user5_13_1890[0]"
                         , "/library[0]/CA1[0]/Seg3_user5_13_1891[0]"
                         , "/library[0]/CA1[0]/Seg4_user5_13_1892[0]"
                         , "/library[0]/CA1[0]/Seg5_user5_13_1893[0]"
                         , "/library[0]/CA1[0]/Seg6_user5_13_1894[0]"
                         , "/library[0]/CA1[0]/Seg7_user5_13_1895[0]"
                         , "/library[0]/CA1[0]/Seg8_user5_13_1896[0]"
                         , "/library[0]/CA1[0]/Seg9_user5_13_1897[0]"
                         , "/library[0]/CA1[0]/Seg10_user5_13_1898[0]"
                         , "/library[0]/CA1[0]/Seg11_user5_13_1899[0]"
                         , "/library[0]/CA1[0]/Seg12_user5_13_1900[0]"
                         , "/library[0]/CA1[0]/Seg13_user5_13_1901[0]"
                         , "/library[0]/CA1[0]/Seg0_apical_dendrite_34_1332[0]"
                         , "/library[0]/CA1[0]/Seg1_apical_dendrite_34_1333[0]"
                         , "/library[0]/CA1[0]/Seg2_apical_dendrite_34_1334[0]"
                         , "/library[0]/CA1[0]/Seg3_apical_dendrite_34_1335[0]"
                         , "/library[0]/CA1[0]/Seg4_apical_dendrite_34_1336[0]"
                         , "/library[0]/CA1[0]/Seg5_apical_dendrite_34_1337[0]"
                         , "/library[0]/CA1[0]/Seg6_apical_dendrite_34_1338[0]"
                         , "/library[0]/CA1[0]/Seg7_apical_dendrite_34_1339[0]"
                         , "/library[0]/CA1[0]/Seg8_apical_dendrite_34_1340[0]"
                         , "/library[0]/CA1[0]/Seg9_apical_dendrite_34_1341[0]"
                         , "/library[0]/CA1[0]/Seg10_apical_dendrite_34_1342[0]"
                         , "/library[0]/CA1[0]/Seg11_apical_dendrite_34_1343[0]"
                         , "/library[0]/CA1[0]/Seg12_apical_dendrite_34_1344[0]"
                         ]
                       , 0.0
                       , 1.0
                       , [0.0, 1.0, 0.0, 1.0]
                       , [0.0, 0.0, 1.0, 1.0]
                       )
def callback(morphology, viewer):
    morphology.set_color( "group-1"
                        , np.random.random_sample((33,)) * (100.0 - 20.0) + 20.0
                        )
    morphology.set_color( "group-2"
                        , np.random.random_sample((36,))
                        )
    return True

viewer = moogli.DynamicMorphologyViewerWidget(morphology)
viewer.show()
viewer.set_callback(callback)
app.exec_()
