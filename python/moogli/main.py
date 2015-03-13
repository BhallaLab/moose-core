import _moogli

"""
class Visualizer(MorphologyViewer):

    def __init__(parent = None, callback = None):
        self.callback   = callback
        self.morphology = moogli.Morphology("morph", 1)
        desktop = QtGui.QApplication.desktop()
        self.visualizer  = MorphologyEditor( self.morphology
                                           , desktop.screenGeometry().width()
                                           , desktop.screenGeometry().height()
                                           )
        self._timer = QtCore.QTimer(self)
        self._update()

    def read_morphology_from_moose(path = ""):
        import moose
        compartments = moose.wildcardFind(path + "/##[ISA=CompartmentBase]")
        for compartment in compartments:
            try:
                parent_compartment = compartment.neighbors["raxial"][0]
                proximal_diameter  = parent_compartment.diameter
            except IndexError:
                proximal_diameter = compartment.diameter
            self.morphology.add_compartment( compartment.path
                                           , compartment.parent.path
                                           , compartment.x0          * 10000000
                                           , compartment.y0          * 10000000
                                           , compartment.z0          * 10000000
                                           , proximal_diameter       * 10000000
                                           , compartment.x           * 10000000
                                           , compartment.y           * 10000000
                                           , compartment.z           * 10000000
                                           , compartment.diameter    * 10000000
                                           )

    def _update(self):
        if self.callback is not None:
            self.callback(morphology, self)
        self.frame()
        self._timer.timeout.connect(self._update)
        self._timer.start(0)

def main():
    app = QtGui.QApplication(sys.argv)
    filename = os.path.join( os.path.split(os.path.realpath(__file__))[0]
                           , "../../Demos/neuroml/CA1/CA1.morph.pop.xml")


    # filename = os.path.join( os.path.split(os.path.realpath(__file__))[0]
    #                        , "../neuroml/PurkinjeCellPassivePulseInput/PurkinjePassive.net.xml")
    # filename = os.path.join( os.path.split(os.path.realpath(__file__))[0]
    #                        , "../neuroml/OlfactoryBulbPassive/OBpassive_numgloms3_seed750.0.xml")


    visualizer = Visualizer()
    visualizer.read_morphology_from_moose()
    visualizer.show()
    return app.exec_()
    # popdict, projdict = moose.neuroml.loadNeuroML_L123(filename)
    # modelRoot   = moose.Neutral("/" + os.path.splitext(os.path.basename(filename))[0])
    # element = moose.Neutral(modelRoot.path + "/model")
    # if(moose.exists("/cells"))  : moose.move("/cells"  , element.path)
    # if(moose.exists("/elec"))   : moose.move("/elec"   , modelRoot.path)
    # if(moose.exists("/library")): moose.move("/library", modelRoot.path)

if __name__ == "__main__":
    main()
"""

DISTAL          = 0
AVERAGED        = 1
PROXIMAL_DISTAL = 2


def read_morphology_from_moose(name = "", path = "", radius = DISTAL):
    import moose
    morphology = _moogli.Morphology(name, 1)
    compartments = moose.wildcardFind(path + "/##[ISA=CompartmentBase]")
    for compartment in compartments:
        distal_diameter = compartment.diameter
        try:
            parent_compartment = compartment.neighbors["raxial"][0]
            proximal_diameter  = parent_compartment.diameter
        except IndexError:
            proximal_diameter = distal_diameter

        if   radius == DISTAL          :
            proximal_diameter = distal_diameter
        elif radius == AVERAGED        :
            distal_diameter = proximal_diameter =  ( distal_diameter
                                                   + proximal_diameter
                                                   ) / 2.0

        morphology.add_compartment(     compartment.path
                                       , compartment.parent.path
                                       , compartment.x0          * 10000000
                                       , compartment.y0          * 10000000
                                       , compartment.z0          * 10000000
                                       , proximal_diameter       * 10000000
                                       , compartment.x           * 10000000
                                       , compartment.y           * 10000000
                                       , compartment.z           * 10000000
                                       , distal_diameter         * 10000000
                                       )
    return morphology
