ADD_LIBRARY(src STATIC
    ./globals.cpp
    ./constants.cpp
    core/Compartment.cpp
    core/Moogli.cpp
    core/Morphology.cpp
    core/MorphologyViewer.cpp
    #core/Neuron.cpp
    #core/PickHandler.cpp
    core/SelectInfo.cpp
    core/Selector.cpp
    mesh/CylinderMesh.cpp
    mesh/SphereMesh.cpp
    utility/conversions.cpp
    utility/csv.cpp
    utility/fileutils.cpp
    utility/record.cpp
    utility/stringutils.cpp
    utility/utilities.cpp
)

