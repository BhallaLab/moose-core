ADD_LIBRARY(src STATIC
    ./globals.cpp
    ./constants.cpp
    core/Morphology.cpp
    core/Compartment.cpp
    core/SelectInfo.cpp
    core/KeyboardHandler.cpp
    core/MorphologyViewer.cpp
    core/MorphologyViewerWidget.cpp
    core/Selector.cpp
    mesh/CylinderMesh.cpp
    mesh/SphereMesh.cpp
    utility/conversions.cpp
    utility/record.cpp
    utility/stringutils.cpp
    utility/utilities.cpp
)

