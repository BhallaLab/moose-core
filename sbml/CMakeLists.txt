file(GLOB files_SRC 
    "*.h"
    "*.cpp"
)
IF(LIBSBML_FOUND)
    add_definitions(-DUSE_SBML)
ELSE(LIBSBML_FOUND)
    add_definitions(-UUSE_SBML)
ENDIF(LIBSBML_FOUND)

include_directories(../msg)
include_directories(../basecode)
add_library(moose_sbml ${files_SRC})
