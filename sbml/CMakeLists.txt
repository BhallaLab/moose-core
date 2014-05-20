file(GLOB files_SRC "*.cpp")
include_directories(../msg ../basecode ../external/debug)
add_library(moose_sbml ${files_SRC})
target_link_libraries(moose_sbml ${SBML_LIBRARIES})
