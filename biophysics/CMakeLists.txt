include_directories(../basecode ../synapse ../utility ../)
include_directories(../external/muparser/include ${GSL_INCLUDE_DIRS})
FILE(GLOB BIOPHYSICS_SRC *.cpp )
add_library(biophysics ${BIOPHYSICS_SRC} )
