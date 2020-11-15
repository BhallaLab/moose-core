# Compiler check.
# Must support c++14
if(COMPILER_IS_TESTED)
    return()
endif()

########################### COMPILER MACROS #####################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-strict-aliasing" COMPILER_WARNS_STRICT_ALIASING )

# Turn warning to error: Not all of the options may be supported on all
# versions of compilers. be careful here.
add_definitions(-Wall
    #-Wno-return-type-c-linkage
    -Wno-unused-variable
    -Wno-unused-function
    #-Wno-unused-private-field
    )

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
        message(FATAL_ERROR "Insufficient gcc version. Minimum requried 4.9")
    endif()
    add_definitions( -Wno-unused-local-typedefs )
    add_definitions( -fmax-errors=5 )
elseif(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    add_definitions( -Wno-unused-local-typedef )
endif()

add_definitions(-fPIC)
if(COMPILER_WARNS_STRICT_ALIASING)
    add_definitions( -Wno-strict-aliasing )
endif(COMPILER_WARNS_STRICT_ALIASING)

# Disable some harmless warnings.
CHECK_CXX_COMPILER_FLAG( "-Wno-unused-but-set-variable"
    COMPILER_SUPPORT_UNUSED_BUT_SET_VARIABLE_NO_WARN
    )
if(COMPILER_SUPPORT_UNUSED_BUT_SET_VARIABLE_NO_WARN)
    add_definitions( "-Wno-unused-but-set-variable" )
endif(COMPILER_SUPPORT_UNUSED_BUT_SET_VARIABLE_NO_WARN)
set(COMPILER_IS_TESTED ON)
