########################### COMPILER MACROS #####################################

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )
CHECK_CXX_COMPILER_FLAG( "-Wno-strict-aliasing" COMPILER_WARNS_STRICT_ALIASING )


# Turn warning to error: Not all of the options may be supported on all
# versions of compilers. be careful here.
add_definitions(-Wall
    #-Wno-return-type-c-linkage
    -Wno-unused-variable      
    -Wno-unused-function
    #-Wno-unused-private-field
    )
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

if(COMPILER_SUPPORTS_CXX11)
    message(STATUS "Your compiler supports c++11 features. Enabling it")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions( -DENABLE_CPP11 )
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" )
    endif(APPLE)
elseif(COMPILER_SUPPORTS_CXX0X)
    message(STATUS "Your compiler supports c++0x features. Enabling it")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions( -DENABLE_CPP11 )
    add_definitions( -DBOOST_NO_CXX11_SCOPED_ENUMS -DBOOST_NO_SCOPED_ENUMS )
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" )
    endif(APPLE)
else()
    add_definitions( -DBOOST_NO_CXX11_SCOPED_ENUMS -DBOOST_NO_SCOPED_ENUMS )
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()


