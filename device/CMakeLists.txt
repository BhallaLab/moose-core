include_directories(../basecode ../msg ../external/debug)
add_library(device 
    PulseGen.cpp
    DiffAmp.cpp
    PIDController.cpp
    RC.cpp
    )
