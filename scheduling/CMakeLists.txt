include_directories(../external/cuda)
include_directories(../msg)
include_directories(../basecode)

file(GLOB scheduling_SRC "*.cpp")

if(NOTIFY_PROGRESS)
    ADD_DEFINITIONS("-DNOTIFY_PROGRESS")
endif()

add_library(scheduling ${scheduling_SRC})
