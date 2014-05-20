file(GLOB scheduling_SRC "*.cpp")
include_directories(../msg ../basecode ../external/debug)
add_library(scheduling ${scheduling_SRC})
