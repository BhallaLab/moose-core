file(GLOB scheduling_SRC 
    "*.h"
    "*.cpp"
)
include_directories(../msg)
include_directories(../basecode)
add_library(scheduling ${scheduling_SRC})
