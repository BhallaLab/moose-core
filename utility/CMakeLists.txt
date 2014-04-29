file(GLOB files_SRC 
    "*.h"
    "*.cpp"
)
include_directories(../msg)
include_directories(../basecode)
add_library(utility ${files_SRC})
