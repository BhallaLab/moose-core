file(GLOB random_SRC 
    "*.h"
    "*.cpp"
)
include_directories(../basecode)
add_library(randnum ${random_SRC})
