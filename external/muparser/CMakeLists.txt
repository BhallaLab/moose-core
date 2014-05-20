file(GLOB files_SRC "*.cpp")
include_directories(../msg ../basecode ../debug)
add_library(muparser ${files_SRC})
