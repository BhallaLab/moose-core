file(GLOB files_SRC "src/*.cpp")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(muparser ${files_SRC})

