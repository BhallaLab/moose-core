file(GLOB shellcode_SRC 
    "*.h"
    "*.cpp"
)
include_directories(../basecode ../msg ../external/debug )
add_library(shell ${shellcode_SRC})
target_link_libraries(shell basecode utility biophysics kinetics debug_print)
