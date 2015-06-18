file(GLOB scheduling_SRC 
    "*.h"
    "*.cpp"
)

IF(NOTIFY_AT_EACH_SECOND)
    ADD_DEFINITIONS("-DNOTIFY_AT_EACH_SECOND")
ENDIF()

include_directories(../msg)
include_directories(../basecode)
add_library(scheduling ${scheduling_SRC})
