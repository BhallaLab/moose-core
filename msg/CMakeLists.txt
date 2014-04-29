file(GLOB msg_SRC 
    "*.h"
    "*.cpp"
)
include_directories(../basecode)
add_library(msg 
    Msg.cpp
    ./DiagonalMsg.cpp
    ./OneToAllMsg.cpp 
    ./OneToOneMsg.cpp
    ./SingleMsg.cpp 
    ./SparseMsg.cpp 
    ./SparseMsg.cpp 
    ./OneToOneDataIndexMsg.cpp 
    ./testMsg.cpp
    )
target_link_libraries(msg randnum basecode)
