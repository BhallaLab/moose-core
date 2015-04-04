include_directories(../basecode ../msg  )
add_library(shell 
	Shell.cpp	
	ShellCopy.cpp	
	ShellThreads.cpp	
	LoadModels.cpp 
	SaveModels.cpp 
	Neutral.cpp	
	Wildcard.cpp	
	testShell.cpp	
    )
