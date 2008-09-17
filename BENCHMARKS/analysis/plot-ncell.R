# R code to analyse raw benchmark results obtained from benchmark.sh
# This code computes the runtime (computed as sum of time spent by the program
# in user space and in the kernel space), and plots it against the number of
# cells. Suitable for the traub91 and myelin benchmarks.

bench <- read.csv( 'shrikhand', sep = '\t' )
attach( bench )

script <- "Myelin.g"

size <- NCell[ Script == script & Command == "../../moose" & SimLength < 0.25 ]

Time <- User + System

time.moose.full <- Time[ Script == script & Command == "../../moose" & SimLength > 0.25 ]
time.moose.setup <- Time[ Script == script & Command == "../../moose" & SimLength < 0.25 ]
time.moose <- time.moose.full - time.moose.setup
time.moose.norm <- time.moose / size

time.genesis.full <- Time[ Script == script & Command == "genesis" & SimLength > 0.25 ]
time.genesis.setup <- Time[ Script == script & Command == "genesis" & SimLength < 0.25 ]
time.genesis <- time.genesis.full - time.genesis.setup
time.genesis.norm <- time.genesis / size

slice <- 1:length(size)

#~ plot (
	#~ y = time.moose[slice],
	#~ x = log10( size[slice] ),
	#~ main = "Time v/s Size",
	#~ ylab = "Time",
	#~ xlab = "log( Size )",
	#~ col = "red",
	#~ pch = 19	# filled circles
#~ )
#~ 
#~ points (
	#~ y = time.genesis[slice],
	#~ x = log10( size[slice] ),
	#~ col = "blue",
	#~ pch = "+"
#~ )

data <- cbind( size, time.genesis, time.moose )
rownames( data ) <- 1:length(size)
data
