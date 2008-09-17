# R code to analyse raw benchmark results obtained from benchmark.sh
# This code computes the runtime (computed as sum of time spent by the program
# in user space and in the kernel space), and plots it against the number of
# cells. For different cell sizes, this program must be run again with the
# size.filter variable set to the appropriate values. Suitable for the rallpack1
# and rallpack2 benchmarks.

bench <- read.csv( 'ncell-size', sep = '\t' )
attach( bench )

script <- "rall.2.g"

size.filter <- 9

size <- NCell[ Script == script & Command == "../../moose" & SimLength < 0.25 & Size == size.filter ]

Time <- User + System

time.moose.full <- Time[ Script == script & Command == "../../moose" & SimLength > 0.25 & Size == size.filter ]
time.moose.setup <- Time[ Script == script & Command == "../../moose" & SimLength < 0.25 & Size == size.filter ]
time.moose <- time.moose.full - time.moose.setup
time.moose.norm <- time.moose / size

time.genesis.full <- Time[ Script == script & Command == "genesis" & SimLength > 0.25 & Size == size.filter ]
time.genesis.setup <- Time[ Script == script & Command == "genesis" & SimLength < 0.25 & Size == size.filter ]
time.genesis <- time.genesis.full - time.genesis.setup
time.genesis.norm <- time.genesis / size

slice <- 1:length(size)

data <- cbind( size, time.genesis, time.moose )
rownames( data ) <- 1:length(size)
data
