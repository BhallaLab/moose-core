// moose
//
// Script to test CrossCorr class.
// 
// Following objects are created in this script:
// 	RandomSpike: rspk
// 	DummyDelay:  delay1, delay2
// 	CrossCorr:   cc
//
// These are joined thus:
// 
//      rspk ----> delay1 ----> delay2
//                   |            |
//                   +---> cc <---+
//
// 'delay2' lags 3 steps behind 'delay1' (0.03 seconds)
//

float RANDOM_FIRE_RATE = 0.1
int   DELAY            = 3
int   CC_BIN_COUNT     = 21
float CC_BIN_WIDTH     = 0.01
float THRESHOLD        = 0.5
float SIMULATION_TIME  = 10.0
str   PLOT_FILE        = "correlogram.plot"
int   PLOT_MODE        = 1    // 1:Overwrite  0:Append

create RandomSpike rspk
create DummyDelay delay1
create DummyDelay delay2
create CrossCorr cc

setclock 0 .01 0
setclock 1 .01 1
setclock 2 .01 2
setclock 3 .01 3
useclock /rspk 0
useclock /delay1 1
useclock /delay2 2
useclock /cc 3

call /rspk rateIn {RANDOM_FIRE_RATE}
setfield /##[TYPE=DummyDelay] threshold {THRESHOLD}
setfield /delay2 delay {DELAY}
setfield /cc binCount {CC_BIN_COUNT}
setfield /cc binWidth {CC_BIN_WIDTH}
setfield /cc threshold {THRESHOLD}

addmsg /rspk/stateOut /delay1/spikeIn
addmsg /delay1/spikeOut /delay2/spikeIn
addmsg /delay1/spikeTimeOut /cc/aSpikeIn
addmsg /delay2/spikeTimeOut /cc/bSpikeIn

reset
step {SIMULATION_TIME} -t
call /cc printIn {PLOT_FILE} {PLOT_MODE}
echo "Correlation histogram written to " {PLOT_FILE}
//quit

