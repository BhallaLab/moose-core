//moose

create compartment /proto/foo
create compartment /proto/bar

setfield /proto/foo Em 1234
setfield /proto/bar Em 5678

showfield # name // Should list: shell sched library proto solvers.

showfield /proto/# name // should list foo and bar.

showfield /proto/# Em	// should list 1234 5678

ce /proto

showfield # Em	// should list 1234 5678
