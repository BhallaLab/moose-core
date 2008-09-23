//genesis
//moose

// This test program works the same in GENESIS and MOOSE. Checks that
// global parameters are passed correctly into readcell.
// Note that GENESIS readcell has the nasty side-effect that if these
// parameters are reassigned internally, the globals will also change.
// MOOSE does NOT permit this.
// The cell model has a soma, which we ignore, and two dendrites. One
// of the dendrites uses the globals passed in, and the other uses
// values redefined in the readcell.
// In the readcell, RM, CM, and RA are all doubled and 
// EREST_ACT is set to -0.07


float RM = 1.0
float CM = 0.01
float RA = 1.0
float EREST_ACT = -0.065

create neutral /library

ce /library
create compartment compartment
ce /

readcell globalParms.p /globalParms

openfile "test.plot" w
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
writefile "test.plot" { getfield /globalParms/dend1 Rm }
writefile "test.plot" { getfield /globalParms/dend1 Ra }
writefile "test.plot" { getfield /globalParms/dend1 Cm }
writefile "test.plot" { getfield /globalParms/dend1 Em }
writefile "test.plot" { getfield /globalParms/dend2 Rm }
writefile "test.plot" { getfield /globalParms/dend2 Ra }
writefile "test.plot" { getfield /globalParms/dend2 Cm }
writefile "test.plot" { getfield /globalParms/dend2 Em }
closefile "test.plot"

quit
