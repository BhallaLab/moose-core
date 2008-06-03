// moose
include util.g

float  SIMDT           = 50e-6
float  PLOTDT          = {SIMDT} * 1.0
float  SIMLENGTH       = 0.25
int    N_COMPARTMENT   = 1000
float  CABLE_LENGTH    = 1e-3
float  RA              = 1.0
float  RM              = 4.0
float  CM              = 0.01
float  EM              = -0.065
float  INJECT          = 1e-10
float  DIAMETER        = 1e-6
float  LENGTH          = {CABLE_LENGTH} / {N_COMPARTMENT}

create neutral /cable
make_compartment /cable/c1 {RA} {RM} {CM} {EM} {INJECT} {DIAMETER} {LENGTH}

int i
for ( i = 2; i <= {N_COMPARTMENT}; i = i + 1 )
	make_compartment /cable/c{i} {RA} {RM} {CM} {EM} 0.0 {DIAMETER} {LENGTH}
	link_compartment /cable/c{i - 1} /cable/c{i}
end

echo "Rallpack 1 model set up."

create Neutral /plot
create Table /plot/v1
create Table /plot/vn
setfield /plot/v1,/plot/vn stepmode 3
addmsg /plot/v1/inputRequest /cable/c1/Vm
addmsg /plot/vn/inputRequest /cable/c{N_COMPARTMENT}/Vm

setclock 0 {SIMDT} 0
setclock 1 {PLOTDT} 1
useclock /plot/##[TYPE=Table] 1

reset
step {SIMLENGTH} -t
setfield /plot/v1 print "sim_cable.0"
setfield /plot/vn print "sim_cable.x"
echo "Plots written to 'sim_cable.*'"
//quit
