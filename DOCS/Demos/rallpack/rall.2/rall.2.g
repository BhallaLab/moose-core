// moose
include util.g

float  SIMDT           = 50e-6
float  PLOTDT          = {SIMDT} * 1.0
float  SIMLENGTH       = 0.25
int    MAX_DEPTH       = 10
float  RA              = 1.0
float  RM              = 4.0
float  CM              = 0.01
float  EM              = -0.065
float  INJECT          = 0.1e-9
float  DIAMETER_0      = 16e-6
float  LENGTH_0        = 32e-6

create Neutral /cell

float diameter = {DIAMETER_0}
float length   = {LENGTH_0}
int   label    = 1
make_compartment /cell/c{label} \
	{RA} {RM} {CM} {EM} {INJECT} {diameter} {length}

int i, j
for ( i = 2; i <= MAX_DEPTH; i = i + 1 )
	diameter = {diameter / 2.0 ** (2.0 / 3.0)}
	length   = {length   / 2.0 ** (1.0 / 3.0)}
	
	for ( j = 1; j <= 2 ** (i - 1); j = j + 1 )
		label = label + 1
		make_compartment /cell/c{label} \
			{RA} {RM} {CM} {EM} 0.0 {diameter} {length}
		link_compartment /cell/c{label / 2} /cell/c{label}
	end
end

echo "Rallpack 2 model set up."

create HSolve /solve
/* Unlike GENESIS, where the solver is informed of all compartments,
 * here any single "seed" compartment from the tree suffices.
 */
setfield /solve path /cell/c1

create Neutral /plot
create Table /plot/v1
create Table /plot/vn
setfield /plot/v1,/plot/vn stepmode 3
addmsg /plot/v1/inputRequest /cell/c1/Vm
addmsg /plot/vn/inputRequest /cell/c1023/Vm

setclock 0 {SIMDT} 0
setclock 1 {SIMDT} 1
setclock 2 {PLOTDT} 0
useclock /cell/##[TYPE=Compartment] 0
useclock /solve 1
useclock /plot/##[TYPE=Table] 2
reset

step {SIMLENGTH} -t
setfield /plot/v1 print "sim_branch.0"
setfield /plot/vn print "sim_branch.x"
echo "Plots written to 'sim_branch.*'"
quit
