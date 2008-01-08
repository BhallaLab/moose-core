// moose || genesis

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model: 2 linear cells, with a synapse. 1st cell is excitable, 2nd is not.     %
% Solver is automatically created and setup. Script runs in either MOOSE or     %
% GENESIS, depending on the state of the MOOSE flag defined below.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

int MOOSE = 1

float inj = 1.0e-10
float dt = 50e-6
float iodt = 100e-6
float runtime = 0.05
float EREST_ACT = -0.065

addalias setup_table2 setupgate
addalias tweak_tabchan tweakalpha
addalias tau_tweak_tabchan tweaktau
addalias setup_tabchan setupalpha
addalias setup_tabchan_tau setuptau

function settab2const(gate, table, imin, imax, value)
    str gate
	str table
	int i, imin, imax
	float value
	for (i = (imin); i <= (imax); i = i + 1)
		setfield {gate} {table}->table[{i}] {value}
	end
end

include simplechan.g

if ( !{MOOSE} )
	create neutral /library
end
ce /library
make_Na_mit_usb
make_K_mit_usb
create compartment compartment
ce /

/********************************************************************
**                       Model construction                        **
********************************************************************/
readcell myelin2.p /axon
readcell cable.p /cable

/* create synapse */

ce /axon/n99/i20
create spikegen spike
setfield spike thresh 0.0 \
               abs_refract .02
addmsg . spike INPUT Vm
ce /

ce /cable/c1
create synchan syn
setfield ^ Ek 0 gmax 1e-8 tau1 1e-3 tau2 2e-3
addmsg . syn VOLTAGE Vm
addmsg syn . CHANNEL Gk Ek
ce /

addmsg /axon/n99/i20/spike /cable/c1/syn SPIKE

/********************************************************************
**                       File I/0                                  **
********************************************************************/
create table /plot
call /plot TABCREATE {runtime / iodt} 0 {runtime}
useclock /plot 1
setfield /plot step_mode 3

addmsg /cable/c10 /plot INPUT Vm

/********************************************************************
**                       Simulation control                        **
********************************************************************/

/* Set up the clocks that we are going to use */
setclock 0 {dt}
setclock 1 {iodt}

/* Set the stimulus conditions */
setfield /axon/soma inject {inj}

/* Set up solver if in GENESIS */
if ( !{MOOSE} )
	create hsolve /axon/solve
	setfield /axon/solve path /axon/##[TYPE=compartment] comptmode 1  \
		chanmode 3
	call /axon/solve SETUP
	setmethod 11

	create hsolve /cable/solve
	setfield /cable/solve path /cable/##[TYPE=compartment] comptmode 1  \
		chanmode 3
	call /cable/solve SETUP
	setmethod 11
else
	// Temporary hack to sidestep issues in channel init
	setfield /axon method ee
	reset
	setfield /axon method hsolve
end

/* run the simulation */
reset
step {runtime} -time

/* write plots to file */
openfile "test.plot" a
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
closefile "test.plot"
tab2file test.plot /plot table

echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference plot is included. Present curve is in test.plot.                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"
quit
