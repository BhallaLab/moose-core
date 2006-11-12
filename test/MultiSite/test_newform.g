//moose
include MultiSite_newform.g
SIMDT = 0.001

create MultiSite /kinetics/ms
setfield /kinetics/ms nTotal 1
setfield /kinetics/ms nSites 3
setfield /kinetics/ms nStates 2

// This one does not do anything, put in here to check.
addmsg /kinetics/M_S1/nOut /kinetics/ms/siteIn

// The Ca site
addmsg /kinetics/Ca.M_S2/nOut /kinetics/ms/siteIn

// The phosph site
addmsg /kinetics/M_S3*/nOut /kinetics/ms/siteIn

// Connect up output to the modulated reaction
addmsg /kinetics/ms/scaleOut /kinetics/mod_M_S0/scaleKfIn

// Fill up states array: site 1      site 2      site 3
setfield /kinetics/ms states[0] -1 states[1] 0 states[2] 1
setfield /kinetics/ms states[3] -1 states[4] 1 states[5] -1

// Fill up rates array
setfield /kinetics/ms rates[0] 2 // This is the high rate state
setfield /kinetics/ms rates[1] 0.5 // This is the low rate state


function do_run( time )
	float time
	if ( VARIABLE_DT_FLAG && ( time > TRANSIENT_TIME ) )
		step {TRANSIENT_TIME} -t
		setclock 0 {SIMDT}	0
		setclock 1 {SIMDT}	1
		setclock 4 {SIMDT} 1
		step {time - TRANSIENT_TIME} -t
	else
		step {time} -t
	end
end

setfield /sli_shell isInteractive 1
reset

setfield /kinetics/Ca n 0
do_run 100

float remaining_Ca = {getfield /kinetics/Ca n}
setfield /kinetics/Ca n {remaining_Ca + 1}
do_run 100

remaining_Ca = {getfield /kinetics/Ca n}
setfield /kinetics/Ca n {remaining_Ca + 1}
do_run 100

call /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] printIn newform.plot
