//genesis

echo Loading kkit.g

create Neutral /kinetics
create Neutral /graphs
create Neutral /moregraphs
float FASTDT = 5e-05 
float SIMDT = 0.1
float CONTROLDT = 10
float PLOTDT = 5
float MAXTIME = 6000
float TRANSIENT_TIME = 2
float VARIABLE_DT_FLAG = 0
float DEFAULT_VOL = 1.6667e-21
float VERSION = 11.0

create Tock /tock0
create Tock /tock3

echo kkit.g loaded

function kparms
end

function enddump
	if ( VARIABLE_DT_FLAG )
		setclock 0 {FASTDT} 0
		setclock 1 {FASTDT} 1
		setclock 4 {FASTDT} 1
	else
		setclock 0 {SIMDT}	0
		setclock 1 {SIMDT}	1
		setclock 4 {SIMDT} 1
	end
	setclock 2 { CONTROLDT }
	setclock 3 { PLOTDT }
	useclock /kinetics/##[TYPE=Molecule] 0
	useclock /kinetics/##[TYPE=Reaction],/kinetics/##[TYPE=Enzyme] 1
	useclock /tock0 tick 4
	useclock /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] 3
end

function complete_loading
	reset
	addmsg /sched/cj/ct3/process /tock3/tick
	if ( VARIABLE_DT_FLAG && ( MAXTIME > TRANSIENT_TIME ) )
		step {TRANSIENT_TIME} -t
		setclock 0 {SIMDT}	0
		setclock 1 {SIMDT}	1
		setclock 4 {SIMDT} 1
		step {MAXTIME - TRANSIENT_TIME} -t
	else
		step {MAXTIME} -t
	end
	call /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] printIn kh.plot
end
