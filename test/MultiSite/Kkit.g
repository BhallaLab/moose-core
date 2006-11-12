//genesis

// This file must be renamed if we want to run the old GENESIS-based
// kkit in this directory.

echo Loading kkit.g
setfield /sli_shell isInteractive 0

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

echo kkit.g loaded

function kparms
end

function initdump
end

function xtextload
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
	useclock /kinetics/##[TYPE=Molecule],/kinetics/##[TYPE=Table],/kinetics/##[TYPE=MultiSite] 0
	useclock /kinetics/##[TYPE=Reaction],/kinetics/##[TYPE=Enzyme],/kinetics/##[TYPE=ConcChan] 1
	useclock /graphs/##[TYPE=Plot],/moregraphs/##[TYPE=Plot] 3
end

function complete_loading
	echo loaded kkit file
end

