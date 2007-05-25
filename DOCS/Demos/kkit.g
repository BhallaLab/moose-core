//genesis
echo in kkit.g
float PLOTDT = 1
float SIMDT = 0.01
float FASTDT = 0.01
float CONTROLDT = 1
float MAXTIME = 100
float TRANSIENT_TIME = 2
float VARIABLE_DT_FLAG = 0
float DEFAULT_VOL = 1
float VERSION = 11.0

function kparms
end

function initdump
end

create neutral /kinetics

create neutral /graphs
create neutral /moregraphs

function enddump
	setclock 0 {SIMDT} 0
	setclock 1 {SIMDT} 1
	setclock 2 {PLOTDT}
	setclock 3 {CONTROLDT}

	useclock /kinetics/##[TYPE=Molecule] 0
	useclock /kinetics/##[TYPE=Enzyme],/kinetics/##[TYPE=Reaction] 1
	useclock /graphs/##[TYPE=Table] 2

	setfield /graphs/##[TYPE=Table] step_mode 3

/*
	str name
	foreach name ( { el /graphs/#/#[TYPE=Table] } )
		echo {name}
		echo setfield {name} step_mode 3
		setfield {name} step_mode 3
	end
	setfield /graphs/conc1/E.Co step_mode 3
	setfield /graphs/conc1/P.Co step_mode 3
	setfield /graphs/conc1/S.Co step_mode 3
	*/

	echo done reading dump
	reset
end

function save
	setfield /graphs/##[TYPE=Table] print kkit.plot
	/*
	setfield /graphs/conc1/E.Co print "E.plot"
	setfield /graphs/conc1/S.Co print "S.plot"
	setfield /graphs/conc1/P.Co print "P.plot"
	*/
end

function complete_loading
	reset
	step {MAXTIME} -t
	save
end
