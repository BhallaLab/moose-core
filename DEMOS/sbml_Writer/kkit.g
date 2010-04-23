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

int METHOD = 1
function kparms
end

function initdump
end

// create neutral /kinetics
if ( !{exists /kinetics} )
	if ( METHOD == 1 )
		create KineticManager /kinetics
		setfield /kinetics method rk5
	end
	if ( METHOD == 2 )
		create KineticManager /kinetics
		setfield /kinetics method Gillespie1
	end
	if ( METHOD == 0 )
		create KinCompt /kinetics
	end
end

create neutral /edit
create neutral /file
create neutral /file/modpath
addfield /file/modpath value
create neutral /graphs
create neutral /moregraphs

function enddump
	setclock 0 {SIMDT} 0
	setclock 1 {SIMDT} 1
	setclock 2 {PLOTDT}
	setclock 3 {CONTROLDT}

	useclock /graphs/##[TYPE=Table] 2
	useclock /moregraphs/##[TYPE=Table] 2

	setfield /graphs/##[TYPE=Table] step_mode 3
	setfield /moregraphs/##[TYPE=Table] step_mode 3

	echo done reading dump
	reset
end

function do_save_all_plots( filename )
	str filename
	str name
	foreach name ( {el /graphs/##[TYPE=Table] } )
		openfile {filename} a
		writefile {filename} "/newplot"
		writefile {filename} "/plotname "{name}
		closefile {filename}
		setfield {name} append {filename}
	end
	foreach name ( {el /moregraphs/##[TYPE=Table] } )
		openfile {filename} a
		writefile {filename} "/newplot"
		writefile {filename} "/plotname "{name}
		closefile {filename}
		setfield {name} append {filename}
	end
end

function save
	setfield /graphs/##[TYPE=Table] print kkit.plot
end

function save2
	// str name
	setfield /graphs/##[TYPE=Table] print kkit.plot2
end

function xtextload
end

function complete_loading
	reset
	step {MAXTIME} -t
	do_save_all_plots test.plot
	exit
end
