function init_plots( runtime, clock, iodt )
	float runtime
	int clock
	float iodt
	
	create neutral /plots
	
	addfield ^ _count
	setfield ^ _count 0
	
	addfield ^ _clock
	setfield ^ _clock {clock}
	
	addfield ^ _ndivs
	setfield ^ _ndivs {runtime / iodt}
	
	addfield ^ _runtime
	setfield ^ _runtime {runtime}
end

function add_plot( el, field, file )
	str el, field, file
	int count, clock, ndivs
	float runtime
	
	count = { getfield /plots _count }
	clock = { getfield /plots _clock }
	ndivs = { getfield /plots _ndivs }
	runtime = { getfield /plots _runtime }
	
	setfield /plots _count { count + 1 }
	
	create table /plots/p{count}
	setfield ^ step_mode 3
	call ^ TABCREATE { ndivs } 0 { runtime }
	addmsg {el} ^ INPUT {field}
	useclock ^ {clock}
	
	addfield ^ _file
	setfield ^ _file { file }
	
	addfield ^ _name
	setfield ^ _name { el }
end

function save_plots
	int i, count
	str file
	str name
	
	count = { getfield /plots _count }
	for ( i = 0; i < count; i = i + 1 )
		file = { getfield /plots/p{i} _file }
		name = { getfield /plots/p{i} _name }
		
		openfile {file} a
		writefile {file} "/newplot"
//		writefile {file} "/plotname "{name} // writes full path
		writefile {file} "/plotname "{getpath {name} -tail} //  writes only object name
		closefile {file}

		setfield /plots/p{i} append {file}

		openfile {file} a
		writefile {file} " "
		closefile {file}
	end
end
