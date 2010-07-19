include plotUtil.g

str infile = "Ca1.xml"
str outfile = "Vm.soma.plot"
str outfile1 = "Ca.soma.plot"
str target = "/CA1"

int USE_SOLVER = 1
float SIMLENGTH = 0.10
float SIMDT = 10e-6
float IODT = 100e-6
int IOCLOCK = 2

setclock 0 {SIMDT}
setclock 1 {SIMDT}
setclock 2 {IODT}

readNeuroML { infile } { target }
setfield { target }/soma_0 inject 2.0e-10 

init_plots { SIMLENGTH } { IOCLOCK } { IODT }

add_plot { target }/soma_0 Vm { outfile }  
add_plot { target }/soma_0/CaPool Ca { outfile1} 

if ( ! USE_SOLVER )
	setfield { target } method "ee"
end

ce /library
call K_CConductance TABFILL X 3000 2
call K_CConductance TABFILL Z 3000 2
call K_AHPConductance TABFILL Z 3000 2
call CaConductance TABFILL Y 3000 2
reset

step {SIMLENGTH} -time

/* Clear the output file */
openfile { outfile } w
closefile { outfile }

save_plots
//quit

