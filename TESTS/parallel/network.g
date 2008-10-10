//moose
// should be run on 2 nodes.
float DT = 10e-6
float PLOTDT = 100e-6
float RUNTIME = 0.02

setclock 0 {DT}
setclock 1 {DT}
setclock 2 {PLOTDT}


include hh_tchan.g

if ({exists /library} == 0)
	create neutral /library
end

ce /library
	make_Na_hh_tchan
	make_K_hh_tchan
	create compartment compt
	copy Na_hh_tchan compt
	copy K_hh_tchan compt
	setfield compt Rm 1e8 Ra 1e8 Cm 1e-10 initVm -0.065 Em -0.065 Vm 0
	addmsg compt compt/Na_hh_tchan VOLTAGE Vm
	addmsg compt/Na_hh_tchan compt CHANNEL Gk Ek
	addmsg compt compt/K_hh_tchan VOLTAGE Vm
	addmsg compt/K_hh_tchan compt CHANNEL Gk Ek
	
	copy compt incompt 
	copy compt outcompt
	 
	create synchan outcompt/glu
	setfield outcompt/glu tau1 0.001 tau2 0.002 gmax 1e-8 Ek 0.0
	addmsg outcompt outcompt/glu VOLTAGE Vm
	addmsg outcompt/glu outcompt CHANNEL Gk Ek

	
	create spikegen incompt/axon
	setfield incompt/axon thresh -0.04 abs_refract 0.01
	addmsg incompt incompt/axon INPUT Vm
	/*
	*/

ce /

// copy /library/compt /cell1
// copy /library/compt /cell2

// addmsg /cell1/axon /cell2/glu SPIKE
create neutral /inarray@0
create neutral /outarray@1

createmap /library/incompt inarray 1 10
createmap /library/outcompt outarray 1 10
planarconnect /inarray/incompt[]/axon /outarray/outcompt[]/glu -probability 1 -sourcemask box -100 -100 100 100 -destmask box -100 -100 100 100 // 100 % connectivity
//planarweight /inarray/incompt[]/axon /outarray/outcompt[]/glu -fixed 1
//planardelay /inarray/incompt[]/axon /outarray/outcompt[]/glu -fixed 0

create table /plot@1
create neutral /plots@1
createmap /plot /plots 1 10
int i
int j


for ( i = 0; i < 10; i = i + 1 )
	call /plots/plot[{i}] TABCREATE {RUNTIME / PLOTDT} 0 {RUNTIME}
	useclock /plots/plot[{i}] 2
	setfield /plots/plot[{i}] step_mode 3
	addmsg /outarray/outcompt[{i}]/glu /plots/plot[{i}] INPUT Gk
end

reset

for (i = 0; i < 10; i = i + 1)
	for (j = 0; j < 10; j = j + 1)
		setfield /outarray/outcompt[{i}]/glu synapse[{j}].weight {i + j} synapse[{j}].delay {(i*j*1e-4)}
	end
end


step {RUNTIME/2} -t
setfield /inarray/incompt[] Vm 0.0
step {RUNTIME/2} -t

openfile "test.plot" a
writefile "test.plot" "/newplot"
writefile "test.plot" "/plotname Vm"
closefile "test.plot"
for (i = 0; i < 10; i = i + 1)
	str filename = "test.plot"
	tab2file {filename} /plots/plot[{i}] table	
end 
quit
