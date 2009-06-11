//genesis
//moose

// This test program works the same in GENESIS and MOOSE. Loads in a 
// number of channels and runs them with a square voltage pulse,
// plots out their conductances.
// At this point 18 of the 21 channels work in MOOSE.

float RUNTIME = 0.1
float TOT_RUNTIME = RUNTIME * 3
float DT = 1e-5
float PLOTDT = 1e-4
float EREST_ACT = -0.065

setclock 0 {DT}
setclock 1 {DT}
setclock 2 {PLOTDT}

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

function setup_table( gate, table, xdivs, A, B, C, D, F )
	setupgate {gate} {table} {A} {B} {C} {D} {F} -size {xdivs} \
		-range -0.1 0.05
end

function setup_table3(gate, table, xdivs, xmin, xmax, A, B, C, D, F)
	setupgate {gate} {table} {A} {B} {C} {D} {F} -size {xdivs} \
		-range {xmin} {xmax}
end


include bulbchan.g
include traubchan.g
include traub91chan.g
include hh_tchan.g
// include FNTchan.g // doesn't even seem to work with GENESIS.
include SMSNNchan.g
include yamadachan.g

create neutral /library

ce /library
//	bulbchan here
	echo "==== Making LCa3_mit_usb"
	make_LCa3_mit_usb
	
	echo "==== Making Na_rat_smsnn"
	make_Na_rat_smsnn
	
//	This channel is created in the Yamada section below
//	echo "==== Making KA_bsg_yka"
//	make_KA_bsg_yka
	
//	This channel is created in the Yamada section below
//	echo "==== Making KM_bsg_yka"
//	make_KM_bsg_yka
	
	echo "==== Making K_mit_usb"
	make_K_mit_usb
	
	echo "==== Making K2_mit_usb"
	make_K2_mit_usb
	
	echo "==== Making Na_mit_usb"
	make_Na_mit_usb
	
//	echo "==== Making Kca_mit_usb"
//	make_Kca_mit_usb
//	MOOSE cannot deal with this channel, at this time.
//	echo "==== Making Ca_mit_conc"
//	make_Ca_mit_conc

//	Traubchan here
	echo "==== Making Ca_hip_traub"
	make_Ca_hip_traub
	
	echo "==== Making K_hip_traub"
	make_K_hip_traub
	
	echo "==== Making Kca_hip_traub"
	make_Kca_hip_traub
	
	echo "==== Making Ca_hip_traub91"
	make_Ca_hip_traub91
	
	echo "==== Making Kahp_hip_traub91"
	make_Kahp_hip_traub91
	
	echo "==== Making Kc_hip_traub91"
	make_Kc_hip_traub91
	
	echo "==== Making Na_hip_traub91"
	make_Na_hip_traub91
	
	echo "==== Making Kdr_hip_traub91"
	make_Kdr_hip_traub91
	
	echo "==== Making Ka_hip_traub91"
	make_Ka_hip_traub91

//	hh_tchan here
	echo "==== Making Na_hh_tchan"
	make_Na_hh_tchan
	
	echo "==== Making K_hh_tchan"
	make_K_hh_tchan

//	FNTchan her
//	echo "==== Making NCa_drg_fnt_tab"
//	make_NCa_drg_fnt_tab
//	
//	echo "==== Making NCa_drg_fnt"
//	make_NCa_drg_fnt

//	SMSNN channels here : 
//	Stuhmer, Methfessel, Sakmann, Noda and Numa, Eur Biophys J 1987.
	echo "==== Making Na_rat_smsnn"
	make_Na_rat_smsnn
	
//	yamadachan here: Yamada, Koch, and Adams
//	Methods in Neuronal Modeling, MIT press, ed Koch and Segev.
	echo "==== Making Na_bsg_yka"
	make_Na_bsg_yka
	
//	Currently does not work. TABCREATE fails, and a field assignment subsequently
//	leads to a crash because the A and B interpols are not present.
//	echo "==== Making Ca_bsg_yka"
//	make_Ca_bsg_yka
	
	echo "==== Making KA_bsg_yka"
	make_KA_bsg_yka
	
	echo "==== Making KM_bsg_yka"
	make_KM_bsg_yka
	
	echo "==== Making K_bsg_yka"
	make_K_bsg_yka
ce /

create compartment /compt
setfield /compt \
	Em {EREST_ACT} \
	Rm 1e9	\
	Cm 1	\
	Ra 1e9	\
	initVm	-0.065	\
	Vm	-0.065
create Ca_concen /compt/ca
setfield ^ \
	tau 0.01333 \
	B	17.4e12	\
	Ca_base	0.0 \
	Ca 0.0
		

create table /plot
call /plot TABCREATE {TOT_RUNTIME / PLOTDT} 0 {RUNTIME}
useclock /plot 2
setfield /plot step_mode 3

str chan
str chname
str temp
foreach chan ( { el /library/# } )
	copy {chan} /compt

	chname = { getpath {chan } -tail }
	echo "==== Testing channel: "{ chname }

	addmsg /compt /compt/{chname} VOLTAGE Vm
	addmsg /compt/{chname} /compt CHANNEL Gk Ek
	addmsg /compt/{chname} /plot INPUT Gk

	// Hack to control conc, Many channels don't like this message
	// but GENESIS is bad about zeroing out the Ca input variable, so
	// we need to put this in for consistency.
	addmsg /compt/ca /compt/{chname} CONCEN Ca 
	reset
	setfield /compt Vm -0.065 x 0
	step {RUNTIME} -t
	setfield /compt Vm 0.01	x 0.5
	step {RUNTIME} -t
	setfield /compt Vm -0.065 x 0
	step {RUNTIME} -t
	
	openfile "test_"{chname}".plot" w
	writefile "test_"{chname}".plot" "/newplot"
	writefile "test_"{chname}".plot" "/plotname "{chname}
	closefile "test_"{chname}".plot"

	tab2file "test_"{chname}".plot" /plot table

	delete /compt/{chname}
end

quit
