// moose
setfield /sli_shell isInteractive 0
int INSTANTX = 1
int INSTANTY = 2
int INSTANTZ = 4
float EREST_ACT = -0.065
alias setup_tabchan setupalpha
alias setup_tabchan_tau setuptau
alias tweak_tabchan tweakalpha
alias tau_tweak_tabchan tweaktau

function settab2const(gate, table, imin, imax, value)
	str gate
	str table
	int imin, imax
	float value
	int i

	for (i = imin; i <= imax; i = i + 1)
		setfield {gate} {table}->table[{i}] {value}
	end
end

include moosebulbchan.g

create neutral /library

ce ^

    /* These are some standard channels used in .p files */
	/*
    make_Na_squid_hh
    make_K_squid_hh
    make_Na_mit_hh
    make_K_mit_hh

    make_Na_mit_tchan
    make_K_mit_tchan
	*/

    /* There are some synaptic channels for the mitral cell */
    // make_glu_mit_upi
    // make_GABA_mit_upi

    make_LCa3_mit_usb
    make_K_mit_usb
    make_KA_bsg_yka
    make_K2_mit_usb
    make_Na_mit_usb
    make_Ca_mit_conc
    make_Kca_mit_usb


create neutral /cell
readcell mit.p /cell
setfield /cell/soma Inject 5e-10
setfield /sli_shell isInteractive 1
call /library/# reinitIn

create Plot /soma
addmsg /soma/trigPlot /cell/soma/Vm
create Plot /ca
addmsg /ca/trigPlot /cell/soma/Ca_mit_conc/Ca


setclock 0 1e-5
setclock 1 1e-4
useclock /cell/##[TYPE=Compartment],/cell/##[TYPE=CaConc] 0
useclock /##[TYPE=Plot] 1

reset
step 0.1 -t
call /##[TYPE=Plot] printIn mit1e-5.plot
// quit
