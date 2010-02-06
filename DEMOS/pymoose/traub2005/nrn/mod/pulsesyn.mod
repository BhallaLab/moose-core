COMMENT
pulsesyn.mod 
For use to make pulses for the ectopic current injection into the
axons of the Traub et al 2005 model.  This replaces the role of the
curr_cellname currents in the FORTRAN code.

The point process is located in an axon compartment of the cell
receiving this (default) infrequent background stimulus.  A netstim is
set to the poisson probability desired and to this point process.

The variables amp (current amplitude in nanoamps when on) and
time_interval (milliseconds) (length of time to keep injected current
on for each event) are the only two variables that this point process
expects to be set before running the simulation.

Tom Morse, Michael Hines
ENDCOMMENT
NEURON {
	POINT_PROCESS PulseSyn
	RANGE time_interval,  i, amp, instantaneous_amp, on
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	time_interval = 0.4 (ms) <1e-9,1e9> : the time of one pulse
	amp = 0.4 (nA) : positive values depolarize the cell
}

ASSIGNED {
	i (nA)
	instantaneous_amp (nA)
	on (1) : state of Point Proc. 0 = off, 1 = on
}

INITIAL {
	instantaneous_amp = 0
	on = 0
}

BREAKPOINT {

	i = instantaneous_amp : in groucho.f the curr_cellname currents are
		: present in the diff eqs with the opposite sign
		: as the ampa and nmda therefore to be consistent
		: with this, e.g. the default value of 0.4 having the
		: same (excitatory) effect, the minus sign is included
		: in the net_receive equation marked with a (*)

}

NET_RECEIVE(weight (uS)) {
	if (flag>=1) {
		: self event arrived, terminate pulse
		instantaneous_amp = 0
		on = 0
	} else {
		: stimulus arrived, make or continue pulse
		if (on) {
			: if already processing a pulse then prolong the pulse
			net_move(t + time_interval)
		} else {
			net_send(time_interval, 1) : self event to terminate pulse
			on = 1
		}
		instantaneous_amp = - amp : see comment in BREAKPOINT.  (*)
	}
}
