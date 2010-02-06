COMMENT
sim_const.mod
Provides constant current injection and that is all.
Modified from stim.mod (del and dur and assoc code deleted):
Since this is an electrode current, positive values of i depolarize the cell
and in the presence of the extracellular mechanism there will be a change
in vext since i is not a transmembrane current but a current injected
directly to the inside of the cell.
ENDCOMMENT

NEURON {
	POINT_PROCESS IClamp_const
	RANGE amp, i
	ELECTRODE_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	amp (nA)
}
ASSIGNED { i (nA) }

INITIAL {
	i = amp
}

BREAKPOINT {
	: this point process has a NOP (no-operation) breakpoint
}
