//	Parameter file for testing global passing into the readcell function.
// The globals are supposed to pass in RM and EREST_ACT.
// The rest are set by default.
// The parameters, unlike GENESIS, do NOT get altered when readcell 
// assigns them


soma	none	32	0	0	32
dend1    soma    0   0   100 5

*set_global	RM	2.0
*set_global	RA	2.0
*set_global	CM	0.02
*set_global	EREST_ACT	-0.070

dend2    soma    0   0   -100 5
