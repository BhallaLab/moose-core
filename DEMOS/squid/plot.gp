set multiplot;
set size 0.5,0.5;
set origin 0,0.5; plot 'moose_Vm.plot', 'moose_lowpass.plot'
set origin 0, 0; plot 'moose_inject.plot';
set origin 0.5, 0; plot 'moose_ina.plot', 'moose_ik.plot'
set origin 0.5, 0.5; plot 'moose_gk.plot', 'moose_gna.plot'
unset multiplot;
