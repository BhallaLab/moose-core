set title 'Vm in post synaptic compartment for NMDA receptor';
set ylabel 'mV';
set xlabel 'ms';
plot 'nmda_neuron.dat' u ($1):($2) w l ti 'NEURON', \
     'nmda_moose.dat' u ($0*1e-3):($1*1e3) w l lt 0 ti 'MOOSE';
