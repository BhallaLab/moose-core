set title 'Vm in post synaptic compartment for NMDA receptor';
set ylabel 'mV';
set xlabel 'ms';
plot 'nmda_neuron.plot' u ($1):($3) w l ti 'NEURON', \
     'nmda_moose.plot' u ($1):($3) w l lt 0 ti 'MOOSE';
