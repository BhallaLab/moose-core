create compartment cc

create table /plot
call /plot TABCREATE 100 0.0 1.0

// This fails..
addmsg /cc /plot INPUT Vm

// And this too..
addmsg /plot/msgInput /cc/Vm

quit
