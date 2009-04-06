/*************************************************************************
 * This is a simple IF neuron with constant current injection.
 * The frequency is defined as:
 * 
 *  f = 1 / (refractT + tau * ln((Rm * inject + Em - Vr) / (Rm * inject + Em - Vt)))
 *************************************************************************/


create neutral testIF
pushe testIF
create IntFire if1
setfield if1 \
    initVm -0.05 \
    Vt 0.0 \
    Vr -0.07 \
    Cm 1.0e-6 \
    Rm 1.0e4 \
    Em -0.07 \
    refractT 0.01 \
    inject 1e-5

create table vm_table
setfield vm_table step_mode 3
addmsg if1/Vm vm_table/inputRequest
create table event_table 
setfield event_table step_mode 3
addmsg if1/eventSrc event_table/msgInput
setclock 0 1e-6
reset
step 100e-3 -t

tab2file events.plot event_table table -overwrite
tab2file vm.plot vm_table table -overwrite
pope

