echo "
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(Bug Id: 2116525)

DESCRIPTION:
The Cell object manages the biophysics solver setup. At reset, it creates the
solver, which gets autoscheduled to t0. If Cell is autosheduled to t2 (see note
below), then a 'reset' causes Moose to crash.

FURTHER DETAIL:
The objects are meant to be scheduled in the following way:

           process
t0------------------> HSolve::processFunc
     |
     |     reinit
     L--------------> dummyFunc


           process
t2------------------> dummyFunc
     |
     |     reinit
     L--------------> Cell::reinitFunc

The ordering of ticks is: t0, t1, t2

DEBUGGING:
The call graph is:

Tick::reinit (for t0)
  |
  L----> send1 ----> Tick::reinit( for t1 )
          |           |
          |           L---(send)---> Tick::reinit( for t2 )
          |                            |
          |                            L---(send)---> Cell::reinitFunc
          |                                           Solver created here, and
          |                                           attached to t0
          |
          L----> Msg::end() returns bad value, and a loop in send1() continues
                 for longer than it should.

NOTE:
To reproduce this bug, make the following modification in biophysics/Cell.cpp:
Change the statement creating the SchedInfo (should be around line 97) to:
	static SchedInfo schedInfo[] = { { process, 0, 2 } };
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"

setclock 0 0.01 0
setclock 1 0.01 1
setclock 2 0.02 0

create Cell /cell

/* Create a compartment so that a solver will be created at reset */
create Compartment /cell/cc

reset
