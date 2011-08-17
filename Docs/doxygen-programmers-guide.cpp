/**
\page ProgrammersGuide Programmer's Guide
Main or introductory text for this page.

\section UniqueSectionLabel Section Title Here
Section text inserted here.

\subsection UniqueSubsectionLabel Subsection Title Here
Subsection text inserted here.

\section PG_ProcessLoop	Process Loop
The MOOSE main process loop coordinates script commands, multiple threads
to execute those commands and carry out calculations, and data transfer
between nodes. 

\subsection PG_Threads	Threads
MOOSE runs in multithread mode by default. MOOSE uses pthreads.

1. The main thread (or the calling thread from a parser such as Python)
is always allocated.\n
2. MOOSE estimates the number of CPU cores and sets up that same number 
of compute threads. To override this number, the user can specify at the
command line how many threads to use for computation.\n
If MOOSE is running with MPI, one more thread is allocated
for controlling MPI data transfers.

MOOSE can also run in single-threaded mode. Here everything remains in the
'main' thread or the parser thread, and no other threads are spawned.

\subsection PG_ProcessLoopDetails Multithreading and the Process Loop
The MOOSE process loop coordinates input from the main thread, such as
parser commands, with computation and message passing. MOOSE has one
process loop function (processEventLoop) which it calls on all compute
threads.  All these threads
synchronize on custom-written barriers, during which a special single-
thread function is executed. 

The sequence of operations for a single-node, multithread calculation is
as follows:

1. The Process calls of all the executed objects are called. This typically
	triggers all scheduled calculations, which emit various messages. As
	this is being done on multiple threads, all messages are dumped into
	individual temporary queues, one for each thread.\n
2. The first barrier is hit. Here the swapQ function consolidates all
	the temporary queues into a single one.\n
3. All the individual threads now work on the consolidated queue to digest
	messages directed to the objects under that thread. Possibly further
	messages will be emitted. As before these go into thread-specific
	queues.\n
4. The second barrier is hit. Now the scheduler advances the clock by one
	tick.\n
5. The loop cycles back.

In addition to all this, the parser thread can dump calls into its special
queue at any time. However, the parser queue operates a mutex to 
protect it during the first barrier. During the first barrier, the 
queue entries from the parser thread are also incorporated into the 
consolidated queue, and the parser queue is flushed.

These steps are illustrated below:

@image html MOOSE_threading.gif "MOOSE threading and Process Loop"

\subsection PG_MPIProcessLoopDetails Multinode data transfer, Multithreading and the Process Loop
MOOSE uses MPI to transfer data between nodes. The message queues are
already in a format that can be transferred between nodes, so the main 
issue here is to coordinate the threads, the MPI, and the computation in
a manner that is as efficient as possible.
When carrying out MPI data transfers, things are somewhat more involved.
Here we have to additionally coordinate data transfers between many nodes.
This is done using an MPI loop (mpiEventLoop) which is called on
a single additional thread. MPI needs two buffers: one for sending and
one for receiving data. So as to keep the communications going on in 
the background, the system interleaves data transfers from each node with
computation.  The sequence of operations starts out similar to above:

1. The Process calls of all the executed objects are called. This typically
	triggers all scheduled calculations, which emit various messages. As
	this is being done on multiple threads, all messages are dumped into
	individual temporary queues, one for each thread. MPI thread is idle.\n
2. The first barrier is hit. Here the swapQ function consolidates all
	the temporary queues into a single one.\n
3. Here, rather than digest the local consolidated queue, the system
	initiates an internode data transfer. It takes the node0 consolidated
	queue, and sends it to all other nodes using MPI_Bcast. On node 0,
	the command reads the provided buffer. On all other nodes, the command
	dumps the just-received data from node 0 to the provided buffer.
	The compute threads are idle during this phase.\n
4. Barrier 2 is hit. Here the system swaps buffer pointers so that
	the just-received data is ready to be digested, and the other buffer
	is ready to receive the next chunk of data.\n
5. Here the compute threads digest the data from node 0, while the
	MPI thread sends out data from node 1 to all other nodes.\n
6. Barrier 2 comes round again, buffer pointers swap.\n
7. Compute threads digest data from node 1, while MPI thread sends out
	data from node 2 to all other nodes.\n
... This cycle of swap/(digest+send) is repeated for all nodes.\n

8. Compute threads digest data from the last node. MPI thread is idle.\n
9. In the final barrier, the clock tick is advanced.\n
10. The loop cycles back.

As before, the parser thread can dump data into its own queue, and this
is synchronized during the first barrier.

These steps are illustrated below:

@image html MOOSE_MPI_threading.gif "MOOSE threading and Process Loop with MPI data transfers between nodes."

*/

/**
\page ProgrammersGuide Programmer's Guide

\section YetAnotherSection Yet Another Section
All content on a given "page" need not be in the same comment block, or even in
the same file. This comment block can be moved to another file, and its content
will still end up on the ProgrammersGuide page.
*/

/**
\page AppProgInterface Applications Programming Interface, API.

\section ClockScheduling Clocks, Ticks, and Scheduling
\subsection ClockOverview	Overview
Most of the computation in MOOSE occurs in a special function called 
\e process,
which is implemented in all object classes that advance their internal
state over time. The role of Clocks and Ticks is to set up the sequence of
calling \e process for different objects, which may have different intervals
for updating their internal state. The design of scheduling in moose is
similar to GENESIS.

As a simple example, suppose we had six objects, which had to advance their
internal state with the following intervals:
\li \b A: 5
\li \b B: 2
\li \b C: 2
\li \b D: 1
\li \b E: 3
\li \b F: 5

Suppose we had to run this for 10 seconds. The desired order of updates 
would be:

\verbatim
Time	Objects called
1	D
2	D,B,C
3	D,E
4	D,B,C
5	D,A,F
6	D,B,C,E
7	D
8	D,B,C
9	D,E
10	D,B,C,A,F
\endverbatim

\subsection ClockReinit	Reinit: Reinitializing state variables.
In addition to advancing the simulation, the Clocks and Ticks play a closely
related role in setting initial conditions. It is required that every object
that has a \e process call, must have a matching \e reinit function. When the
command \e doReinit is given from the shell, the simulation is reinitialized
to its boundary conditions. To do so, the \e reinit function is called in the 
same sequence that the \process would have been called at time 0 (zero).
For the example above, this sequence would be:\n
D,B,C,E,A,F

In other words, the ordering is first by dt for the object, and second by 
the sequence of the object in the list.

During reinit, the object is expected to restore all state variables to their
boundary condition. Objects typically also send out messages during reinit
to specify this boundary condition value to dependent objects. For example,
a compartment would be expected to send its initial \e Vm value out to a
graph object to indicate its starting value.

\subsection ClockSetup	Setting up scheduling
The API for setting up scheduling is as follows:\n
1. Create the objects to be scheduled.\n
2. Create Clock Ticks for each time interval using

\verbatim
	doSetClock( TickNumber, dt ).
\endverbatim

	In many cases it is necessary to have a precise sequence of events
	ocurring at the same time interval. In this case, set up two or more
	Clock Ticks with the same dt but successive TickNumbers. They will
	execute in the same order as their TickNumber. \n
	Note that TickNumbers are unique. If you reuse a TickNumber, all that
	will happen is that its previous value of dt will be overridden.
	
	Note also that dt can be any positive decimal number, and does not 
	have to be a multiple of any other dt.

3. Connect up the scheduled objects to their clock ticks:

\verbatim
	doUseClock( path, function, TickNumber )
\endverbatim

Here the \e path is a wildcard path that can specify any numer of objects.\n
The \e function is the name of the \e process message that is to be used. This
is provided because some objects may have multiple \e process messages.
The \e TickNumber identifies which tick to use.

Note that as soon as the \e doUseClock function is issued, both the 
\e process and \e reinit functions are managed by the scheduler as discussed
above.

\subsection ClockSchedExample	Example of scheduling.
As an example, here we set up the scheduling for the same 
set of objects A to F we have discussed above.\n
First we set up the clocks:

\verbatim
	doSetClock( 0, 1 );
	doSetClock( 1, 2 );
	doSetClock( 2, 3 );
	doSetClock( 3, 5 );
\endverbatim

Now we connect up the relevant objects to them.

\verbatim
	doUseClock( "D", "process", 0 );
	doUseClock( "B,C", "process", 1 );
	doUseClock( "E", "process", 2 );
	doUseClock( "A,F", "process", 3 );
\endverbatim

Next we initialize them:

\verbatim
	doReinit();
\endverbatim

During the \e doReinit call, the \e reinit function of the objects would be 
called in the following sequence:
\verbatim
	D, B, C, E, A, F
\endverbatim

Finally, we run the calculation for 10 seconds:

\verbatim
	doStart( 10 );
\endverbatim

*/
