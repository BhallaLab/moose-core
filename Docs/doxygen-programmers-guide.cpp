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
