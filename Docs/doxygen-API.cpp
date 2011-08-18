/**
\page AppProgInterface Applications Programming Interface, API.

\section DataStructs Key Data structures
\subsection DataStructOverview	Overview
MOOSE represents all simulation concepts through objects. The API specifies
how to manipulate these objects. Specifically, it deals with their
creation, destruction, field access, computation, and exchange of data
through messages.

Objects in MOOSE are always wrapped in the Element container class.
Each Element holds an array of Objects, sized from zero to a very large
number limited only by machine memory.

The functions and fields of each class in MOOSE are defined in Finfos:
Field Info classes. These are visible to users as fields.

Data communication between Elements (that is, their constitutent Objects)
is managed by the Msg class: messages.

These three concepts: Elements, Finfos, and Msgs - are manipulated by
the API.

\subsection DataStructElementAccess	Id: Handles for Elements

Each Element is uniquely identified by an \b Id. Ids are consistent
across all nodes and threads of a multiprocessor simulation. Ids are
basically indices to a master array of all Elements. Ids are used by
the Python system too.

\subsection DataStructObjectAccess	DataId: Handles for Objects within elements.

The \b DataId specifies a specific object within the Element. 
The DataId has a data part and a field part.
The data part is the parent object, and is used for any array Element.
The field part is for array fields within individual data entries,
in cases where the array fields themselves are accessed like Elements.
For example, in an IntFire neuron, you could have an
array of a million IntFire neurons on Element A, and each IntFire
neuron might have a random individual number, say, 10000, 15000, 8000,
etc Synapses. To index Synapse 234 on IntFire 999999 you would use a
DataId( 999999, 234).

\subsection DataStructObjId	ObjId: Fully specified handle for objects.

The ObjId is a composite of Id and DataId. It uniquely specifies any
entity in the simulation. It is consistent across nodes. 
In general, one would use the ObjId for most Object manipulation,
field access, and messaging API calls.

\subsection DataStructTrees	Element hierarchies and path specifiers.
Elements are organized into a tree hierarchy, much like a Unix file
system. This is similar to the organization in GENESIS. Since every
Element has a name, it is possible to traverse the hierarchy by specifying
a path. For example, you might access a specific dendrite in a cell as 
follows: 

\verbatim
/network/cell/dendrite[50]
\endverbatim

Note that this path specifier maps onto a single ObjId.

Some commands take a \e wildcard path. This compactly specifies a large
number of ObjIds. Some example wildcards are

\verbatim
/network/##		// All possible children of network, followed recursively
/network/#		// All children of network, only one level.
/network/ce#	// All children of network whose name starts with 'ce'
/network/cell/dendrite[]	// All dendrites, regardless of index
/soma,/axon		// The elements soma and axon
\endverbatim


\section FieldAccess Setting and Getting Field values.
\subsection FieldAccessOverview Overview
There is a family of classes for setting and getting Field values.
These are the 
\li SetGet< A1, A2... >::set( ObjId id, const string& name, arg1, arg2... )
and
\li SetGet< A >::get( ObjId id, const string& name )
functions. Here A1, A2 are the templated classes of function arguments.
A is the return class from the \e get call.

Since Fields are synonymous with functions of MOOSE objects, 
the \e set family of commands is also used for calling object functions.
Note that the \e set functions do not have a return value.

The reason there has to be a family of classes is that all functions in
MOOSE are strongly typed. Thus there are SetGet classes for up to six
arguments.


\subsection FieldAccessExamples Examples of field access.
1. If you want to call a function foo( int A, double B ) on
ObjId oid, you would do:

\verbatim
                SetGet2< int, double >::set( oid, "foo", A, B );
\endverbatim

2. To call a function bar( int A, double B, string C ) on oid:
\verbatim
                SetGet3< int, double, string >::set( oid, "bar", A, B, C );
\endverbatim

3. To assign a field value  "short abc" on object oid:
\verbatim
                Field< short >::set( oid, "abc", 123 );
\endverbatim

4. To get a field value "double pqr" on object oid:
\verbatim
                double x = Field< short >::get( oid, "pqr" );
\endverbatim

5. To assign the double 'xcoord' field on all the objects on
element Id id, which has an array of the objects:
\verbatim
                vector< double > newXcoord;
                // Fill up the vector here.
                Field< double >::setVec( id, "xcoord", newXcoord );
\endverbatim
                Note that the dimensions of newXcoord should match those of
                the target element.

                You can also use a similar call if it is just a function on id:
\verbatim
                SetGet1< double >::setVec( id, "xcoord_func", newXcoord );
\endverbatim

6. To extract the double vector 'ycoord' field from all the objects on id:
\verbatim
                vector< double > oldYcoord; // Do not need to allocate.
                Field< double >::getVec( id, "ycoord", oldYcoord );
\endverbatim

7. To set/get LookupFields, that is fields which have an index to lookup:
\verbatim
                double x = LookupField< unsigned int, double >::get( objId, field, index );
                LookupField< unsigned int, double >::set( objId, field, index, value );
\endverbatim

\section APIcalls API system calls
\subsection FieldAccessOverview Overview
There is a special set of calls on the Shell object, which function as the
main MOOSE programmatic API. These calls are all prefixed with 'do'. Here is
the list of functions:

\li Id doCreate(  string type, Id parent, string name, vector< unsigned int > dimensions );
\li bool doDelete( Id id )
\li MsgId doAddMsg( const string& msgType, ObjId src, const string& srcField, ObjId dest, const string& destField);
\li void doQuit();
\li void doStart( double runtime );
\li void doNonBlockingStart( double runtime );
\li void doReinit();
\li void doStop();
\li void doTerminate();
\li void doMove( Id orig, Id newParent );
\li Id doCopyId orig, Id newParent, string newName, unsigned int n, bool copyExtMsgs);
\li Id doFind( const string& path ) const
\li void doSetClock( unsigned int tickNum, double dt )
\li void doUseClock( string path, string field, unsigned int tick );
\li Id doLoadModel( const string& fname, const string& modelpath );
\li void doSyncDataHandler( Id elm, const string& sizeField, Id tgt );



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
