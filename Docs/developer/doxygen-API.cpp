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

\subsection DataStructDataAccess	DataHandlers: Handles for Objects within elements.
The \b DataHandler is a virtual base class that contains and access data
belonging to an Element. It deals with creation, resizing, lookup and
destruction of the data. It also provides a 'forall' function to
efficiently iterate through all data entries and operate upon them. The
partitioning of data entries between threads and nodes is done by the
DataHandler. The 'forall' call on worker threads only operates on the 
subset of data entries assigned to that thread. It is safe and recommended
to call 'forall' on all worker threads.

\subsection DataStructObjectAccess	DataId: Identifiers for Objects within elements.

The \b DataId specifies a specific object within the Element. It is used
by the DataHandler to look up the data.
The DataId contains an unsigned long long, which acts like a very large
linear index. This linear index is partitioned by the DataHandlers in
various complicated ways. For programmer purposes it is important to note
that the linear index is not always contiguous. 
When an Element or its DataHandler is resized, this invalidates all DataIds
previously found for that Element. This is because the indexing will have
changed.

\subsection DataStructObjId	ObjId: Fully specified handle for objects.

The ObjId is a composite of Id and DataId. It uniquely specifies any
entity in the simulation. It is consistent across nodes. 
In general, one would use the ObjId for most Object manipulation,
field access, and messaging API calls.
The ObjId can be initialized using a string path of an object. 
The string path of an object can be looked up from its ObjId.

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

Path specifiers can be arbitrarily nested. Additionally, one can have
multidimensional arrays at any level of nesting. Here is an example 
path with nested arrays:

\verbatim
/network/layerIV/cell[23][34]/dendrite[50]/synchan/synapse[1234]
\endverbatim

\subsection ObjIdAndPaths	ObjIds, paths, and dimensions.

This section refers to the unit test \e testShell.cpp:testObjIdToAndFromPath().

Suppose we create an Element tree with the following dimensions:
\verbatim
f1[0..9]
f2
f3[0..22]
f4
f5[0..8][0..10]
\endverbatim

To do this, we need to first create f1 with dims = {10}, then f2 upon f1 with
dims = {} (or equivalently, using dims = {1}), then f3 upon f2 with dims = {23} and so on.
Note that when you create f2 with dims = {} on f1 with dims = {10}, 
there are actually 10 instances of f2 created. You would access them as

\verbatim
/f1[0]/f2
/f1[1]/f2
/f1[2]/f2
...
\endverbatim

Some useful API calls for dealing with the path:

ObjId::ObjId( const string& path ): Creates the ObjId pointing to an
	already created object on the specified path.

string ObjId::path(): Returns the path string for the specified ObjId.

The next few API calls use the DataHandler, which is accessed from the
ObjId using:

const DataHandler* ObjId::element()->dataHandler().

unsigned int DataHandler::totalEntries(): Returns total size of object
array on specified DataHandler. This is the product of all dimensions.

\verbatim
ObjId f2( "/f1[2]/f2" );
assert( f2.element()->dataHandler()->totalEntries() == 10 );
\endverbatim

Note that some dimensions may be ragged, that is, they might not have the
same number of entries on all indices. This is commonly the case for 
synapses, in which each SynChan may receive different numbers of inputs.
In this case the dimension is larger than the largest SynChan array.
In the example above, we might have

unsigned int DataHandler::sizeOfDim( unsigned int dim ): Returns the size
of the specified dimension.

unsigned int DataHandler::pathDepth(): Returns the depth of the current DataHandler in the element tree. Root is zero.


vector< vector< unsigned int > > pathIndices( DataId di ) const: 
Returns a vector of array indices for the specified DataId.
In the above example:

\verbatim
ObjId f5( "/f1[1]/f2/f3[3]/f4/f5[5][6]" );
vector< vector< unsigned int pathIndices = f5.element()->dataHandler()->pathIndices( f5.dataId );
assert( pathIndices.size() == 6 ); // The zero index is the root element.

// Vectors at each level are:
//	root	f1	f2	f3	f4	f5
// 	{}	{1}	{}	{3}	{}	{5,6}
\endverbatim

\subsection Wildcard_paths	Wildcard paths
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
