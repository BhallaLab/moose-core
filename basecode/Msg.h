/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * Manages data flow between two elements. Is always many-to-many, with
 * assorted variants.
 */
class Msg
{
	public:
		Msg( Element* src, Element* dest );
		virtual ~Msg();

		/**
		 * Call clearQ on e1. Note that it applies at
		 * the Element level, not the index level.
		 */
		virtual const void clearQ();

		/**
		 * Calls Process on e1.
		 */
		virtual const void process( const ProcInfo *p );

		/// call func( arg ) on all targets in e1. Returns next buf pos
		virtual const char* call1( OpFunc func, const char* arg ) const;

		/// call func( arg ) on all targets in e2. Returns next buf pos
		virtual const char* call2( OpFunc func, const char* arg ) const;

		// return pointer to parent connection 1.
		virtual const Connection* parent1() const;

		// return pointer to parent connection 2.
		virtual const Connection* parent2() const;

		// Something here to set up sync message buffers

		// Something here to print message info
		virtual void print( string& output ) const;

		// Duplicate message on new Elements.
		virtual Msg* dup( Element* e1, Element* e2 ) const;

	protected:
		Element* e1_;
		Element* e2_;
		MsgId m1_; // Index of Msg on e1
		MsgId m2_; // Index of Msg on e2
};

/**
 * Mid-level message handler. Manages multiple messages, and appropriate
 * funcs.
 */
class Conn
{
	public:
	private:
		vector< FuncId > f_;
		vector< Msg* > m_;
};

// --------------------------------------------------------------------------
// Once-off assignment calls, from Shell or elsewhere, work like this:
//

// --------------------------------------------------------------------------
// Overall scheduling uses Connections too. Calls a hard-coded clearQ function,
// may do so on multiple threads.

// --------------------------------------------------------------------------
/**
 * Handles Asynchronous sends or 'push' operations, where data does not
 * go every timestep.
 * The Slot is typed.
 * The Conn must do checking at setup time for type appropriateness of
 * functions for matching slots.
 */

slot s = Slot1< double >(Conn#, Init<TargetType>Cinfo()->funcInfo("funcName"));
// This slot constructor does an RTTI check on the func type, and if OK,
// loads in the funcIndex. If bad, emits warning and stores a safety function.
// Problem: handling multiple types of targets. e.g., conc -> plot and xview.
// Get round it: these are simply multiple messages.
s->send( Eref e, double arg ); // Typesafe Replacement for asend function below.
void asend1< double >( Eref e, Slot s, double arg ); // User call for async send. Converts the argument into a char buffer.
void Eref::asend( Slot s, char* arg, unsigned int size); //Looks up Conn from Slot, passes it the funcIndex and args.
void Conn::asend( funcIndex, char* arg, unsigned int size); //Goes through all Msgs with arg. Looks up funcId from slot.funcIndex. Calls:
void Element::addToQ( FuncId funcId, MsgId msgId, const char* arg, unsigned int size ); // Puts MsgId, funcId and data on queue. May select Q based on thread.
// Multi-node Element needs a table to look up for MsgId, pointing to off-node
// targets for that specific MsgId. At this time it also inserts data transfer
// requests to those targets into the postmaster buffers.
// Looks up opFunc, which is also an opportunity to do type checking in case
// lookup shows a problem.
OpFunc Element::getOpFunc( funcId ); // asks Cinfo for opFunc.
OpFunc cinfo::getOpFunc( funcId );

Tick::clearQ(); // Calls clearQ on targets specified by Conn.
Conn::clearQ(); // Calls clearQ on all Msgs.
Msg::clearQ(); // Calls clearQ on e1.
voidElement::clearQ(); // Goes through queues, may select by thread. Calls:
const char* Element::execFunc( FuncId f, const char* buf ); //Finds the target function and calls the Msg to iterate over all tgts. Returns next buf position.
const char* Msg::call1( OpFunc func, const char* arg ) const; // Iterates over targets on e1, and executes function. Returns next queue position. If target is off-node, then skip.


// --------------------------------------------------------------------------
/**
 * Handles synchronous sends, where a fixed amount of data goes every
 * timestep. Typically send a single double. May as well use the
 * Eref version, which puts data into a preassigned place in a buffer.
 */
// Here Slot means an index to the hard-coded location for the buffer.
void ssend1< double >( Eref e, Slot s, double arg ); // Calls Eref::send.
Eref::send1( Slot slot, double arg); // Calls Element::send
Element::send1( Slot slot, unsigned int eIndex, double arg ); // Puts data into buffer

Tick::process(); // Calls Process on targets specified by Conn.
Conn::process( const ProcInfo* p ); // Calls Process on all Msgs.
Msg::process( const ProcInfo* p ); // Calls Process on e1.
Element::process( const ProcInfo* p );// Calls Process on all Data objects. Partitions among threads. p provides info on current thread and numThreads.
virtual Data::process( const ProcInfo* p, Eref e ); // Asks e to look up or sum the incoming data buffers, does comput, sends out other messages.
Eref::oneBuf( Slot slot ); // Looks up value in sync buffer at slot.
Eref::sumBuf( Slot slot ); // Sums up series of values in sync buffer at slot.
Eref::prdBuf( Slot slot, double v ); // multiples v into series of values in sync buffer at slot.

// --------------------------------------------------------------------------
/**
 * Handles 'return' functions, sending data back to originator. Used primarily
 * for getting field values. Also used for doing table lookups and getting
 * values back: a similar case.
 * Also handle arbitrary calls to targets.
 */

// To piggyback onto an existing message
s->send( Eref e, double arg );

// To do a generic field assignment or function call where a message doesn't
// exist:
bool set< double >( Id tgt, FieldId f, double val );
bool set< double >( Id tgt, const string& fieldname, double val );
double x
obj = cinfo->create( objtype, &x );
Msg* m = SingleMsg.add( Id obj, Id tgt );
	// add may need to go offnode
slot s = Slot1< double >(Conn#, Init<TargetType>Cinfo()->funcInfo( "funcName" ));
s->send( Eref e, double arg ); // Typesafe Replacement for asend function below.
delete obj; // should also delete Msg.

// To do a generic 'get' call
bool get< double >( Id tgt, FieldId f, double& val )
obj = cinfo->create( objtype, &x );
Msg* m = SingleMsg.add( Id obj, Id tgt ); // Shouldn't it be a Conn?
slot s = Slot2< FuncId, Id >(Msg#, Init<TargetType>Cinfo()->funcInfo( "getField" ));
s->send( Eref e, myFuncId, myId );
Possibly don't even need to add the message to the target. Except that
if it is off node then I need the machinery to do this for me.

// On remote element, we make the usual s = Slot1<double>. Then:
s->sendTo( Eref me, MsgIndex m, unsigned int tgtIndex, double arg )
We need to get the Mid of the current Msg from the args. This lets us
track the originating Element. The index of the src is there too. So I 
should be able to go straight to Element::addToQ

Should I just use Element::addToQ for sets?


/**
 * Spike and return messages both need to uniquely identify the Src.
 * For return messages we want to find the originating Msg, and go back on it
 * to the Src. 
 * For synapses, the Element Id would do, except that it requires a lookup
 * for synapses to identify the matching weight and delay. Need a quick
 * hash function. Found all are slow.
 */


/**
 * Also need a bidirectional variant if there is heavy reverse traffic.
 * This may be rather rare, since the cases with heavy bidirectional
 * traffic will typically be sync messages, which don't use the queues
 * for sending data.
 */
class SparseMsg: public Msg
{
	public:
		SparseMsg( Element* src, Element* dest );
		void addSpike( unsigned int srcElementIndex, double time ) const;
	private:
		// May have to be a pair of ints, to handle reverse msg indexing.
		// But this indexing is used only to identify src........
		// Can we use one of the SparseMatrix other tables for it?
		SparseMatrix< unsigned int > m_;
};

/**
 * This could be handy: maps onto same index.
 */
class One2OneMsg: public Msg
{
	public:
		One2OneMsg( Element* src, Element* dest );
		void addSpike( unsigned int srcElementIndex, double time ) const;
	private:
		unsigned int synIndex_;
};
