/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include "DeletionMarkerFinfo.h"
#include "GlobalMarkerFinfo.h"
#include "ThisFinfo.h"

/**
 * This sets up initial space on each ArrayElement for 4 messages.
 * We expect always to see the parent, process and usually something else.
 */
static const unsigned int INITIAL_MSG_SIZE = 4;

#ifdef DO_UNIT_TESTS
int ArrayElement::numInstances = 0;
#endif

ArrayElement::ArrayElement(
				Id id,
				const std::string& name, 
				void* data,
				unsigned int numSrc, 
				unsigned int numEntries,
				size_t objectSize
	)
	: Element( id ), name_( name ), 
		data_( data ), 
		msg_( numSrc ), 
		numEntries_(numEntries), 
		objectSize_(objectSize)
{
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif	
		;
}

ArrayElement::ArrayElement(
			Id id,
			const std::string& name, 
			const unsigned int numSrc,
// 			const vector< Msg >& msg, 
// 			const map< int, vector< ConnTainer* > >& dest,
			const vector< Finfo* >& finfo, 
			void *data, 
			int numEntries, 
			size_t objectSize
		): Element (id), name_(name),
			finfo_(1), 
			data_(data), 
			msg_(numSrc),
// 			dest_(dest), 
			numEntries_(numEntries), objectSize_(objectSize)
		{
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif	
		finfo_[0] = finfo[0];
		}

/**
 * Copies a ArrayElement. Does NOT copy data or messages.
 */
ArrayElement::ArrayElement( const ArrayElement* orig, Id id )
		: Element( id ),
		name_( orig->name_ ), 
		finfo_( 1 ),
		data_( 0 ),
		msg_( orig->cinfo()->numSrc() )
{
	assert( finfo_.size() > 0 );
	// Copy over the 'this' finfo
	finfo_[0] = orig->finfo_[0];

///\todo should really copy over the data as well.
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif	
		;
}

ArrayElement::~ArrayElement()
{
#ifndef DO_UNIT_TESTS
	// The unit tests create ArrayElement without any finfos.
	assert( finfo_.size() > 0 );
#endif	
#ifdef DO_UNIT_TESTS
	numInstances--;
#endif

	/**
	 * \todo Lots of cleanup stuff here to implement.
	// Find out what data is, and call its delete command.
	ThisFinfo* tf = dynamic_cast< ThisFinfo* >( finfo_[0] );
	tf->destroy( data() );
	*/	
	if ( data_ ) {
		if ( finfo_.size() > 0 && finfo_[0] != 0 ) {
			ThisFinfo* tf = dynamic_cast< ThisFinfo* >( finfo_[0] );
			if ( tf && tf->noDeleteFlag() == 0 )
				finfo_[0]->ftype()->destroy( data_, 1 );
		}
	}

	/**
	 * Need to explicitly drop messages, because we cannot tie the 
	 * operation to the Msg destructor. This is because the Msg vector
	 * changes size all the time but the Msgs themselves should not
	 * be removed.
	 * Note that we don't use DropAll, because by the time the call has
	 * come here we should have cleared out all the messages going outside
	 * the tree being deleted. Here we just destroy the allocated
	 * ConnTainers and their vectors in all messages.
	 */
	vector< Msg >::iterator m;
	for ( m = msg_.begin(); m!= msg_.end(); m++ )
		m->dropForDeletion();

	// Check if Finfo is one of the transient set, if so, clean it up.
	vector< Finfo* >::iterator i;
	// cout << name() << " " << id() << " f = ";
	for ( i = finfo_.begin(); i != finfo_.end(); i++ ) {
		assert( *i != 0 );
// 		cout << ( *i )->name()  << " ptr= " << *i << " " << endl;
		if ( (*i)->isTransient() ) {
			delete *i;
		}
	}
	// cout << endl;
}

const std::string& ArrayElement::className( ) const
{
	return cinfo()->name();
}

const Cinfo* ArrayElement::cinfo( ) const
{
	const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( finfo_[0] );
	assert( tf != 0 );
	return tf->cinfo();
}

//////////////////////////////////////////////////////////////////
// Msg traversal functions
//////////////////////////////////////////////////////////////////

/**
 * The Conn iterators have to be deleted by the recipient function.
 */
Conn* ArrayElement::targets( int msgNum, unsigned int eIndex ) const
{
	Eref e( const_cast< ArrayElement* >( this ), eIndex );
	if ( msgNum >= 0 && 
		static_cast< unsigned int >( msgNum ) < msg_.size() )
		return new TraverseMsgConn( &msg_[ msgNum ], e );
	else if ( msgNum < 0 ) {
		const vector< ConnTainer* >* d = dest( msgNum );
		if ( d )
			return new TraverseDestConn( d, e );
	}
	return new SetConn( root(), eIndex );
}

/**
 * The Conn iterators have to be deleted by the recipient function.
 */
Conn* ArrayElement::targets( const string& finfoName, unsigned int eIndex ) const
{
	const Finfo* f = cinfo()->findFinfo( finfoName );
	if ( !f )
		return 0;
	return targets( f->msg(), eIndex );
}

unsigned int ArrayElement::numTargets( int msgNum ) const
{
	if ( msgNum >= 0 && 
		static_cast< unsigned int >( msgNum ) < cinfo()->numSrc() )
		return msg_[ msgNum ].numTargets( this );
	else if ( msgNum < 0 ) {
		const vector< ConnTainer* >* d = dest( msgNum );
		if ( d )
			return d->size();
	}
	return 0;
}

unsigned int ArrayElement::numTargets( int msgNum, unsigned int eIndex )
	const
{
	if ( msgNum >= 0 && 
		static_cast< unsigned int >( msgNum ) < cinfo()->numSrc() )
	{ // This is acting as a source for the messages.
		return msg_[ msgNum ].numDest( this, eIndex );
	} else if ( msgNum < 0 )
	{ // acting as a destination.
		const vector< ConnTainer* >* d = dest( msgNum );
		unsigned int ret = 0;
		if ( d ) {
			vector< ConnTainer* >::const_iterator i;
			for ( i = d->begin(); i != d->end(); i++ )
				ret += (*i)->numSrc( eIndex );
			return ret;
		}
	}
	return 0;
}

unsigned int ArrayElement::numTargets( const string& finfoName ) const
{
	const Finfo* f = cinfo()->findFinfo( finfoName );
	if ( !f )
		return 0;
	return numTargets( f->msg() );
}

//////////////////////////////////////////////////////////////////
// Msg functions
//////////////////////////////////////////////////////////////////

const Msg* ArrayElement::msg( unsigned int msgNum ) const
{
	assert ( msgNum < msg_.size() );
	return ( &( msg_[ msgNum ] ) );
}

Msg* ArrayElement::varMsg( unsigned int msgNum )
{
	assert ( msgNum < msg_.size() );
	return ( &( msg_[ msgNum ] ) );
}

/**
 * Returns the base message on the linked list specified by msgNum.
 * Returns 0 on failure.
 * Each Msg has a next_ identifier for a subsequent message. Only the
 * base message, whose index is < numSrc, is to be used for setup.
 */
Msg* ArrayElement::baseMsg( unsigned int msgNum )
{
	assert ( msgNum < msg_.size() );
	unsigned int numSrc = cinfo()->numSrc();
	if ( msgNum < numSrc )
		return ( &( msg_[ msgNum ] ) );
	for ( unsigned int i = 0; i < numSrc; ++i ) {
		if ( msg_[i].linksToNum( this, msgNum ) )
			return &( msg_[i] );
	}
	return &( msg_[ msgNum ] ); // for DynamicFinfos
}

const vector< ConnTainer* >* ArrayElement::dest( int msgNum ) const
{
	if ( msgNum >= 0 )
		return 0;
	map< int, vector< ConnTainer* > >::const_iterator i = dest_.find( msgNum );
	if ( i != dest_.end() ) {
		return &( *i ).second;
	}
	return 0;
}

vector< ConnTainer* >* ArrayElement::getDest( int msgNum ) 
{
	return &dest_[ msgNum ];
}

/*
const Msg* ArrayElement::msg( const string& fName )
{
	const Finfo* f = findFinfo( fName );
	if ( f ) {
		int msgNum = f->msg();
		if ( msgNum < msg_.size() )
			return ( &( msg_[ msgNum ] ) );
	}
	return 0;
}
*/

unsigned int ArrayElement::addNextMsg()
{
	msg_.push_back( Msg() );
	return msg_.size() - 1;
}

unsigned int ArrayElement::numMsg() const
{
	return msg_.size();
}

//////////////////////////////////////////////////////////////////
// Information functions
//////////////////////////////////////////////////////////////////

unsigned int ArrayElement::getTotalMem() const
{
	return sizeof( ArrayElement ) + 
		sizeof( name_ ) + name_.length() + 
		sizeof( finfo_ ) + finfo_.size() * sizeof( Finfo* ) +
		getMsgMem();
}

unsigned int ArrayElement::getMsgMem() const
{
	vector< Msg >::const_iterator i;
	unsigned int ret = 0;
	for ( i = msg_.begin(); i < msg_.end(); i++ ) {
		ret += i->size();
	}
	return ret;
}

bool ArrayElement::isMarkedForDeletion() const
{
	if ( finfo_.size() > 0 )
		return finfo_.back() == DeletionMarkerFinfo::global();
	// This fallback case should only occur during unit testing.
	return 0;
}

bool ArrayElement::isGlobal() const
{
	if ( finfo_.size() > 0 )
		return finfo_.back() == GlobalMarkerFinfo::global();
	// This fallback case should only occur during unit testing.
	return 0;
}


//////////////////////////////////////////////////////////////////
// Finfo functions
//////////////////////////////////////////////////////////////////

/**
 * Returns a finfo matching the target name.
 * Note that this is not a const function because the 'match'
 * function may generate dynamic finfos on the fly. If you need
 * a simpler, const string comparison then use constFindFinfo below,
 * which has limitations for special fields and arrays.
 */
const Finfo* ArrayElement::findFinfo( const string& name )
{
	vector< Finfo* >::reverse_iterator i;
	const Finfo* ret;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			ret = (*i)->match( this, name );
			if ( ret )
					return ret;
	}
	return 0;
}

/**
 * This is a const version of findFinfo. Instead of match it does a
 * simple strcmp against the field name. Cannot handle complex fields
 * like ones with indices.
 */
const Finfo* ArrayElement::constFindFinfo( const string& name ) const
{
	vector< Finfo* >::const_reverse_iterator i;
	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			if ( (*i)->name() == name )
				return *i;
	}

	// If it is not on the dynamically created finfos, maybe it is on
	// the static set.
	return cinfo()->findFinfo( name );
	
	return 0;
}

const Finfo* ArrayElement::findFinfo( const ConnTainer* c ) const
{
	vector< Finfo* >::const_reverse_iterator i;
	const Finfo* ret;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			ret = (*i)->match( this, c );
			if ( ret )
					return ret;
	}
	return 0;
}

const Finfo* ArrayElement::findFinfo( int msgNum ) const
{
	const Cinfo* c = cinfo();
	return c->findFinfo( msgNum );
}

const Finfo* ArrayElement::localFinfo( unsigned int index ) const
{
	if ( index >= finfo_.size() ) 
		return 0;
	return finfo_[ index ];
}

/*
unsigned int ArrayElement::listFinfos( 
				vector< const Finfo* >& flist ) const
{
	vector< Finfo* >::const_iterator i;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	for ( i = finfo_.begin(); i != finfo_.end(); i++ )
	{
		(*i)->listFinfos( flist );
	}

	return flist.size();
}
*/

unsigned int ArrayElement::listFinfos( 
				vector< const Finfo* >& flist ) const
{
	vector< Finfo* >::const_iterator i;
	vector< Finfo* > dynos;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	for ( i = finfo_.begin() + 1; i != finfo_.end(); i++ )
		dynos.push_back( *i );

	for ( i = finfo_.begin(); i != finfo_.end(); i++ )
		(*i)->listFinfos( flist );

	// Replace all earlier entries with later ones if the names match.
	unsigned int j, k;
	unsigned int mainSize = flist.size() - dynos.size();
	// Could do this using STL, but it is too painful to figure out.
	for ( j = 0; j < mainSize; ++j ) {
		for ( k = 0; k < dynos.size(); ++k ) {
			if ( flist[j]->name() == dynos[k]->name() ) {
				flist[j] = dynos[k];
				flist[ mainSize + k ] = 0; // get rid of the dyno entry.
			}
		}
	}
	const Finfo* cond = 0;
	flist.erase( remove( flist.begin(), flist.end(), cond ), flist.end() );

	return flist.size();
}

unsigned int ArrayElement::listLocalFinfos( vector< Finfo* >& flist )
{
	flist.resize( 0 );
	if ( finfo_.size() <= 1 )
		return 0;
	flist.insert( flist.end(), finfo_.begin() + 1, finfo_.end() );
	return flist.size();
}

void ArrayElement::addExtFinfo(Finfo *f){
	//don't think anything just add the finfo to the list
	finfo_.push_back(f);
}

/**
 * Here we need to put in the new Finfo, and also check if it
 * requires allocation of any MsgSrc or MsgDest slots.
 */
void ArrayElement::addFinfo( Finfo* f )
{
	unsigned int num = msg_.size();
	f->countMessages( num );
	if ( num > msg_.size() )
		msg_.resize( num );
	finfo_.push_back( f );
}

/**
 * This function cleans up the finfo f. It removes its messages,
 * deletes it, and removes its entry from the finfo list. Returns
 * true if the finfo was found and removed. At this stage it does NOT
 * permit deleting the ThisFinfo at index 0.
 */
bool ArrayElement::dropFinfo( const Finfo* f )
{
	if ( finfo_.size() < 2 )
		return 0;

	vector< Finfo* >::iterator i;
	for ( i = finfo_.begin() + 1; i != finfo_.end(); i++ ) {
		if ( *i == f ) {
			assert ( f->msg() < static_cast< int >( msg_.size() ) );
			msg_[ f->msg() ].dropAll( this );
			delete *i;
			finfo_.erase( i );
			return 1;
		}
	}
	return 0;
}

void ArrayElement::setThisFinfo( Finfo* f )
{
	if ( finfo_.size() == 0 )
		finfo_.resize( 1 );
	finfo_[0] = f;
}

const Finfo* ArrayElement::getThisFinfo( ) const
{
	if ( finfo_.size() == 0 )
		return 0;
	return finfo_[0];
}


void ArrayElement::prepareForDeletion( bool stage )
{
	if ( stage == 0 ) {
		finfo_.push_back( DeletionMarkerFinfo::global() );
	} else { // Delete all the remote conns that have not been marked.
		vector< Msg >::iterator m;
		for ( m = msg_.begin(); m!= msg_.end(); m++ ) {
			m->dropRemote();
		}

		// Delete the dest connections too
		map< int, vector< ConnTainer* > >::iterator j;
		for ( j = dest_.begin(); j != dest_.end(); j++ ) {
			Msg::dropDestRemote( j->second );
		}
	}
}

/**
 * Debugging function to print out msging info
 */
void ArrayElement::dumpMsgInfo() const
{
	unsigned int i;
	cout << "Element " << name_ << ":\n";
	cout << "msg_: funcid, sizes\n";
	for ( i = 0; i < msg_.size(); i++ ) {
		vector< ConnTainer* >::const_iterator j;
		cout << i << ":	funcid =" << msg_[i].funcId() << ": ";
		for ( j = msg_[i].begin(); j != msg_[i].end(); j++ )
			cout << ( *j )->size() << ", ";
	}
	cout << endl;
}

// Overrides Element version.
Id ArrayElement::id() const {
	return Element::id().assignIndex( Id::AnyIndex );
}

// Holders
void ArrayElement::copyMessages( Element* dup,
	map< const Element*, Element* >& origDup, bool isArray ) const
{
	;
}

void ArrayElement::copyGlobalMessages( Element* dup, bool isArray ) const
{
	;
}

void* ArrayElement::data( unsigned int eIndex ) const
{
	if (eIndex >= numEntries_){
		cout << "ArrayElement: Bad Index...Prepare to crash" << endl;
		return 0;
	}
	return (void *)((char *)data_ + eIndex*objectSize_);
}


#ifdef DO_UNIT_TESTS

/**
 * Here we define a test class that sends 'output' to 'input' at
 * 'process'.
 * It does not do any numerics at all.
 */

Slot outputSlot;
int inFinfoMsg;
int synFinfoMsg;

class Atest {
	public: 
		static void setInput( const Conn* c, double value ) {
			static_cast< Atest* >( c->data() )->input_ = value;
		}
		static double getInput( Eref e ) {
			return static_cast< Atest* >( e.data() )->input_;
		}
		static void setOutput( const Conn* c, double value ) {
			static_cast< Atest* >( c->data() )->output_ = value;
		}
		static double getOutput( Eref e ) {
			return static_cast< Atest* >( e.data() )->output_;
		}
		static void process( Eref e ) {
			send1< double >( e, outputSlot, getOutput( e ) );
		}
		static void wtInput( const Conn* c, double value ) {
			Eref e = c->target();
			Atest* a = static_cast< Atest* >( c->data() );
			assert( a->wt_.size() > c->targetIndex() );
			a->input_ = value + a->wt_[ c->targetIndex() ];
			/*
			cout << endl;
			cout << e->name() << "[" << e.i << "]: input = " << value <<
				", tgtIndex = " << c->targetIndex() << "wts = ";
			for ( unsigned int i = 0; i < a->wt_.size(); i++ )
				cout << a->wt_[i] << ", ";
			cout << endl;
			*/
		}

		void updateNumSynapses( Eref e ) {
			// This is not quite working yet.
			unsigned int n = e->numTargets( synFinfoMsg, e.i );
			if ( n > wt_.size() )
				wt_.resize( n );
		}

		static double getWt( Eref e, const int& index ) {
			Atest* a = static_cast< Atest* >( e.data() );
			a->updateNumSynapses( e );
			if ( index >= 0 && a->wt_.size() > static_cast< unsigned int >( index ) )
				return a->wt_[index];
			return 0.0;
		}

		static void setWt( const Conn* c, double val, const int& index ) {
			Atest* a = static_cast< Atest* >( c->data() );
			a->updateNumSynapses( c->target() );
			if ( index >= 0 && a->wt_.size() > static_cast< unsigned int >( index ) )
				a->wt_[index] = val;
			else
				cout << "Index out of range: " << c->target()->name() << 
					"[" << c->target().i << "], index=" << index << endl;
		}

	private:
		double input_;
		double output_;
		vector< double > wt_;
};

const Cinfo* initAtestCinfo()
{
	static Finfo* aTestFinfos[] = 
	{
		new ValueFinfo( "input", ValueFtype1< double >::global(),
			GFCAST( &Atest::getInput ),
			RFCAST( &Atest::setInput )
		),
		new ValueFinfo( "output", ValueFtype1< double >::global(),
			GFCAST( &Atest::getOutput ),
			RFCAST( &Atest::setOutput )
		),
		new LookupFinfo( "wt", LookupFtype< double, int >::global(),
			GFCAST( &Atest::getWt ),
			RFCAST( &Atest::setWt )
		),
		new SrcFinfo( "outputSrc", Ftype1< double >::global() ),
		new DestFinfo( "msgInput", Ftype1< double >::global(),
			RFCAST( &Atest::setInput )
		),
		new DestFinfo( "wtInput", Ftype1< double >::global(),
			RFCAST( &Atest::wtInput )
		),
	};

	static Cinfo aTest( "Atest", "Upi", "Array Test class",
		initNeutralCinfo(),
		aTestFinfos,
		sizeof( aTestFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Atest >::global()
	);

	return &aTest;
}

/**
 * This tests message passing within an ArrayElement, from one entry
 * to the next. One can force the connOption to a specific value,
 * which works for Simple and Many2Many. Should also work for
 * One2Many and Many2One.
 */
static const unsigned int NUMKIDS = 12;
void arrayElementInternalTest( unsigned int connOption )
{
	cout << "\nTesting Array Elements, option= " << connOption << ": ";

	const Cinfo* aTestCinfo = initAtestCinfo();

	outputSlot = aTestCinfo->getSlot( "outputSrc" );
	inFinfoMsg = aTestCinfo->findFinfo( "msgInput" )->msg();
	synFinfoMsg = aTestCinfo->findFinfo( "wtInput" )->msg();

	Element* n = Neutral::create( "Neutral", "n", 
		Element::root()->id(), Id::scratchId() ); 

	Id childId = Id::scratchId();
	Element* child = 
		Neutral::createArray( "Atest", "foo", n->id(), childId, NUMKIDS );

	ASSERT( child != 0, "Array Element" );
	ASSERT( child == childId(), "Array Element" );
	ASSERT( childId.index() == 0, "Array Element" );
	ASSERT( child->id().index() == Id::AnyIndex, "Array Element" );

	vector< Id > kids;
	bool ret = get< vector< Id > >( n, "childList", kids );
	ASSERT( ret, "Array kids" );
	ASSERT( kids.size() == NUMKIDS, "Array kids" );
	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		ASSERT( kids[i].index() == i, "Array kids" );
		int index;
		bool ret = get< int >( kids[i].eref(), "index", index );
		ASSERT( ret && index == static_cast< int >( i ), "Array kids" );
		double output = i;
		bool sret = set< double >( kids[i].eref(), "output", output );
		output = 0.0;
		ASSERT( sret, "Array kids" );
		sret = set< double >( kids[i].eref(), "input", 0.0 );
		ret = get< double >( kids[i].eref(), "output", output );
		ASSERT( sret && ret && ( output == i ), "Array kid assignment" );
	}

	for ( unsigned int i = 1 ; i < NUMKIDS; i++ ) {
		ret = kids[i-1].eref().add( "outputSrc",
			kids[i].eref(), "msgInput", connOption );
		ASSERT( ret, "Array msg setup" );
	}
	for ( unsigned int i = 0 ; i < NUMKIDS - 1; i++ ) {
		double output = i * i + 1.0;
		bool ret = set< double >( kids[i].eref(), "output", output );
		Atest::process( kids[i].eref() );
		if ( i > 0 ) {
			double input = 0.0;
			double result = ( i - 1 ) * ( i - 1 ) + 1.0;
			ret = get< double >( kids[i].eref(), "input", input );
			ASSERT( ret && ( input == result ), "Array kid messaging" );
		}
	}
	set( n, "destroy" );
}

/**
 * This test checks how two arrays connect up to each other.
 * Different options do different things: 
 * 	Many2Many can have any mapping
 * 	One2OneMap does one-to-one mapping
 * 	All2All makes a fully connected map
 */
void arrayElementMapTest( unsigned int option )
{
	cout << "\nTesting Array Elements mapping connections, option= " <<
		option << ": ";
	// Make the arrays.
	Element* m = Neutral::create( "Neutral", "m", Element::root()->id(), Id::scratchId() ); 
	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), Id::scratchId() ); 

	Id srcId = Id::scratchId();
	Element* src = 
		Neutral::createArray( "Atest", "src", m->id(), srcId, NUMKIDS );

	Id destId = Id::scratchId();
	Element* dest = 
		Neutral::createArray( "Atest", "dest", n->id(), destId, NUMKIDS );

	ASSERT( src != 0 && dest != 0, "Array Map" );
	ASSERT( src == srcId() && dest == destId(), "Array Map" );
	ASSERT( srcId.index() == 0, "Array Map" );
	ASSERT( src->id().index() == Id::AnyIndex, "Array Map" );
	ASSERT( destId.index() == 0, "Array Map" );
	ASSERT( dest->id().index() == Id::AnyIndex, "Array Map" );

	// Get the child lists.
	vector< Id > srcKids;
	vector< Id > destKids;
	bool ret = get< vector< Id > >( m, "childList", srcKids );
	bool sret = get< vector< Id > >( n, "childList", destKids );
	ASSERT( sret && ret, "Array kids" );
	ASSERT( srcKids.size() == NUMKIDS, "Array kids" );
	ASSERT( destKids.size() == NUMKIDS, "Array kids" );

	vector< vector< int > > pattern( NUMKIDS );
	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		pattern[i].resize( NUMKIDS, 0 );
		set< double >( destKids[i].eref(), "input", 0.0 );
	}

	// Set up the connections.
	if ( option == ConnTainer::Many2Many ) {
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
			for ( unsigned int j = 0 ; j < NUMKIDS; j++ ) {
				if ( i + j == NUMKIDS || i == j || ( i + j ) % 3 == 0 ){
					pattern[i][j] = 1;
					ret = srcKids[i].eref().add( "outputSrc",
						destKids[j].eref(), "msgInput", option );
					ASSERT( ret, "Array Many2Many map setup" );
				}
			}
		}
		// Let's check if our message counting code works
		// First work out the correct answer.
		vector< unsigned int > numSrc( NUMKIDS, 0 );
		vector< unsigned int > numDest( NUMKIDS, 0 );
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
			for ( unsigned int j = 0 ; j < NUMKIDS; j++ ) {
				if ( pattern[i][j] != 0 ) {
					numDest[i]++;
					numSrc[j]++;
				}
			}
		}
		// Then check it
		Element* src = srcKids[0].eref().e;
		Element* dest = destKids[0].eref().e;
		unsigned int tgts = 0;
		cout << "+" << flush;
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
			tgts = src->numTargets( outputSlot.msg(), i );
			ASSERT( tgts == numDest[i], "Many2Many numTargets" ); 
			tgts = dest->numTargets( inFinfoMsg, i );
			ASSERT( tgts == numSrc[i], "Many2Many numTargets" ); 
		}
	} else if ( option == ConnTainer::One2OneMap ) {
		ret = srcKids[0].eref().add( "outputSrc",
			destKids[0].eref(), "msgInput", option );
		ASSERT( ret, "Array One2OneMap setup" );
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ )
			pattern[i][i] = 1;
		// Check message counting code
		Element* src = srcKids[0].eref().e;
		Element* dest = destKids[0].eref().e;
		unsigned int tgts = 0;
		cout << "+" << flush;
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
			tgts = src->numTargets( outputSlot.msg(), i );
			ASSERT( tgts == 1, "One2One numTargets" ); 
			tgts = dest->numTargets( inFinfoMsg, i );
			ASSERT( tgts == 1, "One2One numTargets" ); 
		}
	}

	cout << "+" << flush;

	// Stimulate each input and check all the outputs.

	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		double output = i * i + 1.0;
		bool ret = set< double >( srcKids[i].eref(), "output", output );
		Atest::process( srcKids[i].eref() );
		for ( unsigned int j = 0 ; j < NUMKIDS; j++ ) {
			double input = 0.0;
			ret = get< double >( destKids[j].eref(), "input", input );
			if ( pattern[i][j] != 0 ) {
				ASSERT( ret && ( input == output ), "Array map messaging" );
			} else {
				ASSERT( ret && ( input == 0.0 ), "Array map messaging" );
			}
			ret = set< double >( destKids[j].eref(), "input", 0.0 );
		}
	}
	set( n, "destroy" );
	set( m, "destroy" );
}

/**
 * This test checks how two arrays connect up to each other using 
 * synapse-type input, where the message uses an index to identify itself.
 * The looked up 'weight' is assigned to the same value as comes in the
 * msg data.
 *
 * This has the same connectivity patterns as arrayElementMapTest.
 * Different options do different things: 
 * 	Many2Many can have any mapping
 * 	One2OneMap does one-to-one mapping
 * 	All2All makes a fully connected map
 */
void arrayElementSynTest( unsigned int option )
{
	cout << "\nTesting Array Elements synapses and weights, option= " <<
		option << ": ";
	// Make the arrays.
	Element* m = Neutral::create( "Neutral", "m", Element::root()->id(), Id::scratchId() ); 
	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), Id::scratchId() ); 

	Id srcId = Id::scratchId();
	Element* src = 
		Neutral::createArray( "Atest", "src", m->id(), srcId, NUMKIDS );

	Id destId = Id::scratchId();
	Element* dest = 
		Neutral::createArray( "Atest", "dest", n->id(), destId, NUMKIDS );

	ASSERT( src != 0 && dest != 0, "Array Syn" );
	ASSERT( src == srcId() && dest == destId(), "Array Syn" );
	ASSERT( srcId.index() == 0, "Array Syn" );
	ASSERT( src->id().index() == Id::AnyIndex, "Array Syn" );
	ASSERT( destId.index() == 0, "Array Syn" );
	ASSERT( dest->id().index() == Id::AnyIndex, "Array Syn" );

	// Get the child lists.
	vector< Id > srcKids;
	vector< Id > destKids;
	bool ret = get< vector< Id > >( m, "childList", srcKids );
	bool sret = get< vector< Id > >( n, "childList", destKids );
	ASSERT( sret && ret, "Array kids" );
	ASSERT( srcKids.size() == NUMKIDS, "Array kids" );
	ASSERT( destKids.size() == NUMKIDS, "Array kids" );

	vector< vector< int > > pattern( NUMKIDS );
	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		pattern[i].resize( NUMKIDS, 0 );
		set< double >( destKids[i].eref(), "input", 0.0 );
	}
			
	// Set up the connections.
	if ( option == ConnTainer::Many2Many || option == ConnTainer::Simple ) {
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
			for ( unsigned int j = 0 ; j < NUMKIDS; j++ ) {
				if ( i + j == NUMKIDS || i == j || ( i + j ) % 3 == 0 ) {
					pattern[i][j] = 1;
					ret = srcKids[i].eref().add( "outputSrc",
						destKids[j].eref(), "wtInput", option );
					ASSERT( ret, "Array Many2Many map setup" );

					int numTgts = 
						destKids[j].eref().e->numTargets(
						synFinfoMsg, destKids[j].eref().i );
					assert( numTgts > 0 );
					// Set up the syn weights
					double x = i * i + 2.0;
					lookupSet< double, int >( destKids[j].eref(), "wt", x, numTgts - 1 );
					// cout << i << ", " << j << ", numTgts= " << numTgts << ", wt= " << x << endl;
				}
			}
		}
	} else if ( option == ConnTainer::One2OneMap ) {
		ret = srcKids[0].eref().add( "outputSrc",
			destKids[0].eref(), "wtInput", option );
		ASSERT( ret, "Array One2OneMap setup" );
		for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
			pattern[i][i] = 1;
			// Set up the syn weights
			double x = i * i + 2.0;
			lookupSet< double, int >( destKids[i].eref(), "wt", x, 0 );
		}
	}

	cout << "+" << flush;

	// Stimulate each input and check all the outputs.

	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		double output = i * i + 1.0;
		bool ret = set< double >( srcKids[i].eref(), "output", output );
		Atest::process( srcKids[i].eref() );
		for ( unsigned int j = 0 ; j < NUMKIDS; j++ ) {
			double input = 0.0;
			ret = get< double >( destKids[j].eref(), "input", input );
			if ( pattern[i][j] != 0 ) {
				ASSERT( ret && ( input == 2 * output+ 1 ), 
					"Array Syn messaging" );
			} else {
				ASSERT( ret && ( input == 0.0 ), "Array Syn messaging" );
			}
			ret = set< double >( destKids[j].eref(), "input", 0.0 );
		}
	}
	set( n, "destroy" );
	set( m, "destroy" );
}

void arrayElementTest()
{
	initAtestCinfo();
	FuncVec::sortFuncVec();
	arrayElementInternalTest( ConnTainer::Simple ); 
	arrayElementInternalTest( ConnTainer::Many2Many ); 
	arrayElementMapTest( ConnTainer::Many2Many ); 
	arrayElementMapTest( ConnTainer::One2OneMap ); 
	arrayElementSynTest( ConnTainer::Simple ); 
	arrayElementSynTest( ConnTainer::Many2Many ); 
 	arrayElementSynTest( ConnTainer::One2OneMap ); 
}
#endif
