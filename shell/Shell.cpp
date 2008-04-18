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
#include "IdManager.h"
#include "../element/Neutral.h"
#include "../element/Wildcard.h"
#include "Shell.h"
#include "ReadCell.h"
#include "SimDump.h"
#include "Ftype3.h"
#include "../utility/randnum/Probability.h"
#include "../utility/randnum/Uniform.h"
#include "../utility/randnum/Exponential.h"
#include "../utility/randnum/Normal.h"
#include "math.h"
//////////////////////////////////////////////////////////////////////
// Shell MOOSE object creation stuff
//////////////////////////////////////////////////////////////////////

const Cinfo* initShellCinfo()
{
	/**
	 * This is a shared message to talk to the GenesisParser and
	 * perhaps to other parsers like the one for SWIG and Python
	 */
	static Finfo* parserShared[] =
	{
		// Setting cwe
		new DestFinfo( "cwe", Ftype1< Id >::global(),
						RFCAST( &Shell::setCwe ) ),
		// Getting cwe back: First handle a request
		new DestFinfo( "trigCwe", Ftype0::global(), 
						RFCAST( &Shell::trigCwe ) ),
		// Then send out the cwe info
		new SrcFinfo( "cweSrc", Ftype1< Id >::global() ),

		// doing pushe: pushing current element onto stack and using
		// argument for new cwe. It sends back the cweSrc.
		new DestFinfo( "pushe", Ftype1< Id >::global(),
						RFCAST( &Shell::pushe ) ),
		// Doing pope: popping element off stack onto cwe. 
		// It sends back the cweSrc.
		new DestFinfo( "pope", Ftype0::global(), 
						RFCAST( &Shell::pope ) ),

		// Getting a list of child ids: First handle a request with
		// the requested parent elm id.
		new DestFinfo( "trigLe", Ftype1< Id >::global(), 
						RFCAST( &Shell::trigLe ) ),
		// Then send out the vector of child ids.
		new SrcFinfo( "leSrc", Ftype1< vector< Id > >::global() ),
		
		// Creating an object
		new DestFinfo( "create",
				Ftype3< string, string, Id >::global(),
				RFCAST( &Shell::staticCreate ) ),
		// Creating an array of objects
		new DestFinfo( "createArray",
				Ftype4< string, string, Id, vector<double> >::global(),
				RFCAST( &Shell::staticCreateArray ) ),
		new DestFinfo( "planarconnect",
				Ftype3< string, string, double >::global(),
				RFCAST( &Shell::planarconnect ) ),
		new DestFinfo( "planardelay",
				Ftype3< string, string, vector <double> >::global(),
				RFCAST( &Shell::planardelay ) ),
		new DestFinfo( "planarweight",
				Ftype3< string, string, vector<double> >::global(),
				RFCAST( &Shell::planarweight ) ),
		new DestFinfo( "getSynCount",
				Ftype1< Id >::global(),
				RFCAST( &Shell::getSynCount2 ) ),
		// The create func returns the id of the created object.
		new SrcFinfo( "createSrc", Ftype1< Id >::global() ),
		// Deleting an object
		new DestFinfo( "delete", Ftype1< Id >::global(), 
				RFCAST( &Shell::staticDestroy ) ),

		new DestFinfo( "add",
				Ftype2< Id, string >::global(),
				RFCAST( &Shell::addField ) ),
		// Getting a field value as a string: handling request
		new DestFinfo( "get",
				Ftype2< Id, string >::global(),
				RFCAST( &Shell::getField ) ),
		// Getting a field value as a string: Sending value back.
		new SrcFinfo( "getSrc", Ftype1< string >::global() ),

		// Setting a field value as a string: handling request
		new DestFinfo( "set",
				Ftype3< Id, string, string >::global(),
				RFCAST( &Shell::setField ) ),
		// Assigning a file into a table.
		// ElementId, filename, skiplines
		new DestFinfo( "file2tab",
				Ftype3< Id, string, unsigned int >::global(),
				RFCAST( &Shell::file2tab ) ),

		// Handle requests for setting values for a clock tick.
		// args are clockNo, dt, stage
		new DestFinfo( "setClock",
				Ftype3< int, double, int >::global(),
				RFCAST( &Shell::setClock ) ),

		// Handle requests to assign a path to a given clock tick.
		// args are tick id, path, function
		new DestFinfo( "useClock",
				Ftype3< Id, vector< Id >, string >::global(),
				RFCAST( &Shell::useClock ) ),
		
		// Getting a wildcard path of elements: handling request
		new DestFinfo( // args are path, flag true for breadth-first list
				"el",
				Ftype2< string, bool >::global(),
				RFCAST( &Shell::getWildcardList ) ),
		// Getting a wildcard path of elements: Sending list back.
		// This goes through the exiting list for elists set up in le.
		//TypeFuncPair( Ftype1< vector< Id > >::global(), 0 ),

		////////////////////////////////////////////////////////////
		// Running simulation set
		////////////////////////////////////////////////////////////
		new DestFinfo( "resched",
				Ftype0::global(), RFCAST( &Shell::resched ) ),
		new DestFinfo( "reinit",
				Ftype0::global(), RFCAST( &Shell::reinit ) ),
		new DestFinfo( "stop",
				Ftype0::global(), RFCAST( &Shell::stop ) ),
		new DestFinfo( "step",
				Ftype1< double >::global(), // Arg is runtime
				RFCAST( &Shell::step ) ),
		new DestFinfo( "requestClocks",
				Ftype0::global(), &Shell::requestClocks ),
		// Sending back the list of clocks times
		new SrcFinfo( "returnClocksSrc",
			Ftype1< vector< double > >::global() ),
		new DestFinfo( "requestCurrTime",
				Ftype0::global(), RFCAST( &Shell::requestCurrTime ) ),
				// Returns it in the default string return value.

		////////////////////////////////////////////////////////////
		// Message info functions
		////////////////////////////////////////////////////////////
		// Handle request for message list:
		// id elm, string field, bool isIncoming
		new DestFinfo( "listMessages",
				Ftype3< Id, string, bool >::global(),
				RFCAST( &Shell::listMessages ) ),
		// Return message list and string with remote fields for msgs
		new SrcFinfo( "listMessagesSrc",
			Ftype2< vector < Id >, string >::global() ),

		////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions
		////////////////////////////////////////////////////////////
		new DestFinfo( "copy",
			Ftype3< Id, Id, string >::global(), RFCAST( &Shell::copy ) ),
		new DestFinfo( "copyIntoArray",
			Ftype4< Id, Id, string, vector <double> >::global(), RFCAST( &Shell::copyIntoArray ) ),
		new DestFinfo( "move",
			Ftype3< Id, Id, string >::global(), RFCAST( &Shell::move ) ),
		////////////////////////////////////////////////////////////
		// Cell reader
		////////////////////////////////////////////////////////////
		// Args are: file cellpath globalParms
		new DestFinfo( "readcell",
			Ftype3< string, string, vector< double > >::global(), 
					RFCAST( &Shell::readCell ) ),
		////////////////////////////////////////////////////////////
		// Channel setup functions
		////////////////////////////////////////////////////////////
		new DestFinfo( "setupAlpha",
			Ftype2< Id, vector< double > >::global(), 
					RFCAST( &Shell::setupAlpha ) ),
		new DestFinfo( "setupTau",
			Ftype2< Id, vector< double > >::global(), 
					RFCAST( &Shell::setupTau ) ),
		new DestFinfo( "tweakAlpha",
			Ftype1< Id >::global(), RFCAST( &Shell::tweakAlpha ) ),
		new DestFinfo( "tweakTau",
			Ftype1< Id >::global(), RFCAST( &Shell::tweakTau ) ),
		new DestFinfo( "setupGate",
			Ftype2< Id, vector< double > >::global(), 
					RFCAST( &Shell::setupGate ) ),
		////////////////////////////////////////////////////////////
		// SimDump facility
		////////////////////////////////////////////////////////////
		new DestFinfo(	"readDumpFile",
			Ftype1< string >::global(), // arg is filename
					RFCAST( &Shell::readDumpFile ) ),
		new DestFinfo(	"writeDumpFile",
			// args are filename, path to dump
			Ftype2< string, string >::global(), 
					RFCAST( &Shell::writeDumpFile ) ),
		new DestFinfo(	"simObjDump",
			// arg is a set of fields for the desired class
			// The list of fields is a space-delimited list and 
			// can be extracted using separateString.
			Ftype1< string >::global(), RFCAST( &Shell::simObjDump ) ),
		new DestFinfo(	"simUndump",
					// args is sequence of args for simundump command.
			Ftype1< string >::global(), RFCAST( &Shell::simUndump ) ),
		new DestFinfo( "openfile",
				Ftype2< string, string >::global(),
				RFCAST( &Shell::openFile ) ),
		new DestFinfo( "writefile",
				Ftype2< string, string >::global(),
				RFCAST( &Shell::writeFile ) ),
		new DestFinfo( "listfiles",
				Ftype0::global(),
				RFCAST( &Shell::listFiles ) ),
		new DestFinfo( "closefile",
				Ftype1< string >::global(),
				RFCAST( &Shell::closeFile ) ),	
		new DestFinfo( "readfile",
				Ftype2< string, bool >::global(),
				RFCAST( &Shell::readFile) ),	
		////////////////////////////////////////////////////////////
		// field assignment for a vector of objects
		////////////////////////////////////////////////////////////
		// Setting a field value as a string: handling request
		new DestFinfo( "setVecField",
				Ftype3< vector< Id >, string, string >::global(),
				RFCAST( &Shell::setVecField ) ),
		new DestFinfo( "loadtab",
				Ftype1< string >::global(),
				RFCAST( &Shell::loadtab ) ),	
		new DestFinfo( "tabop",
				Ftype4< Id, char, double, double >::global(),
				RFCAST( &Shell::tabop ) ),	
	};

	/**
	 * This handles serialized data, typically between nodes. The
	 * arguments are a single long string. Takes care of low-level
	 * operations such as message set up or the gory details of copies
	 * across nodes.
	 */
	static Finfo* serialShared[] =
	{
		new DestFinfo( "rawAdd", // Addmsg as a raw string.
			Ftype1< string >::global(),
			RFCAST( &Shell::rawAddFunc )
		),
		new DestFinfo( "rawCopy", // Copy an entire object sent as a string
			Ftype1< string >::global(),
			RFCAST( &Shell::rawCopyFunc )
		),
		new DestFinfo( "rawTest", // Test function
			Ftype1< string >::global(),
			RFCAST( &Shell::rawTestFunc )
		),
	};

	static Finfo* masterShared[] = 
	{
		new SrcFinfo( "get",
			// objId, field
			Ftype2< Id, string >::global() ),
		new DestFinfo( "recvGet",
			Ftype1< string >::global(),
			RFCAST( &Shell::recvGetFunc )
		),
		new SrcFinfo( "set",
			// objId, field, value
			Ftype3< Id, string, string >::global() ),
		new SrcFinfo( "add",
				// srcObjId, srcFiekd, destObjId, destField
			Ftype4< Id, string, Id, string >::global()
		),
		new SrcFinfo( "create", 
			// type, name, parentId, newObjId.
			Ftype4< string, string, Id, Id >::global()
		),
	};

	static Finfo* slaveShared[] = 
	{
		new DestFinfo( "get",
			// objId, field
			Ftype2< Id, string >::global(),
			RFCAST( &Shell::slaveGetField )
			),
		new SrcFinfo( "recvGet",
			Ftype1< string >::global()
		),
		new DestFinfo( "set",
			// objId, field, value
			Ftype3< Id, string, string >::global(),
			RFCAST( &Shell::setField )
		),
		new DestFinfo( "add",
				// srcObjId, srcFiekd, destObjId, destField
			Ftype4< Id, string, Id, string >::global(),
			RFCAST( &Shell::addFunc )
		),
		new DestFinfo( "create", 
			// type, name, parentId, newObjId.
			Ftype4< string, string, Id, Id >::global(),
			RFCAST( &Shell::slaveCreateFunc )
		),
	};

	static Finfo* shellFinfos[] =
	{
		new ValueFinfo( "cwe", ValueFtype1< Id >::global(),
				reinterpret_cast< GetFunc >( &Shell::getCwe ),
				RFCAST( &Shell::setCwe ) ),

		new DestFinfo( "xrawAdd", // Addmsg as a raw string.
			Ftype1< string >::global(),
			RFCAST( &Shell::rawAddFunc )
		),
		new DestFinfo( "poll", // Infinite loop, meant for slave nodes
			Ftype0::global(),
			RFCAST( &Shell::pollFunc )
		),
		new SrcFinfo( "pollSrc", 
			// # of steps. 
			// This talks to /sched/pj:step to poll the postmasters
			Ftype1< int >::global()
		),

		new SharedFinfo( "parser", parserShared, 
				sizeof( parserShared ) / sizeof( Finfo* ) ), 
		new SharedFinfo( "serial", serialShared,
				sizeof( serialShared ) / sizeof( Finfo* ) ), 
		new SharedFinfo( "master", masterShared,
				sizeof( masterShared ) / sizeof( Finfo* ) ), 
		new SharedFinfo( "slave", slaveShared,
				sizeof( slaveShared ) / sizeof( Finfo* ) ), 
	};

	static Cinfo shellCinfo(
		"Shell",
		"Upi Bhalla, NCBS",
		"Shell object. Manages general simulator commands.",
		initNeutralCinfo(),
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Shell >::global()
	);

	return &shellCinfo;
}

static const Cinfo* shellCinfo = initShellCinfo();


static const Slot cweSlot = initShellCinfo()->getSlot( "parser.cweSrc" );
static const Slot elistSlot = initShellCinfo()->getSlot( "parser.leSrc" );

// Returns the id of the created object
static const Slot createSlot =
	initShellCinfo()->getSlot( "parser.createSrc" );
static const Slot getFieldSlot =
	initShellCinfo()->getSlot( "parser.getSrc" );
static const Slot clockSlot =
	initShellCinfo()->getSlot( "parser.returnClocksSrc" );
static const Slot listMessageSlot =
	initShellCinfo()->getSlot( "parser.listMessagesSrc" );
static const Slot rCreateSlot =
	initShellCinfo()->getSlot( "master.create" );
static const Slot rGetSlot = initShellCinfo()->getSlot( "master.get" );
static const Slot rSetSlot = initShellCinfo()->getSlot( "master.set" );
static const Slot rAddSlot = initShellCinfo()->getSlot( "master.add" );
static const Slot recvGetSlot =
	initShellCinfo()->getSlot( "slave.recvGet" );

static const Slot pollSlot =
	initShellCinfo()->getSlot( "pollSrc" );


void printNodeInfo( const Conn* c );

//////////////////////////////////////////////////////////////////////
// Initializer
//////////////////////////////////////////////////////////////////////

Shell::Shell()
	: cwe_( Id() ), recentElement_( Id() )
{
	simDump_ = new SimDump;
}

//////////////////////////////////////////////////////////////////////
// General path to eid conversion utilities
//////////////////////////////////////////////////////////////////////

/**
 * This needs to be on the Shell for future use, because we may
 * need to look up remote nodes
 */
Id Shell::parent( Id eid )
{
	Element* e = eid();
	Id ret;
	// Check if eid is on local node, otherwise go to remote node
	// ret = Neutral::getParent(e)
	if ( get< Id >( e, "parent", ret ) )
		return ret;
	return Id::badId();
}

/**
 * Returns the id of the element at the end of the specified path.
 * On failure, returns badId()
 * It is a static func as a utility for parsers.
 * It takes a pre-separated vector of names.
 * It ignores names that are just . or /
 */
Id Shell::traversePath( Id start, vector< string >& names )
{
	assert( !start.bad() );
	vector< string >::iterator i;
	for ( i = names.begin(); i != names.end(); i++ ) {
		if ( *i == "." || *i == "/" ) {
			continue;
		} else if ( *i == ".." ) {
			start = parent( start );
		} else {
			Id ret;
			Element* e = start();
			//Neutral::getChildByName(e, *i);
			
			lookupGet< Id, string >( e, "lookupChild", ret, *i );
			//if ( ret.zero() || ret.bad() ) cout << "Shell:traversePath: The id is bad" << endl;
			if ( ret.zero() || ret.bad() ){
					return Id::badId();
			}
			start = ret;
		}
	}
	return start;
}

// Requires a path argument without a starting space
// Perhaps this should be in the interpreter?
Id Shell::innerPath2eid( 
		const string& path, const string& separator ) const
{
	if ( path == separator || path == "/root" )
			return Id();

	if ( path == "" || path == "." )
			return cwe_;

	if ( path == "^" )
			return recentElement_;

	if ( path == ".." ) {
			if ( cwe_.zero() )
				return cwe_;
			return parent( cwe_ );
	}

	vector< string > names;

	Id start; // Initializes to zero
	if ( path.substr( 0, separator.length() ) == separator ) {
		separateString( path.substr( separator.length() ), names, separator );
	} else if ( path.substr( 0, 5 ) == "/root" ) {
		separateString( path.substr( 5 ), names, separator );
	} else {
		start = cwe_;
		separateString( path, names, separator );
	}
	return traversePath( start, names );
}

// This is the static version of the function.
Id Shell::path2eid( const string& path, const string& separator )
{
	/*
	Id shellId;
	bool ret = lookupGet< Id, string >(
				Element::root(), "lookupChild", shellId, "shell" );
	assert( ret );
	assert( !shellId.bad() );
	Shell* s = static_cast< Shell* >( shellId()->data() );
	*/
	Shell* s = static_cast< Shell* >( ( Id::shellId() )()->data( 0 ) );
	return s->innerPath2eid( path, separator );
}

string Shell::eid2path( Id eid ) 
{
	if ( eid.zero() )
		return string( "/" );
	static const string slash = "/";
	string n( "" );
	while ( !eid.zero() ) {
		n = slash + eid()->name() + n;
		eid = parent( eid );
	}
	return n;
}

/**
 * Returns that component of path that precedes the last separator.
 * If there is nothing there, or no separator, returns an empty string.
 */
string Shell::head( const string& path, const string& separator )
{
	string::size_type pos = path.rfind( separator );
	if ( pos == string::npos )
			return "";

	return path.substr( 0, pos );
}

/**
 * Returns that component of path that follows the last separator.
 * If there is nothing there, or no separator, returns the entire path.
 */
string Shell::tail( const string& path, const string& separator )
{
	string::size_type pos = path.rfind( separator );
	if ( pos == string::npos )
			return path;

	return path.substr( pos + separator.length() );
}

//////////////////////////////////////////////////////////////////////
// Special low-level operations that Shell handles using raw
// serialized strings from PostMaster.
//////////////////////////////////////////////////////////////////////

void Shell::rawAddFunc( const Conn* c, string s )
{
	Element* post = c->source().e;
	assert( post->className() == "PostMaster" );
	unsigned int mynode;
	unsigned int remotenode;
	get< unsigned int >( post, "localNode", mynode );
	get< unsigned int >( post, "remoteNode", remotenode );
	// cout << ".";
	// cout << "Shell::rawAddFunc( " << s << " ), on " << mynode << ", from " << remotenode << "\n";
	vector< string > svec;
	separateString( s, svec, " " );
	unsigned int j = 0; // This is for breakpointing for parallel debug
	while ( j > 0 ) // for breakpointing in parallel debug.
		;
	// svec seq is : srcid, targetId, targetField, srcType
	Id destId = Id::str2Id( svec[1] );
	if ( destId.bad() ) {
		cout << "Error: Shell::rawAddFunc: msgdest is a bad elm on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	} 
	Element* dest = destId();
	if ( dest == 0 ) {
		cout << "Error: Shell::rawAddFunc: msgdest ptr for id " << destId << " is empty on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	} 
	if ( dest->className() == "PostMaster" ) { //oops, off node.
		cout << "Error: Shell::rawAddFunc: msgdest is off node on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	}
	const Finfo *destField = dest->findFinfo( svec[2] );
	if ( destField == 0 ) {
		cout << "Error: Shell::rawAddFunc: targetField does not exist on dest on " << mynode << " from " << remotenode << " with str " << s << "\n";
		return;
	}

	string typeSig = "";
	val2str< const Ftype* >( destField->ftype()->baseFtype(), typeSig );
	if ( typeSig != svec[3] ) {
		cout << "Error: Shell::rawAddFunc: field type mismatch: '" <<
			typeSig << "' != '" << svec[3] << "' on " << mynode << " from " << remotenode << "\n";
		return;
	}
	
	// post->findFinfo( "data" )->add( post, dest, destField );
	// cout << "Shell::rawAddFunc: Successfully added msg on remote node\n";
}

void Shell::rawCopyFunc( const Conn* c, string s )
{
	cout << "Shell::rawCopyFunc( " << s << " )\n";
}

void Shell::rawTestFunc( const Conn* c, string s )
{
	Element* post = c->source().e;
	ASSERT( post->className() == "PostMaster", "rawTestFunc" );
	unsigned int mynode;
	unsigned int remotenode;
	get< unsigned int >( post, "localNode", mynode );
	get< unsigned int >( post, "remoteNode", remotenode );
	char teststr[30];
	sprintf( teststr, "My name is Michael Caine %d,%d", 
		remotenode, mynode );

	// cout << "Shell::rawTestFunc( " << s << " )," << teststr << "\n";
	
	ASSERT( s == teststr, "Shell::rawTestFunc" );
	// cout << "Shell::rawTestFunc( " << s << " )\n";
}

void Shell::pollFunc( const Conn* c )
{
	while( 1 ) {
		// cout << "." << flush;
		send1< int >( c->target(), pollSlot, 1 );
		// Surprisingly, the usleep seems to worsen the responsiveness.
		// usleep( 10 );
	}
}

//////////////////////////////////////////////////////////////////////
// Moose fields for Shell
//////////////////////////////////////////////////////////////////////

void Shell::setCwe( const Conn* c, Id id )
{
	/// \todo: Need some work here to fix up with new id scheme.
	// This should only be called on master node.
	if ( !id.bad() ) {
		Shell* s = static_cast< Shell* >( c->data() );
		s->cwe_ = id;
	} else {
		cout << "Error: Attempt to change to nonexistent element.\n";
	}
}

Id Shell::getCwe( const Element* e )
{
	assert( e != 0 );
	const Shell* s = static_cast< const Shell* >( e->data( 0 ) );
	return s->cwe_;
}

void Shell::trigCwe( const Conn* c )
						
{
	Shell* s = static_cast< Shell* >( c->data() );
	sendBack1< Id >( c, cweSlot, s->cwe_ );
	// sendTo1< Id >( c->target(), cweSlot, c->targetIndex(), s->cwe_);
}

void Shell::pushe( const Conn* c, Id id )
{
	Shell* s = static_cast< Shell* >( c->data() );
	if ( !id.bad() ) {
		s->workingElementStack_.push_back( s->cwe_ );
		s->cwe_ = id;
	} else {
		cout << "Error: Attempt to pushe to nonexistent element.\n";
	}
	sendBack1< Id >( c, cweSlot, s->cwe_ );
	// sendTo1< Id >( c->targetElement(), 0, cweSlot, c->targetIndex(), s->cwe_);
}

void Shell::pope( const Conn* c )
{
	Shell* s = static_cast< Shell* >( c->data() );
	if ( s->workingElementStack_.size() > 0 ) {
		s->cwe_ = s->workingElementStack_.back();
		if ( s->cwe_.bad() ) { 
			// In case we went back to an element that got deleted in
			// the interim.
			s->cwe_ = Id();
		}
		s->workingElementStack_.pop_back();
	} else {
		cout << "Error: empty element stack.\n";
	}
	sendBack1< Id >( c, cweSlot, s->cwe_ );
	// sendTo1< Id >( c->targetElement(), 0, cweSlot, c->targetIndex(), s->cwe_ );
}


//////////////////////////////////////////////////////////////////////
// Create and destroy are possibly soon to be deleted. These may have
// to go over to the Neutral, but till we've sorted out the SWIG
// interface we'll keep it in case it gets used there.
//////////////////////////////////////////////////////////////////////


void Shell::trigLe( const Conn* c, Id parent )
						
{
	Element* pa = parent();
	// Here we do something intelligent for off-node le.
	if ( pa ) {
		vector< Id > ret;
		if ( get< vector< Id > >( pa, "childList", ret ) ) {
			sendBack1< vector< Id > >( c, elistSlot, ret );
			// Element* e = c->targetElement();
			// sendTo1< vector< Id > >( e, 0, elistSlot, c->targetIndex(), ret );
		}
	}
}

// Static function
void Shell::staticCreate( const Conn* c, string type,
					string name, Id parent )
{
	Element* e = c->target().e;
	Shell* s = static_cast< Shell* >( c->data() );

	// This is where the IdManager does clever load balancing etc
	// to assign child node.
	Id id = Id::childId( parent );
	// Id id = Id::scratchId();
	Element* child = id();
	if ( child == 0 ) { // local node
		bool ret = s->create( type, name, parent, id );
		if ( ret ) { // Tell the parser it was created happily.
			sendBack1< Id >( c, createSlot, id );
		//	sendTo1< Id >( e, 0, createSlot, c->targetIndex(), id );
		}
	} else {
		// Shell-to-shell messaging here with the request to
		// create a child.
		// This must only happen on node 0.
		assert( e->id().node() == 0 );
		assert( id.node() > 0 );
		OffNodeInfo* oni = static_cast< OffNodeInfo* >( child->data( 0 ) );
		// Element* post = oni->post;
		unsigned int target = 0;
		/*
		unsigned int target = 
		e->connSrcBegin( rCreateSlot.msg() ) - e->lookupConn( 0 ) +
			id.node() - 1;
			*/
		sendTo4< string , string, Id, Id>( 
			e, rCreateSlot, target,
			type, name, 
			parent, oni->id );
		// Here it needs to fork till the object creation is complete.
		delete oni;
		delete child;
	}
}

// Static function
// parameter has following clumped in the order mentioned, Nx, Ny, dx, dy, xorigin, yorigin
// creates array of simple elements. Will swtich to arrayelements later.
void Shell::staticCreateArray1( const Conn* c, string type,
					string name, Id parent, vector <double> parameter )
{
	
	Eref eref = c->target();
// 	Element* e = c->targetElement();
	Shell* s = static_cast< Shell* >( c->data() );

	int n = (int) (parameter[0]*parameter[1]);
	
	for (int i = 0; i < n; i++){
		Id id = Id::childId( parent );
		Element* child = id();
		char sname[20];
		sprintf(sname, "%s[%d]", name.c_str(), i); 
		if ( child == 0 ) { // local node
			bool ret = s->create( type, sname, parent, id );
			if ( ret ) { // Tell the parser it was created happily.
				//GenesisParserWrapper::recvCreate(conn, id)
				sendBack1< Id >( c, createSlot, id);
				//sendTo1< Id >( e, 0, createSlot, c->targetIndex(), id);
			}
		}
		else {
			// Shell-to-shell messaging here with the request to
			// create a child.
			// This must only happen on node 0.
			assert( eref.id().node() == 0 );
			assert( id.node() > 0 );
			OffNodeInfo* oni = static_cast< OffNodeInfo* >( child->data( 0 ) );
			// Element* post = oni->post;
			unsigned int target = 0;
			//eref.connSrcBegin( rCreateSlot.msg() ) - e->lookupConn( 0 ) +
			//	id.node() - 1;
			sendTo4< string, string, Id, Id>( 
				eref, rCreateSlot, target,
				type, sname, 
				parent, oni->id );
			// Here it needs to fork till the object creation is complete.
			delete oni;
			delete child;
		}
	}
}


// Static function
// parameter has following clumped in the order mentioned, Nx, Ny, dx, dy, xorigin, yorigin
void Shell::staticCreateArray( const Conn* c, string type,
					string name, Id parent, vector <double> parameter )
{

	Eref eref = c->target();
	//Element* e = c->targetElement();
	Shell* s = static_cast< Shell* >( c->data() );

	// This is where the IdManager does clever load balancing etc
	// to assign child node.
	Id id = Id::childId( parent );
	// Id id = Id::scratchId();
	Element* child = id();
	if ( child == 0 ) { // local node
		int n = (int) (parameter[0]*parameter[1]);
		bool ret = s->createArray( type, name, parent, id, n );
		assert(parameter.size() == 6);
// 		ArrayElement* f = static_cast <ArrayElement *> (e);
// 		f->setNoOfElements((int)(parameter[0]), (int)(parameter[1]));
// 		f->setDistances(parameter[2], parameter[3]);
// 		f->setOrigin(parameter[4], parameter[5]);
		if ( ret ) { // Tell the parser it was created happily.
			//GenesisParserWrapper::recvCreate(conn, id)
			sendBack1< Id >( c, createSlot, id);
		}
	} else {
		// Shell-to-shell messaging here with the request to
		// create a child.
		// This must only happen on node 0.
		assert( eref.id().node() == 0 );
		assert( id.node() > 0 );
		OffNodeInfo* oni = static_cast< OffNodeInfo* >( child->data() );
		// Element* post = oni->post;
		unsigned int target = 0;
		//e->connSrcBegin( rCreateSlot.msg() ) - e->lookupConn( 0 ) +
		//	id.node() - 1;
		sendTo4< string , string, Id, Id>( 
			eref, rCreateSlot, target,
			type, name, 
			parent, oni->id );
		// Here it needs to fork till the object creation is complete.
		delete oni;
		delete child;
	}
}

void Shell::planarconnect(const Conn* c, string source, string dest, double probability){
	vector <Id> src_list, dst_list;
	simpleWildcardFind( source, src_list );
	simpleWildcardFind( dest, dst_list );
	for(size_t i = 0; i < src_list.size(); i++) {
		if (src_list[i]()->className() != "SpikeGen" ){
			cout << "The source element must be SpikeGen" << endl;
			return;
		}
		for(size_t j = 0; j < dst_list.size(); j++) {
			//cout << src_list[i]->id().path() << " " << dst_list[i]->id().path() << endl;
			if (dst_list[j]()->className() != "SynChan"){
				cout <<  "The dest element must be SynChan" << endl;
				return;
			}
			if ((rand()%100)/100.0 <= probability){
// 				cout << i+1 << " " << j+1 << endl;
				src_list[i]()->findFinfo("event")->add(src_list[i](), dst_list[j](), dst_list[j]()->findFinfo("synapse"));
			}
		}
	}
}

// void Shell::planardelay(const Conn* c, string source, double delay)
// {
// 	static const Cinfo* sgCinfo = Cinfo::find( "SpikeGen" );
// 	// static const Finfo* eventFinfo = sgCinfo->findFinfo( "event" );
// 	static const Slot eventSlot = sgCinfo->getSlot( "event" );
// 
// 	vector <Element* > srcList;
// 	simpleWildcardFind( source, srcList );
// 	for ( size_t i = 0 ; i < srcList.size(); i++){
// 		if ( srcList[ i ]->className() != "SpikeGen"){
// 			cout << "Shell::planardelay: Error: Source is not SpikeGen" << endl; 	
// 			return;
// 		}
// 
// 		vector< ConnTainer* >::const_iterator j;
// 		const Msg* m = srcList[ i ]->msg( eventSlot.msg() );
// 
// 		for( j = m->begin(); j != m->end(); j++ ) {
// 			// Need to sort out the iteration through all targets, here.
// 			// Many targets will be ArrayElements and should have
// 			// special fast assignment for all indices.
// 			// for ( Conn* k = ( *j )->conn( eIndex, m->isDest() ); j->good(); j++ )
// 				// setDelay( k );
// 		}
// 		
// 		/*
// 		vector <Conn> conn;
// 		srcList[i]->findFinfo("event")->outgoingConns(srcList[i], conn);
// 		for (size_t j = 0; j < conn.size(); j++){
// 			unsigned int numSynapses;
// 			Element *dest = conn[j].targetElement();
// 			if (destination != ""){
// 				bool found = false;
// 				for (vector<Element *>::iterator iter = dst_list.begin(); 
// 					iter != dst_list.end(); iter++)
// 					if (*iter == dest) {found = true; break;}
// 				if (!found) continue;
// 			}
// 			bool ret = get< unsigned int >( dest, "numSynapses", numSynapses );
// 			if (!ret) {cout << "Shell::planardelay: Error2" << endl; return;}
// 			for (size_t k = 0 ; k < numSynapses; k++){
// 				double number = 0;
// 				if (delaychoice){
// 					cout << "planardelay:: radial not implemented."<< endl;
// 					// Not decided what to do
// 				}
// 				else {
// 					number = delay;
// 				}
// 				if (randchoice){
// 					double random = p->getNextSample();
// 					while (random > maxallowed ) random = p->getNextSample();
// 					if (absoluterandom)
// 						{number += random;}
// 					else 
// 						{number += number*random;}
// 				}
// 				if (add){
// 					double delay_old = 0;
// 					ret = lookupGet< double, unsigned int >( dest, "delay", delay_old, k );
// 					if (!ret) {
// 						cout << "planardelay:: Error3" << endl;
// 					}
// 					number += delay_old;
// 				}
// 				lookupSet< double, unsigned int >( dest, "delay", number, k );
// 			}
// 		}
// 		*/
// 	}
// }


void Shell::planardelay(const Conn& c, string source, string destination, vector <double> parameter){
	assert (parameter.size() == 11);
	double delay = parameter[0];
// 	double conduction_velocity = parameter[1];
	bool add = parameter[2];
	double scale = parameter[3];
	double stdev = parameter[4];
	double maxdev = parameter[5];
	double mid = parameter[6];
	double max = parameter[7];
	bool absoluterandom = parameter[8];
	int delaychoice = int(parameter[9]);
	int randchoice = int(parameter[10]);
	double maxallowed;
	Probability *p;
	switch (randchoice){
		case 0:
			break;
		case 1:
			p = new Uniform(-scale, scale);
			maxallowed = scale;
			break;
		case 2: 
			p = new Normal(0, stdev);
			maxallowed = maxdev;
			break;
		case 3: 
			p = new Exponential(log(2)/mid);
			maxallowed = max;
			break;
	}
	
	static const Cinfo* sgCinfo = Cinfo::find( "SpikeGen" );
	static const Slot eventSlot = sgCinfo->getSlot( "event" );
	vector <Id> srcList;
	vector <Id> dst_list;
	simpleWildcardFind( source, srcList );
	if (destination != "")
		simpleWildcardFind( destination, dst_list );
	for (size_t i = 0 ; i < srcList.size(); i++){
		if (srcList[i]()->className() != "SpikeGen"){cout << "Shell::planardelay: Source is not SpikeGen" << endl; return;}
		vector< ConnTainer* >::const_iterator j;
		const Msg* m = srcList[ i ]()->msg( eventSlot.msg() );
		//srcList[i]->findFinfo("event")->outgoingConns(srcList[i], conn);
		for( j = m->begin(); j != m->end(); j++ ) {
			unsigned int numSynapses;
			//Element *dest = (*j)->e2();
			Eref eref;
			for ( Conn* k = ( *j )->conn( /*eIndex*/0, m->isDest() ); k->good(); k++ ){
				eref = k->target();
				
			}
			if (destination != ""){
				bool found = false;
				for (vector<Id>::iterator iter = dst_list.begin(); 
					iter != dst_list.end(); iter++)
					if (*iter == eref.id()) {found = true; break;}
				if (!found) continue;
			}
			bool ret = get< unsigned int >( eref, "numSynapses", numSynapses );
			if (!ret) {cout << "Shell::planardelay: Could not access number of synapses." << endl; return;}
			for (size_t k = 0 ; k < numSynapses; k++){
				double number = 0;
				if (delaychoice){
					cout << "planardelay:: radial not implemented."<< endl;
					// Not decided what to do
				}
				else {
					number = delay;
				}
				if (randchoice){
					double random = p->getNextSample();
					while (random > maxallowed ) random = p->getNextSample();
					if (absoluterandom)
						{number += random;}
					else 
						{number += number*random;}
				}
				if (add){
					double delay_old = 0;
					ret = lookupGet< double, unsigned int >( eref, "delay", delay_old, k );
					if (!ret) {
						cout << "planardelay:: Error3" << endl;
					}
					number += delay_old;
				}
				lookupSet< double, unsigned int >( eref, "delay", number, k );
			}
		}
	}
}

void Shell::planarweight(const Conn& c, string source, string  destination, vector <double> parameter){
	assert (parameter.size() == 12);
	double weight = parameter[0];
// 	double decay_rate = parameter[1];
// 	double max_weight = parameter[2];
// 	double min_weight = parameter[3];
	double scale = parameter[4];
	double stdev = parameter[5];
	double maxdev = parameter[6];
	double mid = parameter[7];
	double max = parameter[8];
	bool absoluterandom = parameter[9];
	int weightchoice = int(parameter[10]);
	int randchoice = int(parameter[11]);
	double maxallowed;
	Probability *p;
	switch (randchoice){
		case 0:
			break;
		case 1:
			p = new Uniform(-scale, scale);
			maxallowed = scale;
			break;
		case 2: 
			p = new Normal(0, stdev);
			maxallowed = maxdev;
			break;
		case 3: 
			p = new Exponential(log(2)/mid);
			maxallowed = max;
			break;
	}
	
	static const Cinfo* sgCinfo = Cinfo::find( "SpikeGen" );
	static const Slot eventSlot = sgCinfo->getSlot( "event" );
	vector <Id> srcList;
	vector <Id> dst_list;
	simpleWildcardFind( source, srcList );
	if (destination != "")
		simpleWildcardFind( destination, dst_list );
	for (size_t i = 0 ; i < srcList.size(); i++){
		if (srcList[i]()->className() != "SpikeGen"){cout << "Shell::planarweight: Source is not SpikeGen" << endl; return;}
		vector< ConnTainer* >::const_iterator j;
		const Msg* m = srcList[ i ]()->msg( eventSlot.msg() );
		//srcList[i]->findFinfo("event")->outgoingConns(srcList[i], conn);
		for( j = m->begin(); j != m->end(); j++ ) {
			unsigned int numSynapses;
			//Element *dest = (*j)->e2();
			Eref eref;
			for ( Conn* k = ( *j )->conn( /*eIndex*/0, m->isDest() ); k->good(); k++ ){
				eref = k->target();
				
			}
			if (destination != ""){
				bool found = false;
				for (vector<Id>::iterator iter = dst_list.begin(); 
					iter != dst_list.end(); iter++)
					if (*iter == eref.id()) {found = true; break;}
				if (!found) continue;
			}
			bool ret = get< unsigned int >( eref, "numSynapses", numSynapses );
			if (!ret) {cout << "Shell::planarweight: Could not access number of synapses." << endl; return;}
			for (size_t k = 0 ; k < numSynapses; k++){
				double number = 0;
				if (weightchoice){
					cout << "planarweight:: decay not implemented."<< endl;
				}
				else {
					number = weight;
				}
				if (randchoice){
					double random = p->getNextSample();
					while (random > maxallowed ) random = p->getNextSample();
					if (absoluterandom)
						{number += random;}
					else 
						{number += number*random;}
				}
				lookupSet< double, unsigned int >( eref, "weight", number, k );
			}
		}
	}
}




// does not do - destination is a SynChan test
void Shell::getSynCount2(const Conn* c, Id dest){
	Element* dst = dest();
	unsigned int numSynapses;
	bool b = get< unsigned int >( dst, "numSynapses", numSynapses );
	if (!b) {
		cout << "Shell:: syncount failed at" << dst->name() <<endl; 
		return;
	}
	char e[10];
	sprintf (e, "%d", numSynapses);
	string ret = e;
	sendBack1< string >( c, getFieldSlot, ret );
	/*
	sendTo1< string >( c->targetElement(),
				getFieldSlot, c->targetIndex(), 0, ret );
	*/
}




// Static function
void Shell::staticDestroy( const Conn* c, Id victim )
{
	Shell* s = static_cast< Shell* >( c->data() );
	s->destroy( victim );
}

/**
 * This function adds a ExtFieldFinfo
 */

void Shell::addField( const Conn* c, Id id, string fieldname )
{
	if ( id.bad() )
		return;
	string ret;
	Element* e = id();
	//cout << e->name() << endl;
	
	// Appropriate off-node stuff here.
	Finfo* f = new ExtFieldFinfo(fieldname, Ftype1<string>::global());
	e->addExtFinfo( f );
}

// Static function
/**
 * This function handles request to get a field value. It triggers
 * a return function to the calling object, as a string.
 * The reason why we take this function to the Shell at all is because
 * we will eventually need to be able to handle this for off-node
 * object requests.
 */
void Shell::getField( const Conn* c, Id id, string field )
{
	if ( id.bad() )
		return;
	string ret;
	Eref eref(id(), id.index());
	
	// Appropriate off-node stuff here.

	const Finfo* f = eref.e->findFinfo( field );
	// Error messages are the job of the parser. So we just return
	// the value when it is good and leave the rest to the parser.
	if ( f )
		if ( f->strGet( eref, ret ) ){
			sendBack1< string >( c, getFieldSlot, ret );
			//GenesisParserWrapper::recvField (conn, ret);
			// sendTo1< string >( c->targetElement(), 0,
				// getFieldSlot, c->targetIndex(), ret );
		}
}

////////////////////////////////////////////////////////////////////////
// Functions for implementing Master/Slave set
////////////////////////////////////////////////////////////////////////

// To be called from the node on which the Master shell resides.
void testMess( Element* e, unsigned int numNodes )
{
/*
	// Here we create a set of neutrals on all the slave nodes.
	unsigned int startConn = e->connSrcBegin( rCreateSlot.msg() ) - 
		e->lookupConn( 0 );
	vector< Id > offNodeObjs( numNodes );
	for ( unsigned int i = 1; i < numNodes; i++ ) {
		offNodeObjs[ i ] = Id::makeIdOnNode( i );
		sendTo4< string , string, Id, Id>( 
			e, rCreateSlot, startConn + i - 1, 
			"Neutral", "OffNodeCreateTest", 
			Id::str2Id( "0" ), offNodeObjs[ i ] );
	}

	// This should return 'shell'
	send2< Id, string >( e, rGetSlot, Id::str2Id( "1" ), "name" );

	// send1< string >( e, recvGetSlot, "fieldvalue" );

	// Poll the postmasters.
	send1< int >( e, pollSlot, 1 );

	// Here we assign new names to each of these neutrals
	startConn = e->connSrcBegin( rSetSlot.msg() ) - e->lookupConn( 0 );
	for ( unsigned int i = 1; i < numNodes; i++ ) {
		char name[20];
		sprintf( name, "OffNodeCreateTest_%d", i );
		sendTo3< Id, string, string >( e, rSetSlot,
			startConn + i - 1,
			offNodeObjs[ i ], "name", name );
	}
	send1< int >( e, pollSlot, 1 );

	// Here we check the names of the neutrals.
	startConn = e->connSrcBegin( rGetSlot.msg() ) - e->lookupConn( 0 );
	for ( unsigned int i = 1; i < numNodes; i++ ) {
		char name[20];
		sprintf( name, "OffNodeCreateTest_%d", i );
		sendTo2< Id, string >( e, rGetSlot,
			startConn + i - 1,
			offNodeObjs[ i ], "name" );
	}
	send1< int >( e, pollSlot, 1 );

	send4< Id, string, Id, string >( e, rAddSlot, 
		Id::str2Id( "5432" ), "srcfield", Id::str2Id( "9876" ),
		"destfield" );

	send1< int >( e, pollSlot, 1 );
*/
}

void printNodeInfo( const Conn* c )
{
	Element* post = c->source().e;
	assert( post->className() == "PostMaster" );
	unsigned int mynode;
	unsigned int remotenode;
	get< unsigned int >( post, "localNode", mynode );
	get< unsigned int >( post, "remoteNode", remotenode );

	cout << "on " << mynode << " from " << remotenode << ":";
}

void Shell::slaveGetField( const Conn* c, Id id, string field )
{
	printNodeInfo( c );
	// cout << "in slaveGetFunc on " << id << " with field :" << field << "\n";
	if ( id.bad() )
		return;
	string ret;
	Element* e = id();
	if ( e == 0 )
		return;

	const Finfo* f = e->findFinfo( field );
	if ( f )
		if ( f->strGet( e, ret ) )
			sendBack1< string >( c, recvGetSlot, ret );
}

void Shell::recvGetFunc( const Conn* c, string value )
{
	printNodeInfo( c );
	cout << "in recvGetFunc with field value :'" << value << "'\n";
	// send off to parser maybe.
	// Problem if multiple parsers.
	// Bigger problem that this is asynchronous now.
	// Maybe it is OK if only one parser.
	// sendTo1< string >( c.targetElement(), getFieldSlot, 0, value );
	send1< string >( c->target(), getFieldSlot, value );
}

void Shell::slaveCreateFunc ( const Conn* c, 
				string objtype, string objname, 
				Id parent, Id newobj )
{
	printNodeInfo( c );
	cout << "in slaveCreateFunc :" << objtype << " " << objname << " " << parent << " " << newobj << "\n";

	Shell* s = static_cast< Shell* >( c->data() );

	bool ret = s->create( objtype, objname, parent, newobj );
	if ( ret ) { // Tell the master node it was created happily.
		// sendTo2< Id, bool >( e, createCheckSlot, c.targetIndex(), newobj, 1 );
	} else { // Tell master node that the create failed.
		// sendTo2< Id, bool >( e, createCheckSlot, c.targetIndex(), newobj, 0 );
	}
	// bool ret = s->create( objtype, objname, parent, newobj );
	// assert( ret );
	// Need to send return back to master node, and again we have
	// asynchrony. Actually return not needed because master assigns
	// id. All we need is to verify that it was created OK.
	/*
	if ( ret ) {
		sendTo1< Id >( c.targetElement(),
					createSlot, c.targetIndex(), ret );
	}
	*/
}

void Shell::addFunc ( const Conn* c, 
				Id src, string srcField,
				Id dest, string destField )
{
	printNodeInfo( c );
	cout << "in slaveAddFunc :" << src << " " << srcField << 
		" " << dest << " " << destField << "\n";
}
// Static function
/**
 * This copies the element tree from src to parent. If name arg is 
 * not empty, it renames the resultant object. It first verifies
 * that the planned new object has a different name from any of
 * the existing children of the prospective parent.
 */
void Shell::copy( const Conn* c, 
				Id src, Id parent, string name )
{
	// Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	Element* e = src()->copy( parent(), name );
	if ( e ) { // Send back the id of the new element base
		sendBack1< Id >( c, createSlot, e->id() );
	}
}

void Shell::copyIntoArray( const Conn* c, 
				Id src, Id parent, string name, vector <double> parameter )
{
	// Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	int n = (int) (parameter[0]*parameter[1]);
	for (int i = 0; i < n; i ++){
		char sname[20];
		sprintf(sname, "%s[%d]", src()->name().c_str(), i);
		Element* e = src()->copy( parent(), sname );
		//assign the other parameters to the arrayelement
		if ( e )  // Send back the id of the new element base
			sendBack1< Id >( c, createSlot, e->id() );
	}
}


/**
 * This function copies the prototype element in form of an array.
 * It is similar to copy() only that it creates an array of copies 
 * elements
*/

void Shell::copyIntoArray1( const Conn* c, 
				Id src, Id parent, string name, vector <double> parameter )
{
	/*
	// Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	int n = (int) (parameter[0]*parameter[1]);
	Element* e = src()->copyIntoArray( parent(), name, n );
	//assign the other parameters to the arrayelement
	assert(parameter.size() == 6);
	ArrayElement* f = static_cast <ArrayElement *> (e);
	f->setNoOfElements((int)(parameter[0]), (int)(parameter[1]));
	f->setDistances(parameter[2], parameter[3]);
	f->setOrigin(parameter[4], parameter[5]);
	if ( e )  // Send back the id of the new element base
		sendBack1< Id >( c->targetElement(), createSlot, c, e->id() );
*/
}

// Static placeholder.
/**
 * This moves the element tree from src to parent. If name arg is 
 * not empty, it renames the resultant object. It first verifies
 * that the planned new object has a different name from any of
 * the existing children of the prospective parent.
 * Unlike the 'copy', this function is handled by the shell and may
 * involve interesting node relocation issues.
 */
void Shell::move( const Conn* c, Id src, Id parent, string name )
{
	assert( !src.bad() );
	assert( !parent.bad() );
	// Cannot move object onto its own descendant
	Element* e = src();
	Element* pa = parent();
	if ( pa->isDescendant( e ) ) {
		cout << "Error: move '" << e->name() << "' to '" << 
				pa->name() << 
				"': cannot move object onto itself or descendant\n";
		return;
	}
	Id srcPaId = Neutral::getParent( e );
	assert ( !srcPaId.bad() );
	if ( srcPaId == parent ) { // Just rename it.
		assert ( name != "" ); // New name must exist.
		if ( Neutral::getChildByName( pa, name ).bad() ) {
			// Good, we do not have name duplication.
			e->setName( name );
			// OK();
			return;
		} else {
			// Bad, we do have name duplication. This should not happen
			// because this ought to mean that we are moving the 
			// object as a child of the named object. 
			assert( 0 );
		}
	} else { // Move the object onto a new parent.
		string temp = name;
		if ( name == "" )
			temp = e->name();
		if ( Neutral::getChildByName( pa, temp ).bad() ) {
			// Good, we do not have name duplication.
			if ( name != "" )
				e->setName( name );
			/// \todo: Here we don't take into acount multiple parents.
			// Here we drop all parents.
			e->dropAll( "child" );

			bool ret = pa->findFinfo( "childSrc" )->add(
				pa, e, e->findFinfo( "child" ) );
			assert ( ret );
			// OK();
			return;
		} else {
			// Bad, we do have name duplication. GENESIS
			// allows this but we will not.
			cout << "Error: move '" << e->name() << "' to '" << 
				pa->name() << "': same name child already exists.\n";
			return;
		}
	}
}

// Static function
/**
 * This function handles request to set a field value.
 * The reason why we take this function to the Shell is because
 * we will eventually need to be able to handle this for off-node
 * object requests.
 */
void Shell::setField( const Conn* c, Id id, string field, string value )
{
	assert( id.good() );
	Element* e = id();
	if ( !e ) {
		cout << "Shell::setField:Error: Element not found: " 
			<< id << endl;
		return;
	}
	// Appropriate off-node stuff here.

	const Finfo* f = e->findFinfo( field );
	if ( f ) {
		if ( !f->strSet( e, value ) )
			cout << "setField: Error: cannot set field " << e->name() <<
					"." << field << " to " << value << endl;
	} else {
		cout << "setField: Error: cannot find field: " << id.path() << ", " << e->name() <<
				"." << field << endl;
	}
}

// Static function
/**
 * This function handles request to set identical field value for a 
 * vector of objects. Used for the GENESIS SET function.
 */
void Shell::setVecField( const Conn* c, 
				vector< Id > elist, string field, string value )
{
	vector< Id >::iterator i;
	for ( i = elist.begin(); i != elist.end(); i++ ) {
		// Cannot use i->good() here because we might set fields on /root.
		assert( !i->bad() ); 
		//Element* e = ( *i )();
		Eref eref( ( *i )(), ( *i ).index() );
		// Appropriate off-node stuff here.
	
		const Finfo* f = eref.e->findFinfo( field );
		if ( f ) {
			if ( !f->strSet( eref, value ) )
				cout << "setVecField: Error: cannot set field " << i->path() <<
						"." << field << " to " << value << endl;
		} else {
			cout << "setVecField: Error: cannot find field: " << i->path() <<
				"." << field << endl;
		}
	}
}

/**
 * This function handles request to load a file into an Interpol object
 */
void Shell::file2tab( const Conn& c, 
				Id id, string filename, unsigned int skiplines )
{
	assert( id.good() );
	Element* e = id();
	if ( !e ) {
		cout << "Shell::file2tab:Error: Element not found: " 
			<< id << endl;
		return;
	}
	// Appropriate off-node stuff here.

	if ( !set< string, unsigned int >( e, "load", filename, skiplines ) ) {
			cout << "Shell::file2tab Error: cannot set field " <<
				e->name() << ".load\n";
	}
}


// Static function
/**
 * Assigns dt and optionally stage to a clock tick. If the Tick does
 * not exist, create it. The ClockJob and Tick0 are created by default.
 * I keep this function in the Shell because we'll want something
 * similar in Python. Otherwise it can readily go into the
 * GenesisParserWrapper.
 */
void Shell::setClock( const Conn* c, int clockNo, double dt,
				int stage )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	char line[20];
	sprintf( line, "t%d", clockNo );
	string TickName = line;
	string clockPath = string( "/sched/cj/" + TickName );
	Id id = sh->innerPath2eid( clockPath, "/" );
	Id cj = sh->innerPath2eid( "/sched/cj", "/" );
	Element* tick = 0;
	if ( id.zero() || id.bad() ) {
		tick = Neutral::create( 
						"Tick", TickName, cj(), Id::scratchId() );
	} else {
		tick = id();
	}
	assert ( tick != 0 && tick != Element::root() );
	set< double >( tick, "dt", dt );
	set< int >( tick, "stage", stage );
	set( cj(), "resched" );
	// Call the function
}

// static function
/**
 * Sets up the path controlled by a given clock tick. The final 
 * argument sets up the target finfo for the message. Usually this
 * is 'process' but some objects need multi-phase clocking so we
 * add the 'function' argument to specify what the target finfo is.
 * The function does a unique merge of the path
 * with the existing targets of the clock tick by checking if the
 * elements on the path are already tied to this tick. (This avoids
 * the N^2 problem of matching them against the list). If they are
 * on some other tick that message is dropped and this new one added.
 * The function does not reinit the clocks or reschedule them: the
 * simulation can resume right away.
 * It is the job of the parser to provide defaults
 * and to decode the path list from wildcards.
 */
void Shell::useClock( const Conn* c,
	Id tickId, vector< Id > path, string function )
{
	assert( !tickId.zero() );
	Element* tick = tickId();
	assert ( tick != 0 );
	const Finfo* tickProc = tick->findFinfo( "process" );

	// vector< Conn > list;

	// Scan through path and check for existing process connections.
	// If they are to the same tick, skip the object
	// If they are to a different tick, delete the connection.
	vector< Id >::iterator i;
	for (i = path.begin(); i != path.end(); i++ ) {
		bool ret;
		assert ( !i->zero() );
		Element* e = ( *i )( );
		assert ( e && e != Element::root() );
		const Finfo* func = e->findFinfo( function );
		if ( func ) {
			Conn* c = e->targets( func->msg() );
			if ( !c->good() ) {
				ret = tickProc->add( tick, e, func );
				assert( ret );
			} else {
				if ( c->target().e != tick ) {
					e->dropAll( func->msg() );
					tick->add( tickProc->msg(), e, func->msg() );
				}
			}
			delete c;

			/*
			Msg* m = e->varMsg( func->msg() );
			if ( m->size() == 0 ) {
			// if ( func->numIncoming( e ) == 0 )
				ret = tickProc->add( tick, e, func );
				assert( ret );
			} else {
				if ( ( *m->begin() )->e1() != tick ) {
					m->dropAll( e );
					tickProc->add( tick, e, func );
				}
			}
			*/
		} else {
			// This cannot be an 'assertion' error because the 
			// user might do a typo.
			cout << "Error: Shell::useClock: unknown function " << function << " in " << i->path() << endl;
		}
	}
}

/**
 * This function converts the path from relative, recent or other forms
 * to a canonical absolute path form that the wildcards can handle.
 */
void Shell::digestPath( string& path )
{
	// Here we deal with all sorts of awful cases involving current and
	// parent element paths.
	if ( path.length() == 0 )
		return;
	if ( path[0] == '/' ) // already canonical form.
		return;
	if ( path.length() == 1 ) {
		if ( path == "." ) {
			path = cwe_.path();
		} else if ( path == "^" ) {
			path = recentElement_.path();
		} else {
			path = cwe_.path() + "/" + path;
		}
	} else if ( path.length() == 2 ) {
		if ( path[0] == '.' ) {
			if ( path[1] == '/' ) { // start from cwe
				path = cwe_.path();
			} else if ( path[1] == '.' ) {
				if ( cwe_ == Id() ) {
					path = "/";
				} else  {
					path = cwe_.path();
					string::size_type pos = path.rfind( '/' );
					path = path.substr( 0, pos );
				}
			}
		} else {
			path = cwe_.path() + "/" + path;
		}
	} else {
		string::size_type pos = path.find_first_of( '/' );
		if ( pos == 1 ) {
			if ( path[0] == '^' ) 
				path = recentElement_.path() + path.substr( 1 );
			else if ( path[0] == '.' )
				path = cwe_.path() + path.substr( 1 );
			else
				path = cwe_.path() + "/" + path;
		} else if ( pos == 2 && path[0] == '.' && path[1] == '.' ) {
			if ( cwe_ == Id() ) {
				path = path.substr( 2 );
			} else { // get parent of cwe and tag path onto it.
				string temp = cwe_.path();
				string::size_type pos = temp.rfind( '/' );
				path = temp.substr( 0, pos ) + path.substr( 2 );
			}
		} else if ( pos != 0 ) {
			path = cwe_.path() + "/" + path;
		}
	}
	// Handle issues with initial double slash.
	if ( path[0] == '/' && path[1] == '/' )
		path = path.substr( 1 );
}
// static function
/** 
 * getWildcardList obtains a wildcard list specified by the path.
 * Normally the list is tested for uniqueness and sorted by pointer -
 * it becomes effectively random.
 * The flag specifies if we want a list in breadth-first order,
 * in which case commas are not permitted.
 */
void Shell::getWildcardList( const Conn* c, string path, bool ordered )
{
	vector<Id> list;
	//vector< Id > ret;

	static_cast< Shell* >( c->data() )->digestPath( path );

	// Finally, refer to the wildcard functions in Wildcard.cpp.
	if ( ordered )
		simpleWildcardFind( path, list );
	else
		wildcardFind( path, list );

	//ret.resize( list.size() );
	vector< Id >::iterator i;
	vector< Element* >::iterator j;

	//for (i = ret.begin(), j = list.begin(); j != list.end(); i++, j++ )
	//		*i = ( *j )->id();
	
	//GenesisParserWrapper::recvElist(conn, elist)
	sendBack1< vector< Id > >( c, elistSlot, list );
}

/**
 * Utility function to find the ClockJob pointer
 */
Element* findCj()
{
	Id schedId;
	lookupGet< Id, string >( 
		Element::root(), "lookupChild", schedId, "sched" );
	assert( !schedId.bad() );
	Id cjId;
	lookupGet< Id, string >( 
		schedId(), "lookupChild", cjId, "cj" );
	assert( !cjId.bad() );
	return cjId();
}

void Shell::resched( const Conn* c )
{
	// Should be a msg
	Element* cj = findCj();
	set( cj, "resched" );
	Id kinetics( "/kinetics" );
	if ( kinetics.good() )
		set( kinetics(), "resched" );
}

void Shell::reinit( const Conn* c )
{
	// Should be a msg
	Element* cj = findCj();
	set( cj, "reinit" );
}

void Shell::stop( const Conn* c )
{
	// Element* cj = findCj();
	// set( cj, "stop" ); // Not yet implemented
}

void Shell::step( const Conn* c, double time )
{
	// Should be a msg
	Element* cj = findCj();
	set< double >( cj, "start", time );
}

/**
 * requestClocks builds a list of all clock times in order of clock
 * number. Puts these into a vector of doubles to send back to the
 * calling parser.
 * \todo: Need to fix requestClocks as it will give the wrong index
 * if we have non-contiguous clock ticks.
 */
void Shell::requestClocks( const Conn* c )
{
	// Here we fill up the clock timings.
	Element* cj = findCj();

	Conn* ct = cj->targets( "childSrc" );
	vector< double > times;

	while ( ct->good() ) {
		double dt;
		if ( get< double >( ct->target(), "dt", dt ) )
			times.push_back( dt );
		ct->increment();
	}
	delete ct;
	sendBack1< vector< double > >( c, clockSlot, times );

	/*
	Element* cj = findCj();
	vector< Conn > kids;
	vector< Conn >::iterator i;
	vector< double > times;
	cj->findFinfo( "childSrc" )->outgoingConns( cj, kids );
	double dt;
	for ( i = kids.begin(); i != kids.end(); i++ ) {
		if ( get< double >( i->targetElement(), "dt", dt ) )
			times.push_back( dt );
	}

	send1< vector< double > >( c->targetElement(), clockSlot, times );
	*/
}

void Shell::requestCurrTime( const Conn* c )
{
	Element* cj = findCj();
	string ret;
	const Finfo* f = cj->findFinfo( "currentTime" );
	assert( f != 0 );
	f->strGet( cj, ret );
	sendBack1< string >( c, getFieldSlot, ret );
}

/**
 * listMessages builds a list of messages associated with the 
 * specified element on the named field, and sends it back to
 * the calling parser. It extracts the
 * target element from the connections, and puts this into a
 * vector of unsigned ints.
 */
void Shell::listMessages( const Conn* c,
				Id id, string field, bool isIncoming )
{
	assert( !id.bad() );
	Element* e = id();
	const Finfo* f = e->findFinfo( field );
	assert( f != 0 );

	vector< pair< Element*, unsigned int > > list;
	vector< Id > ret;
	string remoteFields = "";
	string separator = "";

	Conn* tc = e->targets( f->msg() );
	while( tc->good() ) {
		Eref tgt = tc->target();
		ret.push_back( tgt.id() );
		const Finfo* targetFinfo = tgt.e->findFinfo( tc->targetMsg() );
		assert( targetFinfo != 0 );
		remoteFields = remoteFields + separator + targetFinfo->name();
		separator = ", ";
		tc->increment();
	}
	delete tc;

	/*
	const Msg* m = e->msg( f->msg() );
	vector< ConnTainer* >::const_iterator i;
	for ( i = m->begin(); i != m->end(); i++ ) {
		Element* temp = ( isIncoming ) ? ( *i )->e1() : ( *i )->e2();
		ret.push_back( temp->id() );
		const Finfo* targetFinfo = temp->findFinfo( *i );
		assert( targetFinfo != 0 );
		if ( i == m->begin() )
			remoteFields = remoteFields + targetFinfo->name();
		else
			remoteFields = remoteFields + ", " + targetFinfo->name();
	}
	*/

	sendBack2< vector< Id >, string >(
		c, listMessageSlot, ret, remoteFields );
}

void Shell::readCell( const Conn* c, string filename, string cellpath,
	vector< double > globalParms )
{
	ReadCell rc( globalParms );
	
	rc.read( filename, cellpath );
}

void Shell::setupAlpha( const Conn* c, Id gateId,
				vector< double > parms )
{
	static const Finfo* setupAlphaFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "setupAlpha" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::setupAlpha: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	set< vector< double > >( gate, setupAlphaFinfo, parms );
}

void Shell::setupTau( const Conn* c, Id gateId,
				vector< double > parms )
{
	static const Finfo* setupTauFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "setupTau" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::setupTau: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	set< vector< double > >( gate, setupTauFinfo, parms );
}

void Shell::tweakAlpha( const Conn* c, Id gateId )
{
	static const Finfo* tweakAlphaFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "tweakAlpha" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::tweakAlpha: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	set( gate, tweakAlphaFinfo );
}

void Shell::tweakTau( const Conn* c, Id gateId )
{
	static const Finfo* tweakTauFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "tweakTau" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::tweakTau: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	set( gate, tweakTauFinfo );
}

void Shell::setupGate( const Conn* c, Id gateId,
				vector< double > parms )
{
	static const Finfo* setupGateFinfo = 
			Cinfo::find( "HHGate")->findFinfo( "setupGate" );
	assert( !gateId.bad() );
	Element* gate = gateId();
	if ( gate->className() != "HHGate" ) {
		cout << "Error: Shell::setupGate: element is not an HHGate: "
				<< gate->name() << endl;
		return;
	}
	set< vector< double > >( gate, setupGateFinfo, parms );
}

//////////////////////////////////////////////////////////////////
// SimDump functions
//////////////////////////////////////////////////////////////////
/**
 * readDumpFile loads in a simulation from a GENESIS simdump file.
 * Works specially for reading in Kinetikit dump files.
 * In many old simulations the dump files are loaded in as regular script 
 * files. Here it is the job of the GENESIS parser to detect that these
 * are simDump files and treat them accordingly.
 * In a few cases the simulation commands and the simdump file have been
 * mixed up in a single file. MOOSE does not handle such cases.
 * This uses a local instance of SimDump, and does not interfere
 * with the private version in the Shell.
 */
void Shell::readDumpFile( const Conn* c, string filename )
{
	SimDump localSid;
	
	localSid.read( filename );
}

/**
 * writeDumpFile takes the specified comma-separated path and generates
 * an old-style GENESIS simdump file. Used mostly for dumping kinetic
 * models to kkit format.
 * This is equivalent to the simdump command from the GENESIS parser.
 * This uses the private SimDump object on the Shell because the
 * simObjDump function may need to set its state.
 */
void Shell::writeDumpFile( const Conn* c, string filename, string path )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->simDump_->write( filename, path );
}

/**
 * This function sets up the sequence of fields used in a dumpfile
 * This uses the private SimDump object on the Shell because the
 * writeDumpFile and simObjDump functions may need to use this state
 * information.
 * First argument is the function call, second is the name of the class.
 */
void Shell::simObjDump( const Conn* c, string fields )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->simDump_->simObjDump( fields );
}
/**
 * This function reads in a single dumpfile line.
 * It is only for the special case where the GenesisParser is reading
 * a dumpfile as if it were a script file.
 * This uses the private SimDump object on the Shell because the
 * simObjDump function may need to set its state.
 */
void Shell::simUndump( const Conn* c, string args )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->simDump_->simUndump( args );
}

void Shell::loadtab( const Conn* c, string data )
{
	Shell* sh = static_cast< Shell* >( c->data() );
	sh->innerLoadTab( data );
}

void Shell::tabop( const Conn* c, Id tab, char op, double min, double max )
{
	set< char, double, double >( tab(), "tabop", op, min, max );
}

//////////////////////////////////////////////////////////////////
// File handling functions
//////////////////////////////////////////////////////////////////

///\todo These things should NOT be globals.
map <string, FILE*> Shell::filehandler;
vector <string> Shell::filenames;
vector <string> Shell::modes;
vector <FILE*> Shell::filehandles;

void Shell::openFile( const Conn* c, string filename, string mode )
{
	FILE* o = fopen( filename.c_str(), mode.c_str() );
	if (o == NULL){
		cout << "Error: Shell::openFile Cannot openfile " << filename << endl;
		return;
	}
	map<string, FILE*>::iterator iter = filehandler.find(filename);
	if (iter != filehandler.end() ){
		cout << "File " << filename << " already being used." << endl;
		return;
	}
	//filehandler[filename] = o;
	filenames.push_back(filename);
	modes.push_back(mode);
	filehandles.push_back(o);
}



void Shell::writeFile( const Conn* c, string filename, string text )
{
	size_t i = 0;
	while (filenames[i] != filename && ++i);
	if ( i < filenames.size() ){
		if ( !( modes[i] == "w" || modes[i] == "a" ) ) {
			cout << "Error:: The file has not been opened in write mode" << endl;
			return;
		}
		fprintf(filehandles[i], "%s", text.c_str());
	}
	else {
		cout << "Error:: File "<< filename << " not opened!!" << endl;
		return;
	}
}

void Shell::closeFile( const Conn* c, string filename ){
	size_t i = 0;
	while (filenames[i] != filename && ++i);
	if ( i < filenames.size() ){
		if ( fclose(filehandles[i]) != 0 ) {
			cout << "Error:: Could not close the file." << endl;
			return;
		}
		filenames.erase( filenames.begin() + i );
		modes.erase( modes.begin() + i );
		filehandles.erase( filehandles.begin() + i );
	}
	else {
		cout << "Error:: File "<< filename << " not opened!!" << endl;
		return;
	}
}

void Shell::listFiles( const Conn* c ){
	string ret = "";
	for ( size_t i = 0; i < filenames.size(); i++ ) 
		ret = ret + filenames[i] + "\n";
	sendBack1< string >( c, getFieldSlot, ret );	
}


/*
Limitation: lines should be shorter than 1000 chars
*/
void Shell::readFile( const Conn* c, string filename, bool linemode ){
	size_t i = 0;
	while (filenames[i] != filename && ++i);
	if ( i < filenames.size() ){
		char str[1000];
		if (linemode){
			fgets( str, 1000, filehandles[i] );
		}
		else 
			fscanf( filehandles[i], "%s", str);
		string ret = str;
		if (ret[ ret.size() -1 ] == '\n' && !linemode)
			ret.erase( ret.end() - 1 );
		sendBack1< string >( c, getFieldSlot, ret );
	}
	else {
		cout << "Error:: File "<< filename << " not opened!!" << endl;
		return;
	}
}



//////////////////////////////////////////////////////////////////
// Helper functions.
//////////////////////////////////////////////////////////////////

/**
 * Creates a child element with the specified id, and schedules it.
 * Regular function.
 */
bool Shell::create( const string& type, const string& name, 
		Id parent, Id id )
{
	Element* p = parent();
	Element* child = Neutral::create( type, name, p, id );
	if ( child ) {
		recentElement_ = child->id();
		return 1;
	}
	return 0;
}

// Regular function
bool Shell::createArray( const string& type, const string& name, 
		Id parent, Id id, int n )
{	
	Element* p = parent();
	Element* child = Neutral::createArray( type, name, p, id, n );
	if ( child ) {
		recentElement_ = child->id();
		return 1;
	}
	return 0;
	
	/*const Cinfo* c = Cinfo::find( type );
	Element* p = parent();
	if ( !p ) {
		cout << "Error: Shell::create: No parent " << p << endl;
		return 0;
	}

	const Finfo* childSrc = p->findFinfo( "childSrc" );
	if ( !childSrc ) {
		// Sorry, couldn't resist it.
		cout << "Error: Shell::create: parent cannot handle child\n";
		return 0;
	}
	if ( c != 0 && p != 0 ) {
		Element* e = c->createArray( id, name, n, 0 );
		bool ret = childSrc->add( p, e, e->findFinfo( "child" ) );
		assert( ret );
		recentElement_ = id;
		ret = c->schedule( e );
		assert( ret );
		return 1;
	} else  {
		cout << "Error: Shell::create: Unable to find type " <<
			type << endl;
	}
	return 0;
	*/
}



// Regular function
void Shell::destroy( Id victim )
{
	// cout << "in Shell::destroy\n";
	Element* e = victim();
	if ( !e ) {
		cout << "Error: Shell::destroy: No element " << victim << endl;
		return;
	}

	set( e, "destroy" );
}



//////////////////////////////////////////////////////////////////////
// Deleted stuff.
//////////////////////////////////////////////////////////////////////

#ifdef OLD_SHELL_FUNCS
void Shell::pwe() const
{
	cout << cwe_ << endl;
}

void Shell::ce( Id dest )
{
	if ( dest() )
		cwe_ = dest;
}

void Shell::le ( Id eid )
{
	Element* e = eid( );
	if ( e ) {
		vector< Id > elist;
		vector< Id >::iterator i;
		get( e, "childList", elist );
		for ( i = elist.begin(); i != elist.end(); i++ ) {
			if ( ( *i )() != 0 )
				cout << ( *i )()->name() << endl;
		}
	}
}
#endif

#ifdef DO_UNIT_TESTS

#include <math.h>
#include "../element/Neutral.h"

void testShell()
{
	cout << "\nTesting Shell";

	Element* root = Element::root();
	ASSERT( root->id().zero() , "creating /root" );

	vector< string > vs;
	separateString( "a/b/c/d/e/f/ghij/k", vs, "/" );

	/////////////////////////////////////////
	// Test path parsing
	/////////////////////////////////////////

	ASSERT( vs.size() == 8, "separate string" );
	ASSERT( vs[0] == "a", "separate string" );
	ASSERT( vs[1] == "b", "separate string" );
	ASSERT( vs[2] == "c", "separate string" );
	ASSERT( vs[3] == "d", "separate string" );
	ASSERT( vs[4] == "e", "separate string" );
	ASSERT( vs[5] == "f", "separate string" );
	ASSERT( vs[6] == "ghij", "separate string" );
	ASSERT( vs[7] == "k", "separate string" );

	separateString( "a->b->ghij->k", vs, "->" );
	ASSERT( vs.size() == 4, "separate string" );
	ASSERT( vs[0] == "a", "separate string" );
	ASSERT( vs[1] == "b", "separate string" );
	ASSERT( vs[2] == "ghij", "separate string" );
	ASSERT( vs[3] == "k", "separate string" );
	
	Shell sh;

	/////////////////////////////////////////
	// Test element creation in trees
	// This used to be a set of unit tests for Shell, but now
	// the operations have been shifted over to Neutral.
	// I still set up the creation operations because they are
	// used later for path lookup
	/////////////////////////////////////////

	Id n = Id::lastId();
	Id a = Id::scratchId();
	bool ret = 0;

	ret = sh.create( "Neutral", "a", Id(), a );
	ASSERT( ret, "creating a" );
	ASSERT( a.id_ == n.id_ + 1 , "creating a" );
	ASSERT( ( sh.parent( a ) == 0 ), "finding parent" );

	Id b = Id::scratchId();
	ret = sh.create( "Neutral", "b", Id(), b );
	ASSERT( ret, "creating b" );
	ASSERT( b.id_ == n.id_ + 2 , "creating b" );

	Id c = Id::scratchId();
	ret = sh.create( "Neutral", "c", Id(), c );
	ASSERT( ret, "creating c" );
	ASSERT( c.id_ == n.id_ + 3 , "creating c" );

	Id a1 = Id::scratchId();
	ret = sh.create( "Neutral", "a1", a, a1 );
	ASSERT( ret, "creating a1" );
	ASSERT( a1.id_ == n.id_ + 4 , "creating a1" );
	ASSERT( ( sh.parent( a1 ) == a ), "finding parent" );

	Id a2 = Id::scratchId();
	ret = sh.create( "Neutral", "a2", a, a2 );
	ASSERT( ret, "creating a2" );
	ASSERT( a2.id_ == n.id_ + 5 , "creating a2" );

	/////////////////////////////////////////
	// Test path lookup operations
	/////////////////////////////////////////

	string path = sh.eid2path( a1 );
	ASSERT( path == "/a/a1", "a1 eid2path" );
	path = sh.eid2path( a2 );
	ASSERT( path == "/a/a2", "a2 eid2path" );

	Id eid = sh.innerPath2eid( "/a/a1", "/" );
	ASSERT( eid == a1, "a1 path2eid" );
	eid = sh.innerPath2eid( "/a/a2", "/" );
	ASSERT( eid == a2, "a2 path2eid" );

	/////////////////////////////////////////
	// Test digestPath
	/////////////////////////////////////////
	/*
	Id foo = Id::scratchId(); // first make another test element.
	ret = sh.create( "Neutral", "foo", a2, foo );
	ASSERT( ret, "creating /a/a2/foo" );
	Id f = Id::scratchId(); // first make another test element.
	ret = sh.create( "Neutral", "f", a2, f );
	ASSERT( ret, "creating /a/a2/f" );
	*/

	sh.cwe_ = a2;
	sh.recentElement_ = a1;
	path = "";
	sh.digestPath( path );
	ASSERT( path == "", "path == blank" );

	path = ".";
	sh.digestPath( path );
	ASSERT( path == "/a/a2", "path == /a/a2" );

	path = "^";
	sh.digestPath( path );
	ASSERT( path == "/a/a1", "path == /a/a1" );

	path = "f";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/f", "path == /a/a2/f" );

	path = "./";
	sh.digestPath( path );
	ASSERT( path == "/a/a2", "path == /a/a2" );

	path = "..";
	sh.digestPath( path );
	ASSERT( path == "/a", "path == /a" );

	path = "ax";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/ax", "path == /a/a2/ax" );

	path = "/a";
	sh.digestPath( path );
	ASSERT( path == "/a", "path == /a" );

	sh.cwe_ = Id();
	path = "..";
	sh.digestPath( path );
	ASSERT( path == "/", "path == /" );

	path = "^/b/c/d";
	sh.digestPath( path );
	ASSERT( path == "/a/a1/b/c/d", "path == /a/a1/b/c/d" );

	sh.cwe_ = a2;
	path = "./b/c/d";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/b/c/d", "path == /a/a2/b/c/d" );

	path = "bba/bba";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/bba/bba", "path == /a/a2/bba/bba" );

	path = "../below";
	sh.digestPath( path );
	ASSERT( path == "/a/below", "path == /a/below" );

	sh.cwe_ = Id();
	path = "../rumbelow";
	sh.digestPath( path );
	ASSERT( path == "/rumbelow", "path == /rumbelow" );
	
	sh.cwe_ = a2;
	path = "x/y/z";
	sh.digestPath( path );
	ASSERT( path == "/a/a2/x/y/z", "path == /a/a2/x/y/z" );

	path = "/absolute/x/y/z";
	sh.digestPath( path );
	ASSERT( path == "/absolute/x/y/z", "path == /absolute/x/y/z" );
	
	/////////////////////////////////////////
	// Test destroy operation
	/////////////////////////////////////////
	sh.destroy( a );
	sh.destroy( b );
	sh.destroy( c );
	ASSERT( a() == 0, "destroy a" );
	ASSERT( a1() == 0, "destroy a1" );
	ASSERT( a2() == 0, "destroy a2" );

	/////////////////////////////////////////
	// Test the loadTab operation
	/////////////////////////////////////////
	Element* tab = Neutral::create( "Table", "t1", Element::root(),
		Id::scratchId() );
	static const double EPSILON = 1.0e-9;
	static double values[] = 
		{ 1, 1.0628, 1.1253, 1.1874, 1.2487, 1.309,
			1.3681, 1.4258, 1.4817, 1.5358, 1.5878 };
	sh.innerLoadTab( "/t1 table 1 10 0 10		1 1.0628 1.1253 1.1874 1.2487 1.309 1.3681 1.4258 1.4817 1.5358 1.5878" );
	int ival;
	ret = get< int >( tab, "xdivs", ival );
	ASSERT( ret, "LoadTab" );
	ASSERT( ival == 10 , "LoadTab" );
	for ( unsigned int i = 0; i < 11; i++ ) {
		double y = 0.0;
		ret = lookupGet< double, unsigned int >( tab, "table", y, i );
		ASSERT( ret, "LoadTab" );
		ASSERT( fabs( y - values[i] ) < EPSILON , "LoadTab" );
	}
	set( tab, "destroy" );

}

#endif // DO_UNIT_TESTS
