/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <math.h>

#include <setjmp.h>
#include <FlexLexer.h>
#include "script.h"

#include "../shell/Shell.h"
#include "GenesisParser.h"
#include "GenesisParserWrapper.h"
#include "../element/Neutral.h"

const Cinfo* initGenesisParserCinfo()
{
	/**
	 * This is a shared message to talk to the Shell.
	 */
	static Finfo* parserShared[] =
	{
		// Setting cwe
		new SrcFinfo( "cwe", Ftype1< Id >::global() ),
		// Getting cwe back: First trigger a request
		new SrcFinfo( "trigCwe", Ftype0::global() ),
		// Then receive the cwe info
		new DestFinfo( "recvCwe", Ftype1< Id >::global(),
					RFCAST( &GenesisParserWrapper::recvCwe ) ),

		// Getting a list of child ids: First send a request with
		// the requested parent elm id.
		new SrcFinfo( "trigLe", Ftype1< Id >::global() ),
		// Then recv the vector of child ids. This function is
		// shared by several other messages as all it does is dump
		// the elist into a temporary local buffer.
		new DestFinfo( "recvElist", 
					Ftype1< vector< Id > >::global(), 
					RFCAST( &GenesisParserWrapper::recvElist ) ),

		///////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions.
		///////////////////////////////////////////////////////////////
		// Creating an object: Send out the request.
		new SrcFinfo( "create",
				Ftype3< string, string, Id >::global() ),
		// Creating an object: Recv the returned object id.
		new DestFinfo( "recvCreate",
					Ftype1< Id >::global(),
					RFCAST( &GenesisParserWrapper::recvCreate ) ),
		// Deleting an object: Send out the request.
		new SrcFinfo( "delete", Ftype1< Id >::global() ),

		///////////////////////////////////////////////////////////////
		// Value assignment: set and get.
		///////////////////////////////////////////////////////////////
		// Getting a field value as a string: send out request:
		new SrcFinfo( "get", Ftype2< Id, string >::global() ),
		// Getting a field value as a string: Recv the value.
		new DestFinfo( "recvField",
					Ftype1< string >::global(),
					RFCAST( &GenesisParserWrapper::recvField ) ),
		// Setting a field value as a string: send out request:
		new SrcFinfo( "set", // object, field, value 
				Ftype3< Id, string, string >::global() ),


		///////////////////////////////////////////////////////////////
		// Clock control and scheduling
		///////////////////////////////////////////////////////////////
		// Setting values for a clock tick: setClock
		new SrcFinfo( "setClock", // clockNo, dt, stage
				Ftype3< int, double, int >::global() ),
		// Assigning path and function to a clock tick: useClock
		new SrcFinfo( "useClock", // tick id, path, function
				Ftype3< Id, vector< Id >, string >::global() ),

		// Getting a wildcard path of elements: send out request
		// args are path, flag true for breadth-first list.
		new SrcFinfo( "el", Ftype2< string, bool >::global() ),
		// The return function for the wildcard past is the shared
		// function recvElist

		new SrcFinfo( "resched", Ftype0::global() ), // resched
		new SrcFinfo( "reinit", Ftype0::global() ), // reinit
		new SrcFinfo( "stop", Ftype0::global() ), // stop
		new SrcFinfo( "step", Ftype1< double >::global() ),
				// step, arg is time
		new SrcFinfo( "requestClocks", 
					Ftype0::global() ), //request clocks
		new DestFinfo( "recvClocks", 
					Ftype1< vector< double > >::global(), 
					RFCAST( &GenesisParserWrapper::recvClocks ) ),
		
		///////////////////////////////////////////////////////////////
		// Message info functions
		///////////////////////////////////////////////////////////////
		// Request message list: id elm, string field, bool isIncoming
		new SrcFinfo( "listMessages", 
					Ftype3< Id, string, bool >::global() ),
		// Receive message list and string with remote fields for msgs
		new DestFinfo( "recvMessageList",
					Ftype2< vector < Id >, string >::global(), 
					RFCAST( &GenesisParserWrapper::recvMessageList ) ),

		///////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions.
		///////////////////////////////////////////////////////////////
		// This function is for copying an element tree, complete with
		// messages, onto another.
		new SrcFinfo( "copy", Ftype3< Id, Id, string >::global() ),
		// This function is for moving element trees.
		new SrcFinfo( "move", Ftype3< Id, Id, string >::global() ),

		///////////////////////////////////////////////////////////////
		// Cell reader: filename cellpath
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readcell", Ftype2< string, string >::global() ),

		///////////////////////////////////////////////////////////////
		// Channel setup functions
		///////////////////////////////////////////////////////////////
		// setupalpha
		new SrcFinfo( "setupAlpha", 
					Ftype2< Id, vector< double > >::global() ),
		// setuptau
		new SrcFinfo( "setupTau", 
					Ftype2< Id, vector< double > >::global() ),
		// tweakalpha
		new SrcFinfo( "tweakAlpha", Ftype1< Id >::global() ),
		// tweaktau
		new SrcFinfo( "tweakTau", Ftype1< Id >::global() ),

		///////////////////////////////////////////////////////////////
		// SimDump facilities
		///////////////////////////////////////////////////////////////
		// readDumpFile
		new SrcFinfo( "readDumpFile", 
					Ftype1< string >::global() ),
		// writeDumpFile
		new SrcFinfo( "writeDumpFile", 
					Ftype2< string, string >::global() ),
		// simObjDump
		new SrcFinfo( "simObjDump",
					Ftype1< string >::global() ),
		// simundump
		new SrcFinfo( "simUndump",
					Ftype1< string >::global() ),

		///////////////////////////////////////////////////////////////
		// Setting field values for a vector of objects
		///////////////////////////////////////////////////////////////
		// Setting a vec of field values as a string: send out request:
		new SrcFinfo( "setVecField", // object, field, value 
			Ftype3< vector< Id >, string, string >::global() ),
	};
	
	static Finfo* genesisParserFinfos[] =
	{
		new SharedFinfo( "parser", parserShared,
				sizeof( parserShared ) / sizeof( Finfo* ) ),
		new DestFinfo( "readline",
			Ftype1< string >::global(),
			RFCAST( &GenesisParserWrapper::readlineFunc ) ),
		new DestFinfo( "process",
			Ftype0::global(),
			RFCAST( &GenesisParserWrapper::processFunc ) ), 
		new DestFinfo( "parse",
			Ftype1< string >::global(),
			RFCAST( &GenesisParserWrapper::parseFunc ) ), 
		new SrcFinfo( "echo", Ftype1< string>::global() ),

	};

	static Cinfo genesisParserCinfo(
		"GenesisParser",
		"Upinder S. Bhalla, NCBS, 2004-2007",
		"Object to handle the old Genesis parser",
		initNeutralCinfo(),
		genesisParserFinfos,
		sizeof(genesisParserFinfos) / sizeof( Finfo* ),
		ValueFtype1< GenesisParserWrapper >::global()
	);

	return &genesisParserCinfo;
}

static const Cinfo* genesisParserCinfo = initGenesisParserCinfo();
static const unsigned int setCweSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.cwe" );
static const unsigned int requestCweSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.trigCwe" );
static const unsigned int requestLeSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.trigLe" );
static const unsigned int createSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.create" );
static const unsigned int deleteSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.delete" );
static const unsigned int requestFieldSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.get" );
static const unsigned int setFieldSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.set" );
static const unsigned int setClockSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.setClock" );
static const unsigned int useClockSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.useClock" );
static const unsigned int requestWildcardListSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.el" );
static const unsigned int reschedSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.resched" );
static const unsigned int reinitSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.reinit" );
static const unsigned int stopSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.stop" );
static const unsigned int stepSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.step" );
static const unsigned int requestClocksSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.requestClocks" );
static const unsigned int listMessagesSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.listMessages" );
static const unsigned int copySlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.copy" );
static const unsigned int moveSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.move" );
static const unsigned int readCellSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.readcell" );

static const unsigned int setupAlphaSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.setupAlpha" );
static const unsigned int setupTauSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.setupTau" );
static const unsigned int tweakAlphaSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.tweakAlpha" );
static const unsigned int tweakTauSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.tweakTau" );

static const unsigned int readDumpFileSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.readDumpFile" );
static const unsigned int writeDumpFileSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.writeDumpFile" );
static const unsigned int simObjDumpSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.simObjDump" );
static const unsigned int simUndumpSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.simUndump" );

static const unsigned int setVecFieldSlot = 
	initGenesisParserCinfo()->getSlotIndex( "parser.setVecField" );

//////////////////////////////////////////////////////////////////
// Now we have the GenesisParserWrapper functions
//////////////////////////////////////////////////////////////////

/**
 * This initialization also adds the id of the forthcoming element
 * that the GenesisParserWrapper is going into. Note that the 
 * GenesisParserWrapper is made just before the Element is, so the
 * index is used directly. Note also that we assume that no funny
 * threading happens here.
 */
GenesisParserWrapper::GenesisParserWrapper()
		: returnCommandValue_( "" ), returnId_(),
		cwe_(), createdElm_()
{
		loadBuiltinCommands();
}

void GenesisParserWrapper::readlineFunc( const Conn& c, string s )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c.targetElement()->data() );

	data->AddInput( s );
}

void GenesisParserWrapper::processFunc( const Conn& c )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c.targetElement()->data() );

	data->Process();
}

void GenesisParserWrapper::parseFunc( const Conn& c, string s )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c.targetElement()->data() );

	data->ParseInput( s );
}

void GenesisParserWrapper::setReturnId( const Conn& c, Id id )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c.targetElement()->data() );

	data->returnId_ = id;
}


//////////////////////////////////////////////////////////////////
// GenesisParserWrapper utilities
//////////////////////////////////////////////////////////////////

char* copyString( const string& s )
{
	char* ret = ( char* ) calloc ( s.length() + 1, sizeof( char ) );
	strcpy( ret, s.c_str() );
	return ret;
}

void GenesisParserWrapper::print( const string& s, bool noNewLine )
{
	if ( testFlag_ ) {
		printbuf_ = printbuf_ + s + " ";
	} else {
		cout << s;
		if ( !noNewLine )
				cout << endl;
	}
}

//////////////////////////////////////////////////////////////////
// GenesisParserWrapper Message recv functions
//////////////////////////////////////////////////////////////////

void GenesisParserWrapper::recvCwe( const Conn& c, Id cwe )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c.targetElement()->data() );
	gpw->cwe_ = cwe;
}

//
//This is used for Le, for WildcardList, and others
void GenesisParserWrapper::recvElist( const Conn& c, vector< Id > elist)
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c.targetElement()->data() );
	gpw->elist_ = elist;
}

void GenesisParserWrapper::recvCreate( const Conn& c, Id e )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c.targetElement()->data() );
	gpw->createdElm_ = e;
}

void GenesisParserWrapper::recvField( const Conn& c, string value )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c.targetElement()->data() );
	gpw->fieldValue_ = value;
}

void GenesisParserWrapper::recvClocks( 
				const Conn& c, vector< double > dbls)
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c.targetElement()->data() );
	gpw->dbls_ = dbls;
}

void GenesisParserWrapper::recvMessageList( 
				const Conn& c, vector< Id > elist, string s)
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c.targetElement()->data() );
	gpw->elist_ = elist;
	gpw->fieldValue_ = s;
}

//////////////////////////////////////////////////////////////////
// GenesisParserWrapper Builtin commands
//////////////////////////////////////////////////////////////////

/**
 * This map converts the old GENESIS message syntax into the MOOSE
 * syntax for the source side of a message.
 */
map< string, string >& sliSrcLookup()
{
	static map< string, string > src;

	if ( src.size() > 0 )
		return src;
	
	src[ "REAC A B" ] = "sub";	// for reactions
	src[ "SUBSTRATE n" ] = "";
	src[ "SUBSTRATE n vol" ] = "reac"; // For concchans.
	src[ "PRODUCT n vol" ] = "reac"; // For concchans.
	src[ "NUMCHAN n" ] = "nOut"; // From molecules to concchans.
	src[ "REAC B A" ] = "prd";
	src[ "PRODUCT n" ] = "";

	src[ "REAC sA B" ] = "sub";	// for enzymes
	src[ "SUBSTRATE n" ] = "";
	src[ "REAC eA B" ] = "";	// Target is molecule. Ignore as it
								// only applies to explicit enz.
	src[ "ENZYME n" ] = "reac"; // target is an enzyme. Use it.
	src[ "PRODUCT n" ] = "";
	src[ "MM_PRD pA" ] = "prd";

	src[ "SUMTOTAL n nInit" ] = "nSrc";	// for molecules
	src[ "SUMTOTAL output output" ] = "outputSrc";	// for tables
	src[ "SLAVE output" ] = "out";	// for tables
	src[ "INTRAMOL n" ] = "nOut"; 	// target is an enzyme.
	src[ "CONSERVE n nInit" ] = ""; 	// Deprecated
	src[ "CONSERVE nComplex nComplexInit" ] = ""; 	// Deprecated

	// Some messages for compartments.
	src[ "AXIAL Vm" ] = "axial";
	src[ "AXIAL previous_state" ] = "axial";
	src[ "RAXIAL Ra Vm" ] = "";
	src[ "RAXIAL Ra previous_state" ] = "";
	src[ "INJECT output" ] = "output";

	// Some messages for channels.
	src[ "VOLTAGE Vm" ] = "";
	src[ "CHANNEL Gk Ek" ] = "channel";

	// Some messages for gates, used in the squid demo. This 
	// is used to set the reset value of Vm in the gates, which is 
	// done already through the existing messaging.
	src[ "EREST Vm" ] = "";

	// Some messages for tables
	src[ "INPUT Vm" ] = "Vm";

	// Messages for having tables pretend to be an xplot
	src[ "PLOT Co" ] = "conc";
	src[ "PLOT n" ] = "n";
	return src;
}

/**
 * This map converts the old GENESIS message syntax into the MOOSE
 * syntax for the destination side of a message.
 */
map< string, string >& sliDestLookup()
{
	static map< string, string > dest;

	if ( dest.size() > 0 )
		return dest;

	dest[ "SUBSTRATE n vol" ] = "influx"; // For channels.
	dest[ "PRODUCT n vol" ] = "efflux";
	dest[ "NUMCHAN n" ] = "nIn"; // From molecules to concchans.
	
	dest[ "REAC A B" ] = "reac";	// for reactions
	dest[ "SUBSTRATE n" ] = "";
	dest[ "REAC B A" ] = "reac";
	dest[ "PRODUCT n" ] = "";

	dest[ "REAC sA B" ] = "reac";	// for enzymes
	dest[ "SUBSTRATE n" ] = "";
	dest[ "REAC eA B" ] = "";		// Target is enzyme, but only used
									// for explicit enzymes. Ignore.
	dest[ "ENZYME n" ] = "enz"; 	// Used both for explicit and MM.
	dest[ "PRODUCT n" ] = "";
	dest[ "MM_PRD pA" ] = "prd";

	dest[ "SUMTOTAL n nInit" ] = "sumTotal";	// for molecules
	dest[ "SUMTOTAL output output" ] = "sumTotal";	// for molecules
	dest[ "SLAVE output" ] = "sumTotalIn";	// for molecules
	dest[ "INTRAMOL n" ] = "intramolIn"; 	// target is an enzyme.
	dest[ "CONSERVE n nInit" ] = ""; 	// Deprecated
	dest[ "CONSERVE nComplex nComplexInit" ] = ""; 	// Deprecated

	// Some messages for compartments.
	dest[ "AXIAL Vm" ] = "raxial";
	dest[ "AXIAL previous_state" ] = "raxial";
	dest[ "RAXIAL Ra Vm" ] = "";
	dest[ "RAXIAL Ra previous_state" ] = "";
	dest[ "INJECT output" ] = "injectMsg";

	// Some messages for channels.
	dest[ "VOLTAGE Vm" ] = "";
	dest[ "CHANNEL Gk Ek" ] = "channel";

	// Some messages for gates, used in the squid demo. This 
	// is used to set the reset value of Vm in the gates, which is 
	// done already through the existing messaging.
	dest[ "EREST Vm" ] = "";

	// Some messages for tables
	dest[ "INPUT Vm" ] = "inputRequest";

	// Messages for having tables pretend to be an xplot
	dest[ "PLOT Co" ] = "inputRequest";
	dest[ "PLOT n" ] = "inputRequest";

	return dest;
}

map< string, string >& sliClassNameConvert()
{
	static map< string, string > classnames;

	if ( classnames.size() > 0 )
		return classnames;
	
	classnames[ "neutral" ] = "Neutral";
	classnames[ "pool" ] = "Molecule";
	classnames[ "kpool" ] = "Molecule";
	classnames[ "reac" ] = "Reaction";
	classnames[ "kreac" ] = "Reaction";
	classnames[ "enz" ] = "Enzyme";
	classnames[ "kenz" ] = "Enzyme";
	classnames[ "kchan" ] = "ConcChan";
	classnames[ "conc_chan" ] = "ConcChan";
	classnames[ "Ca_concen" ] = "CaConc";
	classnames[ "compartment" ] = "Compartment";
	classnames[ "hh_channel" ] = "HHChannel";
	classnames[ "tabchannel" ] = "HHChannel";
	classnames[ "vdep_channel" ] = "HHChannel";
	classnames[ "vdep_gate" ] = "HHGate";
	classnames[ "table" ] = "Table";
	classnames[ "xbutton" ] = "Sli";
	classnames[ "xdialog" ] = "Sli";
	classnames[ "xlabel" ] = "Sli";
	classnames[ "xform" ] = "Sli";
	classnames[ "xtoggle" ] = "Sli";
	classnames[ "xshape" ] = "Sli";
	classnames[ "xgraph" ] = "Sli";
	classnames[ "x1dialog" ] = "Sli";
	classnames[ "x1button" ] = "Sli";
	classnames[ "x1shape" ] = "Sli";
	classnames[ "xtext" ] = "Sli";

	return classnames;
}

/*
map< string, string >& sliFieldNameConvert()
{
	static map< string, string > fieldnames;

	if ( fieldnames.size() > 0 )
		return fieldnames;

	return fieldnames;
}
*/


string sliMessage(
	const string& msgType, map< string, string >& converter )
{
	map< string, string >::iterator i;
	i = converter.find( msgType );
	if ( i != converter.end() ) {
		if ( i->second.length() == 0 ) // A redundant message 
			return "";
		else 
			return i->second; // good message.
	} else {
		// Trim off bits of the message, see if it recognizes it.
		// Require that at least one string be known.
		// Always go with more specific first, then more general.
		vector< string > args;
		separateString( msgType, args, " " );
		if ( args.size() == 1 ) {
			cout << "Error:sliMessage: Unknown message " <<
				msgType << "\n";
		} else {
			vector< string >::iterator j;
			string subType = "";
			for ( j = args.begin(); j != args.end() - 1; j++ ) {
				if ( j == args.begin() )
					subType = *j;
				else
					subType = subType + " " + *j;
			}
			return sliMessage( subType, converter );
		}
	}
	return "";
}

void do_add( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	return gpw->doAdd( argc, argv, s );
}

bool GenesisParserWrapper::innerAdd(
	Id src, const string& srcF, Id dest, const string& destF )
{
	if ( !src.bad() && !dest.bad() ) {
		Element* se = src();
		Element* de = dest();
		const Finfo* sf = se->findFinfo( srcF );
		if ( !sf ) return 0;
		const Finfo* df = de->findFinfo( destF );
		if ( !df ) return 0;

		return se->findFinfo( srcF )->add( se, de, de->findFinfo( destF )) ;
	}
	return 0;
}

void GenesisParserWrapper::doAdd(
				int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
		string srcE = Shell::head( argv[1], "/" );
		string srcF = Shell::tail( argv[1], "/" );
		string destE = Shell::head( argv[2], "/" );
		string destF = Shell::tail( argv[2], "/" );
		Id src( srcE );
		Id dest( destE );
		// Id src = path2eid( srcE, s );
		// Id dest = path2eid( destE, s );

		// Should ideally send this off to the shell.
		if ( !innerAdd( src, srcF, dest, destF ) )
	 		cout << "Error in doAdd " << argv[1] << " " << argv[2] << endl;
	} else if ( argc > 3 ) {
	// Old-fashioned addmsg. Backward Compatibility conversions here.
	// usage: addmsg source-element dest-delement msg-type [msg-fields]
	// Most of these are handled using the info in the msg-type and
	// msg fields. Often there are redundant messages which are now
	// handled by shared messages. The redundant one is ignored.
		string msgType = argv[3];

		for ( int i = 4; i < argc; i++ )
			msgType = msgType + " " + argv[ i ];

		string srcF = sliMessage( msgType, sliSrcLookup() );
		string destF = sliMessage( msgType, sliDestLookup() );

		if ( srcF.length() > 0 && destF.length() > 0 ) {
			Id src( argv[1] );
			Id dest( argv[2] );
			// Id src = path2eid( argv[1], s );
			// Id dest = path2eid( argv[2], s );
	// 		cout << "in do_add " << src << ", " << dest << endl;
			if ( !innerAdd( src, srcF, dest, destF ) )
	 			cout << "Error in do_add " << argv[1] << " " << argv[2] << " " << msgType << endl;
		}
		//	s->addFuncLocal( src, dest );
	} else {
		cout << "usage:: " << argv[0] << " src dest\n";
		cout << "deprecated usage:: " << argv[0] << " source-element dest-element msg-type [msg-fields]";
	}
}

void do_drop( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
		// s->dropFuncLocal( argv[1], argv[2] );
		cout << "In do_drop " << argv[1] << ", " << argv[2] << endl;
	} else {
		cout << "usage:: " << argv[0] << " src dest\n";
	}
}

/**
 * setfield [obj] field value [field value] ...
 */
void do_set( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	return gpw->doSet( argc, argv, s );
}

void GenesisParserWrapper::doSet( int argc, const char** argv, Id s )
{
	static map< string, string > tabmap;
	tabmap[ "X_A" ] = "xGate/A";
	tabmap[ "X_B" ] = "xGate/B";
	tabmap[ "Y_A" ] = "yGate/A";
	tabmap[ "Y_B" ] = "yGate/B";
	tabmap[ "Z_A" ] = "zGate/A";
	tabmap[ "Z_B" ] = "zGate/B";

	Element* sh = s();
	int start = 2;
	if ( argc < 3 ) {
		cout << argv[0] << ": Too few command arguments\n";
		cout << "usage:: " << argv[0] << " [path] field value ...\n";
		return;
	}
	if ( argc % 2 == 1 ) { // 'path' is left out, use current object.
		send0( sh, requestCweSlot );
		elist_.resize( 0 );
		elist_.push_back( cwe_ );
		// e = cwe_;
		start = 1;
	} else  {
		string path = argv[1];
		send2< string, bool >( sh, requestWildcardListSlot, path, 0 );
		if ( elist_.size() == 0 ) {
			return;
		}
		// e = GenesisParserWrapper::path2eid( argv[1], s );
		// Try wildcards
		// if ( e == BAD_ID )
			// return;
		start = 2;
	}

	// Hack here to deal with the special case of filling tables
	// in tabchannels. Example command looks like:
	// 		setfield Ca Y_A->table[{i}] {y}
	// so here we need to do setfield Ca/yGate/A table[{i}] {y}
	
	for ( int i = start; i < argc; i += 2 ) {
		// s->setFuncLocal( path + "/" + argv[ i ], argv[ i + 1 ] );
		string field = argv[i];
		string value = argv[ i+1 ];
		string::size_type pos = field.find( "->table" );
		if ( pos == string::npos )
				pos = field.find( "->calc_mode" );
		if ( pos == string::npos )
				pos = field.find( "->sy" );
		if ( pos != string::npos ) { // Fill table
			map< string, string >::iterator i = 
					tabmap.find( field.substr( 0, 3 ) );
			if ( i != tabmap.end() ) {
				string path;
				if ( start == 1 )
					path = "./" + i->second;
				else
					path = string( argv[1] ) + "/" + i->second;
				// Id e = GenesisParserWrapper::path2eid( path, s );
				Id e( path );
				field = field.substr( pos + 2 );
				send3< Id, string, string >( s(),
					setVecFieldSlot, e, field, value );
				continue;
			}
		}
		// cout << "in do_set " << path << "." << field << " " <<
				// value << endl;
		send3< vector< Id >, string, string >( s(),
			setVecFieldSlot, elist_, field, value );
	}
}

/** 
 * tabCreate handles the allocation of tables.
 * Returns true on success
 * For channels the GENESIS command is something like 
 * call /squid/Na TABCREATE X {NDIVS} {VMIN} {VMAX}
 *
 * For tables this is 
 * call /Vm TABCREATE {NDIVS} {XMIN} {XMAX}
 */
bool GenesisParserWrapper::tabCreate( int argc, const char** argv, Id s)
{
	string path = argv[1];
	// Id id = path2eid( path, s );
	Id id( path );
	if ( !id.zero() && !id.bad() ) {
		send2< Id, string >( s(),
			requestFieldSlot, id, "class" );
		if ( fieldValue_.length() == 0 ) // Nothing came back
			return 0;
		if ( fieldValue_ == "HHChannel" && argc == 7 ) {
			// TABCREATE on HHChannel requires the assignment of
			// something like
			// setfield /squid/Na/xGate/A xmin {XMIN}
			// setfield /squid/Na/xGate/A xmax {XMAX}
			// setfield /squid/Na/xGate/A xdivs {XDIVS}
			if ( strlen( argv[3] ) != 1 ) {
				cout << "usage: call element TABCREATE gate xdivs xmin xmax\n";
				cout << "Error: Gate should be X, Y or Z\n";
				return 0;
			}
			string tempA;
			string tempB;
			if ( string( argv[3] ) == "X" ) {
				tempA = path + "/xGate/A";
				tempB = path + "/xGate/B";
			} else if ( string( argv[3] ) == "Y" ) {
				tempA = path + "/yGate/A";
				tempB = path + "/yGate/B";
			} else if ( string( argv[3] ) == "Z" ) {
				tempA = path + "/zGate/A";
				tempB = path + "/zGate/B";
			}
			// id = path2eid( tempA, s );
			id = Id( tempA );
			if ( id.zero() || id.bad() ) return 0; //Error msg here
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xdivs", argv[4] );
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xmin", argv[5] );
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xmax", argv[6] );
			// id = path2eid( tempB, s );
			id = Id( tempB );
			if ( id.zero() || id.bad() ) return 0; //Error msg here
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xdivs", argv[4] );
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xmin", argv[5] );
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xmax", argv[6] );
			return 1;
		}
		if ( fieldValue_ == "Table" && argc == 6 ) {
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xdivs", argv[3] );
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xmin", argv[4] );
			send3< Id, string, string >( s(),
				setFieldSlot, id, "xmax", argv[5] );
			return 1;
		}
	}
	return 0;
}

void do_call( int argc, const char** const argv, Id s )
{
	if ( argc < 3 ) {
		cout << "usage:: " << argv[0] << " path field/Action [args...]\n";
		return;
	}
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	// Ugly hack to avoid LOAD call for notes on kkit dumpfiles
	if ( strcmp ( argv[2], "LOAD" ) == 0 ) {
		// cout << "in do_call LOAD " << endl;
		return;
	}
	
	// Ugly hack to handle the TABCREATE calls, which get converted
	// to three setfields, or six if it is a tabchannel. 
	if ( strcmp ( argv[2], "TABCREATE" ) == 0 ) {
		if ( !gpw->tabCreate( argc, argv, s ) )
				cout << "Error: TABCREATE failed\n";
	}

	// Ugly hack to handle the TABFILL call, which need to be redirected
	// to the two interpols of the HHGates.
	// Deprecated.
	// The call looks like:
	// call KM_bsg_yka TABFILL X 3000 0
	//
	if ( strcmp ( argv[2], "TABFILL" ) == 0 ) {
		if ( argc != 6 ) {
			cout << "usage: " << argv[0] << 
					" element TABFILL gate divs mode\n";
			return;
		}
		string elmpath = argv[1];
		string argstr = argv[4];
		argstr = argstr + "," + argv[5];
		int ndivs = atoi( argv[4] );
		if ( ndivs < 1 )
				return;
		if ( argv[3][0] == 'X' )
				elmpath = elmpath + "/xGate";
		else if ( argv[3][0] == 'Y' )
				elmpath = elmpath + "/yGate";
		else if ( argv[3][0] == 'Z' )
				elmpath = elmpath + "/zGate";
		else {
			cout << "Error: " << argv[0] << 
					" unknown gate '" << argv[3] << "'\n";
			return;
		}
		string temp = elmpath + "/A";
		// Id gate = GenesisParserWrapper::path2eid( temp, s );
		Id gate( temp );
		if ( gate.bad() ) {
			cout << "Error: " << argv[0] << 
					" could not find object '" << temp << "'\n";
			return;
		}
		send3< Id, string, string >( s(),
			setFieldSlot, gate, "tabFill", argstr );

		temp = elmpath + "/B";
		// gate = GenesisParserWrapper::path2eid( temp, s );
		gate = Id( temp );
		if ( gate.bad() ) {
			cout << "Error: " << argv[0] << 
					" could not find object '" << temp << "'\n";
			return;
		}
		send3< Id, string, string >( s(),
			setFieldSlot, gate, "tabFill", argstr );

		return;
	}
}

int do_isa( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
		cout << "in do_isa " << argv[1] << ", " << argv[2] << endl;
		// return s->isaFuncLocal( argv[1], argv[2] );
	} else {
		cout << "usage:: " << argv[0] << " type field\n";
	}
	return 0;
}

bool GenesisParserWrapper::fieldExists(
			Id eid, const string& field, Id s )
{
	send2< Id, string >( s(), requestFieldSlot, eid, "fieldList" );
	if ( fieldValue_.length() == 0 ) // Nothing came back
		return 0;
	return ( fieldValue_.find( field ) != string::npos );
}

int do_exists( int argc, const char** const argv, Id s )
{
	if ( argc == 2 ) { // Checking for element
		// Id eid = GenesisParserWrapper::path2eid( argv[1], s );
		Id eid( argv[1] );
		return ( !eid.bad() );
	} else if ( argc == 3 ) { // checking for element and field.
		// Id eid = GenesisParserWrapper::path2eid( argv[1], s );
		Id eid( argv[1] );
		if ( !eid.bad() ) {
			GenesisParserWrapper* gpw =
				static_cast< GenesisParserWrapper* >( s()->data() );
			return gpw->fieldExists( eid, argv[2], s );
		}
	} else {
		cout << "usage:: " << argv[0] << " element [field]\n";
	}
	return 0;
}

/**
 * getfield [obj] field
 */
char* do_get( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	return gpw->doGet( argc, argv, s );
}

char* GenesisParserWrapper::doGet( int argc, const char** argv, Id s )
{
	string field;
	string value;
	Id e;
	if ( argc == 3 ) {
		// e = GenesisParserWrapper::path2eid( argv[1], s );
		e = Id( argv[1] );
		if ( e.bad() )
			return copyString( "" );
		field = argv[2];
	} else if ( argc == 2 ) {
		send0( s(), requestCweSlot );
		e = cwe_;
		field = argv[ 1 ];
	} else {
		cout << "usage:: " << argv[0] << " [element] field\n";
		return copyString( "" );
	}
	fieldValue_ = "";
	send2< Id, string >( s(),
		requestFieldSlot, e, field );
	if ( fieldValue_.length() == 0 ) // Nothing came back
		return 0;
	return copyString( fieldValue_.c_str() );
}

char* do_getmsg( int argc, const char** const argv, Id s )
{
	static string space = " ";
	if ( argc < 3 ) {
		cout << "usage:: " << argv[0] << " element -incoming -outgoing -slot msg-number slot-number -count -type msg-number -destination msg-number -source msg-number -find srcelem type\n";
		return "";
	}
	string field = argv[ 1];
	string options = argv[ 2 ];
	for ( int i = 3; i < argc; i++ )
		options += space + argv[ i ];

	// return copyString( s->getmsgFuncLocal( field, options ) );
	return copyString( "" );
}

void do_showmsg( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	gpw->doShowMsg( argc, argv, s );
}

/**
 * Displays information about incoming/outgoing messages.
 * Unlike GENESIS, we can put in additional info about the message
 * type, but we cannot put in all the current arguments without
 * special effort.
 * Form of output is
 * INCOMING MESSAGES onto <mypath>
 * MSG[0]:	into <destField> from [<path1>, <path2>, ...].<srcField>
 * ...
 *
 * OUTGOING MESSAGES
 * MSG[0]:	from <srcField> into [<path1>, <path2>, ...].<destField>
 * ...
 *
 * Algorithm: ask Shell for list of all fields on object.
 * Increment through each requesting incoming/outgoing messages.
 * This uses the Shell::listMessages function.
 */

void printMessageList( const string& f1, const string& f2,
		vector< Id >& elist, unsigned int& msgNum, bool isIncoming )
{
	vector< string > list;
	separateString( f2, list, ", " );
	if ( elist.size() > 0 ) {
		assert( elist.size() == list.size() );
		unsigned int i;
		if ( isIncoming )
			cout << "MSG[" << msgNum << "]:	into " << f1 << " from: ";
		else
			cout << "MSG[" << msgNum << "]:	from " << f1 << " into: ";
		string lastField = "";
		for ( i = 0; i < elist.size(); i++ ) {
			if ( lastField != list[i] ) {
				if ( lastField != "" )
					cout << " ]." << lastField << "\n	[ ";
				else
					cout << "[ ";
			} else {
					cout << ", ";
			}
			// cout << GenesisParserWrapper::eid2path( elist[i] );
			cout << elist[i].path();
			lastField = list[i];
		}
		cout << " ]." << lastField << "\n";
		msgNum++;
	}
}

void GenesisParserWrapper::doShowMsg( int argc, const char** argv, Id s)
{
	Id e;
	if ( argc == 1 ) {
		send0( s(), requestCweSlot );
		e = cwe_;
	} else {
		// e = path2eid( argv[1], s );
		e = Id( argv[1] );
		if ( e.bad() ) {
			cout << "Error: " << argv[0] << ": unknown element " <<
					argv[1] << endl;
			return;
		}
	}
	send2< Id, string >( s(), requestFieldSlot, e, "fieldList" );
	vector< string > list;
	vector< string >::iterator i;
	separateString( fieldValue_, list, ", " );

	// cout << "INCOMING MESSAGES onto " << eid2path( e ) << endl;
	cout << "INCOMING MESSAGES onto " << e.path() << endl;
	unsigned int msgNum = 0;
	for ( i = list.begin(); i != list.end(); i++ ) {
		if ( *i == "fieldList" )
			continue;
		send3< Id, string, bool >( s(), listMessagesSlot, e, *i, 1 );
		// The return message puts the elements in elist_ and the 
		// target field names in fieldValue_
		printMessageList( *i, fieldValue_, elist_, msgNum, 1 );
	}

	// cout << "OUTGOING MESSAGES from " << eid2path( e ) << endl;
	cout << "OUTGOING MESSAGES from " << e.path() << endl;
	msgNum = 0;
	for ( i = list.begin(); i != list.end(); i++ ) {
		if ( *i == "fieldList" )
			continue;
		send3< Id, string, bool >( s(), listMessagesSlot, e, *i, 0 );
		// The return message puts the elements in elist_ and the 
		// target field name in fieldValue_
		printMessageList( *i, fieldValue_, elist_, msgNum, 0);
	}
}

void do_create( int argc, const char** const argv, Id s )
{
	if ( argc != 3 ) {
		cout << "usage:: " << argv[0] << " class name\n";
		return;
	}
	string className = argv[1];
	if ( !Cinfo::find( className ) )  {
		// Possibly it is aliased for backward compatibility.
		map< string, string >::iterator i = 
			sliClassNameConvert().find( argv[1] );
		if ( i != sliClassNameConvert().end() ) {
			className = i->second;
			if ( className == "Sli" ) {
				// We bail out of these classes as MOOSE does not
				// yet handle them.
				cout << "Do not know how to handle class: " << 
						className << endl;
				return;
			}
		} else {
			cout << "GenesisParserWrapper::do_create: Do not know class: " << className << endl;
			return;
		}
	}

	string name = Shell::tail( argv[2], "/" );
	if ( name.length() < 1 ) {
		cout << "Error: invalid object name : " << name << endl;
		return;
	}
	string parent = Shell::head( argv[2], "/" );
	// Id pa = GenesisParserWrapper::path2eid( parent, s );
	Id pa( parent );

	send3< string, string, Id >( s(),
		createSlot, className, name, pa );

		// The return function recvCreate gets the id of the
		// returned elm, but
		// the GenesisParser does not care.
}

void do_delete( int argc, const char** const argv, Id s )
{
	if ( argc == 2 ) {
		// Id victim = GenesisParserWrapper::path2eid( argv[1], s );
		Id victim( argv[1] );
		if ( victim != Id() )
			send1< Id >( s(), deleteSlot, victim );
	} else {
		cout << "usage:: " << argv[0] << " Element/path\n";
	}
}

/**
 * This function figures out destination and name for moves
 * and copies. The rules are:
 * 1. Target can be existing element, in which case name is retained.
 *  In this case the childname is set to the empty string "".
 * 2. Target might not exist. In this case the parent of the target
 * 	must exist, and the object e is going to be renamed. This new
 * 	name is returned in the childname.
 */
bool parseCopyMove( int argc, const char** const argv, Id s,
		Id& e, Id& pa, string& childname )
{
	if ( argc == 3 ) {
		// e = GenesisParserWrapper::path2eid( argv[1], s );
		e = Id( argv[1] );
		if ( !e.zero() && !e.bad() ) {
			childname = "";
			// pa = GenesisParserWrapper::path2eid( argv[2], s );
			pa = Id( argv[2] );
			if ( pa.bad() ) { // Possibly we are renaming it too.
				string pastr = argv[2];
				if ( pastr.find( "/" ) == 0 ) {
					pastr = "/";
				} else {
					pastr = Shell::head( argv[2], "/" );
				}
				if ( pastr == "" )
						pastr = ".";
				// pa = GenesisParserWrapper::path2eid( pastr, s );
				pa = Id( pastr );
				if ( pa.bad() ) { // Nope, even that doesn't work.
					cout << "Error: " << argv[0] << 
							": Parent element " << argv[2] << 
							" not found\n";
					return 0;
				}
				childname = Shell::tail( argv[2], "/" );
			}
			return 1;
		} else {
			cout << "Error: " << argv[0] << ": source element " <<
					argv[1] << " not found\n";
		}
	} else {
		cout << "usage:: " << argv[0] << " src dest\n";
	}
	return 0;
}

void do_move( int argc, const char** const argv, Id s )
{
	Id e;
	Id pa;
	string name;
	if ( parseCopyMove( argc, argv, s, e, pa, name ) ) {
		send3< Id, Id, string >( s(), moveSlot, e, pa, name );
	}
}

void do_copy( int argc, const char** const argv, Id s )
{
	Id e;
	Id pa;
	string name;
	if ( parseCopyMove( argc, argv, s, e, pa, name ) ) {
		send3< Id, Id, string >( s(), copySlot, e, pa, name );
	}
}

void do_copy_shallow( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
		// s->copyShallowFuncLocal( argv[1], argv[2] );
	} else {
		cout << "usage:: " << argv[0] << " src dest\n";
	}
}

void do_copy_halo( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
		// s->copyHaloFuncLocal( argv[1], argv[2] );
	} else {
		cout << "usage:: " << argv[0] << " src dest\n";
	}
}

void do_ce( int argc, const char** const argv, Id s )
{
	if ( argc == 2 ) {
		// Id e = GenesisParserWrapper::path2eid( argv[1], s );
		Id e( argv[1] );
		send1< Id >( s(), setCweSlot, e );
	} else {
		cout << "usage:: " << argv[0] << " Element\n";
	}
}

void do_pushe( int argc, const char** const argv, Id s )
{
	if ( argc == 2 ) {
		// s->pusheFuncLocal( argv[1] );
	} else {
		cout << "usage:: " << argv[0] << " Element\n";
	}
}

void do_pope( int argc, const char** const argv, Id s )
{
	if ( argc == 1 ) {
		// s->popeFuncLocal( );
	} else {
		cout << "usage:: " << argv[0] << "\n";
	}
}

void do_alias( int argc, const char** const argv, Id s )
{
	string alias = "";
	string old = "";
	if ( argc == 3 ) {
		alias = argv[1];
		old = argv[2];
	} else if ( argc == 2 ) {
		alias = argv[1];
	}
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	gpw->alias( alias, old );
}

void do_quit( int argc, const char** const argv, Id s )
{
	// s->quitFuncLocal( );
		exit( 0 );
}

void do_stop( int argc, const char** const argv, Id s )
{
	if ( argc == 1 ) {
		send0( s(), stopSlot );
	} else {
		cout << "usage:: " << argv[0] << "\n";
	}
}

void do_reset( int argc, const char** const argv, Id s )
{
	if ( argc == 1 ) {
		send0( s(), reschedSlot );
		send0( s(), reinitSlot );
		;
	} else {
		cout << "usage:: " << argv[0] << "\n";
	}
}

void do_step( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	gpw->step( argc, argv );
}

void GenesisParserWrapper::step( int argc, const char** const argv )
{
	double runtime;
	Element* e = element()();
	if ( argc == 3 ) {
		if ( strcmp( argv[ 2 ], "-t" ) == 0 ) {
			runtime = strtod( argv[ 1 ], 0 );
		} else {
			cout << "usage:: " << argv[0] << 
					" time/nsteps [-t -s(default ]\n";
			return;
		}
	} else {
		send0( e, requestClocksSlot ); 
		assert( dbls_.size() > 0 );
		// This fills up a vector of doubles with the clock duration.
		// Find the shortest dt.
		double min = 1.0e10;
		vector< double >::iterator i;
		for ( i = dbls_.begin(); i != dbls_.end(); i++ )
			if ( min > *i )
					min = *i ;
		if ( argc == 1 )
			runtime = min;
		else 
			runtime = min * strtol( argv[1], 0, 10 );
	}
	if ( runtime < 0 ) {
		cout << "Error: " << argv[0] << ": negative time is illegal\n";
		return;
	}

	send1< double >( e, stepSlot, runtime );
}

void do_setclock( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
			send3< int, double, int >( s(),
				setClockSlot, 
				atoi( argv[1] ), atof( argv[2] ), 0 );
	} else if ( argc == 4 ) {
			send3< int, double, int >( s(),
				setClockSlot, 
				atoi( argv[1] ), atof( argv[2] ), atoi( argv[3] ) );
	} else {
		cout << "usage:: " << argv[0] << " clockNum dt [stage]\n";
	}
}

void GenesisParserWrapper::showClocks( Element* e )
{
	send0( e, requestClocksSlot ); 
	// This fills up a vector of doubles with the clock duration.
	cout << "ACTIVE CLOCKS\n";
	cout << "-------------\n";
	for ( unsigned int i = 0; i < dbls_.size(); i++ )
		cout << "[" << i << "]	: " << dbls_[i] << endl;
}

void do_showclocks( int argc, const char** const argv, Id s )
{
	if ( argc == 1 ) {
		Element* e = s();
		GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
				( e->data() );
		gpw->showClocks( e );

	} else {
		cout << "usage:: " << argv[0] << "\n";
	}
}

void do_useclock( int argc, const char** const argv, Id s )
{
	char tickName[40];
	string func = "process";
	if ( argc == 3 ) {
		sprintf( tickName, "/sched/cj/t%s", argv[2] );
	} else if ( argc == 4 ) {
		sprintf( tickName, "/sched/cj/t%s", argv[2] );
		func = argv[3];
	} else {
		cout << "usage:: " << argv[0] << " path clockNum [funcname]\n";
		return;
	}

	// Id tickId = GenesisParserWrapper::path2eid( tickName, s );
	Id tickId( tickName );
	if ( tickId.bad() ) {
		cout << "Error:" << argv[0] << ": Invalid clockNumber " <<
				tickName << "\n";
		return;
	}

	string path = argv[1];
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	gpw->useClock( tickId, path, func, s );
}

void GenesisParserWrapper::useClock(
	Id tickId, const string& path, const string& func, Id s )
{
	Element* e = s();

	// Here we use the default form which takes comma-separated lists
	// but may scramble the order.
	// This request elicits a return message to put the list in the
	// elist_ field.

	send2< string, bool >( e, requestWildcardListSlot, path, 0 );

	send3< Id, vector< Id >, string >( 
		s(), useClockSlot, tickId, elist_, func );
}

void do_show( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	gpw->doShow( argc, argv, s );
}

/**
 * This function shows values for all fields of an element. It
 * sends out a request for the FieldList, which comes back as one big
 * string. It parses out the string and requests values for each field
 * in turn to print out.
 */
void GenesisParserWrapper::showAllFields( Id e, Id s )
{
	char temp[80];
	
	// Ask for the list of fields as one big string
	send2< Id, string >( s(), requestFieldSlot, e, "fieldList" );
	vector< string > list;
	vector< string >::iterator i;
	separateString( fieldValue_, list, ", " );
	for ( i = list.begin(); i != list.end(); i++ ) {
		if ( *i == "fieldList" )
			continue;
		fieldValue_ = "";
		send2< Id, string >( s(), requestFieldSlot, e, *i );
		if ( fieldValue_.length() > 0 ) {
			sprintf( temp, "%-25s%s", i->c_str(), "= " );
			print( temp + fieldValue_ );
		}
	}
}

/**
 * Decide if it is a specific field, or all.
 * If specific, get the value for that specific field and print it.
 * If all, first get the list of all fields (which is a field too),
 * then get the value for each specific field in turn.
 * The first arg could be a field, or it could be the object.
 */
void GenesisParserWrapper::doShow( int argc, const char** argv, Id s )
{
	Id e;
	int firstField = 2;
	char temp[80];

	if ( argc < 2 ) {
		print( "Usage: showfield [object/wildcard] [fields] -all" );
		return;
	}


	if ( argc == 2 ) { // show fields of cwe.
		send0( s(), requestCweSlot );
		e = cwe_;
		firstField = 1;
	} else {
		// e = path2eid( argv[1], s );
		e = Id( argv[1] );
		if ( e.bad() ) {
			e = cwe_;
			firstField = 1;
		} else {
			firstField = 2;
		}
	}

	// print( "[ " + eid2path( e ) + " ]" );
	print( "[ " + e.path() + " ]" );

	for ( int i = firstField; i < argc; i++ ) {
		if ( strcmp( argv[i], "*") == 0 ) {
			showAllFields( e, s );
		} else { // get specific field here.
			fieldValue_ = "";
			send2< Id, string >( s(), requestFieldSlot, e, argv[i] );
			if ( fieldValue_.length() > 0 ) {
				sprintf( temp, "%-25s%s", argv[i], "= " );
				print( temp + fieldValue_ );
			} else {
				cout << "'" << argv[i] << "' is not an element or the field of the working element\n";
				return;
			}
		}
	}
}

void do_le( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data() );
	gpw->doLe( argc, argv, s );
}

void GenesisParserWrapper::doLe( int argc, const char** argv, Id s )
{
	if ( argc == 1 ) { // Look in the cwe first.
		send0( s(), requestCweSlot );
		send1< Id >( s(), requestLeSlot, cwe_ );
	} else if ( argc >= 2 ) {
		// Id e = path2eid( argv[1], s );
		Id e( argv[1] );
		/// \todo: Use better test for a bad path than this.
		if ( e.bad() ) {
			print( string( "cannot find object '" ) + argv[1] + "'" );
			return;
		}
		send1< Id >( s(), requestLeSlot, e );
	}
	vector< Id >::iterator i = elist_.begin();
	// This operation should really do it in a parallel-clean way.
	/// \todo If any children, we should suffix the name with a '/'
	for ( i = elist_.begin(); i != elist_.end(); i++ )
		print( ( *i )()->name() );
	elist_.resize( 0 );
}

void do_pwe( int argc, const char** const argv, Id s )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data() );
	gpw->doPwe( argc, argv, s );
}

void GenesisParserWrapper::doPwe( int argc, const char** argv, Id s )
{
	send0( s(), requestCweSlot );
	// Here we need to wait for the shell to service this message
	// request and put the requested value in the local cwe_.
	
	// print( GenesisParserWrapper::eid2path( cwe_ ) );
	print( cwe_.path() );
}

void do_listcommands( int argc, const char** const argv, Id s )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data() );
	gpw->listCommands( );
}

void do_listobjects( int argc, const char** const argv, Id s )
{
	;// s->listClassesFuncLocal( );
}

void do_echo( int argc, const char** const argv, Id s )
{
	// vector< string > vec;
	int options = 0;
	if ( argc > 1 && strncmp( argv[ argc - 1 ], "-n", 2 ) == 0 ) {
		options = 1; // no newline
		argc--;
	}

	string temp = "";
	for (int i = 1; i < argc; i++ ) {
		temp = temp + argv[i];
		if ( i < argc - 1 )
			temp = temp + " ";
	}

	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data() );
	gpw->print( temp, options );

	/*
	for (int i = 1; i < argc; i++ )
		vec.push_back( argv[i] );
		*/

	// s->echoFuncLocal( vec, options );
}

void do_tab2file( int argc, const char** const argv, Id s )
{
	if ( argc < 4 ) {
		cout << argv[0] << ": Too few command arguments\n";
		cout << "usage: " << argv[0] << " file-name element table -table2 table -table3 table -mode mode -nentries n -overwrite\n\n";
		cout << "mode can be y, xy, alpha, beta, tau, minf index. 'y' is default\n";
		return;
	}
	string fname = argv[1];
	string elmname = argv[2];
	string tabname = argv[3];

	// Here we will later have to put in checks for tables on channels.
	// Also lots of options remain.
	// For now we just call the print command on the interpol
	// Id e = GenesisParserWrapper::path2eid( elmname, s );
	Id e( elmname );
	if ( !e.zero() && !e.bad() )
		send3< Id, string, string >( s(),
			setFieldSlot, e, "print", fname );
	else
		cout << "Error: " << argv[0] << ": element not found: " <<
				elmname << endl;
}

/**
 * This returns a space-separated list of individual element paths
 * generated from the wildcard list of the 'path' argument.
 * The path argument can be a comma-separated series of individual
 * element paths, or wildcard paths.
 * There is no guarantee about ordering of the result.
 * There is a guarantee about there being no repeats.
 *
 * The original GENESIS version has additional arguments
 * 	-listname listname'
 * which I have no clue about and have not implemented.
 */
char* do_element_list( int argc, const char** const argv, Id s )
{
	static string space = " ";
	if ( argc != 2 ) {
		cout << "usage:: " << argv[0] << " path\n";
		return copyString( "" );
	}
	string path = argv[1];
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data() );
	string ret;
	gpw->elementList( ret, path, s );

	// return copyString( s->getmsgFuncLocal( field, options ) );
	return copyString( ret.c_str() );
}

void GenesisParserWrapper::elementList(
		string& ret, const string& path, Id s)
{
 	send2< string, bool >( s(), requestWildcardListSlot, path, 0 );
	bool first = 1;
	vector< Id >::iterator i;
	for ( i = elist_.begin(); i != elist_.end(); i++ ) {
		if ( first )
			ret = i->path(); // ret = eid2path( *i );
		else
			ret = ret + " " + i->path();
			// ret = ret + " " + eid2path( *i );
		first = 0;
	}
}

/**
 * This really sets up the scheduler to keep track of all specified
 * objects in the wildcard, including those created after this command
 * was issued. Here we fudge it using the same functionality of
 * useclock.
 */
void do_addtask( int argc, const char** const argv, Id s )
{
	if (argc != 5 ) {
		cout << "usage:: " << argv[0] << " Task path -action action\n";
		return;
	}
	string action;
	if ( strcmp( argv[4], "INIT" ) == 0 )
		action = "init";
	else if ( strcmp( argv[4], "PROCESS" ) == 0 )
		action = "process";
	else {
		cout << "Error:: " << argv[0] << ": action " << argv[4] <<
				" not known\n";
		return;
	}

	const char* nargv[] = { argv[0], argv[2], "0", action.c_str() };
	do_useclock( 4, nargv, s );
	// for useclock the args are:
	// cout << "usage:: " << argv[0] << " path clockNum [funcname]\n";
}

void do_readcell( int argc, const char** const argv, Id s )
{
	if (argc != 3 ) {
		cout << "usage:: " << argv[0] << " filename cellpath\n";
		return;
	}

	string filename = argv[1];
	string cellpath = argv[2];

 	send2< string, string >( s(), readCellSlot, filename, cellpath );
}

Id findChanGateId( int argc, const char** const argv, Id s ) 
{
	assert( argc >= 3 );
	// In MOOSE we only have tabchannels with gates, more like the
	// good old tabgates and vdep_channels. Functionally equivalent,
	// and here we merge the two cases.
	
	string gate = argv[1];
	if ( argv[2][0] == 'X' )
			gate = gate + "/xGate";
	else if ( argv[2][0] == 'Y' )
			gate = gate + "/yGate";
	else if ( argv[2][0] == 'Z' )
			gate = gate + "/zGate";
	// Id gateId = GenesisParserWrapper::path2eid( gate, s );
	Id gateId( gate );
	if ( gateId.bad() ) // Don't give up, it might be a tabgate
		gateId = Id( argv[1] );
		// gateId = GenesisParserWrapper::path2eid( argv[1], s );
	if ( gateId.bad() ) { // Now give up
			cout << "Error: findChanGateId: unable to find channel/gate '" << argv[1] << "/" << argv[2] << endl;
			return gateId;
	}
	return gateId;
}

void setupChanFunc( int argc, const char** const argv, Id s, 
				unsigned int slot )
{
	if (argc < 13 ) {
		cout << "usage:: " << argv[0] << " channel-element gate AA AB AC AD AF BA BB BC BD BF -size n -range min max\n";
		return;
	}

	Id gateId = findChanGateId( argc, argv, s );
	if ( gateId.bad() )
			return;

	vector< double > parms;
	parms.push_back( atof( argv[3] ) ); // AA
	parms.push_back( atof( argv[4] ) ); // AB
	parms.push_back( atof( argv[5] ) ); // AC
	parms.push_back( atof( argv[6] ) ); // AD
	parms.push_back( atof( argv[7] ) ); // AF
	parms.push_back( atof( argv[8] ) ); // BA
	parms.push_back( atof( argv[9] ) ); // BB
	parms.push_back( atof( argv[10] ) ); // BC
	parms.push_back( atof( argv[11] ) ); // BD
	parms.push_back( atof( argv[12] ) ); // BF

	double size = 3000.0;
	double min = -0.100;
	double max = 0.050;
	int nextArg;
	if ( argc >=15 ) {
		if ( strncmp( argv[13], "-s", 2 ) == 0 ) {
			size = atof( argv[14] );
			nextArg = 15;
		} else {
			nextArg = 13;
		}

		if ( strncmp( argv[nextArg], "-r", 2 ) == 0 ) {
			if ( argc == nextArg + 3 ) {
				min = atof( argv[nextArg + 1] );
				max = atof( argv[nextArg + 2] );
			}
		}
	}
	parms.push_back( size );
	parms.push_back( min );
	parms.push_back( max );

 	send2< Id, vector< double > >( s(), slot, gateId, parms );
}

void do_setupalpha( int argc, const char** const argv, Id s )
{
	setupChanFunc( argc, argv, s, setupAlphaSlot );
}

void do_setuptau( int argc, const char** const argv, Id s )
{
	setupChanFunc( argc, argv, s, setupTauSlot );
}

void tweakChanFunc( int argc, const char** const argv, Id s, 
				unsigned int slot )
{
	if (argc < 3 ) {
		cout << "usage:: " << argv[0] << " channel-element gate\n";
		return;
	}

	Id gateId = findChanGateId( argc, argv, s );
	if ( gateId.bad() )
			return;

 	send1< Id >( s(), slot, gateId );
}

void do_tweakalpha( int argc, const char** const argv, Id s )
{
	tweakChanFunc( argc, argv, s, tweakAlphaSlot );
}

void do_tweaktau( int argc, const char** const argv, Id s )
{
	tweakChanFunc( argc, argv, s, tweakTauSlot );
}





/**
 * Equivalent to simdump
 */
void doWriteDump( int argc, const char** const argv, Id s )
{
	if (argc < 3 ) {
		cout << "usage:: " << argv[0] << " file -path path -all\n";
		return;
	}

	bool doAll = 0;
	string filename = argv[1];
	string path = "";
	for ( int i = 2; i < argc; i++ ) {
		if ( strncmp( argv[i], "-p", 2 ) == 0 ) {
			if ( argc > i + 1 ) {
				i++;
				if ( path.length() == 0 )
					path = argv[ i ];
				else
					path = path + "," + argv[ i ];
			}
		}
		if ( strncmp( argv[i], "-a", 2 ) == 0 ) {
			doAll = 1;
		}
	}
	if ( doAll )
	path = "/##";
	send2< string, string >( s(), writeDumpFileSlot, filename, path );
}

/**
 * The simdump command. Here we just munge all args into one big
 * string to send to the shell
 */
void doSimUndump( int argc, const char** const argv, Id s )
{
	if (argc < 4 ) {
		cout << "usage:: " << argv[0] << " object element ... x y z";
		return;
	}

	string args = "";
	for ( int i = 0; i < argc; i++ ) {
		if ( args.length() == 0 ) {
			args = argv[ i ];
		} else {
			string temp = argv[i];
			if ( temp.find( " " ) == string::npos )
				args = args + " " + temp;
			else 
				args = args + " \"" + temp + "\"";
		}
	}
	send1< string >( s(), simUndumpSlot, args );
}

void doSimObjDump( int argc, const char** const argv, Id s )
{
	if (argc < 3 ) {
		cout << "usage:: " << argv[0] << " object  -coords -default -noDUMP";
		return;
	}
	bool coordsFlag = 0;
	bool defaultFlag = 0;
	bool noDumpFlag = 0;

	string args = argv[0];
	for ( int i = 1; i < argc; i++ ) {
		if ( argv[i][0] == '-' ) {
			if ( argv[i][1] == 'c' )
				coordsFlag = 1;
			if ( argv[i][1] == 'd' )
				defaultFlag = 1;
			if ( argv[i][1] == 'n' )
				noDumpFlag = 1;
			continue;
		}
		args = args + " " + argv[ i ];
	}
	if ( coordsFlag ) {
		args = args + " x y z";
	}
	// At this point we don't handle default and noDump
	send1< string >( s(), simObjDumpSlot, args );
}

/**
 * New function, not present in GENESIS. Used when we want to treat
 * the dumpfile as an isolated entity rather than as just another
 * script file. Think of it as a counterpart of readcell.
 */
void doReadDump( int argc, const char** const argv, Id s )
{
	if (argc != 2 ) {
		cout << "usage:: " << argv[0] << " file\n";
		return;
	}

	string filename = argv[1];

	send1< string >( s(), readDumpFileSlot, filename );
}









float do_exp( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
; //		s->error( "exp: Too few command arguments\nusage: exp number" );
		return 0.0;
	} else {	
		return exp( atof( argv[ 1 ] ) );
	}
}

float do_log( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
; //		s->error( "log: Too few command arguments\nusage: log number" );
		return 0.0;
	} else {	
		return log( atof( argv[ 1 ] ) );
	}
}

float do_log10( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
; // 		s->error( "log10: Too few command arguments\nusage: log10 number" );
		return 0.0;
	} else {	
		return log10( atof( argv[ 1 ] ) );
	}
}

float do_sin( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
		; // s->error( "sin: Too few command arguments\nusage: sin number" );
		return 0.0;
	} else {	
		return sin( atof( argv[ 1 ] ) );
	}
}

float do_cos( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
		// s->error( "cos: Too few command arguments\nusage: cos number" );
		return 0.0;
	} else {	
		return cos( atof( argv[ 1 ] ) );
	}
}

float do_tan( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
		// s->error( "tan: Too few command arguments\nusage: tan number" );
		return 0.0;
	} else {	
		return tan( atof( argv[ 1 ] ) );
	}
}

float do_sqrt( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
		// s->error( "sqrt: Too few command arguments\nusage: sqrt number" );
		return 0.0;
	} else {	
		return sqrt( atof( argv[ 1 ] ) );
	}
}

float do_pow( int argc, const char** const argv, Id s )
{
	if ( argc != 3 ) {
		// s->error( "pow: Too few command arguments\nusage: exp base exponent" );
		return 0.0;
	} else {	
		return pow( atof( argv[ 1 ] ), atof( argv[ 2 ] ) );
	}
}

float do_abs( int argc, const char** const argv, Id s )
{
	if ( argc != 2 ) {
		// s->error( "abs: Too few command arguments\nusage: abs number" );
		return 0.0;
	} else {	
		return fabs( atof( argv[ 1 ] ) );
	}
}

char** doArgv( int argc, const char** const argv, Id s )
{
	return 0;
}

int doArgc( int argc, const char** const argv, Id s )
{
	return 0;
}






// Old GENESIS Usage: addfield [element] field-name -indirect element field -description text
// Here we'll have a subset of it:
// addfield [element] field-name -type field_type
void do_addfield( int argc, const char** const argv, Id s )
{
	if ( argc == 2 ) {
		// const char * nargv[] = { argv[0], ".", argv[1] };
//		s->commandFuncLocal( 3, nargv );
	} else if ( argc == 3 ) {
		// s->commandFuncLocal( argc, argv );
	} else if ( argc == 4 && strncmp( argv[2], "-f", 2 ) == 0 ) {
		// const char * nargv[] = { argv[0], ".", argv[1], argv[3] };
//		s->commandFuncLocal( 4, nargv );
	} else if ( argc == 5 && strncmp( argv[3], "-f", 2 ) == 0 ) {
		// const char * nargv[] = { argv[0], argv[1], argv[3], argv[4] };
//		s->commandFuncLocal( 4, nargv );
//	} else {
		; // s->error( "usage: addfield [element] field-name -type field_type" );
	}
}

void doShellCommand ( int argc, const char** const argv, Id s )
{
; //	s->commandFuncLocal( argc, argv );
}

void do_complete_loading( int argc, const char** const argv, Id s )
{
}

void do_xshow ( int argc, const char** const argv, Id s )
{
	// s->error( "not implemented yet." );
}

void do_xhide ( int argc, const char** const argv, Id s )
{
	// s->error( "not implemented yet." );
}

void do_xshowontop ( int argc, const char** const argv, Id s )
{
	// s->error( "not implemented yet." );
}

void do_xupdate ( int argc, const char** const argv, Id s )
{
	// s->error( "not implemented yet." );
}

void do_xcolorscale ( int argc, const char** const argv, Id s )
{
	// s->error( "not implemented yet." );
}

void do_x1setuphighlight ( int argc, const char** const argv, Id s )
{
	// s->error( "not implemented yet." );
}

void do_xsendevent ( int argc, const char** const argv, Id s )
{
	//s->error( "not implemented yet." );
}

//////////////////////////////////////////////////////////////////
// GenesisParserWrapper load command
//////////////////////////////////////////////////////////////////

void GenesisParserWrapper::loadBuiltinCommands()
{
	AddFunc( "addmsg", do_add, "void" );
	AddFunc( "deletemsg", do_drop, "void" );
	AddFunc( "setfield", do_set, "void" );
	AddFunc( "getfield", reinterpret_cast< slifunc >( do_get ), "char*" );
	AddFunc( "getmsg", reinterpret_cast< slifunc >( do_getmsg ), "char*" );
	AddFunc( "showmsg", do_showmsg, "void" );
	AddFunc( "call", do_call, "void" );
	AddFunc( "isa", reinterpret_cast< slifunc >( do_isa ), "int" );
	AddFunc( "exists", reinterpret_cast< slifunc >( do_exists ), "int");
	AddFunc( "showfield", do_show, "void" );
	AddFunc( "create", do_create, "void" );
	AddFunc( "delete", do_delete, "void" );
	AddFunc( "move", do_move, "void" );
	AddFunc( "copy", do_copy, "void" );
	AddFunc( "copy_shallow", do_copy_shallow, "void" );
	AddFunc( "copy_halo", do_copy_halo, "void" );
	AddFunc( "ce", do_ce, "void" );
	AddFunc( "pushe", do_pushe, "void" );
	AddFunc( "pope", do_pope, "void" );
	AddFunc( "addalias", do_alias, "void" );
	AddFunc( "alias", do_alias, "void" );
	AddFunc( "quit", do_quit, "void" );
	AddFunc( "stop", do_stop, "void" );
	AddFunc( "reset", do_reset, "void" );
	AddFunc( "step", do_step, "void" );
	AddFunc( "setclock", do_setclock, "void" );
	AddFunc( "useclock", do_useclock, "void" );
	AddFunc( "showclocks", do_showclocks, "void" );
	AddFunc( "le", do_le, "void" );
	AddFunc( "pwe", do_pwe, "void" );
	AddFunc( "listcommands", do_listcommands, "void" );
	AddFunc( "listobjects", do_listobjects, "void" );
	AddFunc( "echo", do_echo, "void" );
	AddFunc( "tab2file", do_tab2file, "void" );
	AddFunc( "el",
			reinterpret_cast< slifunc >( do_element_list ), "char*" );
	AddFunc( "element_list",
			reinterpret_cast< slifunc >( do_element_list ), "char*" );
	AddFunc( "addtask", do_addtask, "void" );

	AddFunc( "readcell", do_readcell, "void" );

	AddFunc( "setupalpha", do_setupalpha, "void" );
	AddFunc( "setup_tabchan", do_setupalpha, "void" );
	AddFunc( "setuptau", do_setuptau, "void" );
	AddFunc( "tweakalpha", do_tweakalpha, "void" );
	AddFunc( "tweaktau", do_tweaktau, "void" );

	AddFunc( "simdump", doWriteDump, "void" );
	AddFunc( "simundump", doSimUndump, "void" );
	AddFunc( "simobjdump", doSimObjDump, "void" );
	AddFunc( "readdump", doReadDump, "void" ); // New command: not in GENESIS

	AddFunc( "argv", 
		reinterpret_cast< slifunc >( doArgv ), "char**" );
	AddFunc( "argc", 
		reinterpret_cast< slifunc >( doArgc ), "int" );

	AddFunc( "loadtab", doShellCommand, "void" );
	AddFunc( "addfield", do_addfield, "void" );
	AddFunc( "complete_loading", do_complete_loading, "void" );
	AddFunc( "exp", reinterpret_cast< slifunc>( do_exp ), "float" );
	AddFunc( "log", reinterpret_cast< slifunc>( do_log ), "float" );
	AddFunc( "log0", reinterpret_cast< slifunc>( do_log10 ), "float" );
	AddFunc( "sin", reinterpret_cast< slifunc>( do_sin ), "float" );
	AddFunc( "cos", reinterpret_cast< slifunc>( do_cos ), "float" );
	AddFunc( "tan", reinterpret_cast< slifunc>( do_tan ), "float" );
	AddFunc( "sqrt", reinterpret_cast< slifunc>( do_sqrt ), "float" );
	AddFunc( "pow", reinterpret_cast< slifunc>( do_pow ), "float" );
	AddFunc( "abs", reinterpret_cast< slifunc>( do_abs ), "float" );
	AddFunc( "xshow", do_xshow, "void" );
	AddFunc( "xhide", do_xhide, "void" );
	AddFunc( "xshowontop", do_xshowontop, "void" );
	AddFunc( "xupdate", do_xupdate, "void" );
	AddFunc( "xcolorscale", do_xcolorscale, "void" );
	AddFunc( "x1setuphighlight", do_x1setuphighlight, "void" );
	AddFunc( "xsendevent", do_xsendevent, "void" );
}

//////////////////////////////////////////////////////////////////
// GenesisParserWrapper Field commands
//////////////////////////////////////////////////////////////////

/**
 * Looks up the Id of the object specified by the string s.
 * In most cases this call does not need to know about the 
 * GenesisParserWrapper element g, but if it refers to the current
 * working element of the shell then it does need g.
 */
/*
Id GenesisParserWrapper::path2eid( const string& path, Id g )
{
	Element* e = g();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
		( e->data() );
	return gpw->innerPath2eid( path, g );
}
*/

/*
Id GenesisParserWrapper::innerPath2eid( const string& path, Id g )
{
	static string separator = "/";

	if ( path == separator || path == separator + "root" )
			return Id();

	if ( path == "" || path == "." ) {
		send0( g(), requestCweSlot );
		return cwe_;
	}

	if ( path == "^" )
		return createdElm_;

	if ( path == ".." ) {
		send0( g(), requestCweSlot );
		if ( cwe_ == Id() )
			return Id();
		return Shell::parent( cwe_ );
	}

	vector< string > names;

	Id start;
	if ( path.substr( 0, separator.length() ) == separator ) {
		start = Id();
		separateString( path.substr( separator.length() ), names, separator );
	} else if ( path.substr( 0, 5 ) == "/root" ) {
		start = Id();
		separateString( path.substr( 5 ), names, separator );
	} else {
		send0( g(), requestCweSlot );
		start = cwe_;
		separateString( path, names, separator );
	}
	Id ret = Shell::traversePath( start, names );
	// if ( ret == BAD_ID ) print( string( "cannot find object '" ) + path + "'" );
	return ret;
}
*/

/*
 * Should really refer to the shell for this in case we need to do
 * node traversal.
 */
/*
static Id parent( Id e )
{
	Element* elm = e();
	Id ret;
	
	// Check if eid is on local node, otherwise go to remote node
	if ( get< unsigned int >( elm, "parent", ret ) )
		return ret;
	return 0;
}
*/

/*
string GenesisParserWrapper::eid2path( Id eid ) 
{
	static const string slash = "/";
	string n( "" );

	if ( eid == Id() )
		return "/";

	while ( eid != Id() ) {
		n = slash + eid()->name() + n;
		eid = parent( eid );
	}
	return n;
}
*/

/**
 * Looks up the shell attached to the parser specified by g.
 * Static func.
 */
Element* GenesisParserWrapper::getShell( Id g )
{
	/// \todo: shift connDestBegin function to base Element class.
	SimpleElement* e = dynamic_cast< SimpleElement *>( g() );
	assert ( e != 0 && e != Element::root() );
	vector< Conn >::const_iterator i = e->connDestBegin( 3 );
	Element* ret = i->targetElement();
	return ret;
}

//////////////////////////////////////////////////////////////////
// Utility function for creating a GenesisParserWrapper, and 
// connecting it up to the shell.
//////////////////////////////////////////////////////////////////

/**
 * This function is called from main() if there is a genesis parser.
 * It passes in the initial string issued to the program, which
 * the Genesis parser treats as a file argument for loading.
 * Then the parser goes to its infinite loop using the Process call.
 */
Element* makeGenesisParser()
{
	// set< string, string >( Element::root(), "create", "Shell", "shell");
	// Element* shell = Element::lastElement();
	Id shellId = Id::shellId();
	/*
	lookupGet< Id, string >( Element::root(), "lookupChild",
		shellId, "shell" );
		*/
	assert( !shellId.bad() );
	Element* shell = shellId();
	Element* sli = Neutral::create( "GenesisParser", "sli", shell );
	// set< string, string >( shell, "create", "GenesisParser", "sli");
	// Element* sli = Element::lastElement();
	static_cast< GenesisParserWrapper* >( sli->data() )->
		setElement( sli->id() );

	unsigned int ret = shell->findFinfo( "parser" )->add( shell, sli, 
		sli->findFinfo( "parser" ) );
	assert( ret );

#ifdef DO_UNIT_TESTS
	static_cast< GenesisParserWrapper* >( sli->data() )->unitTest();
#endif

	return sli;

	// The original infinite event loop. Now shifted out to the 'main'
	// set( sli, "process" );
}

//////////////////////////////////////////////////////////////////
// GenesisParserWrapper unit tests
//////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
void GenesisParserWrapper::gpAssert(
		const string& command, const string& ret )
{
	testFlag_ = 1;
	printbuf_ = "";
	ParseInput( command );
	if ( ! (printbuf_ == ret) )
		cout << "printbuf_ = '" << printbuf_ << "', ret = '" << ret <<
			"'\n" << flush;
	assert( printbuf_ == ret );
	testFlag_ = 0;
	cout << ".";
}

void GenesisParserWrapper::unitTest()
{
#ifdef USE_MPI
	string lestr = "shell postmasters sched library proto solvers ";
#else
	string lestr = "shell sched library proto solvers ";
#endif
	cout << "\nDoing GenesisParserWrapper tests";
	gpAssert( "le", lestr );
	gpAssert( "create neutral /foo", "" );
	gpAssert( "le", lestr + "foo " );
	gpAssert( "ce /foo", "" );
	gpAssert( "le", "" );
	gpAssert( "pwe", "/foo " );
	gpAssert( "create neutral ./bar", "" );
	gpAssert( "le", "bar " );
	gpAssert( "le /foo", "bar " );
	gpAssert( "le ..", lestr + "foo " );
	gpAssert( "ce bar", "" );
	gpAssert( "pwe", "/foo/bar " );
	gpAssert( "ce ../..", "" );
	// gpAssert( "le", "sched shell foo " );
	gpAssert( "le", lestr + "foo " );
	gpAssert( "delete /foo", "" );
	gpAssert( "le", lestr );
	gpAssert( "le /foo", "cannot find object '/foo' " );
	gpAssert( "echo foo", "foo " );
	gpAssert( "echo bar -n", "bar " );
	gpAssert( "echo {2 + 3}", "5 " );
	gpAssert( "echo {sqrt { 13 - 4 }}", "3 " );
	gpAssert( "echo {sin 1.5 }", "0.997495 " );
	gpAssert( "echo {log 3 }", "1.09861 " );
	gpAssert( "create compartment /compt", "" );
	gpAssert( "echo {getfield /compt Vm}", "-0.06 " );
	gpAssert( "setfield /compt Vm 1.234", "" );
	gpAssert( "echo {getfield /compt Vm}", "1.234 " );
	gpAssert( "setfield /compt Cm 3.1415", "" );
	gpAssert( "echo {getfield /compt Cm}", "3.1415 " );
	gpAssert( "ce /compt", "" );
	gpAssert( "echo {getfield Cm}", "3.1415 " );
	gpAssert( "echo {getfield Vm}", "1.234 " );
	gpAssert( "setfield Rm 0.1", "" );
	gpAssert( "echo {getfield Rm}", "0.1 " );
	gpAssert( "ce /", "" );
	gpAssert( "showfield /compt Vm",
					"[ /compt ] Vm                       = 1.234 " );
	gpAssert( "showfield compt Em Cm Rm",
					"[ /compt ] Em                       = -0.06 Cm                       = 3.1415 Rm                       = 0.1 " );
	gpAssert( "alias shf showfield", "" );
	gpAssert( "shf /compt Em",
					"[ /compt ] Em                       = -0.06 " );
	gpAssert( "alias gf getfield", "" );
	gpAssert( "alias", "gf\tgetfield shf\tshowfield " );
	gpAssert( "alias gf", "getfield " );
	gpAssert( "le /sched/cj", "t0 t1 t2 t3 t4 t5 " );
//	Autoscheduling causes solver to spawn here
//	gpAssert( "setclock 1 0.1", "" );
	gpAssert( "le /sched/cj", "t0 t1 t2 t3 t4 t5 " );
	gpAssert( "echo {getfield /sched/cj/t0 dt}", "1 " );
//	Autoscheduling interferes with these tests
//	gpAssert( "echo {getfield /sched/cj/t1 dt}", "0.1 " );
//	gpAssert( "useclock /##[TYPE=Compartment] 1", "" );
	gpAssert( "delete /compt", "" );
	gpAssert( "le", lestr );

	cout << "\n";
}
#endif
