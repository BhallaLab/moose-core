/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include <math.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <climits>

#include <setjmp.h>
#include <FlexLexer.h>

#include "basecode/moose.h"
#include "element/Neutral.h"
#include "shell/Shell.h"

#include "script.h"
#include "GenesisParser.h"
#include "func_externs.h"
#include "GenesisParserWrapper.h"

using namespace std;

extern string trim(const string&);
extern const string& getClassDoc(const string& className, const string& fieldName);
extern const string& getCommandDoc(const string& command);
extern void print_help(const string&);

/*
static const char nullChar( '\0' );
static const char* nullCharPtr = &nullChar;
*/

const Cinfo* initGenesisParserCinfo()
{
	static Finfo* parserShared[] =
	{
		new SrcFinfo( "cwe", Ftype1< Id >::global(),
			"Setting cwe" ),
		new SrcFinfo( "trigCwe", Ftype0::global(),
			"Getting cwe back: First trigger a request" ),
		new DestFinfo( "recvCwe", Ftype1< Id >::global(),
					RFCAST( &GenesisParserWrapper::recvCwe ),
					"Then receive the cwe info" ),
		new SrcFinfo( "pushe", Ftype1< Id >::global(),
			"Setting pushe. This returns with the new cwe." ),
		new SrcFinfo( "pope", Ftype0::global(),
			"Doing pope. This returns with the new cwe." ),
		new SrcFinfo( "trigLe", Ftype1< Id >::global(),
			"Getting a list of child ids: First send a request with the requested parent elm id." ),
		new DestFinfo( "recvElist", 
					Ftype1< vector< Id > >::global(), 
					RFCAST( &GenesisParserWrapper::recvElist ),
					"Then recv the vector of child ids. This function is shared by several other messages as all "
					"it does is dump the elist into a temporary local buffer." ),
		///////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions.
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "create",
				Ftype4< string, string, int, Id >::global(),
				"Creating an object: Send out the request. "
				"args: type, name, node, parent" ),
		new SrcFinfo( "createArray",
				Ftype4< string, string, Id, vector <double> >::global(),
				"Creating an object: Recv the returned object id." ),
		new SrcFinfo( "planarconnect", Ftype3< string, string, double >::global() ),
		new SrcFinfo( "planardelay", Ftype3< string, string, vector <double> >::global() ),
		new SrcFinfo( "planarweight", Ftype3< string, string, vector<double> >::global() ),
		new SrcFinfo( "getSynCount", Ftype1< Id >::global() ),
		new DestFinfo( "recvCreate",
					Ftype1< Id >::global(),
					RFCAST( &GenesisParserWrapper::recvCreate ) ),
		// Deleting an object: Send out the request.
		new SrcFinfo( "delete", Ftype1< Id >::global() ),

		///////////////////////////////////////////////////////////////
		// Value assignment: set and get.
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "addField", Ftype2<Id, string>::global(),
			"Create a dynamic field on the specified object" ),
		new SrcFinfo( "get", Ftype2< Id, string >::global(),
			"Getting a field value as a string: send out request:" ),
		new DestFinfo( "recvField",
					Ftype1< string >::global(),
					RFCAST( &GenesisParserWrapper::recvField ),
					"Getting a field value as a string: Recv the value." ),
		new SrcFinfo( "set", // object, field, value 
				Ftype3< Id, string, string >::global(),
				"Setting a field value as a string: send out request:" ),
		new SrcFinfo( "file2tab", // object, filename, skiplines 
				Ftype3< Id, string, unsigned int >::global() ),

		///////////////////////////////////////////////////////////////
		// Clock control and scheduling
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "setClock", // clockNo, dt, stage
				Ftype3< int, double, int >::global(),
				"Setting values for a clock tick: setClock" ),
		new SrcFinfo( "useClock", // tickname, path, function
				Ftype3< string, string, string >::global(),
				"Assigning path and function to a clock tick: useClock" ),
		new SrcFinfo( "el", Ftype2< string, bool >::global(),
				"Getting a wildcard path of elements: send out request args are path, flag true for "
				"breadth-first list." ),
		new SrcFinfo( "resched", Ftype0::global(),
				"The return function for the wildcard past is the shared function recvElist"), // resched
		new SrcFinfo( "reinit", Ftype0::global() ), // reinit
		new SrcFinfo( "stop", Ftype0::global() ), // stop
		new SrcFinfo( "step", Ftype1< double >::global() ),
		new SrcFinfo( "requestClocks", 
					Ftype0::global(),
					"step, arg is time" ), //request clocks
		new DestFinfo( "recvClocks", 
					Ftype1< vector< double > >::global(), 
					RFCAST( &GenesisParserWrapper::recvClocks ) ),
		new SrcFinfo( "requestCurrentTime", Ftype0::global() ),
		new SrcFinfo( "quit", Ftype0::global(), 
			"Returns time in the default return value." ),
		
		///////////////////////////////////////////////////////////////
		// Message functions
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "addMsg", 
					Ftype4< vector< Id >, string, vector< Id >, string >::global(),
					"Create a message. srcId, srcField, destId, destField" ),
		new SrcFinfo( "deleteMsg", Ftype2< Fid, int >::global(),
					"Delete a message based on number " ),
		new SrcFinfo( "deleteEdge", 
					Ftype4< Id, string, Id, string >::global(),
					"Delete a message based on src id.field and dest id.field "
					"This is how to specify an edge, so call it deleteEdge" ),
		new SrcFinfo( "listMessages", 
					Ftype3< Id, string, bool >::global(),
					"Request message list: id elm, string field, bool isIncoming" ),
		new DestFinfo( "recvMessageList",
					Ftype2< vector < Id >, string >::global(), 
					RFCAST( &GenesisParserWrapper::recvMessageList ),
					"Receive message list and string with remote fields for msgs" ),

		///////////////////////////////////////////////////////////////
		// Object heirarchy manipulation functions.
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "copy", Ftype3< Id, Id, string >::global(),
					"This function is for copying an element tree, complete with messages, onto another." ),
		new SrcFinfo( "copyIntoArray", Ftype4< Id, Id, string, vector <double> >::global() ),
		new SrcFinfo( "move", Ftype3< Id, Id, string >::global(),
					"This function is for moving element trees." ),

		///////////////////////////////////////////////////////////////
		// Cell reader: filename cellpath
		///////////////////////////////////////////////////////////////
		
		new SrcFinfo( "readcell", 
			Ftype4< string, string, vector< double >, int >::global(),
			"filename, cellpath, parms, node" ),

		///////////////////////////////////////////////////////////////
		// Channel setup functions
		///////////////////////////////////////////////////////////////
		
		new SrcFinfo( "setupAlpha", 
					Ftype2< Id, vector< double > >::global(),
					"setupalpha" ),
		new SrcFinfo( "setupTau", 
					Ftype2< Id, vector< double > >::global(),
					"setuptau" ),
		new SrcFinfo( "tweakAlpha", Ftype1< Id >::global(),
					"tweakalpha" ),
		new SrcFinfo( "tweakTau", Ftype1< Id >::global(),
					"tweaktau" ),
		new SrcFinfo( "setupGate", 
					Ftype2< Id, vector< double > >::global() ),

		///////////////////////////////////////////////////////////////
		// SimDump facilities
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readDumpFile", 
					Ftype1< string >::global(),
					"readDumpFile" ),
		new SrcFinfo( "writeDumpFile", 
					Ftype2< string, string >::global(),
					"writeDumpFile" ),
		new SrcFinfo( "simObjDump",
					Ftype1< string >::global(),
					"simObjDump" ),
		new SrcFinfo( "simUndump",
					Ftype1< string >::global(),
					"simundump" ),
		new SrcFinfo( "openfile", 
			Ftype2< string, string >::global() ),
		new SrcFinfo( "writefile", 
			Ftype2< string, string >::global() ),
		new SrcFinfo( "flushfile", 
			Ftype1< string >::global() ),
		new SrcFinfo( "listfiles", 
			Ftype0::global() ),
		new SrcFinfo( "closefile", 
			Ftype1< string >::global() ),
		new SrcFinfo( "readfile", 
			Ftype2< string, bool >::global() ),
		///////////////////////////////////////////////////////////////
		// Setting field values for a vector of objects
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "setVecField", // object, field, value 
			Ftype3< vector< Id >, string, string >::global(),
			"Setting a vec of field values as a string: send out request:" ),
		new SrcFinfo( "loadtab", 
			Ftype1< string >::global() ),
		new SrcFinfo( "tabop", 
			Ftype4< Id, char, double, double >::global() ),
		///////////////////////////////////////////////////////////////
		// SBML
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readsbml", 
			Ftype3< string, string, int >::global() ),
		new SrcFinfo( "writesbml", 
			Ftype3< string, string, int >::global() ),

		///////////////////////////////////////////////////////////////
		// NeuroML
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "readneuroml", 
			Ftype3< string, string, int >::global() ),
		new SrcFinfo( "writeneuroml", 
			Ftype3< string, string, int >::global() ),
		///////////////////////////////////////////////////////////////
		// Misc
		///////////////////////////////////////////////////////////////
		new SrcFinfo( "createGate", 
			Ftype2< Id, string >::global(),
			"Args: HHGate id, Interpol A id, Interpol B id. "
			"Request an HHGate explicitly to create Interpols, with the given "
			"ids. This is used when the gate is a global object, and so the "
			"interpols need to be globals too. Comes in use in TABCREATE in the "
			"parallel context." ),
	};
	
	static Finfo* genesisParserFinfos[] =
	{
		new SharedFinfo( "parser", parserShared,
				sizeof( parserShared ) / sizeof( Finfo* ),
				"This is a shared message to talk to the Shell." ),
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
	static string doc[] =
	{
		"Name", "GenesisParser",
		"Author", "Upinder S. Bhalla, NCBS, 2004-2007",
		"Description", "Object to handle the old Genesis parser",
	};

	static Cinfo genesisParserCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initNeutralCinfo(),
		genesisParserFinfos,
		sizeof(genesisParserFinfos) / sizeof( Finfo* ),
		ValueFtype1< GenesisParserWrapper >::global()
	);

	return &genesisParserCinfo;
}

static const Cinfo* genesisParserCinfo = initGenesisParserCinfo();
static const Slot setCweSlot = 
	initGenesisParserCinfo()->getSlot( "parser.cwe" );
static const Slot requestCweSlot = 
	initGenesisParserCinfo()->getSlot( "parser.trigCwe" );
static const Slot requestLeSlot = 
	initGenesisParserCinfo()->getSlot( "parser.trigLe" );
static const Slot pusheSlot = 
	initGenesisParserCinfo()->getSlot( "parser.pushe" );
static const Slot popeSlot = 
	initGenesisParserCinfo()->getSlot( "parser.pope" );
static const Slot createSlot = 
	initGenesisParserCinfo()->getSlot( "parser.create" );
static const Slot createArraySlot = 
	initGenesisParserCinfo()->getSlot( "parser.createArray" );
static const Slot planarconnectSlot = 
	initGenesisParserCinfo()->getSlot( "parser.planarconnect" );
static const Slot planardelaySlot = 
	initGenesisParserCinfo()->getSlot( "parser.planardelay" );
static const Slot planarweightSlot = 
	initGenesisParserCinfo()->getSlot( "parser.planarweight" );
static const Slot getSynCountSlot = 
	initGenesisParserCinfo()->getSlot( "parser.getSynCount" );
static const Slot deleteSlot = 
	initGenesisParserCinfo()->getSlot( "parser.delete" );
static const Slot addfieldSlot = 
	initGenesisParserCinfo()->getSlot( "parser.addField" );
static const Slot requestFieldSlot = 
	initGenesisParserCinfo()->getSlot( "parser.get" );
static const Slot setFieldSlot = 
	initGenesisParserCinfo()->getSlot( "parser.set" );
static const Slot file2tabSlot = 
	initGenesisParserCinfo()->getSlot( "parser.file2tab" );
static const Slot setClockSlot = 
	initGenesisParserCinfo()->getSlot( "parser.setClock" );
static const Slot useClockSlot = 
	initGenesisParserCinfo()->getSlot( "parser.useClock" );
static const Slot requestWildcardListSlot = 
	initGenesisParserCinfo()->getSlot( "parser.el" );
static const Slot reschedSlot = 
	initGenesisParserCinfo()->getSlot( "parser.resched" );
static const Slot reinitSlot = 
	initGenesisParserCinfo()->getSlot( "parser.reinit" );
static const Slot stopSlot = 
	initGenesisParserCinfo()->getSlot( "parser.stop" );
static const Slot stepSlot = 
	initGenesisParserCinfo()->getSlot( "parser.step" );
static const Slot requestClocksSlot = 
	initGenesisParserCinfo()->getSlot( "parser.requestClocks" );
static const Slot requestCurrentTimeSlot = 
	initGenesisParserCinfo()->getSlot( "parser.requestCurrentTime" );
static const Slot quitSlot = 
	initGenesisParserCinfo()->getSlot( "parser.quit" );

static const Slot addMessageSlot = 
	initGenesisParserCinfo()->getSlot( "parser.addMsg" );
static const Slot deleteMessageSlot = 
	initGenesisParserCinfo()->getSlot( "parser.deleteMsg" );
static const Slot deleteEdgeSlot = 
	initGenesisParserCinfo()->getSlot( "parser.deleteEdge" );
static const Slot listMessagesSlot = 
	initGenesisParserCinfo()->getSlot( "parser.listMessages" );

static const Slot copySlot = 
	initGenesisParserCinfo()->getSlot( "parser.copy" );
static const Slot copyIntoArraySlot = 
	initGenesisParserCinfo()->getSlot( "parser.copyIntoArray" );
static const Slot moveSlot = 
	initGenesisParserCinfo()->getSlot( "parser.move" );
static const Slot readCellSlot = 
	initGenesisParserCinfo()->getSlot( "parser.readcell" );

static const Slot setupAlphaSlot = 
	initGenesisParserCinfo()->getSlot( "parser.setupAlpha" );
static const Slot setupTauSlot = 
	initGenesisParserCinfo()->getSlot( "parser.setupTau" );
static const Slot tweakAlphaSlot = 
	initGenesisParserCinfo()->getSlot( "parser.tweakAlpha" );
static const Slot tweakTauSlot = 
	initGenesisParserCinfo()->getSlot( "parser.tweakTau" );
static const Slot setupGateSlot = 
	initGenesisParserCinfo()->getSlot( "parser.setupGate" );

static const Slot readDumpFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.readDumpFile" );
static const Slot writeDumpFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.writeDumpFile" );
static const Slot simObjDumpSlot = 
	initGenesisParserCinfo()->getSlot( "parser.simObjDump" );
static const Slot simUndumpSlot = 
	initGenesisParserCinfo()->getSlot( "parser.simUndump" );

static const Slot openFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.openfile" );
static const Slot writeFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.writefile" );
static const Slot flushFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.flushfile" );
static const Slot listFilesSlot = 
	initGenesisParserCinfo()->getSlot( "parser.listfiles" );
static const Slot closeFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.closefile" );
static const Slot readFileSlot = 
	initGenesisParserCinfo()->getSlot( "parser.readfile" );
static const Slot setVecFieldSlot = 
	initGenesisParserCinfo()->getSlot( "parser.setVecField" );
static const Slot loadtabSlot = 
	initGenesisParserCinfo()->getSlot( "parser.loadtab" );
static const Slot tabopSlot = 
	initGenesisParserCinfo()->getSlot( "parser.tabop" );
static const Slot readSbmlSlot = 
	initGenesisParserCinfo()->getSlot( "parser.readsbml" );
static const Slot writeSbmlSlot = 
	initGenesisParserCinfo()->getSlot( "parser.writesbml" );
static const Slot readNeuromlSlot = 
	initGenesisParserCinfo()->getSlot( "parser.readneuroml" );
static const Slot writeNeuromlSlot = 
	initGenesisParserCinfo()->getSlot( "parser.writeneuroml" );
static const Slot createGateSlot = 
	initGenesisParserCinfo()->getSlot( "parser.createGate" );

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
		cwe_(), createdElm_(),
		testFlag_( 0 )
{
		loadBuiltinCommands();
}

void GenesisParserWrapper::readlineFunc( const Conn* c, string s )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c->data() );

	data->AddInput( s );
}

void GenesisParserWrapper::processFunc( const Conn* c )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c->data() );

	data->Process();
}

void GenesisParserWrapper::parseFunc( const Conn* c, string s )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c->data() );

	data->ParseInput( s );
}

void GenesisParserWrapper::setReturnId( const Conn* c, Id id )
{
	GenesisParserWrapper* data =
	static_cast< GenesisParserWrapper* >( c->data() );

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

void GenesisParserWrapper::recvCwe( const Conn* c, Id cwe )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c->data() );
	gpw->cwe_ = cwe;
}

//
//This is used for Le, for WildcardList, and others
void GenesisParserWrapper::recvElist( const Conn* c, vector< Id > elist)
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c->data() );
	gpw->elist_ = elist;
}

void GenesisParserWrapper::recvCreate( const Conn* c, Id e )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c->data() );
	gpw->createdElm_ = e;
}

void GenesisParserWrapper::recvField( const Conn* c, string value )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c->data() );
	gpw->fieldValue_ = value;
}

void GenesisParserWrapper::recvClocks( 
				const Conn* c, vector< double > dbls)
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c->data() );
	gpw->dbls_ = dbls;
}

void GenesisParserWrapper::recvMessageList( 
				const Conn* c, vector< Id > elist, string s)
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( c->data() );
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

	// This is a bit nasty, as in GENESIS the message is directly
	// from the enzyme, but in MOOSE it is from the child enzComplex
	// molecule. We ignore it for now as it is a corner case.
	src[ "SUMTOTAL nComplex nComplexInit" ] = ""; // for enzymes

	src[ "SLAVE output" ] = "outputSrc";	// for tables
	src[ "SUM output" ] = "outputSrc";	// for tables
	src[ "SLAVE n" ] = "nSrc";	// for direct connections between mols.
	src[ "INTRAMOL n" ] = "nOut"; 	// target is an enzyme.
	src[ "CONSERVE n nInit" ] = ""; 	// Deprecated
	src[ "CONSERVE nComplex nComplexInit" ] = ""; 	// Deprecated

	// Some messages for compartments.
	src[ "AXIAL Vm" ] = "axial";
	src[ "AXIAL previous_state" ] = "axial";
	src[ "RAXIAL Ra Vm" ] = "";
	src[ "RAXIAL Ra previous_state" ] = "";
	src[ "INJECT output" ] = "outputSrc";
	
	// Some messages for channels.
	src[ "VOLTAGE Vm" ] = "";
	src[ "CHANNEL Gk Ek" ] = "channel";
	src[ "SynChan.Mg_block.CHANNEL Gk Ek" ] = "origChannel";
	src[ "MULTGATE m" ] = ""; // Have to handle specially. Don't make msg
	src[ "MULTGATE output" ] = "gate"; 
	src[ "useX.MULTGATE" ] = "gate";
	src[ "useY.MULTGATE" ] = "gate";
	src[ "useZ.MULTGATE" ] = "gate";
	src[ "CONCEN Ca" ] = "concSrc";
        // Messages for Nernst object
        src[ "CIN Ca" ] = "concSrc";
        src[ "COUT Ca" ] = "concSrc";
        src[ "EK E" ] = "ESrc";
	// Message for synchan
	src[ "SpikeGen.SPIKE" ] = "event";
	// Message for CaConcen
	src[ "I_Ca Ik" ] = "IkSrc";
	
	// Messages for RandomSpike
	src[ "RandomSpike.SPIKE"] = "event";
	// Some messages for gates, used in the squid demo. This 
	// is used to set the reset value of Vm in the gates, which is 
	// done already through the existing messaging.
	src[ "EREST Vm" ] = "";
        
	// Messages for PulseGen
	src[ "PulseGen.INPUT output" ] = "outputSrc";
        src[ "PulseGen.INPUT Vm" ] = "outputSrc";
	// Messages for DiffAmp
	src[ "DiffAmp.INPUT output" ] = "outputSrc";
	src[ "PLUS output" ] = "outputSrc";
	src[ "MINUS output" ] = "outputSrc";
        // Messages for Compartment/Vm to DiffAmp/plus or minus
        src[ "PLUS Vm" ] = "VmSrc";
        src[ "MINUS Vm" ] = "VmSrc";
	// For compatibility taking output from RC
	src[ "INPUT state" ] = "state";
	src[ "PLUS state" ] = "outputSrc";
	src[ "MINUS state" ] = "outputSrc";

	// Messages for PID
	src[ "CMD output" ] = "outputSrc";
	src[ "SNS output" ] = "outputSrc";
	src[ "GAIN output" ] = "outputSrc";
	src[ "CMD Vm" ] = "VmSrc";
	src[ "SNS Vm" ] = "VmSrc";
	src[ "GAIN Vm" ] = "VmSrc";
	src[ "CMD state" ] = "outputSrc";
	src[ "SNS state" ] = "outputSrc";
	src[ "GAIN state" ] = "outputSrc";

	// Messages for RC - included in COmpartment's message
	// src[ "RC.INJECT output" ]
	
	// Some messages for tables, specially used for I/O
	src[ "SpikeGen.INPUT Vm" ] = "VmSrc";
	src[ "SpikeGen.INPUT output" ] = "outputSrc";
	src[ "RandomSpike.INPUT Vm" ] = "eventSrc";	
	src[ "INPUT Vm" ] = "Vm";
	src[ "INPUT Im" ] = "Im";
	src[ "INPUT Ca" ] = "Ca";
	src[ "INPUT Ik" ] = "Ik";
	src[ "INPUT Gk" ] = "Gk";
	src[ "INPUT Ek" ] = "Ek";
	src[ "INPUT Gbar" ] = "Gbar";
	src[ "INPUT X" ] = "X";
	src[ "INPUT Y" ] = "Y";
	src[ "INPUT Z" ] = "Z";
	src[ "INPUT n" ] = "n";
	src[ "INPUT Co" ] = "conc";
	src[ "INPUT state" ] = "state";
	src[ "INPUT output" ] = "output";
	src[ "INPUT cmd" ] = "command";
	src[ "INPUT sns"] = "sensed";
	src[ "INPUT e" ] = "error";
	src[ "INPUT e_integral" ] = "integral";
	src[ "INPUT e_deriv" ] = "derivative";
	src[ "INPUT e_previous" ] = "e_previous";
	// Messages for having tables pretend to be an xplot
	src[ "PLOT Co" ] = "conc";
	src[ "PLOT n" ] = "n";
	src[ "PLOT Vm" ] = "Vm";
	src[ "PLOT Im" ] = "Im";
	src[ "PLOT Ca" ] = "Ca";
	src[ "PLOT Ik" ] = "Ik";
	src[ "PLOT Gk" ] = "Gk";
	src[ "PLOT Ek" ] = "Ek";
	src[ "PLOT Gbar" ] = "Gbar";
	src[ "PLOT X" ] = "X";
	src[ "PLOT Y" ] = "Y";
	src[ "PLOT Z" ] = "Z";
	src[ "PLOT n" ] = "n";
	src[ "PLOT Co" ] = "conc";
	src[ "PLOT state" ] = "state";
	src[ "PLOT output" ] = "output";
	src[ "PLOT e" ] = "error";
	src[ "PLOT e_integral" ] = "integral";
	src[ "PLOT e_deriv" ] = "deriv";
	src[ "PLOT e_previous" ] = "e_previous";
	// Messages for doing table operations
	src[ "PRD Gk" ] = "GkSrc";
	
	// Messages for GHK
	src[ "PERMEABILITY Gk" ] = "ghk";	// From HHChannel
	src[ "Cin Ca" ] = "concSrc"; // From CaConc
	// Messages for GHK - to accept values from a table
	src[ "PERMEABILITY output" ] = "outputSrc";
	
	src[ "SAVE Ik" ] = "Ik"; // Use with AscFile
	src[ "SAVE C" ] = "Ca";

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

	// This is a bit nasty, as in GENESIS the message is directly
	// from the enzyme, but in MOOSE it is from the child enzComplex
	// molecule. We ignore it for now as it is a corner case.
	dest[ "SUMTOTAL nComplex nComplexInit" ] = "";	// for enzymes

	dest[ "SLAVE output" ] = "sumTotal";	// for molecules
	dest[ "SUM output" ] = "sum";	// for tables
	dest[ "SLAVE n" ] = "sumTotal";	// for molecules
	dest[ "INTRAMOL n" ] = "intramolIn"; 	// target is an enzyme.
	dest[ "CONSERVE n nInit" ] = ""; 	// Deprecated
	dest[ "CONSERVE nComplex nComplexInit" ] = ""; 	// Deprecated

	// Some messages for compartments.
	dest[ "AXIAL Vm" ] = "raxial";
	dest[ "AXIAL previous_state" ] = "raxial";
	dest[ "RAXIAL Ra Vm" ] = "";//"axial";
	dest[ "RAXIAL Ra previous_state" ] = ""; //"axial";
	dest[ "INJECT output" ] = "injectMsg";
	
	// Some messages for channels.
	dest[ "VOLTAGE Vm" ] = "";
	dest[ "CHANNEL Gk Ek" ] = "channel";

	// Special messages for spikegen and synapse
	dest[ "SpikeGen.SPIKE" ] = "synapse";
	dest[ "SpikeGen.INPUT Vm" ] = "Vm";
	dest[ "SpikeGen.INPUT output" ] = "Vm";
	// Messages for RandomSpike
	dest[ "RandomSpike.SPIKE" ] = "synapse";
	
	// Some of these funny comparisons are inserted when the code finds
	// cases which need special work.
	dest[ "SynChan.Mg_block.CHANNEL Gk Ek" ] = "origChannel";
	dest[ "MULTGATE m" ] = "";		// This needs to be handled specially,
	// so block the use of this message.
	dest[ "useX.MULTGATE" ] = "xGate";
	dest[ "useY.MULTGATE" ] = "yGate";
	dest[ "useZ.MULTGATE" ] = "zGate";
	dest[ "MULTGATE output" ] = "zGate";	// Rare use case from table.
	dest[ "CONCEN Ca" ] = "concen";
	// for CaConc object
	dest[ "I_Ca Ik" ] = "current";

        // Messages for Nernst object
        dest[ "CIN Ca" ] = "CinMsg";
        dest[ "COUT Ca" ] = "CoutMsg";
        dest[ "EK E" ] = "EkDest"; // from Nernst to HHChannel
	// Some messages for gates, used in the squid demo. This 
	// is used to set the reset value of Vm in the gates, which is 
	// done already through the existing messaging.
	dest[ "EREST Vm" ] = "";

	
	// Messages for PulseGen
	dest[ "PulseGen.INPUT output" ] = "input";

	// Messages for DiffAmp
	dest[ "PLUS output" ] = "plusDest";
	dest[ "MINUS output" ] = "minusDest";
	dest[ "GAIN output" ] = "gainDest";
        // Compartment/Vm -> DiffAmp/plus or minus
	dest[ "PLUS Vm" ] = "plusDest";
	dest[ "MINUS Vm" ] = "minusDest";
	// These are to take output from RC
	dest[ "PLUS state" ] = "plusDest";
	dest[ "MINUS state" ] = "minusDest";
	dest[ "GAIN state" ] = "gainDest";
	
	// Messages for PIDController
	dest[ "CMD output" ] = "commandDest";
	dest[ "SNS output" ] = "sensedDest";
	dest[ "GAIN output" ] = "gainDest";
	dest[ "SNS Vm" ] = "sensedDest";
	dest[ "CMD Vm" ] = "commandDest";
	dest[ "GAIN Vm" ] = "gainDest";
	dest[ "CMD state" ] = "commandDest";
	dest[ "SNS state" ] = "sensedDest";
	dest[ "CMD state" ] = "commandDest";
	dest[ "SNS state" ] = "sensedDest";
	dest[ "GAIN state" ] = "gainDest";
	//	dest[ "GAIN state" ] = "gainDest"; // already in DiffAmp

	// Message for RC - already included in Compartment
	// But the nomenclature violates the general scheme
	// dest[ "INJECT output" ] = "injectMsg";
	
	// Some messages for tables
	dest[ "INPUT Vm" ] = "inputRequest";
	dest[ "INPUT Im" ] = "inputRequest";
	dest[ "INPUT Ca" ] = "inputRequest";
	dest[ "INPUT Ik" ] = "inputRequest";
	dest[ "INPUT Gk" ] = "inputRequest";
	dest[ "INPUT Ek" ] = "inputRequest";
	dest[ "INPUT Gbar" ] = "inputRequest";
	dest[ "INPUT X" ] = "inputRequest";
	dest[ "INPUT Y" ] = "inputRequest";
	dest[ "INPUT Z" ] = "inputRequest";
	dest[ "INPUT n" ] = "inputRequest";
	dest[ "INPUT Co" ] = "inputRequest";
	dest[ "INPUT output" ] = "inputRequest";
	dest[ "INPUT state" ] = "inputRequest";
	dest[ "INPUT cmd" ] = "inputRequest";
	dest[ "INPUT sns"] = "inputRequest";
	dest[ "INPUT e" ] = "inputRequest";
	dest[ "INPUT e_integral" ] = "inputRequest";
	dest[ "INPUT e_deriv" ] = "inputRequest";
	dest[ "INPUT e_previous" ] = "inputRequest";

	// Messages for having tables pretend to be an xplot
	dest[ "PLOT Vm" ] = "inputRequest";
	dest[ "PLOT Im" ] = "inputRequest";
	dest[ "PLOT Ca" ] = "inputRequest";
	dest[ "PLOT Ik" ] = "inputRequest";
	dest[ "PLOT Gk" ] = "inputRequest";
	dest[ "PLOT Ek" ] = "PLOTRequest";
	dest[ "PLOT Gbar" ] = "PLOTRequest";
	dest[ "PLOT X" ] = "PLOTRequest";
	dest[ "PLOT Y" ] = "PLOTRequest";
	dest[ "PLOT Z" ] = "PLOTRequest";
	dest[ "PLOT n" ] = "inputRequest";
	dest[ "PLOT Co" ] = "inputRequest";
	dest[ "PLOT output" ] = "inputRequest";
	dest[ "PLOT state" ] = "inputRequest";
	dest[ "PLOT e" ] = "inputRequest";
	dest[ "PLOT e_integral" ] = "inputRequest";
	dest[ "PLOT e_deriv" ] = "inputRequest";
	dest[ "PLOT e_previous" ] = "inputRequest";
	// Messages for doing table operations
	dest[ "PRD Gk" ] = "prd";

	// Messages for GHK
	dest[ "PERMEABILITY Gk" ] = "ghk"; // From HHChannel
	dest[ "Cin Ca" ] = "CinDest"; // From CaConc
	// Messages for GHK - to accept values from a table
	dest[ "PERMEABILITY output" ] = "ghk";

	dest[ "SAVE Ik" ] = "save"; // AscFile
	dest[ "SAVE C" ] = "save";

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
	classnames[ "symcompartment" ] = "SymCompartment";
	classnames[ "leakage" ] = "Leakage";
	classnames[ "hh_channel" ] = "HHChannel";
        classnames[ "nernst" ] = "Nernst";
	classnames[ "tabchannel" ] = "HHChannel";
	classnames[ "tab2Dchannel" ] = "HHChannel2D";
	classnames[ "vdep_channel" ] = "HHChannel";
	classnames[ "vdep_gate" ] = "HHGate";
	classnames[ "tabgate" ] = "HHGate";
	classnames[ "randomspike" ] = "RandomSpike";
	classnames[ "spikegen" ] = "SpikeGen";
	classnames[ "pulsegen" ] = "PulseGen";
	classnames[ "diffamp" ] = "DiffAmp";
	classnames[ "PID" ] = "PIDController";
	classnames[ "synchan" ] = "SynChan";
	classnames[ "table" ] = "Table";
	classnames[ "asc_file" ] = "AscFile";
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
	classnames[ "ghk" ] = "GHK";
        classnames[ "efield" ] = "Efield";
	return classnames;
}


map< string, string >& sliFieldNameConvert()
{
	static map< string, string > fieldnames;

	if ( fieldnames.size() > 0 )
		return fieldnames;
	
	fieldnames["Molecule.Co"] = "conc";
	fieldnames["Molecule.CoInit"] = "concInit";
	fieldnames["Molecule.vol"] = "volumeScale";
	fieldnames["SpikeGen.thresh"] = "threshold";
	fieldnames["SpikeGen.output_amp"] = "amplitude";
	fieldnames["Table.table->dx"] = "dx";
	fieldnames["Table.table->invdx"] = "";
	fieldnames["Table.table->xmin"] = "xmin";
	fieldnames["Table.table->xmax"] = "xmax";
	fieldnames["Table.table->table"] = "table";
	fieldnames["Table.table->xdivs"] = "xdivs";
	fieldnames["Compartment.dia"] = "diameter";
	fieldnames["Compartment.len"] = "length";
	fieldnames["SymCompartment.dia"] = "diameter";
	fieldnames["SymCompartment.len"] = "length";
	fieldnames["SynChan.gmax"] = "Gbar";
	fieldnames["HHChannel.gbar"] = "Gbar";
	fieldnames["HHChannel.Z_conc"] = "useConcentration";
        fieldnames["Nernst.valency"] = "valence";
        fieldnames["Nernst.T"] = "Temperature";
	fieldnames["PulseGen.level1"] = "firstLevel";
	fieldnames["PulseGen.width1"] = "firstWidth";
	fieldnames["PulseGen.delay1"] = "firstDelay";	
	fieldnames["PulseGen.level2"] = "secondLevel";
	fieldnames["PulseGen.width2"] = "secondWidth";
	fieldnames["PulseGen.delay2"] = "secondDelay";
	fieldnames["PulseGen.baselevel"] = "baseLevel";
	fieldnames["PulseGen.trig_time"] = "trigTime";
	fieldnames["PulseGen.trig_mode"] = "trigMode";
	fieldnames["PulseGen.previous_input"] = "prevInput";
	fieldnames["RandomSpike.min_amp"] = "minAmp";
	fieldnames["RandomSpike.max_amp"] = "maxAmp";
	fieldnames["RandomSpike.reset_value"] = "resetValue";
	fieldnames["RandomSpike.abs_refract"] = "absRefract";
	fieldnames["PIDController.cmd"] = "command";
	fieldnames["PIDController.sns"] = "sensed";
	fieldnames["PIDController.tau_i"] = "tauI";
	fieldnames["PIDController.tau_d"] = "tauD";
	fieldnames["PIDController.e"] = "error";
	fieldnames["PIDController.e_integral"] = "integral";
	fieldnames["PIDController.e_deriv"] = "derivative";
        fieldnames["Efield.field"] = "potential";
	return fieldnames;
}



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
			( e->data( 0 ) );
	return gpw->doAdd( argc, argv, s );
}

bool GenesisParserWrapper::innerAdd( Id s,
	Id src, const string& srcF, Id dest, const string& destF )
{
	vector< Id > srcList( 1, src );
	vector< Id > destList( 1, dest );
	if ( !src.bad() && !dest.bad() ) {
		send4< vector< Id >, string, vector< Id >, string >( s(), addMessageSlot,
			srcList, srcF, destList, destF );
		return 1;
	}
	return 0;
}

bool hasCa( string s ) {
	return ( s.find( "ca" ) != string::npos || 
		s.find( "Ca" ) != string::npos ||  
		s.find( "CA" ) != string::npos );
}


string GenesisParserWrapper::handleMultGate( 
	int argc, const char** const argv, Id s,
	string& gate, double& gatePower )
{
	// The MULTGATE message has the crucial information
	// about the power to use for the gate. But the info about
	// which gate to use is only specified with the extended field
	// addmsgs which come later. The whole thing is hideously ugly.

	// We need to infer the following:
	// - Which gate to use. 
	// - Whether to set the useConcentration flag
	// - Which power to use. This is the last arg in the message.

	// Which gate to use: If there is a 'ca' in the name of the
	// gate, then definitely use zGate and set useConc flag.
	// If there is a 'ca' in the name of the channel, and the
	// power is 1, then use zGate and set useConc 
	// flag. If there is no Ca or the zGate is used, then use
	// x.
	// Use conc flag follows from rules above.

	// - Which power to use. This is the last arg in the message.
	gatePower = atof( argv[5] );

	string msgType = "";
	// Now the rest of the heuristics.
	string srcName = Shell::tail( argv[1], "/" ); // This is the gate
	string destName = argv[2]; // This is the channel
	Id dest( destName );
	bool useConc = 0;
	bool zGateUsed = 0;
	bool xGateUsed = 0;
	send2< Id, string >( s(), requestFieldSlot, dest, "Zpower" );
	if ( fieldValue_.length() > 0 ) {
		if ( fieldValue_ != "0" )
			zGateUsed = 1;
	}

	send2< Id, string >( s(), requestFieldSlot, dest, "Xpower" );
	if ( fieldValue_.length() > 0 ) {
		if ( fieldValue_ != "0" )
			xGateUsed = 1;
	}
	
	useConc = hasCa( srcName );
	if ( hasCa( destName ) && gatePower == 1.0 ) {
		useConc = 1;
	}
	if ( useConc && zGateUsed == 0 ) {
		msgType = "useZ.MULTGATE";
		gate = "Z";
	} else if ( xGateUsed == 0 ) {
		msgType = "useX.MULTGATE";
		gate = "X";
	} else {
		msgType = "useY.MULTGATE";
		gate = "Y";
	}

	if ( useConc )
		send3< Id, string, string >( s(),
			setFieldSlot, dest, "useConcentration", "1" );

	return msgType;
}

void GenesisParserWrapper::doAdd(
				int argc, const char** const argv, Id s )
{
	//if (argc >= 3) cout << argv[1] << " " << argv[2] << endl;
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
		if ( !innerAdd( s, src, srcF, dest, destF ) )
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

		Id src( argv[1] );
		Id dest( argv[2] );
		if ( src.bad() ) {
	 		cout << "Error in doAdd: " << argv[1]
			     << " does not exist." << endl;
			return;
		}
		if ( dest.bad() ) {
	 		cout << "Error in doAdd: " << argv[2]
			     << " does not exist." << endl;
			return;
		}

		send2< Id, string >( s(), requestFieldSlot, src, "class" );
		string srcClassName = fieldValue_;
		send2< Id, string >( s(), requestFieldSlot, dest, "class" );
		string destClassName = fieldValue_;

		if ( msgType == "CHANNEL Gk Ek" && srcClassName == "SynChan" && destClassName == "Mg_block" )
			msgType = srcClassName + "." + destClassName + "." + msgType;
		if ( msgType == "SPIKE" && srcClassName == "SpikeGen" )
			msgType = srcClassName + "." + msgType;
		if ( msgType == "INPUT Vm" && destClassName == "SpikeGen" )
			msgType = destClassName + "." + msgType;
                if ( msgType == "SPIKE" && srcClassName == "RandomSpike" )
                         msgType = srcClassName + "." + msgType;
                if ( msgType == "INPUT Vm" && destClassName == "RandomSpike")
                         msgType = destClassName + "." + msgType;
                // ugly hack to differentiate between INPUT output
                // message to Table (where source is the ValueFinfo
                // "output" and the same message to other classes,
                // with an input message, like RC.
                // Another hack to separate pulsegen.outputSrc ->
                // spikegen.Vm ( addmsg pulsegen spikegen INPUT output
                // - here the target will be Vm
                if ( msgType == "INPUT output" && srcClassName == "PulseGen" && destClassName == "SpikeGen" )
                        msgType = destClassName + "." + msgType;
                else if ( msgType == "INPUT output" && srcClassName == "PulseGen" && destClassName != "Table" )
                        msgType = srcClassName + "." + msgType;
                else if ( msgType == "INPUT output" && destClassName == "PulseGen" )
                        msgType = destClassName + "." + msgType;
                else if ( msgType == "INPUT output" && srcClassName == "DiffAmp" && destClassName != "Table" )
                        msgType = srcClassName + "." + msgType;
		bool usingMULTGATE = 0;
		string gate = "";
		double power = 0;
		// Typically used between tabgate and vdep_channel.
		if ( msgType.substr( 0, 10 ) == "MULTGATE m" ) {
			msgType = handleMultGate( argc, argv, s, gate, power );
			usingMULTGATE = 1;
		}

		string srcF = sliMessage( msgType, sliSrcLookup() );
		string destF = sliMessage( msgType, sliDestLookup() );

		if ( srcF.length() > 0 && destF.length() > 0 ) {
			//Id src( argv[1] );
			//Id dest( argv[2] );
			// Id src = path2eid( argv[1], s );
			// Id dest = path2eid( argv[2], s );
	// 		cout << "in do_add " << src << ", " << dest << endl;
			if ( !innerAdd( s, src, srcF, dest, destF ) )
	 			cout << "Error in do_add " << argv[1] << " " << argv[2] << " " << msgType << endl;
		}

		if ( gate.length() == 1 && power > 0.0 ) {
		// finish off multgate by assigning the gate power.
			string field = gate + "power";
			send3< Id, string, string >( s(), setFieldSlot,
				dest, field, argv[5] );
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
			( e->data( 0 ) );
	return gpw->doSet( argc, argv, s );
}

void GenesisParserWrapper::doSet( int argc, const char** argv, Id s )
{
	static map< string, string > tabmap;
	if ( tabmap.empty() ) {
		tabmap[ "X_A" ] = "xGate/A";
		tabmap[ "X_B" ] = "xGate/B";
		tabmap[ "Y_A" ] = "yGate/A";
		tabmap[ "Y_B" ] = "yGate/B";
		tabmap[ "Z_A" ] = "zGate/A";
		tabmap[ "Z_B" ] = "zGate/B"; 
		tabmap[ "bet" ] = "B"; 	// Short for beta: truncated at 3 chars.
		tabmap[ "alp" ] = "A"; 	// Short for alpha
	}
	
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
		//Shell::getWildCardList(conn, path, 0);
		send2< string, bool >( sh, requestWildcardListSlot, path, 0 );
		if ( elist_.size() == 0 ) {
			cout << "Error: " << argv[0] << " : cannot find element " <<
				path << endl;
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

		// This is a little non-general, using only the first elist
		// entry for a conversion that may apply a wildcard. However,
		// it applies only to the situation where we want to do a field
		// name conversion, so hopefully it covers most use cases.
		send2< Id, string >( s(), requestFieldSlot, elist_[0], "class" );
		string className = fieldValue_;

		// const string& className = elist_[0]()->className();
		map< string, string >::iterator iter = 
			sliFieldNameConvert().find( className + "." + field );
		if ( iter != sliFieldNameConvert().end() ) 
			field = iter->second;
		//hack for synapse[i].weight type of fields
		if (field.substr(0, 8) == "synapse[") {
			size_t pos = field.find(']');
			if (pos != string::npos)
				field = field.substr(pos+2) + '[' + field.substr(8, pos - 8) + ']';
		}
		string value = argv[ i+1 ];
		if ( field.substr( 0, 12 ) == "table->table" ) { // regular table
			field = field.substr( 7 ); // snip off the initial table->
		} else {
			string::size_type pos = field.find( "->" );
			/*
			string::size_type pos = field.find( "->table" );
			if ( pos == string::npos )
					pos = field.find( "->calc_mode" );
			if ( pos == string::npos )
					pos = field.find( "->sy" );
			*/
			if ( pos != string::npos ) { // Fill table
				map< string, string >::iterator j = 
						tabmap.find( field.substr( 0, 3 ) );
				if ( j != tabmap.end() ) {
					string path;
					if ( start == 1 )
						path = "./" + j->second;
					else
						path = string( argv[1] ) + "/" + j->second;
					// Id e = GenesisParserWrapper::path2eid( path, s );
					// Here we should expand the path with the new 
					// additions. Note that we need to use a temporary
					// elist for this so as not to interfere with the 
					// original.
					Id e( path );
					if ( e.bad() ) {
						cerr <<
							"Error: GenesisParserWrapper::doSet: Object " <<
							path <<
							" not found.\n";
						continue;
					}
					vector< Id > el;
					el.push_back( e );
					field = field.substr( pos + 2 );
					send3< vector< Id >, string, string >( s(),
						setVecFieldSlot, el, field, value );
					continue;
				}
			}
		}
		// cout << "in do_set " << path << "." << field << " " <<
				// value << endl;
		
		if ( field.length() > 0 )
		//Shell::setVecField(conn, elist_, field, value)
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
bool GenesisParserWrapper::tabCreate( int argc, const char** argv, Id s )
{
	string elmPath = argv[ 1 ];
	Id elmId( elmPath );
	if ( !elmId.zero() && !elmId.bad() ) {
		send2< Id, string >( s(), requestFieldSlot, elmId, "class" );
		if ( fieldValue_.length() == 0 ) // Nothing came back
			return 0;
		
		if ( fieldValue_ == "Table" ){
                    if ( argc != 6 ) {
			cerr << "Error: GenesisParserWrapper::tabCreate: usage:"
				<< " call element TABCREATE xdivs xmin xmax\n";
			return 0;
                    }
                    send3< Id, string, string >( s(),
                                                 setFieldSlot, elmId, "xdivs", argv[ 3 ] );
                    send3< Id, string, string >( s(),
                                                 setFieldSlot, elmId, "xmin", argv[ 4 ] );
                    send3< Id, string, string >( s(),
                                                 setFieldSlot, elmId, "xmax", argv[ 5 ] );
                    return 1;
		}
		
		if ( fieldValue_ == "HHChannel" ){
                    if ( argc != 7 || strlen( argv[ 3 ] ) != 1 )
                    {
                        cerr << "Error: GenesisParserWrapper::tabCreate: usage:"
                             << " call element TABCREATE gate xdivs xmin xmax\n";
				return 0;
                    }
                    char gate = toupper( argv[ 3 ][ 0 ] );
                    if ( gate != 'X' && gate != 'Y' && gate != 'Z' ) {
				cerr << "Error: GenesisParserWrapper::tabCreate: usage:"
                                     << " call element TABCREATE gate xdivs xmin xmax."
                                     << " Gate should be X, Y or Z\n";
				return 0;
                    }
                    return channelTabCreate(s, elmId, gate, string(argv[4]), string(argv[5]), string(argv[6]));
		}

		if ( fieldValue_ == "HHChannel2D" ){
			if ( argc != 10 || strlen( argv[ 3 ] ) != 1 )
			{
				cerr << "Error: GenesisParserWrapper::tabCreate: usage:"
					<< " call element TABCREATE gate vdivs vmin vmax concdivs concmin concmax\n";
				return 0;
			}
                    char gate = toupper( argv[ 3 ][ 0 ] );
                    if ( gate != 'X' && gate != 'Y' && gate != 'Z' ) {
                        cerr << "Error: GenesisParserWrapper::tabCreate: usage:"
                             << " call element TABCREATE gate xdivs xmin xmax ydivs ymin ymax."
                             << " Gate should be X, Y or Z\n";
                        return 0;
                    }
                    return channelTabCreate(s, elmId, gate, string(argv[4]), string(argv[5]), string(argv[6]), string(argv[7]), string(argv[8]), string(argv[9]));
		}
        }
        return 0;
}

bool GenesisParserWrapper::channelTabCreate(Id s, Id elmId, char gate, string xdivs, string xmin, string xmax, string ydivs, string ymin, string ymax)
{
    
    /*
     * One way to create the gate and the interpols is to just set, lets
     * say, the Xpower field on the channel to some value. This will
     * implicitly create the corresponding (X in this case) gate and its
     * interpolation tables.
     * 
     * This turns out to be a problem when doing TABCREATE on a channel
     * that is global (for example, by being in /library). Implicit
     * creation means that every node will choose separate ids for the
     * locally generated objects.
     * 
     * To avoid this, we create the gate and its interpols explicitly below.
     * 
     * A final twist: If the channel, and hence the gate, is not a global
     * after all, then we leave the gate to its own devices. The gate will
     * detect that it is not global, and create its children Interpols
     * A & B implicitly. Otherwise we generate 2 global Ids, and send
     * them to the gates on all nodes, requesting them to create the
     * Interpols.
     */
			
    /*
     * Creating gate
     */
    string gateName = string( 1, tolower( gate ) ) + "Gate";
    string gatePath = elmId.path() + "/" + gateName;
    Id gateId( gatePath );
    if ( gateId.zero() || gateId.bad() ) {
        /*
        // Implicit creation of gate and interpols
        vector< Id > el( 1, elmId );
        send3< vector< Id >, string, string >( s(),
        setVecFieldSlot, el, string( argv[ 3 ] ) + "power", "1.0" );
        */
				
        send2< Id, string >( s(),
                             createGateSlot, elmId, string( 1, gate ) );
				
        gateId = Id( gatePath );
    }
    if ( gateId.zero() || gateId.bad() ) {
        cerr << "Error: GenesisParserWrapper::tabCreate:"
             << " Unable to create gate " << gate
             << " under channel " << elmId.path() << ".\n";
        return 0;
    }
			
    /*
     * Interpols created. Set fields.
     */
    string tables[ ] = { "A", "B" };
    for ( unsigned int i = 0; i < 2; i++ ) {
        string tabName = tables[ i ];
        string tabPath = gatePath + "/" + tabName;
        Id tabId( tabPath );
				
        if ( tabId.zero() || tabId.bad() ) {
            cerr << "Error: GenesisParserWrapper::tabCreate:"
                 << " Unable to create Interpol " << tabName
                 << " under gate " << gatePath << ".\n";
            return 0;
        }
				
        send3< Id, string, string >( s(),
                                     setFieldSlot, tabId, "xdivs", xdivs );
        send3< Id, string, string >( s(),
                                     setFieldSlot, tabId, "xmin", xmin );
        send3< Id, string, string >( s(),
                                     setFieldSlot, tabId, "xmax", xmax );
        if (!ydivs.empty() && !ymin.empty() && !ymax.empty()) {
            send3< Id, string, string >( s(),
                                         setFieldSlot, tabId, "ydivs", ydivs );
            send3< Id, string, string >( s(),
                                         setFieldSlot, tabId, "ymin", ymin );
            send3< Id, string, string >( s(),
                                         setFieldSlot, tabId, "ymax", ymax );
        }
    }
    return 1;
}

void do_call( int argc, const char** const argv, Id s )
{
	if ( argc < 3 ) {
		cout << "usage:: " << argv[0] << " path field/Action [args...]\n";
		return;
	}
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data( 0 ) );
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
		return;
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

	// syntax: call table TABOP op [min max]
	if ( strcmp ( argv[2], "TABOP" ) == 0 ) { 
		Id table( argv[1] );
		if ( table.bad() ) {
			cout << "Error: " << argv[0] << 
					" could not find object '" << table << "'\n";
			return;
		}
		if ( argc == 4 ) {
			send4< Id, char, double, double >( s(),
				tabopSlot, table, argv[3][0], 0.0, 0.0 );
		} else if ( argc == 6 ) {
			double min = atof( argv[4] );
			double max = atof( argv[5] );
			send4< Id, char, double, double >( s(),
				tabopSlot, table, argv[3][0], min, max );
		} else {
			cout << "usage: " << argv[0] << 
					" element TABOP op [min max]\n";
			cout << "valid operations:\n";
			cout << "a = average, m = min, M = Max, r= range\n";
			cout << "s = slope, i = intercept, f = freq\n";
			cout << "S = Sqrt(sum of squares)\n";
		}
		return;
	}

	// Fallback: generic setfield.
	Id id( argv[1] );
	if ( !id.good() ) {
		cout << "Error: call: could not find object: " << argv[1] << endl;
		return;
	}

	string field = argv[2];
	string argstr = "";
	if ( argc >= 4 ) {
		argstr = argv[3];
		for( int i = 4; i < argc; ++i ) {
			argstr = argstr + "	" + argv[i];
		}
	}
	send3< Id, string, string >( s(), setFieldSlot, id, field, argstr );
}

int do_isa( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
            Id eid(argv[2]);
            if (eid.bad()){
                cout << "Error: " << argv[2] << " : no such element." << endl;
                return 0;
            }
            const Cinfo * thisCinfo = eid()->cinfo();
            const Cinfo * otherCinfo = Cinfo::find(argv[1]);
            if (otherCinfo){
                return thisCinfo->isA(otherCinfo);
            }
            // Try translating from GENESIS name to MOOSE name
            map<string, string>::iterator it = sliClassNameConvert().find(argv[1]);
            if (it == sliClassNameConvert().end()){
                cout << "Error: " << argv[2] << " : no such class." << endl;
                return 0;
            }
            otherCinfo = Cinfo::find(it->second);
            if (otherCinfo) {
                return thisCinfo->isA(otherCinfo);
            }
	} else {
            cout << "usage:: " << argv[0] << " type field\n";
	}
	return 0;
}

bool GenesisParserWrapper::fieldExists(
			Id eid, const string& field, Id s )
{
	//conversion of field
	send2< Id, string >( s(), requestFieldSlot, eid, "class" );
	string className = fieldValue_;

	map< string, string >::iterator i = 
			sliFieldNameConvert().find( className + "." + field );
	string newfield = field;
	if ( i != sliFieldNameConvert().end() ) 
		newfield = i->second;
	//conversion of field, if needed, complete
		
	send2< Id, string >( s(), requestFieldSlot, eid, "fieldList" );
	if ( fieldValue_.length() == 0 ) // Nothing came back
		return 0;
	return ( fieldValue_.find( newfield ) != string::npos );
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
				static_cast< GenesisParserWrapper* >( s()->data( 0 ) );
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
			( e->data( 0 ) );
	return gpw->doGet( argc, argv, s );
}



char* GenesisParserWrapper::doGet( int argc, const char** argv, Id s )
{
	static map< string, string > tabmap;
	tabmap[ "X_A" ] = "xGate/A";
	tabmap[ "X_B" ] = "xGate/B";
	tabmap[ "Y_A" ] = "yGate/A";
	tabmap[ "Y_B" ] = "yGate/B";
	tabmap[ "Z_A" ] = "zGate/A";
	tabmap[ "Z_B" ] = "zGate/B"; 
	tabmap[ "bet" ] = "B"; 	// Short for beta: truncated at 3 chars.
	tabmap[ "alp" ] = "A"; 	// Short for alpha

	string field;
	string value;
	Id e;
	if ( argc == 3 ) {
		// e = GenesisParserWrapper::path2eid( argv[1], s );
		e = Id( argv[1] );
		if ( e.bad() ) 
			cout << "Element '" << argv[1] << "' not found\n";
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
		requestFieldSlot, e, "class" );
	map< string, string >::iterator i = 
			sliFieldNameConvert().find( fieldValue_ + "." + field );
	if ( i != sliFieldNameConvert().end() )
		field = i->second;

	// Hack section to handle table field lookups.
	if ( field.substr( 0, 12 ) == "table->table" ) { // regular table
		field = field.substr( 7 ); // snip off the initial table->
	} else {
		string::size_type pos = field.find( "->" );
		if ( pos != string::npos ) { // Other table fields.
			map< string, string >::iterator j = 
					tabmap.find( field.substr( 0, 3 ) );
			if ( j != tabmap.end() ) {
				string path;
				if ( argc == 2 )
					path = "./" + j->second;
				else
					path = string( argv[1] ) + "/" + j->second;
				e = Id( path );
				field = field.substr( pos + 2 );
			}
		}
	}

	fieldValue_ = "";
	send2< Id, string >( s(),
		requestFieldSlot, e, field );
	if ( fieldValue_.length() == 0 ) // Nothing came back
		return 0;
	return copyString( fieldValue_.c_str() );
}

const char* do_getmsg( int argc, const char** const argv, Id s )
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
			( e->data( 0 ) );
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

/**
 * Identify child node specification in the name@nodeNum format.
 * Return node.
 * Modify name to strip out node info.
 * 
 * If nodeNum < 0, it signifies all nodes (i.e., global)
 */
unsigned int parseNodeNum( string& name )
{
	unsigned int childNode = Id::UnknownNode; // Tell the system to figure out child node.
	int givenNode;
	if ( name.rfind( "@" ) != string::npos ) {
		string nodeNum = Shell::tail( name, "@" );
		if ( nodeNum.length() > 0 ) {
			name = Shell::head( name, "@" );
			givenNode = atoi( nodeNum.c_str() );
			if ( givenNode < 0 )
				childNode = Id::GlobalNode;
			else
				childNode = static_cast< unsigned int >( givenNode );
		}
		else
			childNode = 0;
	}
	return childNode;
}


void do_create( int argc, const char** const argv, Id s )
{
	if ( argc != 3 ) {
		cout << "usage:: " << argv[0] << " class name\n";
		return;
	}

	assert( strlen( argv[2] ) > 0 );
	string className = argv[1];
	if ( !Cinfo::find( className ) ) {
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

	unsigned int childNode = parseNodeNum( name );
	/*
	int childNode = -1; // Tell the system to figure out child node.
	if ( name.rfind( "@" ) != string::npos ) {
		string nodeNum = Shell::tail( name, "@" );
		if ( nodeNum.length() > 0 ) {
			childNode = atoi( nodeNum.c_str() );
			name = Shell::head( name, "@" );
		}
		else
			childNode = 0;
	}
	*/

	string parent = Shell::head( argv[2], "/" );
	if ( argv[2][0] == '/' && parent == "" )
		parent = "/";

	Id pa( parent ); // This includes figuring out the node number.

	send4< string, string, int, Id >( s(), createSlot, 
		className, name, childNode, pa );

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
				if ( pastr.find( "/" ) == string::npos ) {
					pastr = ".";
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
		//Shell::copy(conn, e, pa, name);
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
		if ( e.bad() ) {
			cout << "Error - cannot change to '" << argv[1] << "'\n";
		} else {
			send1< Id >( s(), setCweSlot, e );
		}
	} else {
		cout << "usage:: " << argv[0] << " Element\n";
	}
}

void do_pushe( int argc, const char** const argv, Id s )
{
	GenesisParserWrapper* gpw = 
		static_cast< GenesisParserWrapper* > ( s()->data( 0 ) );
	if ( argc == 2 ) {
		// s->pusheFuncLocal( argv[1] );
		Id e( argv[1] );
		if ( e.bad() ) {
			gpw->print( string( "Error - cannot change to '" ) +
				string( argv[1] ) + "'" );
		} else {
			send1< Id >( s(), pusheSlot, e );

			// Disabling printing the CWE after pushe/pope
			// gpw->printCwe();
		}
	} else {
		cout << "usage:: " << argv[0] << " Element\n";
	}
}

void do_pope( int argc, const char** const argv, Id s )
{
	if ( argc == 1 ) {
		// s->popeFuncLocal( );
		send0( s(), popeSlot );
		// Disabling printing the CWE after pushe/pope
		// GenesisParserWrapper* gpw = 
		//	static_cast< GenesisParserWrapper* > ( s()->data( 0 ) );
		// gpw->printCwe();
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
			( e->data( 0 ) );
	gpw->alias( alias, old );
}

void do_quit( int argc, const char** const argv, Id s )
{
	// s->quitFuncLocal( );
	send0( s(), quitSlot );
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
			( e->data( 0 ) );
	gpw->step( argc, argv );
}

void GenesisParserWrapper::step( int argc, const char** const argv )
{
	double runtime;
	Element* e = element()();
	if ( argc == 3 ) {
		if ( strcmp( argv[ 2 ], "-t" ) == 0 || strcmp( argv[ 2 ], "-time" ) == 0) {
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

// Args are clock number, dt, optionally stage.
// If we do not specify stage, then the system uses the pre-existing
// stage. This is specified by a dummy stage value of -1.
void do_setclock( int argc, const char** const argv, Id s )
{
	if ( argc == 3 ) {
			send3< int, double, int >( s(),
				setClockSlot, 
				//
				atoi( argv[1] ), atof( argv[2] ), -1 );
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
				( e->data( 0 ) );
		gpw->showClocks( e );

	} else {
		cout << "usage:: " << argv[0] << "\n";
	}
}

/**
 * This function simply sends the strings over to the Shell. This is
 * necessary because we don't know ahead of time what the tick ids
 * will be on different nodes. Furthermore, the scheduling operations
 * need to go to individual nodes anyway, so we'll let them do the
 * wildcarding for the path, too.
 */
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

	/*
	// Id tickId = GenesisParserWrapper::path2eid( tickName, s );
	Id tickId( tickName );
	if ( tickId.bad() ) {
		cout << "Error:" << argv[0] << ": Invalid clockNumber " <<
				tickName << "\n";
		return;
	}
	*/

	string path = argv[1];
	send3< string, string, string >( 
		s(), useClockSlot, tickName, path, func );

	/*
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data( 0 ) );
	gpw->useClock( tickName, path, func, s );
	*/
}

/*
void GenesisParserWrapper::useClock(
	string tickName, const string& path, const string& func, Id s )
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
*/

void do_show( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data( 0 ) );
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
		//Shell::getField(conn, e->id(), *i)
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
	char temp[200];
	int firstField = 2;

	if ( argc < 2 ) {
		print( "Usage: showfield [object/wildcard] [fields] -all" );
		return;
	}

	if ( argc == 2 ) { // show fields of cwe.
		send0( s(), requestCweSlot );
		elist_.resize( 1, cwe_ );
		// e = cwe_;
		firstField = 1;
	} else {
		string path = argv[1];
		//Shell::getWildCardList(conn, path, 0);
		send2< string, bool >( s(), requestWildcardListSlot, path, 0 );
		if ( elist_.size() == 0 ) {
			cout << "Error: " << argv[0] << " : cannot find element " <<
				path << endl;
			return;
		}
		firstField = 2;

		/*
		// e = path2eid( argv[1], s );
		e = Id( argv[1] );
		if ( e.bad() ) {
			e = cwe_;
			firstField = 1;
		} else {
			firstField = 2;
		}
		*/
	}
	// print( "[ " + eid2path( e ) + " ]" );
	
	vector< Id >::iterator j;
	vector< Id > tempList = elist_; // In case it gets bashed elsewhere.
	for ( j = tempList.begin(); j != tempList.end(); j++ ) {
		e = *j;
		print( "[ " + e.path() + " ]" );
		for ( int i = firstField; i < argc; i++ ) {
			if ( strcmp( argv[i], "*") == 0 || 
				strcmp( argv[i], "-all" ) == 0 ||
				strcmp( argv[i], "-a" ) == 0 ) {
				showAllFields( e, s );
			} else { // get specific field here.
				fieldValue_ = "";
				send2< Id, string >( s(), requestFieldSlot, e, "class" );
				string className = fieldValue_;

				fieldValue_ = "";
				//Shell::getField(conn, e, argv[i])
				string field = argv[i];
				map< string, string >::iterator iter = 
					sliFieldNameConvert().find( className + "." + field );
				if ( iter != sliFieldNameConvert().end() ) 
					field = iter->second;
					
				send2< Id, string >( s(), requestFieldSlot, e, field );
				if ( fieldValue_.length() > 0 ) {
				///\todo: Should use C++ formatting to avoid overflow.
					sprintf( temp, "%-25s%s", field.c_str(), "= " );//printing the new field name 
					print( temp + fieldValue_ );
				} else {
					cout << "'" << field << "' is not an element or the field of the working element\n";
					return;
				}
			}
		}
	}
}

void do_le( int argc, const char** const argv, Id s )
{
	Element* e = s();
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( e->data( 0 ) );
	gpw->doLe( argc, argv, s );
}

void GenesisParserWrapper::doLe( int argc, const char** argv, Id s )
{
	Id current;
	if ( argc == 1 ) { // Look in the cwe first.
		send0( s(), requestCweSlot );
		send1< Id >( s(), requestLeSlot, cwe_ );
		current = cwe_;
	} else if ( argc >= 2 ) {
		// Id e = path2eid( argv[1], s );
		Id e( argv[1] );
		current = e;
		/// \todo: Use better test for a bad path than this.
		if ( e.bad() ) {
			print( string( "cannot find object '" ) + argv[1] + "'" );
			return;
		}
		send1< Id >( s(), requestLeSlot, e );
	}

	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );

	vector< Id >::iterator i = elist_.begin();
	// This operation should really do it in a parallel-clean way.
	/// \todo If any children, we should suffix the name with a '/'
	for ( i = elist_.begin(); i != elist_.end(); i++ ) {
		send2< Id, string >( s(), requestFieldSlot, *i, "name" );
		print( gpw->getFieldValue() );
	}
	elist_.resize( 0 );
}

void do_pwe( int argc, const char** const argv, Id s )
{
	// Here we need to wait for the shell to service this message
	// request and put the requested value in the local cwe_.
	send0( s(), requestCweSlot );
	// print it out.
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
	gpw->printCwe();
}

/*
void GenesisParserWrapper::doPwe( int argc, const char** argv, Id s )
{
	send0( s(), requestCweSlot );
	// Here we need to wait for the shell to service this message
	// request and put the requested value in the local cwe_.
	
	// print( GenesisParserWrapper::eid2path( cwe_ ) );
	print( cwe_.path() );
}
*/

void GenesisParserWrapper::printCwe()
{
	print( cwe_.path() );
}

void do_listcommands( int argc, const char** const argv, Id s )
{
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
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
			( s()->data( 0 ) );
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
	const char* printMode = "append";
	if ( argc > 4 ) {
		if ( argv[ argc - 1 ][0] == '-' && argv[ argc - 1 ][1] == 'o' )
			printMode = "print"; // overwrite.
	}

	// Here we will later have to put in checks for tables on channels.
	// Also lots of options remain.
	// For now we just call the print command on the interpol
	// Id e = GenesisParserWrapper::path2eid( elmname, s );
        
        switch (tabname[0])
        {
            case 'X':
                elmname += "/xGate";
                break;
                
            case 'Y':
                elmname += "/yGate";
                break;
                
            case 'Z':
                elmname += "/zGate";
                break;                    
        }
        if ('X' == tabname[0] || 'Y' == tabname[0] || 'Z' == tabname[0])
        {
            switch (tabname[2])
            {
                case 'A':
                    elmname += "/A";
                    break;
                case 'B':
                    elmname += "/B";
                    break;
            }
        }
        
	Id e( elmname );
	if ( !e.zero() && !e.bad() )
		send3< Id, string, string >( s(),
			setFieldSlot, e, printMode, fname );
	else
		cout << "Error: " << argv[0] << ": element not found: " <<
				elmname << endl;
}


/**
 * file2tab: Loads a file into a table.
 */
void do_file2tab( int argc, const char** const argv, Id s )
{
	static map< string, string > tabmap;
	if ( tabmap.empty() ) {
		tabmap[ "X_A" ] = "xGate/A";
		tabmap[ "X_B" ] = "xGate/B";
		tabmap[ "Y_A" ] = "yGate/A";
		tabmap[ "Y_B" ] = "yGate/B";
		tabmap[ "Z_A" ] = "zGate/A";
		tabmap[ "Z_B" ] = "zGate/B"; 
		tabmap[ "bet" ] = "B"; 	// Short for beta: truncated at 3 chars.
		tabmap[ "alp" ] = "A"; 	// Short for alpha
	}

	if ( argc < 4 ) {
		cout << argv[0] << ": Too few command arguments\n";
		cout << "usage: " << argv[0] << " file-name element table -table2 table -table3 table -skiplines number\n\n";
		return;
	}
	string fname = argv[1];
	string elmname = argv[2];
	string tabname = argv[3];
	unsigned int skiplines = 0;
	if ( argc > 5 && string( argv[4] ) == "-skiplines" )
		skiplines = atoi( argv[5] );
	

	Id e( elmname );
	
	// Hack here to deal with the special case of filling tables
	// in tabchannels. Example command looks like:
	//		file2tab {filename} Ca Y_A
	// so here we need to do
	//		file2tab {filename} Ca/yGate/A table
	if ( e()->className() == "HHChannel" && tabmap.find( tabname ) != tabmap.end() ) {
		elmname = elmname + "/" + tabmap[ tabname ];
		e = Id( elmname );
	}
	
	if ( !e.zero() && !e.bad() )
		send3< Id, string, unsigned int >( s(),
			file2tabSlot, e, fname, skiplines );
	else
		cerr << "Error: " << argv[0] << ": element not found: " <<
				elmname << endl;
}

/**
 * Utility function for returning an empty argv list
 */
char** NULLArgv()
{
	char** argv = (char** )malloc( sizeof( char* ) );
	argv[0] = 0;
	return argv;
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
char** do_element_list( int argc, const char** const argv, Id s )
{
	static string space = " ";
	if ( argc != 2 ) {
		cout << "usage:: " << argv[0] << " path\n";
		return NULLArgv();
	}
	string path = argv[1];
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
	return gpw->elementList( path, s );

	// return copyString( s->getmsgFuncLocal( field, options ) );
//	return copyString( ret.c_str() );
}
/**
 * Allocates and returns an array of char* for the element list
 * entries, just like the old GENESIS sim_list.c::do_construct_list
 * function. There too I did not know how the list was supposed to be
 * freed.
 */
char** GenesisParserWrapper::elementList( const string& path, Id s)
{
 	//Shell::getWildCardList(conn, path, 0(ordered))
 	send2< string, bool >( s(), requestWildcardListSlot, path, 0 );
	// bool first = 1;
	vector< Id >::iterator i;
	// Need to use old-style allocation because the genesis parser 
	// just might free it in the old style.
	if ( elist_.size() == 0 )
		return( NULLArgv() );
	char** ret = ( char** )calloc( elist_.size() + 1, sizeof( char * ) );
	char** j = ret;
	for ( i = elist_.begin(); i != elist_.end(); i++ ) {
		*j++ = copyString( i->path().c_str() );
		/*
		if ( first )
			ret = i->path(); // ret = eid2path( *i );
		else
			ret = ret + " " + i->path();
			// ret = ret + " " + eid2path( *i );
		first = 0;
		*/
	}
	return ret;
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

// Extracts the global values for the readcell parameters if defined.
void GenesisParserWrapper::getReadcellGlobals( 
	vector< double >& globalParms )
{
	globalParms.resize( 0 );
	globalParms.resize( 5, 0.0 );

	Result *rp;
	Symtab* GlobalSymbols = getGlobalSymbols();

	rp=SymtabLook(GlobalSymbols,"CM");
	if (rp)
		globalParms[0] = GetScriptDouble("CM");
		
	rp=SymtabLook(GlobalSymbols,"RM");
	if (rp)
		globalParms[1] = GetScriptDouble("RM");
		
	rp=SymtabLook(GlobalSymbols,"RA");
	if (rp)
		globalParms[2] = GetScriptDouble("RA");
		
	rp=SymtabLook(GlobalSymbols,"EREST_ACT");
	if (rp)
		globalParms[3] = GetScriptDouble("EREST_ACT");
		
	rp=SymtabLook(GlobalSymbols,"ELEAK");
	if (rp)
		globalParms[4] = GetScriptDouble("ELEAK");
	else
		globalParms[4] = globalParms[3];
}

void do_readcell( int argc, const char** const argv, Id s )
{
	if (argc != 3 ) {
		cout << "usage:: " << argv[0] << " filename cellpath\n";
		return;
	}
	string cellpath = argv[2];
	unsigned int childNode = parseNodeNum( cellpath );

	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
	vector< double > globalParms;
	gpw->getReadcellGlobals( globalParms );


	string filename = argv[1];

 	send4< string, string, vector< double >, int >( 
		s(), readCellSlot, filename, cellpath, globalParms, childNode );
}

Id findChanGateId( int argc, const char** const argv, Id s ) 
{
	assert( argc >= 3 );
	// In MOOSE we only have tabchannels with gates, more like the
	// good old tabgates and vdep_channels. Functionally equivalent,
	// and here we merge the two cases.
	
	char gateChar = argv[2][0];
	string gate = argv[1];
	string type = "";
	if ( gateChar == 'X' )
			type =  "xGate";
	else if ( gateChar == 'Y' )
			type = "yGate";
	else if ( gateChar == 'Z' )
			type = "zGate";
	
	gate = gate + "/" + type;
	Id gateId( gate );
	if ( gateId.bad() ) {// Don't give up, it might be a tabgate or might not have been created
		GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
		Id id = Id( argv[1] );
		
		send2< Id, string >( s(), requestFieldSlot, id, "class" );
		string className = gpw->getFieldValue();
		if ( className == "HHGate" )
			gateId = id;
		else if ( className == "HHChannel" ) {
			if (type != "") {
				send2< Id, string >( s(),
					createGateSlot, id, string( 1, gateChar ) );
				gateId = Id(gate);
			}
			else
				; // error
		}
	}
	
	if ( gateId.bad() ) { // Now give up
			cout << "Error: findChanGateId: unable to find channel/gate '" << argv[1] << "/" << argv[2] << endl;
			return gateId;
	}
	return gateId;
}

void setupChanFunc( int argc, const char** const argv, Id s, 
				Slot slot )
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

void tweakChanFunc( int argc, const char** const argv, Id s, Slot slot )
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

void do_setupgate( int argc, const char** const argv, Id s ) 
{
	if (argc < 8 ) {
		cout << "usage:: " << argv[0] << " channel-element table A B C D F -size n -range min max -noalloc\n";
		return;
	}

	Id gateId = findChanGateId( argc, argv, s );
	if ( gateId.bad() ) {
		cout << "Error: " << argv[0] << ": Could not find gate " <<
			argv[1] << "." << argv[2] << endl;
			return;
	}

	vector< double > parms;
	parms.push_back( atof( argv[3] ) ); // A
	parms.push_back( atof( argv[4] ) ); // B
	parms.push_back( atof( argv[5] ) ); // C
	parms.push_back( atof( argv[6] ) ); // D
	parms.push_back( atof( argv[7] ) ); // F

	double size = 3000.0;
	double min = -0.100;
	double max = 0.050;
	int nextArg;
	if ( argc >=10 ) {
		if ( strncmp( argv[8], "-s", 2 ) == 0 ) {
			size = atof( argv[9] );
			nextArg = 10;
		} else {
			nextArg = 8;
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

	if ( strcmp( argv[2], "beta" ) == 0 ) {
		parms.push_back( 1.0 );
	} else if ( strcmp( argv[2], "alpha" ) == 0 ) {
		parms.push_back( 0.0 );
	} else if ( strcmp( argv[2], "table" ) == 0 ) {
		cout << "Warning: " << argv[0] << ": Moose cannot yet handle tables: " <<
			argv[1] << "." << argv[2] << endl;
			return;
	} else {
		cout << "Error: " << argv[0] << ": Could not find gate " <<
			argv[1] << "." << argv[2] << endl;
			return;
	}

 	send2< Id, vector< double > >( s(), setupGateSlot, gateId, parms );
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

float do_version( int argc, const char** const argv, Id s )
{
	return 3.0;
}

char *do_revision(int argc, const char** const argv, Id s)
{
    return copyString(SVN_REVISION);
}
char** doArgv( int argc, const char** const argv, Id s )
{
    cout << argv[0] << ": Not yet implemented!" << endl;
    return 0;
}

int doArgc( int argc, const char** const argv, Id s )
{
    cout << argv[0] << ": Not yet implemented!" << endl;
    return 0;
}






// Old GENESIS Usage: addfield [element] field-name -indirect element field -description text
// Here we'll have a subset of it:
// addfield [element] field-name -type field_type
void do_addfield( int argc, const char** const argv, Id s )
{
/*	if ( argc == 2 ) {
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
*/
	string classname = argv[1];
	string fieldname = argv[2];
	send2<Id, string>(s(), addfieldSlot, Id(classname), fieldname);
	
}

void do_loadtab ( int argc, const char** const argv, Id s )
{
	// cout << "in do_loadtab: argc = " << argc << endl;
	string line = "";
	for ( int i = 1 ; i < argc; i++ ) {
		line.append( argv[i] );
		line.append( " " );
	}

	send1< string >( s(), loadtabSlot, line );
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

void do_xps ( int argc, const char** const argv, Id s ) {
	cout << "Not yet implemented!!" << endl;
}

void do_createmap(int argc, const char** const argv, Id s) {
	
	const char* source, *dest;
	int Nx, Ny;
	double dx = 1.0, dy = 1.0;
	double xorigin = 0.0, yorigin = 0.0;
	bool object = false;
	vector <double> parameter;
	
	if ( argc < 5 ) {
		cout << "Usage: " << argv[0] << " source dest Nx Ny -delta dx dy -origin x y\n";
		return;
	}
	
	source = argv[1];
	dest = argv[2];
	Nx = atoi(argv[3]);
	Ny = atoi(argv[4]);
	
	for(int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-delta") == 0) {
                    if (i+2 >= argc) {
			cout << "Parser:: -delta option requires two arguments..Ignoring it" << endl;
                    }
                    else 
                    {
                        dx = atof(argv[i+1]);
                        dy = atof(argv[i+2]);
                    }
                    
		}	
		else if (strcmp(argv[i], "-origin") == 0) {
			if (i+2 >= argc) {
				cout << "Parser:: -origin option requires two arguments...Ignoring it." << endl;
			}
                        else
                        {
                            xorigin = atof(argv[i+1]);
                            yorigin = atof(argv[i+2]);
                        }
                        
		}
		else if (strcmp(argv[i], "-object") == 0) {
			object = true;
		}
	}
	
	parameter.push_back(Nx);
	parameter.push_back(Ny);
	parameter.push_back(dx);
	parameter.push_back(dy);
	parameter.push_back(xorigin);
	parameter.push_back(yorigin);
	
	string parent = dest;	
	Id pa(parent);
	if ( pa.bad()) {
		string headpath = Shell::head( dest, "/" );
		Id head(headpath);
		if (head.bad()) {
			cout << "'" << headpath  << "'" << " is not defined."  << dest << "." << endl;
			return;
		}
		send4< string, string, int, Id >( s(),
			createSlot, "Neutral", Shell::tail(dest, "/"), head.node(), head );
	}
	if (object) {
		string className = source;
		if ( !Cinfo::find( className ) ) {
			/*WORK*/
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
		//string name = Shell::tail(dest, "/");
		string name;
		if (className == "Neutral") {name = "proto";}
		else name = source;
		if ( name.length() < 1 ) {
			cout << "Error: invalid object name : " << name << endl;
			return;
		}
		//string parent = Shell::head( argv[2], "/" );
		
		pa = Id(parent);
		if (pa.bad()) cout << "Too bad" <<endl;
		//Shell::staticCreateArray( conn, className, name, pa, n )
		send4< string, string, Id, vector <double> >( s(),
		createArraySlot, className, name, pa, parameter );
	}
	else{
		string name;
		Id e;
		Id pa;
		if ( parseCopyMove( 3, argv, s, e, pa, name ) ) {
			//Shell::copy(conn, e, pa, name);
			send4< Id, Id, string, vector <double> >( s(), 
				copyIntoArraySlot, e, pa, name, parameter );
		}
	}
	
	
}

void do_planarconnect( int argc, const char** const argv, Id s )
{
/*
source-path dest-path -relative -sourcemask {box,ellipse} x1 y1 x2 y2 -sourcehole {box,ellipse} x1 y1 x2 y2 -destmask {box,ellipse} x1 y1 x2 y2 -desthole {box,ellipse} x1 y1 x2 y2 -probability p
planarconnect / dest-path -relative -sourcemask box -1 -1 1 1 -destmask box 1 1 2 2 -probability 0.5
*/
	string source, dest;
	source = argv[1];
	dest = argv[2];
	double probability = 1;
	for (int i = 3; i < argc; i++ ) {
		if (strcmp(argv[i], "-probability") == 0 && (argc != i+1))
			probability = atof(argv[i+1]);
	}
	//Shell::planarconnect(conn, source, dest, probability)
	send3<string, string, double>(s(), 
		planarconnectSlot, source, dest, probability);
	//GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >( e->data( 0 ) );
	//return gpw->doPlanarConnect( argc, argv, s );
	
}

void do_planardelay( int argc, const char** const argv, Id s )
{
	if (argc < 3) {
		cout << "usage:: " << endl;
		cout << "planarweight sourcepath [destination_path]\t\\ " << endl;
		cout << "\t-fixed delay\t\t\t\t\\ " << endl;
                cout << "\t-radial conduction_velocity\t\t\\ " << endl;
                cout << "\t-add\t\t\t\t\t\\ " << endl;
                cout << "\t-uniform scale\t\t\t\t\\ " << endl;
                cout << "\t-gaussian stdev maxdev\t\t\t\\ " << endl;
                cout << "\t-exponential mid max\t\t\t\\ " << endl;
                cout << "\t-absoluterandom" << endl;
		return;
	}
	string source = argv[1];
	string destination = argv[2];
	int start = 3;
	if (destination == "-fixed"       || destination == "-radial"    || 
	    destination == "-uniform"     || destination == "-gaussian" || 
	    destination == "-exponential" || destination == "-absoluterandom" ||
	    destination == "-add"
	   ) {
		start = 2;
		destination = "";
	}
	double delay = 0;
	double conduction_velocity = 0;
	double scale = 0;
	double stdev = 0;
	double maxdev = 0;
	double mid = 0;
	double max = 0;
	bool absoluterandom = false;
	bool add = false;
	double delaychoice = 0;
	double randchoice = 0;
	
	int delayoptions = 0;
	int randoptions = 0;
	for (int i = start; i < argc; i++) {
		if (strcmp(argv[i], "-fixed") == 0) {
			if (argc - i <= 1) {
				cout << "Error: Delay missing" << endl;
				continue;
			}
			delay = atof(argv[i+1]);
			i+=1;
			delayoptions++;
			delaychoice = 0;
			continue;
		}
		if (strcmp(argv[i], "-radial") == 0) {
			if (argc - i <= 1) {
				cout << "Error: Conduction velocity not given" << endl;
				continue;
			}
			conduction_velocity = atof(argv[i+3]);
			i+=1;
			delayoptions++;
			delaychoice = 1;
			continue;
		}
		if (strcmp(argv[i], "-uniform") == 0) {
			if (argc - i <= 1) {
				cout << "Error: Scale missing" << endl;
				continue;
			}
			scale = atof(argv[i+1]);
			i+=1;
			randoptions++;
			randchoice = 1;
			continue;
		}
		if (strcmp(argv[i], "-gaussian") == 0) {
			if (argc - i <= 2) {
				cout << "Error: All the parameters for gaussian not given" << endl;
				continue;
			}
			stdev = atof(argv[i+1]);
			maxdev = atof(argv[i+2]);
			i+=2;
			randoptions++;
			randchoice = 2;
			continue;
		}
		if (strcmp(argv[i], "-exponential") == 0) {
			if (argc - i <= 2) {
				cout << "Error: All the parameters for exponential not given" << endl;
				continue;
			}
			mid = atof(argv[i+1]);
			max = atof(argv[i+2]);
			i+=2;
			randoptions++;
			randchoice = 3;
			continue;
		}
		if (strcmp(argv[i], "-absoluterandom") == 0) {
			absoluterandom = true;
			continue;
		}
		if (strcmp(argv[i], "-add") == 0) {
			add = true;
			continue;
		}
	}
	if (delayoptions > 1) {
		cout << "planardelay::Must have exactly one of -fixed, -radial options." << endl;
		return;
	}
	if (randoptions > 1) {
		cout << "planardelay::Must have at most one of -uniform, -gaussian, -exponential options." << endl;
		return;
	}
	vector <double> parameter(11, 0);
	parameter[0] = delay;
	parameter[1] = conduction_velocity;
	if (add) parameter[2] = 1;
	parameter[3] = scale;
	parameter[4] = stdev;
	parameter[5] = maxdev;
	parameter[6] = mid;
	parameter[7] = max;
	if (absoluterandom) parameter[8] = 1;
	parameter[9] = delaychoice;
	parameter[10] = randchoice;
	send3<string, string, vector<double> >(s(), planardelaySlot, source, destination, parameter);
}

void do_planarweight( int argc, const char** const argv, Id s )
{
	if (argc < 3) {
		cout << "usage:: " << endl;
		cout << "planarweight sourcepath [destination_path]\t\\ " << endl;
		cout << "\t-fixed weight\t\t\t\t\\ " << endl;
                cout << "\t-decay decay_rate max_weight min_weight\t\\ " << endl;
                cout << "\t-uniform scale\t\t\t\t\\ " << endl;
                cout << "\t-gaussian stdev maxdev\t\t\t\\ " << endl;
                cout << "\t-exponential mid max\t\t\t\\ " << endl;
                cout << "\t-absoluterandom				  " << endl;
		return;
	}
	string source = argv[1];
	string destination = argv[2];
	int start = 3;
	if (destination == "-fixed"       || destination == "-decay"    || 
	    destination == "-uniform"     || destination == "-gaussian" || 
	    destination == "-exponential" || destination == "-absoluterandom") {
		start = 2;
		destination = "";
	}
	
	int randoptions = 0;
	int weightoptions = 0;
	
	double weightchoice = 0;
	double randchoice = 0;
	double fixedweight = 0;
	double decay_rate = 0;
	double max_weight = 0;
	double min_weight = 0;
	double scale = 0;
	double stdev = 0;
	double maxdev = 0;
	double mid = 0;
	double max = 0;
	bool absoluterandom = false;
	
	for (int i = start; i < argc; i++) {
		if (strcmp(argv[i], "-fixed") == 0) {
			if (argc - i <= 1) {
				cout << "Error: Weight missing" << endl;
				continue;
			}
			fixedweight = atof(argv[i+1]);
			i+=1;
			weightoptions++;
			weightchoice = 0;
			continue;
		}
		if (strcmp(argv[i], "-decay") == 0) {
			if (argc - i <= 3) {
				cout << "Error: All the parameters for decay not given" << endl;
				continue;
			}
			decay_rate = atof(argv[i+1]);
			max_weight = atof(argv[i+2]);
			min_weight = atof(argv[i+3]);
			i+=3;
			weightoptions++;
			weightchoice = 1;
			continue;
		}
		if (strcmp(argv[i], "-uniform") == 0) {
			if (argc - i <= 1) {
				cout << "Error: Scale missing" << endl;
				continue;
			}
			scale = atof(argv[i+1]);
			i+=1;
			randoptions++;
			randchoice = 1;
			continue;
		}
		if (strcmp(argv[i], "-gaussian") == 0) {
			if (argc - i <= 2) {
				cout << "Error: All the parameters for gaussian not given" << endl;
				continue;
			}
			stdev = atof(argv[i+1]);
			maxdev = atof(argv[i+2]);
			i+=2;
			randoptions++;
			randchoice = 2;
			continue;
		}
		if (strcmp(argv[i], "-exponential") == 0) {
			if (argc - i <= 2) {
				cout << "Error: All the parameters for exponential not given" << endl;
				continue;
			}
			mid = atof(argv[i+1]);
			max = atof(argv[i+2]);
			i+=2;
			randoptions++;
			randchoice = 3;
			continue;
		}
		if (strcmp(argv[i], "-absoluterandom") == 0) {
			absoluterandom = true;
			continue;
		}
	}
	if (weightoptions > 1) {
		cout << "Must have exactly one of -fixed, -decay options." << endl;
		return;
	}
	if (randoptions > 1) {
		cout << "Must have at most one of -uniform, -gaussian, -exponential options." << endl;
		return;
	}
	
	vector <double> parameter(12, 0);
	parameter[0] = fixedweight;
	parameter[1] = decay_rate;
	parameter[2] = max_weight;
	parameter[3] =  min_weight;
	parameter[4] = scale;
	parameter[5] = stdev;
	parameter[6] = maxdev;
	parameter[7] = mid;
	parameter[8] = max;
	if (absoluterandom)
		parameter[9] = 1;
	parameter[10] = weightchoice;
	parameter[11] = randchoice;
	send3<string, string, vector<double> >(s(), planarweightSlot, source, destination, parameter);
}

int do_getsyncount( int argc, const char** const argv, Id s )
{
	cout << "getsyncount:: under repair\n";
	/*
	vector <Conn> conn;
	if (argc == 3) {
		Element* src = Id(argv[1])();
		Element* dst = Id(argv[2])();
		if (!(src->className () == "SpikeGen" && dst->className() == "SynChan")) {
			cout << "getsyncount:: The elements' class types are not matching" << endl;
			return 0;
		}
		src->findFinfo("event")->outgoingConns(src, conn);
		unsigned int count = 0;
		for(size_t i = 0; i < conn.size(); i++) {
			if(conn[i].targetElement() == dst)
				count++;
		}
		return count;
	}
	else if (argc == 2) {
		Element* dest = Id(argv[1])();
		if (dest->className() != "SynChan") {
			cout << "getsyncount:: The src element is not of type SynChan." << endl;
			return 0;
		}
		send1 <Id> (s(), 0, getSynCountSlot, dest->id());
		GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
		string value = gpw->getFieldValue();
		int numSynapses = atoi(value.c_str());
		return numSynapses;
	}
	*/
	return 0;
	//send2<string, string>(s(), 0, planarweightSlot, source, dest);
}

char* do_getsynsrc( int argc, const char** const argv, Id s ) {
//getsynsrc <postsynaptic-element> <index>
	cout << "getsynsrc : under repair\n";
	return copyString( "foo" );
	/*
	if (argc != 3) {
		cout << "usage:: getsynsrc <postsynaptic-element> <index>" << endl;
		string s = "";
		return copyString(s.c_str()); 
	}
	vector <Conn> conn;
	string ret = "";
	Element *dst = Id(argv[1])();
	if (dst->className() != "SynChan") {
		cout << "getsynsrc:: The element's type is not matching"<< endl;
		string s = "";
		return copyString(s.c_str()); 
	}
	dst->findFinfo("synapse")->incomingConns(dst, conn);
	unsigned int index = atoi(argv[2]);
	if (index >= conn.size()) {
		assert( 0 )
	}
	Element *src = conn[index].targetElement();
	ret = src->id().path();
	return copyString(ret.c_str());
	*/
}


char* do_getsyndest( int argc, const char** const argv, Id s ) {
//getsynsrc <presynaptic-element> <index>
	cout << "getsynsrc: under repair\n";
	return copyString( "foo" );
	/*
	if (argc != 3) {
		cout << "usage:: getsynsrc <presynaptic-element> <index>" << endl;
		string s = "";
		return copyString(s.c_str()); 
	}
	vector <Conn> conn;
	string ret = "";
	Element *src = Id(argv[1])();
	if (src->className() != "SpikeGen") {
		cout << "getsyndest:: The element's type is not matching"<< endl;
		string s = "";
		return copyString(s.c_str()); 
	}
	src->findFinfo("event")->outgoingConns(src, conn);
	unsigned int index = atoi(argv[2]);
	if (index >= conn.size()) { 
		assert( 0 ) }
	Element *dst = conn[index].targetElement();
	ret = dst->id().path();
	return copyString(ret.c_str());
	*/
}

int do_getsynindex( int argc, const char** const argv, Id s ) {
//getsynsrc <presynaptic-element> <postsynaptic-element> [-number n]
	cout << "getsynsrc: under repair\n";
	return 0;

/*
	if (argc != 3) {
		cout << "usage:: getsynsrc <presynaptic-element> <postsynaptic-element> [-number n]" << endl;
		return -1; 
	}
	vector <Conn> conn;
	Element *src = Id(argv[1])();
	Element *dst = Id(argv[2])();
	if (src == 0 || dst == 0) {
		cout << "Wrong paths!!" << endl;
		return -1;
	}
	if (!(src->className() == "SpikeGen" && dst->className() == "SynChan")) {
		cout << "getsynindex:: The elements' type is not matching"<< endl;
		return 0;
	}
	src->findFinfo("event")->outgoingConns(src, conn);
	for(size_t i = 0; i < conn.size(); i++) {
		if (conn[i].targetElement() == dst) {
			return i;
		}
	}
	return -1;
	*/
}

int do_strcmp(int argc, const char** const argv, Id s ) {
	if (argc != 3) {
		cout << "usage:: strcmp <str1> <str2>" << endl;
		return 0;
	}
	return strcmp(argv[1], argv[2]);
}

int do_strlen(int argc, const char** const argv, Id s ) {
	if (argc != 2) {
		cout << "usage:: strlen <str>" << endl;
		return 0;
	}
	return strlen(argv[1]);
}

char* do_strcat(int argc, const char** const argv, Id s ) {
	if (argc != 3) {
		cout << "usage:: strcat <str1> <str2>" << endl;
		return 0;
	}
	string str1 = argv[1];
	string str2 = argv[2];
	string concat = str1 + str2;
	return copyString(concat.c_str());
}

char* do_substring(int argc, const char** const argv, Id s ) {
	if (argc < 3 || argc > 4) {
		cout << "usage:: substring <str1> <start> <end>" << endl;
		return 0;
	}
	string str = argv[1];
	//check whether argv[2] and argv[3] are in correct format
	size_t start = atoi(argv[2]);
	size_t end = str.size();
	if (argc == 4) 
		end = atoi(argv[3]);
	if (start > end) {
		cout << "You cannot start after end" << endl;
		return 0;
	}
	if (start > str.size() || end > str.size()) {
		cout << "You string has only " << str.size() << " chars"  << endl;
	}
	string substr = str.substr(start, end-start+1);	
	return copyString(substr.c_str());
}

char* do_getpath(int argc, const char** const argv, Id s ) {
	if ( argc != 3 ) {
		cout << "usage:: " << argv[0] << " path -tail -head" << endl;
		return 0;
	}
	bool doHead = 0;
	if ( strncmp( argv[2], "-t", 2 ) == 0 )
		doHead = 0;
	else if ( strncmp( argv[2], "-h", 2 ) == 0 )
		doHead = 1;
	else {
		cout << "usage:: " << argv[0] << " path -tail -head" << endl;
		return 0;
	}
	string path = argv[1];
	string::size_type pos = path.find_last_of( "/" );

	if (pos == string::npos ) {
		if ( doHead ) 
			return copyString( "" );
		else
			return copyString( argv[1] );
	}

	/*
	if ( pos == path.length() - 1 ) {
		if ( doHead ) 
			return copyString( argv[1] );
		else
			return copyString( "" );
	}
	*/
	
	string temp;
	if ( doHead )
		temp = path.substr( 0, pos + 1 );
	else
		temp = path.substr( pos + 1 );

	return copyString( temp.c_str() );
}

 
int do_findchar(int argc, const char** const argv, Id s ) {
	if (argc != 3) {
		cout << "usage:: strcmp <str> <char>" << endl;
		return 0;
	}
	string str1 = argv[1];
	string str2 = argv[2];
	int pos = str1.find(str2, 0); 	
	if ( pos == ( int )( string::npos ) )
		pos = -1;
	return pos;
}
/*
Function: opens file
Question: Where will the file handle go?
*/

void do_openfile(int argc, const char** const argv, Id s) {
	if ( argc != 3 ) {
		cout << "usage:: openfile <filename> <mode>" << endl;
		return;
	}
	string filename = argv[1];
	string mode = argv[2];
	send2<string, string>(s(), openFileSlot, filename, mode);
}

void do_writefile(int argc, const char** const argv, Id s) {
	//writefile <filename> text
	if ( argc < 2 ) {
		cout << "usage: " << argv[0] << 
			" filename [arg1 ...] [-n[onewline] [-f[ormat] format]\n";
		return;
	}
	bool newline = true;
	bool userformat = false;
	string format = "%s";
	int max = argc;
	for (int i = 2; i < argc; i++) {
		if (strncmp(argv[i], "-n", 2) == 0) {
			newline = false;
			if (max > i) 
				max = i;
		}
		if (strncmp(argv[i], "-f", 2) == 0) {
			if (i+1 >= argc) {
				cout << "writefile::format not mentioned." << endl;
				continue;
			}
			userformat = true;
			format = argv[i+1];
			if (max > i)
				max = i;
			i++;
		}
	}
	
	
	string filename = argv[1];
	string text = "";
	for ( int i = 2; i < max; i++ ) { 
		char e[100];
		sprintf(e, format.c_str(), argv[i]);
		text = text + e;
		if (!userformat && i < max - 1)
			text = text + " ";
	}
	if (newline)
		text = text + "\n";
	send2 < string, string > ( s(), writeFileSlot, filename, text );
}


void do_flushfile(int argc, const char** const argv, Id s) {
	if ( argc != 2 ) {
		cout << "usage:: flushfile <filename>" << endl;
	}
	string filename = argv[1];
	send1< string > ( s(), flushFileSlot, filename );
}

void do_closefile(int argc, const char** const argv, Id s) {
	if ( argc != 2 ) {
		cout << "usage:: closefile <filename>" << endl;
	}
	string filename = argv[1];
	send1< string > ( s(), closeFileSlot, filename );
}

void do_listfiles(int argc, const char** const argv, Id s) {
	if ( argc != 1 ) {
		cout << "usage:: openfile <filename> <mode>" << endl;
	}
	send0( s(), listFilesSlot );
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
	cout << gpw->getFieldValue();	
}

string GenesisParserWrapper::getFieldValue() {
	return fieldValue_;
}

char* do_readfile(int argc, const char** const argv, Id s) {
	if ( argc < 2 ) {
		cout << "usage:: readfile <filename> -linemode" << endl;
		return 0;
	}
	string filename = argv[1];
	bool linemode = false;
	if (argc == 3)
		if (strcmp(argv[2], "-linemode") == 0) 
			linemode = true;
	send2< string , bool > ( s(), readFileSlot, filename, linemode );
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
	string text = "" + gpw->getFieldValue();
	return copyString(text.c_str());
}

char* do_getarg( int argc, const char** const argv, Id s ) {
	if ( !( argc >= 2  && strcmp(argv[argc-1], "-count") == 0) &&
		!( argc >= 3  && strcmp(argv[argc-2], "-arg") == 0 ) ) {
		cout << "usage:: getarg <list of argument> <-count OR -arg #>" << endl;
		return 0;
	}
	if ( strcmp(argv[argc-1], "-count") == 0 ) {
		char e[5];
		sprintf(e, "%d", argc-2);
		return strdup(e);
	}
	if ( strcmp(argv[argc-2], "-arg") == 0 ) {
		// check whether argv[argc -1] is a number
		int index = atoi(argv[argc-1]);
		if (index > argc-2) {
			cout << "getarg:: Improper index" << endl;
			return 0;
		}
		return strdup(argv[index]);
	}
	return 0;
}

int do_randseed( int argc, const char** const argv, Id s ) {
	if (argc == 1) {
		return static_cast <int> (mtrand()*294967296);
	}
	if (argc == 2) {
		//check whether argv[1] is an int in string!!
		int seed = atoi(argv[1]);
		//copied the error message from genesis
		cout << "WARNING from init_rng: changing global seed value! Independence of streams is not guaranteed" << endl;
		mtseed(seed);
		return 0;
	}
	cout << "usage:: randseed [seed]" << endl;
	return 0;
}

float do_rand( int argc, const char** const argv, Id s ) {
	if (argc != 3) {
		cout << "usage:: rand <lo> <hi>" << endl;
		return 0;
	}
	//check whether argv[1] and argv[2] are in proper formats
	double lo = atof(argv[1]);
	double hi = atof(argv[2]);
	//return lo + rand()*(hi - lo)/RAND_MAX;
	return lo + mtrand()*(hi - lo);
}

int do_random(int argc, const char** const argv, Id s) {
	if (argc != 1) {
		cout << "usage:: random\n\treturns a random integer uniformly distributed in the range 0, 0x7fffffff." << endl;
		return 0;
	}
	return (long)(genrand_int32()>>1);
} 

void do_disable( int argc, const char** const argv, Id s ) {
	cout << "disable not yet implemented!!" << endl;
}

void do_setup_table2( int argc, const char** const argv, Id s ) {
	cout << "setup_table2 not yet implemented!!" << endl;
}

int do_INSTANTX(int argc, const char** const argv, Id s ) {
	if (argc != 1) {
		cout << "Error:: INSTANTX is a constant" << endl;
	}
	return 1;
}

int do_INSTANTY(int argc, const char** const argv, Id s ) {
	if (argc != 1) {
		cout << "Error:: INSTANTY is a constant" << endl;
	}
	return 2;
}


int do_INSTANTZ(int argc, const char** const argv, Id s ) {
	if (argc != 1) {
		cout << "Error:: INSTANTZ is a constant" << endl;
	}
	return 4;
}

const char* do_VOLT_C1_INDEX(int argc, const char ** const argv, Id s){
    static const char* ret = "VOLT_C1_INDEX";
    if (argc != 1) {
        cout << "Error: VOLT_C1_INDEX is a constant" << endl;
    }
    return copyString(ret);
}

const char* do_VOLT_C2_INDEX(int argc, const char ** const argv, Id s){
    static const char* ret = "VOLT_C2_INDEX";
    if (argc != 1) {
        cout << "Error: VOLT_C1_INDEX is a constant" << endl;
    }
    return copyString(ret);
}

void do_floatformat(int argc, const char** const argv, Id s ) {
	char *format;
	char formtype;
	
	if (argc != 2) {
		cout << "usage::floatformat format-string"<< endl;
		return;
	}
	
	format = strdup(argv[1]);
	formtype = format[strlen(format)-1];

	if (format[0] != '%' || strrchr(format, '%') != format ||
	    (formtype != 'f' && formtype != 'g')) {
		cout << "\texample : floatformat %%0.5g" << endl;
		printf("\tonly f and g formats are allowed\n");
		return;
	}

	set_float_format(format);
}

float do_getstat(int argc, const char** const argv, Id s )
{
	send0( s(), requestCurrentTimeSlot );
	GenesisParserWrapper* gpw = static_cast< GenesisParserWrapper* >
			( s()->data( 0 ) );
	double ret = atof( gpw->getFieldValue().c_str() );	
	return static_cast< float >( ret );
}

void do_showstat(int argc, const char** const argv, Id s )
{
	float time = do_getstat( argc, argv, s );
	cout << "current simulation time = " << time << endl;
}

/**
   prints help on a specied class or a specific field of a class.
*/
void do_help(int argc, const char** const argv, Id s )
{
    if (argc == 1 || argc > 3) {
        cout << "For info on a particular command, type help <commandname>\n"
             << "For info on a particular class, type help <classname> [-full]\n"
             << "For info on a particular field, type help <classname>.<fieldname>"
             << endl;
        
        return;
    }
    string topic(argv[1]);
    string field = "";
    string::size_type field_start = topic.find_first_of(".");
    if ( field_start != string::npos) {
        // May we need to go recursively?
        // Assume for the time being that only one level of field
        // documentation is displayed. No help for channel.xGate.A
        // kind of stuff.
        field = topic.substr(field_start+1); 
        topic = topic.substr(0, field_start);
    }
    else if (argc == 3)
    {
        field = string(argv[2]);
    }
    // check if it is old-style class name
    map< string, string >::iterator sli_iter = sliClassNameConvert().find( topic );
    if ( sli_iter != sliClassNameConvert().end() ) {
        topic = string(sli_iter->second);
    }
    sli_iter = sliFieldNameConvert().find(topic + "." + field);
    if ( sli_iter != sliFieldNameConvert().end() ) {
        field = string(sli_iter->second);
    }
    const string doc = getClassDoc(topic, field);
    if(!doc.empty()) {
        print_help(doc);
        return;
    }
    // fallback to looking for a file with same name in documentation directory
    print_help(getCommandDoc(string(argv[1])));
}

/**
   Run a shell command and return the stdout.
   Note that buffer size is 1KB and only the last 1KB of the output
   is printed.
*/

char* const do_shellcmd(int argc, char** const argv, Id s)
{
	const int buf_size = 1024;
    static char buffer[buf_size];
	#ifndef WIN32
		FILE * cmd_out = NULL;
		if (argc < 2){
			cout << "usage: sh <shell-command> [<command_args>]*" << endl;
			return NULL;
		}
		string commandline;
		for ( int ii = 1; ii < argc; ++ ii){
			commandline.append(argv[ii]);
			commandline.append(" ");
		}
		cmd_out = popen(commandline.c_str(), "r");
		if (!cmd_out){
			return NULL;
		}    
		while (fgets(buffer, buf_size, cmd_out)){
			cout << buffer;
		}
		cout << endl;
		pclose(cmd_out);
		cmd_out = NULL;
	#endif
    return copyString(buffer);
}


char** do_arglist(int argc, const char** const argv, Id s)
{
    if (argc == 1) {
        cout << "usage:: arglist <string>" << endl;
        return NULL;
    }
    string args(argv[1]);
    vector <string> argList;
    separateString(args, argList, " ");
    char **output = (char**)calloc(argList.size() + 1, sizeof(char*));
    char **ptr = output;
    for (vector <string>::iterator ii = argList.begin(); ii != argList.end(); ++ii) {
        if (trim(*ii).length() > 0) {
            *ptr = copyString(ii->c_str());
            ++ptr;
        }        
    }

    return output;
}
void do_readsbml( int argc, const char** const argv, Id s )
{
	if (argc != 3 ) {
		cout << "usage::readSBML filename /kinetics \n";
		return;
	}
	string modelpath=argv[2];
	int childNode = parseNodeNum( modelpath );
	string filename=argv[1];
 	send3< string, string, int >(s(), readSbmlSlot, filename, modelpath, childNode );
}

void do_writesbml( int argc, const char** const argv, Id s )
{
	if (argc != 3 ) {
		cout << "usage::writeSBML filename /kinetics \n";
		return;
	}
	string writeloc=argv[2];
	int childNode = parseNodeNum( writeloc );
	string filename=argv[1];
 	send3< string, string, int >(s(), writeSbmlSlot, filename, writeloc, childNode );
}
void do_readneuroml( int argc, const char** const argv, Id s )
{
	if (argc != 3 ) {
		cout << "usage::readNeuroML filename /model \n";
		return;
	}
	string modelpath=argv[2];
	int childNode = parseNodeNum( modelpath );
	string filename=argv[1];
 	send3< string, string, int >(s(), readNeuromlSlot, filename, modelpath, childNode );
}

void do_writeneuroml( int argc, const char** const argv, Id s )
{
	if (argc != 3 ) {
		cout << "usage::writeNeuroML filename /model \n";
		return;
	}
	string writeloc=argv[2];
	int childNode = parseNodeNum( writeloc );
	string filename=argv[1];
 	send3< string, string, int >(s(), writeNeuromlSlot, filename, writeloc, childNode );
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
        AddFunc( "exit", do_quit, "void" );
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
	AddFunc( "file2tab", do_file2tab, "void" );
	AddFunc( "el",
			reinterpret_cast< slifunc >( do_element_list ), "char**" );
	AddFunc( "element_list",
			reinterpret_cast< slifunc >( do_element_list ), "char*" );
	AddFunc( "addtask", do_addtask, "void" );

	AddFunc( "readcell", do_readcell, "void" );

	AddFunc( "setupalpha", do_setupalpha, "void" );
	AddFunc( "setup_tabchan", do_setupalpha, "void" );
	AddFunc( "setuptau", do_setuptau, "void" );
	AddFunc( "tweakalpha", do_tweakalpha, "void" );
	AddFunc( "tweaktau", do_tweaktau, "void" );
	AddFunc( "setupgate", do_setupgate, "void" );

	AddFunc( "simdump", doWriteDump, "void" );
	AddFunc( "simundump", doSimUndump, "void" );
	AddFunc( "simobjdump", doSimObjDump, "void" );
	AddFunc( "readdump", doReadDump, "void" ); // New command: not in GENESIS

	AddFunc( "argv", 
		reinterpret_cast< slifunc >( doArgv ), "char**" );
	AddFunc( "argc", 
		reinterpret_cast< slifunc >( doArgc ), "int" );

	AddFunc( "loadtab", do_loadtab, "void" );
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
	AddFunc( "version", reinterpret_cast< slifunc>( do_version ), "float");
	AddFunc( "revision", reinterpret_cast< slifunc>( do_revision ), "char*");
	AddFunc( "xshow", do_xshow, "void" );
	AddFunc( "xhide", do_xhide, "void" );
	AddFunc( "xshowontop", do_xshowontop, "void" );
	AddFunc( "xupdate", do_xupdate, "void" );
	AddFunc( "xcolorscale", do_xcolorscale, "void" );
	AddFunc( "x1setuphighlight", do_x1setuphighlight, "void" );
	AddFunc( "xsendevent", do_xsendevent, "void" );
	AddFunc( "createmap", do_createmap, "void" );
	AddFunc( "planarconnect", do_planarconnect, "void" );
	AddFunc( "planardelay", do_planardelay, "void" );
	AddFunc( "planarweight", do_planarweight, "void" );
	AddFunc( "getsyncount", reinterpret_cast< slifunc >(do_getsyncount), "int" );
	AddFunc( "getsynsrc", reinterpret_cast< slifunc >(do_getsynsrc), "char*" );
	AddFunc( "getsyndest", reinterpret_cast< slifunc >(do_getsyndest), "char*" );
	AddFunc( "getsynindex", reinterpret_cast< slifunc >(do_getsynindex), "int" );
	AddFunc( "strcmp", reinterpret_cast< slifunc >(do_strcmp), "int" );
	AddFunc( "strlen", reinterpret_cast< slifunc >(do_strlen), "int" );
	AddFunc( "strcat", reinterpret_cast< slifunc >(do_strcat), "char*" );
	AddFunc( "substring", reinterpret_cast< slifunc >(do_substring), "char*" );
	AddFunc( "getpath", reinterpret_cast< slifunc >(do_getpath), "char*" );
	
	AddFunc( "findchar", reinterpret_cast< slifunc >(do_findchar), "int" );
	AddFunc( "getelementlist", reinterpret_cast< slifunc >(do_element_list ), "char*");
	AddFunc( "openfile", do_openfile, "void" );
	AddFunc( "writefile", do_writefile, "void" );
	AddFunc( "flushfile", do_flushfile, "void" );
	AddFunc( "readfile", reinterpret_cast< slifunc >( do_readfile ), "char*");
	AddFunc( "listfiles", do_listfiles, "void" );
	AddFunc( "closefile", do_closefile, "void" );
	AddFunc( "getarg", reinterpret_cast< slifunc >( do_getarg ), "char*" );
	AddFunc( "randseed", reinterpret_cast< slifunc >( do_randseed ), "int" );
	AddFunc( "rand", reinterpret_cast< slifunc >( do_rand ), "float" );
	AddFunc( "random", reinterpret_cast< slifunc >( do_random ), "int" );
	AddFunc( "xps", do_xps, "void" );
	AddFunc( "disable", do_disable, "void" );
	AddFunc( "setup_table2", do_setup_table2, "void" );
	AddFunc( "INSTANTX", reinterpret_cast< slifunc > ( do_INSTANTX ), "int" );
	AddFunc( "INSTANTY", reinterpret_cast< slifunc > ( do_INSTANTY ), "int" );
	AddFunc( "INSTANTZ", reinterpret_cast< slifunc > ( do_INSTANTZ ), "int" );
        AddFunc( "VOLT_C1_INDEX",  reinterpret_cast< slifunc > ( do_VOLT_C1_INDEX ), "char*");
        AddFunc( "VOLT_C2_INDEX",  reinterpret_cast< slifunc > ( do_VOLT_C2_INDEX ), "char*");
	AddFunc( "floatformat", do_floatformat, "void" );
	AddFunc( "getstat", reinterpret_cast< slifunc > ( do_getstat ), "float" );
	AddFunc( "showstat", do_showstat, "void" );
	AddFunc( "help", do_help, "void" );
        AddFunc( "arglist", reinterpret_cast< slifunc > ( do_arglist ), "char**");
	AddFunc( "readSBML", reinterpret_cast< slifunc > ( do_readsbml ), "void" );
	AddFunc( "writeSBML", reinterpret_cast< slifunc > ( do_writesbml ), "void" );
	AddFunc( "readNeuroML", reinterpret_cast< slifunc > ( do_readneuroml ), "void" );
	AddFunc( "writeNeuroML", reinterpret_cast< slifunc > ( do_writeneuroml ), "void" );
        AddFunc( "sh", reinterpret_cast< slifunc > (do_shellcmd ), "char*" );
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
		send0( g(), 0, requestCweSlot );
		return cwe_;
	}

	if ( path == "^" )
		return createdElm_;

	if ( path == ".." ) {
		send0( g(), 0, requestCweSlot );
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
		send0( g(), 0, requestCweSlot );
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
/*
Element* GenesisParserWrapper::getShell( Id g )
{
	/// \todo: shift connDestBegin function to base Element class.
	SimpleElement* e = dynamic_cast< SimpleElement *>( g() );
	assert ( e != 0 && e != Element::root() );
	vector< Conn >::const_iterator i = e->connDestBegin( 3 );
	Element* ret = i->targetElement();
	return ret;
}
*/

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
	Id shellId = Id::shellId();
	assert( !shellId.bad() );
	Element* sli = Neutral::create( "GenesisParser", "sli", shellId, Id::initId() );
	sli->id().setGlobal();
	static_cast< GenesisParserWrapper* >( sli->data( 0 ) )->
		setElement( sli->id() );

	bool ret = shellId.eref().add( "parser", sli, "parser", 
		ConnTainer::Default );
	assert( ret );

#ifdef DO_UNIT_TESTS
	if ( Shell::myNode() == 0 )
		static_cast< GenesisParserWrapper* >( sli->data( 0 ) )->unitTest();
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
#ifdef USE_MUSIC
	string lestr = "shell sched library proto music ";
#else
	string lestr = "shell sched library proto ";
#endif // USE_MUSIC
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

	gpAssert( "create neutral /hop", "" );
	gpAssert( "le", "bar " );
	gpAssert( "le /foo", "bar " );
	gpAssert( "le ..", lestr + "foo hop " );
	gpAssert( "delete /hop", "" );
	gpAssert( "le ..", lestr + "foo " );

	gpAssert( "ce bar", "" );
	gpAssert( "pwe", "/foo/bar " );
	gpAssert( "ce ../..", "" );
	// gpAssert( "le", "sched shell foo " );
	gpAssert( "le", lestr + "foo " );
	gpAssert( "delete /foo", "" );
	gpAssert( "le", lestr );
	//gpAssert( "le /foo", "cannot find object '/foo' " );
	gpAssert( "echo foo", "foo " );

	// Testing backslash carryover of lines
	gpAssert( "echo foo \\\n", "" );
	gpAssert( "bar", "foo bar " );

	// Testing noline
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
	gpAssert( "le /sched/cj", "t0 t1 " );
//	Autoscheduling causes solver to spawn here
//	gpAssert( "setclock 1 0.1", "" );
	// gpAssert( "le /sched/cj", "t0 t1 t2 t3 t4 t5 " );
	gpAssert( "echo {getfield /sched/cj/t0 dt}", "1 " );
//	Autoscheduling interferes with these tests
//	gpAssert( "echo {getfield /sched/cj/t1 dt}", "0.1 " );
//	gpAssert( "useclock /##[TYPE=Compartment] 1", "" );
	gpAssert( "delete /compt", "" );
	gpAssert( "le", lestr );
	//	gpAssert( "echo {strcmp \"hello\" \"hello\"} ", "0 " );
	//	gpAssert( "echo {strcmp \"hello\" \"hell\"} ", "1 " );
	//	gpAssert( "echo {strcmp \"hell\" \"hello\"} ", "-1 " );

	gpAssert( "echo {getpath /foo/bar -tail}", "bar " );
	gpAssert( "echo {getpath /foo/bar -head}", "/foo/ " );
	gpAssert( "echo {getpath foo -tail}", "foo " );
	gpAssert( "echo {getpath foo -head}", " " );
	gpAssert( "echo {getpath /foo -tail}", "foo " );
	gpAssert( "echo {getpath /foo -head}", "/ " );
	gpAssert( "echo {getpath /foo/ -tail}", " " );
	gpAssert( "echo {getpath /foo/ -head}", "/foo/ " );

	// Checking pushe/pope
//	gpAssert( "pushe /proto", "/proto " );
	gpAssert( "pushe /proto", "" );
	gpAssert( "pwe", "/proto " );
//	gpAssert( "pope", "/ " );
	gpAssert( "pope", "" );
	gpAssert( "pwe", "/ " );
//	gpAssert( "pushe /foobarzod", "Error - cannot change to '/foobarzod' " );
	gpAssert( "pwe", "/ " );

	// Checking copy syntax
	gpAssert( "create neutral /a", "" );
	gpAssert( "create neutral /a/b", "" );
	gpAssert( "create neutral /c", "" );
	gpAssert( "copy /a/b /c/d", "" );
	gpAssert( "le /c", "d " );
	gpAssert( "copy /a/b /c/d", "" );
	gpAssert( "le /c", "d " );
	gpAssert( "le /c/d", "b " );
	gpAssert( "copy /a/b c", "" );
	gpAssert( "le /c", "d b " );

	gpAssert( "ce /a", "" );
	gpAssert( "le", "b " );
	gpAssert( "copy b q", "" );
	gpAssert( "le", "b q " );
	gpAssert( "ce /", "" );
	gpAssert( "delete /a", "" );
	gpAssert( "delete /c", "" );
	
	cout << "\n" << flush;
}
#endif
