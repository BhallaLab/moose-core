
#include "moose.h"
#include <math.h>
#include <string>
#include <setjmp.h>
#include <FlexLexer.h>
#include "script.h"

#include "../shell/Shell.h"
#include "GenesisParser.h"
#include "GenesisParserWrapper.h"
#include "../element/Neutral.h"
#include "func_externs.h"

#include "ParGenesisParser.h"

using namespace std;

const Cinfo* initParGenesisParserCinfo()
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
		new SrcFinfo( "createArray",
				Ftype4< string, string, Id, vector <double> >::global() ),
		new SrcFinfo( "planarconnect", Ftype3< string, string, double >::global() ),
		new SrcFinfo( "planardelay", Ftype2< string, double >::global() ),
		new SrcFinfo( "planarweight", Ftype2< string, double >::global() ),
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
		new SrcFinfo( "copyIntoArray", Ftype4< Id, Id, string, vector <double> >::global() ),
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
		new SrcFinfo( "openfile", 
			Ftype2< string, string >::global() ),
		new SrcFinfo( "writefile", 
			Ftype2< string, string >::global() ),
		new SrcFinfo( "listfiles", 
			Ftype0::global() ),
		new SrcFinfo( "closefile", 
			Ftype1< string >::global() ),
		new SrcFinfo( "readfile", 
			Ftype2< string, bool >::global() ),
		///////////////////////////////////////////////////////////////
		// Setting field values for a vector of objects
		///////////////////////////////////////////////////////////////
		// Setting a vec of field values as a string: send out request:
		new SrcFinfo( "setVecField", // object, field, value 
			Ftype3< vector< Id >, string, string >::global() ),
		new SrcFinfo( "loadtab", 
			Ftype1< string >::global() ),
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

	static Cinfo parGenesisParserCinfo(
		"ParGenesisParser",
		"Mayuresh Kulkarni, CRL, 2007",
		"Parallel version of Genesis Parser",
		initNeutralCinfo(),
		genesisParserFinfos,
		sizeof(genesisParserFinfos) / sizeof( Finfo* ),
		ValueFtype1< ParGenesisParserWrapper >::global()
	);

	return &parGenesisParserCinfo;
}

static const Cinfo* parGenesisParserCinfo = initParGenesisParserCinfo();
static const unsigned int planarconnectSlot = 
	initParGenesisParserCinfo()->getSlotIndex( "parser.planarconnect" );

ParGenesisParserWrapper::ParGenesisParserWrapper()
{
	loadBuiltinCommands();
}


void ParGenesisParserWrapper::MyFunc()
{
	cout<<endl<<"Reached Planarconnect";
}


void do_setrank( int argc, const char** const argv, Id s )
{
	cout<<endl<<"In do_setrank";
}

void do_parplanarconnect( int argc, const char** const argv, Id s )
{
	cout<<endl<<"In do_planarconnect";
	string source, dest;
	source = argv[1];
	dest = argv[2];
	//Shell::planarconnect(conn, source, dest, probability)
	send3<string, string, double>(s(), planarconnectSlot, source, dest, 0.5);

}

void ParGenesisParserWrapper::loadBuiltinCommands()
{
	AddFunc( "setrank", do_setrank, "void" );
	AddFunc( "planarconnect", do_parplanarconnect, "void");
}

