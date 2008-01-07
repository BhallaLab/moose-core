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
#include "ParShell.h"

//////////////////////////////////////////////////////////////////////
// Shell MOOSE object creation stuff
//////////////////////////////////////////////////////////////////////

const Cinfo* initParShellCinfo()
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
				RFCAST( &ParShell::planarconnect ) ),
		new DestFinfo( "planardelay",
				Ftype2< string, double >::global(),
				RFCAST( &ParShell::planardelay ) ),
		new DestFinfo( "planarweight",
				Ftype2< string, double >::global(),
				RFCAST( &ParShell::planarweight ) ),
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
		new SrcFinfo( "getSrc", Ftype1< string >::global(), 0 ),

		// Setting a field value as a string: handling request
		new DestFinfo( "set",
				Ftype3< Id, string, string >::global(),
				RFCAST( &Shell::setField ) ),

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
		"ParShell",
		"Mayuresh Kulkarni, CRL",
		"Parallel version of Shell object",
		initNeutralCinfo(),
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		ValueFtype1< ParShell >::global()
	);

	return &shellCinfo;
}



static const Cinfo* shellCinfo = initParShellCinfo();

ParShell::ParShell()
{
}


void ParShell::planarconnect(const Conn& c, string source, string dest, string spikegenRank, string synchanRank)
{
        int next, previous;
        bool ret;

        Id spkId(source);
        Id synId(dest);

        Element *eSpkGen = spkId();
        Element *eSynChan = synId();

        previous = 0;
        while(1)
        {
                next = spikegenRank.find('|', previous);
                if(next == -1)
                        break;


                ret = set< int >( eSpkGen, "sendRank", atoi(spikegenRank.substr(previous, next-previous).c_str()) );
                previous = next+1;
        }

        previous = 0;
        while(1)
        {
                next = synchanRank.find('|', previous);

                if(next == -1)
                        break;

                ret = set< int >( eSynChan, "recvRank", atoi(synchanRank.substr(previous, next-previous).c_str()) );
                previous = next+1;
        }

}

void ParShell::planardelay(const Conn& c, string source, double delay){
	vector <Element* > src_list;
	simpleWildcardFind( source, src_list );
	for (size_t i = 0 ; i < src_list.size(); i++){
		if (src_list[i]->className() != "ParSynChan"){cout<<"ParShell::planardelay: error!!";}
		unsigned int numSynapses;
		bool ret;
		ret = get< unsigned int >( src_list[i], "numSynapses", numSynapses );
		if (!ret) {cout << "error" <<endl;}
		for (size_t j = 0 ; j < numSynapses; j++){
			lookupSet< double, unsigned int >( src_list[i], "delay", delay, j );
		}
	}
}

void ParShell::planarweight(const Conn& c, string source, double weight){
	vector <Element* > src_list;
	simpleWildcardFind( source, src_list );
	for (size_t i = 0 ; i < src_list.size(); i++){
		if (src_list[i]->className() != "ParSynChan"){cout<<"ParShell::planarweight: error";}
		unsigned int numSynapses;
		bool ret;
		ret = get< unsigned int >( src_list[i], "numSynapses", numSynapses );
		if (!ret) {/*error!*/}
		for (size_t j = 0 ; j < numSynapses; j++){
			lookupSet< double, unsigned int >( src_list[i], "weight", weight, j );
		}
	}
}

