
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

#include <mpi.h>

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
		new SrcFinfo( "add", Ftype2<Id, string>::global() ),
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
		new SrcFinfo( "requestCurrentTime", Ftype0::global() ),
		// Returns time in the default return value.
		
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
			RFCAST( &ParGenesisParserWrapper::parseFunc ) ), 
		new SrcFinfo( "echo", Ftype1< string>::global() ),

	};

	static Cinfo genesisParserCinfo(
		"ParGenesisParser",
		"Mayuresh Kulkarni, CRL, 2007",
		"Parallel version of Genesis Parser",
		initNeutralCinfo(),
		genesisParserFinfos,
		sizeof(genesisParserFinfos) / sizeof( Finfo* ),
		ValueFtype1< ParGenesisParserWrapper >::global()
	);
	return &genesisParserCinfo;
}



 




static const Cinfo* parGenesisParserCinfo = initParGenesisParserCinfo();
static const unsigned int planarconnectSlot = initParGenesisParserCinfo()->getSlotIndex( "parser.planarconnect" );

static	int	arrSpikegenConnections[MAX_MPI_PROCESSES][MAX_MPI_PROCESSES];
static	int	arrSynchanConnections[MAX_MPI_PROCESSES][MAX_MPI_PROCESSES];

ParGenesisParserWrapper::ParGenesisParserWrapper()
{
	loadBuiltinCommands();
	sendrank_ = 0;
}

void do_setrank( int argc, const char** const argv, Id s )
{
}

void do_parquit( int argc, const char** const argv, Id s )
{
		MPI_Finalize();
		exit( 0 );
}

void do_parplanarconnect( int argc, const char** const argv, Id s )
{
	int j,iMyRank;

	string source, dest;
	string spikegenrank, synchanrank;
	char arrTemp[10];

	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);

	source = argv[1];
	dest = argv[2];

        for(j=0; 0 != arrSpikegenConnections[iMyRank][j]; j++)
        {
                 sprintf(arrTemp, "%d", arrSpikegenConnections[iMyRank][j]);
		 spikegenrank += arrTemp;
		 spikegenrank += "|";
        }


        for(j=0; 0 != arrSynchanConnections[iMyRank][j]; j++)
        {
		sprintf(arrTemp, "%d", arrSynchanConnections[iMyRank][j]);
		synchanrank += arrTemp;
		synchanrank += "|";
        }

	send4<string, string, string, string>(s(), planarconnectSlot, source, dest, spikegenrank, synchanrank);

}

bool ParGenesisParserWrapper::checkUnique(int randomVar, int spikeIndex)
{
        int i;
        for(i = 0; 0 != arrSpikegenConnections[spikeIndex][i]; i++) // check for duplication
        {
                if(arrSpikegenConnections[spikeIndex][i] == randomVar)
                {
                        return false;
                }
        }

        return true;
}


void ParGenesisParserWrapper::loadBuiltinCommands()
{
	AddFunc( "setrank", do_setrank, "void");
	AddFunc( "planarconnect", do_parplanarconnect, "void");
	AddFunc( "quit", do_parquit, "void");
}

void ParGenesisParserWrapper::parseFunc( const Conn& c, string s )
{
	ParGenesisParserWrapper &objParParser = *(static_cast< ParGenesisParserWrapper* >( c.targetElement()->data() ));
	int i;
	char **pArgs;

        MPI_Comm_rank(MPI_COMM_WORLD, &objParParser.processrank_);
        MPI_Comm_size(MPI_COMM_WORLD, &objParParser.processcount_);

	if(objParParser.processrank_ == 0)
	{
		objParParser.ParseInput( s );
	}
	else
	{
		stCommand &objCommand = static_cast< ParGenesisParserWrapper* >( c.targetElement()->data() )->objCommand_;
		
		while(1)
		{
			MPI_Bcast(&objCommand, sizeof(int) + MAX_COMMAND_SIZE * MAX_COMMAND_ARGS * sizeof(char), MPI_CHAR, 0, MPI_COMM_WORLD);

			if(objCommand.iRank != 0 && objCommand.iRank != objParParser.processrank_)
			{
				//Ignore command if it is not for the current rank
				continue;
			}

			pArgs = new char* [objCommand.iSize];

			for(i=0; i<objCommand.iSize; i++)
			{
                                pArgs[i] = new char [strlen(objCommand.arrCommand[i])+1];
                                strcpy(pArgs[i], objCommand.arrCommand[i]);

			}

			if(!strcmp(objCommand.arrCommand[0], "planarconnect"))
			{
				static_cast< ParGenesisParserWrapper* >( c.targetElement()->data() )->BCastConnections();
			}

			if(!strcmp(objCommand.arrCommand[0], "step"))
			{
				// Execute Barrier before executing step command
				//cout<<endl<<"Executing barrier at rank: "<<objParParser.processrank_<<endl<<flush;
				MPI_Barrier(MPI_COMM_WORLD);		
			}


			objParParser.ExecuteCommand(objCommand.iSize, pArgs);

                        for(i=0; i<objCommand.iSize; i++)
                        {
                                delete pArgs[i];
                        }

                        delete[] pArgs;

			if(!strcmp(objCommand.arrCommand[0], "quit"))
				break;
		}
	}
}

void ParGenesisParserWrapper::generateRandomConnections()
{
        int connectionCount;
        int i,j,k;
        bool bValid = false;
        int randomVar;


        memset(arrSpikegenConnections, 0, MAX_MPI_PROCESSES * MAX_MPI_PROCESSES);
        memset(arrSynchanConnections, 0, MAX_MPI_PROCESSES * MAX_MPI_PROCESSES);


        for(i=1; i<processcount_; i++)
        {
                bValid = false;
                while(bValid == false)
                {
                      connectionCount = rand()%processcount_;
                      if(connectionCount != 0)
                                bValid = true;
                }

                for(j=0; j<connectionCount; j++)
                {
                        bValid = false;
                        while(bValid == false)
                        {
                                randomVar = rand()%processcount_;
                                if(randomVar != 0 && randomVar != i)
                                {
                                        bValid = true;
                                }
                        }

                        if(checkUnique(randomVar, i) == true)
                        {
                                arrSpikegenConnections[i][j] = randomVar;

                                for(k=0; 0 != arrSynchanConnections[randomVar][k]; k++);

                                arrSynchanConnections[randomVar][k] = i;
                        }
                }
        }
	
	BCastConnections();
	
        /*cout<<endl<<"Planarconnect arguments";
        for(i=1; i < processcount_; i++)
        {
                cout<<endl<<" For Process "<<i;

                strcpy(arrArgs[argc], "");
                for(j=0; 0 != arrSpikegenConnections[i][j]; j++)
                {
                        sprintf(szRandomNumber, "%d", arrSpikegenConnections[i][j]);
                        strcat(arrArgs[argc], szRandomNumber );
                        strcat(arrArgs[argc], "|");
                }

                strcpy(arrArgs[argc+1], "");
                for(j=0; 0 != arrSynchanConnections[i][j]; j++)
                {
                        sprintf(szRandomNumber, "%d", arrSynchanConnections[i][j]);
                        strcat(arrArgs[argc+1], szRandomNumber );
                        strcat(arrArgs[argc+1], "|");
                }

                for(j=0; j < argc+2; j++)
                {
                        cout<<endl<<arrArgs[j]<<flush;
                }

                SendCommand(argc+2);
        }*/

}

bool ParGenesisParserWrapper::RootCommand(char **argv)
{
	if(
		!strcmp("abs", argv[0]) ||
		!strcmp("exp", argv[0]) ||
		!strcmp("log", argv[0]) ||
		!strcmp("log0", argv[0]) ||
		!strcmp("sin", argv[0]) ||
		!strcmp("cos", argv[0]) ||
		!strcmp("tan", argv[0]) ||
		!strcmp("sqrt", argv[0]) ||
		!strcmp("pow", argv[0])
	  )
	{
		return true;
	}
	else if(!strcmp("setrank", argv[0]))
	{
		if(argv[1] == NULL)
			cout<<endl<<"Error: Missing argument to setrank"<<endl<<flush;
		else
		{
			if(atoi(argv[1]) >= processcount_)
				cout<<endl<<"Error: Invalid rank value passed as argument to setrank"<<endl<<flush;
			else
				sendrank_ = atoi(argv[1]);
		}

		return true;
	}

	return false;
}


Result ParGenesisParserWrapper::ExecuteCommand(int argc, char** argv)
{
	FILE            *pfile;
	// int          code;
	short           redirect_mode = 0;
	int             start = 0;
	int             i;
	char            *mode = "a";
	Result          result;
	// int          ival;
	func_entry      *command;

	//Result	result;
	result.r_type = IntType();
	result.r.r_int = 0;
 
	if(argc < 1)
	{
		cout<<endl<<"Error: number of arguments less than 1"<<endl;
		return(result);
	}

	if(processrank_ == 0 && RootCommand(argv) ==false )
	{
		if(argc > MAX_COMMAND_ARGS)
		{
			cout<<endl<<"Error: Max command arguments exceed "<<MAX_COMMAND_ARGS<<endl;
			return result;
		}

		objCommand_.clear();	
		objCommand_.iSize = argc;
		objCommand_.iRank = sendrank_;
		for(int i=0; i<argc; i++)
		{
			strcpy(objCommand_.arrCommand[i], argv[i]);
		}

		SendCommand(argc);
		
		if(!strcmp(objCommand_.arrCommand[0], "step"))
		{
			//Execute Barrier at step command
			//cout<<endl<<"Executing barrier at rank: "<<processrank_<<endl<<flush;
			MPI_Barrier(MPI_COMM_WORLD);
		}

		if(!strcmp(objCommand_.arrCommand[0], "planarconnect"))
		{
			generateRandomConnections();
		}

		if(!strcmp(argv[0], "quit"))
		{
			MPI_Finalize();
			exit(0);
		}
	}
	else
	{
		command = GetCommand(argv[0]);


		if (command && command->HasFunc() ) {
		/*
		** check through the arg list for stdout
		** redirection
		*/
		for(i=0;i<argc;i++){
		    /*
		    ** check for stdout redirection
		    */
		    if(i+1 < argc && ((strcmp(argv[i],"|") == 0) ||
		   (strcmp(argv[i],">") == 0) ||
		   (strcmp(argv[i],">>") == 0))){
			start = i+1;
			if(strcmp(argv[i],"|") == 0){
			    /*
			    ** pipe
			    */
			    redirect_mode = 1;
			    mode = "w";
			}
			if(strcmp(argv[i],">") == 0){
			    /*
			    ** redirect stdout to a file
			    */
			    redirect_mode = 2;
			    mode = "w";
			}
			if(strcmp(argv[i],">>") == 0){
			    /*
			    ** append stdout to a file
			    */
			    redirect_mode = 2;
			    mode = "a";
			}
			break;
		    }
		}
		if(redirect_mode){
			cerr << "Error: Redirection not yet working in MOOSE\n";
		// Here we have a lot of OS-specific stuff. Will deal with later.
	#if 0
		    FILE *fp = NULL;
		    FILE *output;
		    int   savefd;
		    int   stdoutfd = fileno(stdout);

		    normal_tty();
		    /*
		    ** save the stdout FILE structure
		    */
		    switch(redirect_mode){
		    case 1:
			/*
			** open up the pipe
			*/
			fp = G_popen(ArgListToString(argc-start,argv+start),mode);
			break;
		    case 2:
			/*
			** open up the file
			*/
			fp = fopen(argv[start],mode);
			break;
		    }
		    if (fp == NULL){ /* G_popen or fopen failed!!! */
			genesis_tty();
			perror(argv[start]);
			return(result);
		    }
		    /*
		    ** flush the existing data in stdout
		    */
		    fflush(stdout);
		    savefd = dup(stdoutfd);
		    close(stdoutfd);
		    dup(fileno(fp));
		    /*
		    ** call the function
		    */
			command->Execute(start - 1, argv);
		    // func(start-1,argv);
		    /*
		    ** flush the output and close it
		    */
		    fflush(stdout);
		    close(stdoutfd);
		    switch(redirect_mode){
		    case 1:
			G_pclose(fp);
			break;
		    case 2:
			fclose(fp);
			break;
		    }
		    /*
		    ** restore the stdout file structure
		    */
		    dup(savefd);
		    close(savefd);
		    genesis_tty();
	#endif
		} else
		/*
		** call the function
		*/
		return command->Execute( argc, (const char**)argv, element_ );

	    } else 
	    /*
	    ** is it a simulator shell script?
	    */
	    if(ValidScript(pfile = SearchForScript(argv[0],"r"))){
		AddScript(NULL, pfile, argc, argv, FILE_TYPE);
	    } else {
		/*
		** if the function is not locally defined then
		** check to see if system functions should be
		** tried
		*/
		// May 2004. USB.
		// For MOOSE, we cannot permit system calls because they are not
		// cross-platform
	/*
		if(Autoshell()){
		    normal_tty();
		    if((code = ExecFork(argc,argv)) != 0){
			Error();
			printf("code %d\n",code);
		    };
		    genesis_tty();
		    result.r_type = IntType();
		    result.r.r_int = code;
		    return(result);
		} else
	*/
		cerr << "undefined function " << argv[0] << std::endl;
	    }
	    result.r_type = IntType();
	    result.r.r_int = 0;
	    return(result);	

	    //GenesisParserWrapper::ExecuteCommand(argc, argv);

	}
	
	return result;
}

int ParGenesisParserWrapper::SendCommand(int argc)
{
	MPI_Bcast(&objCommand_, sizeof(int) + MAX_COMMAND_SIZE * MAX_COMMAND_ARGS * sizeof(char), MPI_CHAR, 0, MPI_COMM_WORLD);
	return 0;
}

void ParGenesisParserWrapper::BCastConnections()
{
	//int i,j;

	MPI_Bcast(arrSpikegenConnections, MAX_MPI_PROCESSES * MAX_MPI_PROCESSES , MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(arrSynchanConnections, MAX_MPI_PROCESSES * MAX_MPI_PROCESSES, MPI_INT, 0, MPI_COMM_WORLD);

        /*for(i=1; i < processcount_; i++)
        {
                cout<<endl<<" "<<i<<": ";

                for(j=0; 0 != arrSpikegenConnections[i][j]; j++)
                {
                        cout<<arrSpikegenConnections[i][j]<<" , ";
                }

                cout<<" ---- ";

                for(j=0; 0 != arrSynchanConnections[i][j]; j++)
                {
                        cout<<arrSynchanConnections[i][j]<<" , ";
                }
        }*/
}

