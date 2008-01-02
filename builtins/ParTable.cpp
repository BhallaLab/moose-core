/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include <mpi.h>

#include "moose.h"
#include "Interpol.h"
#include "Table.h"
#include "ParTable.h"

/**
 * This is a parallel version of the table class.
 * The generated output is sent back to the root process
 *
 */

const Cinfo* initParTableCinfo()
{
	/** 
	 * This is a shared message to receive Process message from
	 * the scheduler. 
	 */
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
		RFCAST( &ParTable::process ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
		RFCAST( &ParTable::reinit ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	/*
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
				RFCAST( &Table::process ) ),
	    TypeFuncPair( Ftype1< ProcInfo >::global(),
				RFCAST( &Table::reinit ) ),
	};
	*/

	/** 
	 * This is a shared message to request and handle value
	 * messages from fields.
	 */
	static Finfo* inputRequestShared[] =
	{
			// Sends out the request. Issued from the process call.
		new SrcFinfo( "requestInput", Ftype0::global() ),
			// Handle the returned value.
	    new DestFinfo( "handleInput", Ftype1< double >::global(),
				RFCAST( &Table::setInput ) ),
	};

	/*
	static TypeFuncPair inputRequestTypes[] =
	{
			// Sends out the request. Issued from the process call.
		TypeFuncPair( Ftype0::global(), 0 ),
			// Handle the returned value.
	    TypeFuncPair( Ftype1< double >::global(),
				RFCAST( &Table::setInput ) ),
	};
	*/

	static Finfo* tableFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "index", ValueFtype1< int >::global(),
			GFCAST( &ParTable::getIndex ),
			RFCAST( &ParTable::setIndex )
		),
		new ValueFinfo( "input", ValueFtype1< double >::global(),
			GFCAST( &Table::getInput ),
			RFCAST( &Table::setInput )
		),
		new ValueFinfo( "output", ValueFtype1< double >::global(),
			GFCAST( &Table::getOutput ),
			RFCAST( &Table::setOutput )
		),
		new ValueFinfo( "step_mode", ValueFtype1< int >::global(),
			GFCAST( &Table::getStepMode ),
			RFCAST( &Table::setStepMode )
		),
		// Paste over silly old GENESIS inconsistency in naming.
		new ValueFinfo( "stepmode", ValueFtype1< int >::global(),
			GFCAST( &Table::getStepMode ),
			RFCAST( &Table::setStepMode )
		),
		new ValueFinfo( "stepsize", ValueFtype1< double >::global(),
			GFCAST( &Table::getStepsize ),
			RFCAST( &Table::setStepsize )
		),
		new ValueFinfo( "threshold", ValueFtype1< double >::global(),
			GFCAST( &Table::getStepsize ),
			RFCAST( &Table::setStepsize )
		),
		new LookupFinfo( "tableLookup",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Table::getLookup ),
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "inputRequest", inputRequestShared, 
			sizeof( inputRequestShared ) / sizeof( Finfo* ) ),
		/*
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "inputRequest", inputRequestTypes, 2 ),
		*/
		
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		/// Sends the output value every timestep.
		new SrcFinfo( "outputSrc", Ftype1< double >::global() ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		
		/**
		 * Look up and interpolate value from table using x value.
		 * Put result in output, and also send out on outputSrc.
		 */
		new DestFinfo( "msgInput", Ftype1< double >::global(), 
			RFCAST( &Table::setInput )
		),

		/**
		 * Sum this value onto the output field.
		 */
		new DestFinfo( "sum", Ftype1< double >::global(), 
			RFCAST( &Table::sum )
		),

		/**
		 * Multipy this value into the output field.
		 */
		new DestFinfo( "prd", Ftype1< double >::global(), 
			RFCAST( &Table::prd )
		),

		/**
		 * Put value into table index specifiey by second arg.
		 */
		new DestFinfo( "input2", Ftype2< double, unsigned int >::global(), 
			RFCAST( &Table::input2 )
		),

	};

	static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static Cinfo tableCinfo(
	"ParTable",
	"Mayuresh Kulkarni, 2007, CRL",
	"This class is the parallel version of the Table class. \nMoose simulation output being generated at each neuron is sent \nto the root process by every non root node",
	initInterpolCinfo(),
	tableFinfos,
	sizeof( tableFinfos ) / sizeof( Finfo * ),
	ValueFtype1< ParTable >::global(),
		schedInfo, 1
	);

	return &tableCinfo;
}

static const Cinfo* tableCinfo = initParTableCinfo();

static double arrVislnData[MAX_MPI_RECV_RECORD_SIZE];

////////////////////////////////////////////////////////////////////
// Here we set up Table class functions
////////////////////////////////////////////////////////////////////
ParTable::ParTable()
{
	index_ = -1;
	ulTableIndex_ = 0; 
	ulCurrentIndex_ = 1;
	bSelected_ = false;
	bRecvCalled_ = false;
	bSendCalled_ = false;
	lRequest_ = -1;
	ulLastRecordSent_ = 0;
}
////////////////////////////////////////////////////////////////////
// Here we set up Table value fields
////////////////////////////////////////////////////////////////////

void ParTable::setIndex( const Conn& c, int input ) 
{
	static_cast< ParTable* >( c.data() )->index_ = input;
}
int ParTable::getIndex( const Element* e )
{
	return static_cast< ParTable* >( e->data() )->index_;
}


////////////////////////////////////////////////////////////////////
// Here we set up private Table class functions.
////////////////////////////////////////////////////////////////////

void ParTable::process( const Conn& c, ProcInfo p )
{
	static_cast< Table* >( c.data() )->innerProcess( c.targetElement(), p );
	static_cast< ParTable* >( c.data() )->innerProcess( c.targetElement(), p );
}

void ParTable::reinit( const Conn& c, ProcInfo p )
{
	static_cast< Table* >( c.data() )->innerReinit( c, p );
	static_cast< ParTable* >( c.data() )->innerReinit( c, p );
	memset(arrVislnData, 0, sizeof(arrVislnData));
}

void ParTable::innerProcess( Element* e, ProcInfo p )
{
	int iFlag;
	int iMyRank;
	int i;
	static int iSentChunkCount = 0;

	usleep(1000);

	if(stepMode_ == TAB_BUF && index_ != -1)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
		if(!bRecvCalled_)
		{
        		MPI_Irecv (&lRequest_, 1, MPI_DOUBLE, 0, index_, MPI_COMM_WORLD, &recv_request_);
			bRecvCalled_ = true;
		}

		MPI_Test(&recv_request_, &iFlag, &status_);
		if(iFlag == true)
		{
			bRecvCalled_ = false;
			//cout<<endl<<"Process "<<iMyRank<<" received request: "<<lRequest_<<flush;
			if(lRequest_ == 0)
			{
				cout<<endl<<"Process "<<iMyRank<<" : "<<index_<<" received STOP request"<<flush;
				bSelected_ = false;
			}
			else if(lRequest_ == 1)
			{
				cout<<endl<<"Process "<<iMyRank<<" : "<<index_<<" received START request"<<flush;
				bSelected_ = true;
			}
			
			memset(arrVislnData, 0, sizeof(arrVislnData));
			ulCurrentIndex_ = 1;
			iSentChunkCount = 0;
		}

		ulTableIndex_++;
		if(bSelected_)
		{
			arrVislnData[ulCurrentIndex_] = input_;
			ulCurrentIndex_++;

			if( (ulCurrentIndex_+1)%VISLN_CHUNK_SIZE == 0)
			{
				//cout<<endl<<"TableIndex: "<<ulTableIndex_<<"	CurrentIndex: "<<ulCurrentIndex_<<" Offset: "				     <<iSentChunkCount*VISLN_CHUNK_SIZE<<flush;
				arrVislnData[iSentChunkCount*VISLN_CHUNK_SIZE] = ulTableIndex_;
                        	MPI_Isend(	&arrVislnData[iSentChunkCount * VISLN_CHUNK_SIZE], 
						VISLN_CHUNK_SIZE, 
						MPI_DOUBLE, 
						0,
						iMyRank*10+index_, 
						MPI_COMM_WORLD, 
						&send_request_[iSentChunkCount]
					);

				iSentChunkCount++;
				ulCurrentIndex_++;

				if(iSentChunkCount == MAX_MPI_RECV_RECORD_SIZE/VISLN_CHUNK_SIZE)
				{
					memset(arrVislnData, 0, sizeof(arrVislnData));
					ulCurrentIndex_ = 1;
					iSentChunkCount = 0;
					for(i=0; i<MAX_MPI_RECV_RECORD_SIZE/VISLN_CHUNK_SIZE; i++)
					{
	                        		MPI_Test(&send_request_[iSentChunkCount],&iFlag, MPI_STATUS_IGNORE);
						if(iFlag == false)
							cout<<endl<<"Error: Possible data loss sending visualizatin data to node 0 from process: "<<iMyRank<<flush;
					}
				}
			}
			
		}
	}
}



void ParTable::innerReinit( const Conn& c, ProcInfo p )
{
}



