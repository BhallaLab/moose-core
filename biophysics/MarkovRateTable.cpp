/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "VectorTable.h"
#include "../builtins/Interpol2D.h"
#include "MarkovRateTable.h"

//Message source that sends out instantaneous rate information at each time step
//to the solver object. 
static SrcFinfo1< vector< vector< double > > >* instRatesOut()
{
	static SrcFinfo1< vector< vector< double > > > instRatesOut( "instratesOut",	
		"Sends out instantaneous rate information of varying transition rates at each time step." 
		);

	return &instRatesOut;
}

const Cinfo* MarkovRateTable::initCinfo()
{
	/////////////////////
	//SharedFinfos
	/////////////////////
	static DestFinfo handleVm("handleVm", 
			"Handles incoming message containing voltage information.",
			new OpFunc1< MarkovRateTable, double >(&MarkovRateTable::handleVm)
			);

	static Finfo* channelShared[] = 
	{
		&handleVm
	};

	static SharedFinfo channel("channel",
			"This message couples the rate table to the compartment. The rate table needs updates on voltage in order to compute the rate table.", 
			channelShared, sizeof( channelShared ) / sizeof( Finfo* ) 
			);

	/////////////////////
	//DestFinfos
	////////////////////
	static DestFinfo process(	"process",
			"Handles process call",
			new ProcOpFunc< MarkovRateTable >( &MarkovRateTable::process ) ); 

	static DestFinfo reinit( "reinit", 
			"Handles reinit call",
			new ProcOpFunc< MarkovRateTable >( &MarkovRateTable::reinit ) );

	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc", 
			"This is a shared message to receive Process message from the"
			"scheduler. The first entry is a MsgDest for the Process "
			"operation. It has a single argument, ProcInfo, which "
			"holds lots of information about current time, thread, dt and"
			"so on. The second entry is a MsgDest for the Reinit "
			"operation. It also uses ProcInfo.",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	static DestFinfo setuptables("setuptables",
			"Initialization of the class. Allocates memory for all the tables.",
			new OpFunc1< MarkovRateTable, unsigned int >(&MarkovRateTable::setupTables) );

	static DestFinfo handleLigandConc("handleLigandConc",
			"Handles incoming message containing ligand concentration.",
			new OpFunc1< MarkovRateTable, double >(&MarkovRateTable::handleLigandConc) );

	static DestFinfo set1d("set1d",
			"Setting up of 1D lookup table for the (i,j)'th rate.",
			new OpFunc4< MarkovRateTable, unsigned int, unsigned int, Id,
				unsigned int >
			( &MarkovRateTable::setVtChildTable )
		);

	static DestFinfo set2d("set2d",
			"Setting up of 2D lookup table for the (i,j)'th rate.",
			new OpFunc3< MarkovRateTable, unsigned int, unsigned int, Id >
			( &MarkovRateTable::setInt2dChildTable )
		);

	static DestFinfo setconst("setconst", 
			"Setting a constant value for the (i,j)'th rate. Internally, this is"
		  "	stored as a 1-D rate with a lookup table containing 1 entry.",
			new OpFunc3< MarkovRateTable, unsigned int, unsigned int, double >
			( &MarkovRateTable::setConstantRate ) 
			);

	///////////////////////////
	//Field information.		
	//////////////////////////
	static ReadOnlyValueFinfo< MarkovRateTable, vector< vector< double > > >
		Q( "Q", 
			"Instantaneous rate matrix.",
			&MarkovRateTable::getQ
		 );

	//Really no point in allowing access to Vm_ and ligandConc_ when it is
	//available from the MarkovChannel class.
	
	static ReadOnlyValueFinfo< MarkovRateTable, unsigned int >
		size( "size", 
			"Dimension of the families of lookup tables. Is always equal to the number of states in the model.",
		&MarkovRateTable::getSize
		);

	static Finfo* markovRateTableFinfos[] =
	{
		&channel,						//SharedFinfo
		instRatesOut(),			//SrcFinfo
		&proc,							//DestFinfo
		&setuptables,				//DestFinfo
		&handleLigandConc,	//DestFinfo
		&set1d,							//DestFinfo
		&set2d,							//DestFinfo
		&setconst,					//DestFinfo
		&Q,									//ReadOnlyValueFinfo
		&size,							//ReadOnlyValueFinfo
	};

	static Cinfo MarkovRateTableCinfo(
		"MarkovRateTable",
		Neutral::initCinfo(),
		markovRateTableFinfos,
		sizeof(markovRateTableFinfos)/sizeof(Finfo *),
		new Dinfo< MarkovRateTable >
	);

	return &MarkovRateTableCinfo;
}

static const Cinfo* MarkovRateTableCinfo = MarkovRateTable::initCinfo();

MarkovRateTable::MarkovRateTable() :
	Vm_(0),
	ligandConc_(0),
	size_(0)
{ ; }

MarkovRateTable::~MarkovRateTable()
{
	for ( unsigned int i = 0; i < size_; ++i )	
	{
		for ( unsigned int j = 0; j < size_; ++j )
		{
			if ( isRate1d( i, j ) || isRateConstant( i, j ) ) 
				delete vtTables_[i][j];
			if ( isRate2d( i, j ) ) 
				delete int2dTables_[i][j];
		}
	}
}

VectorTable* MarkovRateTable::getVtChildTable( unsigned int i, unsigned int j ) const
{
	if ( isRate1d( i, j ) )
		return vtTables_[i][j];
	else	
	{
		cerr << "Error : No one parameter rate table set for this rate!\n";
		return 0;
	}
}

void MarkovRateTable::innerSetVtChildTable( unsigned int i, unsigned int j,
																						VectorTable vecTable, 
																					  unsigned int ligandFlag )	
{
	if ( vecTable.tableIsEmpty() )
	{
		cerr << "Error : Cannot set with empty table!.\n";
		return;
	}

	if ( areIndicesOutOfBounds( i, j ) )
	{
		cerr << "Error : Table requested is out of bounds!.\n";
		return; 
	}

	//If table hasnt been allocated any memory, do so. 
	if ( vtTables_[i][j] == 0 )
		vtTables_[i][j] = new VectorTable();

	//Checking to see if this rate has already been set with a 2-parameter rate.
	if ( isRate2d( i, j ) ) 
	{
		cerr << "This rate has already been set with a 2 parameter lookup table.\n";
		return;
	}

	*vtTables_[i][j] = vecTable;
	useLigandConc_[i][j] = ligandFlag;
}

Interpol2D* MarkovRateTable::getInt2dChildTable( unsigned int i, unsigned int j ) const
{
	if ( isRate2d( i, j ) )
		return int2dTables_[i][j];
	else	
	{
		cerr << "Error : No two parameter rate table set for this rate!\n";
		return 0;
	}
}

void MarkovRateTable::innerSetInt2dChildTable( unsigned int i, unsigned int j, Interpol2D int2dTable )
{
	if ( areIndicesOutOfBounds( i, j ) ) 
	{
		cerr << "Error : Table requested is out of bounds\n";
		return;
	}
	
	//If table isn't already initialized, do so.
	if ( int2dTables_[i][j] == 0 )
		int2dTables_[i][j] = new Interpol2D();

	//Checking to see if this rate has already been set with a one parameter rate
	//table.
	if ( isRate1d( i, j) ) 
	{
		cerr << "Error : This rate has already been set with a one parameter rate table!\n";
		return;
	}

	*int2dTables_[i][j] = int2dTable;
}



double MarkovRateTable::lookup1d( unsigned int i, unsigned int j, double x )
{
	if ( areIndicesOutOfBounds( i, j ) ) 
		return 0;

	return vtTables_[i][j]->innerLookup( x );
}

double MarkovRateTable::lookup2d( unsigned int i, unsigned int j, double x, double y )
{
	if ( areIndicesOutOfBounds( i, j ) ) 
		return 0;

	return int2dTables_[i][j]->innerLookup( x, y );
}

vector< vector< double > > MarkovRateTable::getQ() const
{
	return Q_;
}

unsigned int MarkovRateTable::getSize() const
{
	return size_;
}

bool MarkovRateTable::isRateZero( unsigned int i, unsigned int j ) const
{
	return ( vtTables_[i][j] == 0 && int2dTables_[i][j] == 0 );
}

bool MarkovRateTable::isRateConstant( unsigned int i, unsigned int j ) const
{
	if ( isRate2d( i, j ) || isRateZero( i, j ) )
		return false;

	return ( vtTables_[i][j]->getDiv() == 0 );
}

bool MarkovRateTable::isRate1d( unsigned int i, unsigned int j ) const
{
	//Second condition is necessary because for a constant rate, the 1D lookup
	//table class is set, but has only one entry. So a constant rate would pass
	//off as a one-parameter rate transition if not for this check.
	
	if ( vtTables_[i][j] == 0 )
		return false;
	else
		return ( vtTables_[i][j]->getDiv() > 0 );

//	return ( vtTables_[i][j] != 0 && vtTables_[i][j]->getDiv() > 0);
}

bool MarkovRateTable::isRateLigandDep( unsigned int i, unsigned int j ) const
{
	return ( isRate1d( i, j ) && useLigandConc_[i][j] > 0 ); 
}

bool MarkovRateTable::isRate2d( unsigned int i, unsigned int j ) const
{
	return ( int2dTables_[i][j] != 0 );
}

bool MarkovRateTable::areIndicesOutOfBounds( unsigned int i, unsigned int j ) const
{
	return ( i > size_ || j > size_ );
}

void MarkovRateTable::updateRates()
{
	double temp;

	//Rather crude update function. Might be easier to store the variable rates in
	//a separate list and update only those, rather than scan through the entire
	//matrix at every single function evaluation. Profile this later. 
	for ( unsigned int i = 0; i < size_; ++i )
	{
		for ( unsigned int j = 0; j < size_; ++j )
		{
			temp = Q_[i][j];
			//If rate is ligand OR voltage dependent.
//			cout << "updateRates\n";
			if ( isRate1d( i, j ) )
			{
				//Use ligand concentration instead of voltage.
				if ( isRateLigandDep( i, j ) )
					Q_[i][j] = lookup1d( i, j, ligandConc_ );
				else
					Q_[i][j] = lookup1d( i, j, Vm_ );
			}
			
			//If rate is ligand AND voltage dependent. It is assumed that ligand
			//concentration varies along the first dimension.
			if ( isRate2d( i, j ) )
				Q_[i][j] = lookup2d( i, j, ligandConc_, Vm_ );

			//Ensures that the row sums to zero.
			if ( temp != Q_[i][j] )
				Q_[i][i] = Q_[i][i] - Q_[i][j] + temp;
		}
	}
}

void MarkovRateTable::initConstantRates() 
{
	for (	unsigned int i = 0; i < size_; ++i )	
	{
		Q_[i][i] = 0;
		for ( unsigned int j = 0; j < size_; ++j )
		{
			if ( isRateConstant( i, j ) )
			{
				Q_[i][j] = lookup1d( i, j, 0.0 );  
				//Doesn't really matter which value is looked up as there is only one
				//entry in the table.
				Q_[i][i] -= Q_[i][j];
			}
		}
	}
}	

istream& operator>>( istream& in, MarkovRateTable& rateTable )
{
	for ( unsigned int i = 0; i < rateTable.size_; ++i )
	{
		for ( unsigned int j = 0; j < rateTable.size_; ++j )
		{
			if ( rateTable.isRate1d( i, j ) )
				in >> *rateTable.vtTables_[i][j];
		}
	}

	for ( unsigned int i = 0; i < rateTable.size_; ++i )
	{
		for ( unsigned int j = 0; j < rateTable.size_; ++j )
		{
			if ( rateTable.isRate2d( i, j ) ) 
				in >> *rateTable.int2dTables_[i][j];
		}
	}

	for ( unsigned int i = 0; i < rateTable.size_; ++i )
	{
		for ( unsigned int j = 0; j < rateTable.size_; ++j )
			in >> rateTable.useLigandConc_[i][j];
	}
	in >> rateTable.Vm_;
	in >> rateTable.ligandConc_;
	in >> rateTable.size_;

	return in;
}

bool MarkovRateTable::isInitialized() const
{
	return ( size_ > 0 );
}

/////////////////////////////////////////////////
//							MsgDest functions.
////////////////////////////////////////////////
void MarkovRateTable::process( const Eref& e, ProcPtr info )
{
	updateRates();
	//Sending the whole matrix out seems a bit like overkill, as only a few rates
	//may vary. Would using a sparse representation speed things up? 
	instRatesOut()->send( e, info, Q_ );
}

void MarkovRateTable::reinit( const Eref& e, ProcPtr info )
{
	if ( isInitialized() )
		initConstantRates();	
	else
		cerr << "MarkovRateTable class has not been initialized!.";

	instRatesOut()->send( e, info, Q_ );
}

void MarkovRateTable::setupTables( unsigned int size )
{
	size_ = size;

	if ( vtTables_.empty() )
		vtTables_ = resize< VectorTable* >( vtTables_, size, 0 );
	if ( int2dTables_.empty() )
		int2dTables_ = resize< Interpol2D* >( int2dTables_, size, 0 );
	if ( useLigandConc_.empty() )
		useLigandConc_ = resize< unsigned int >( useLigandConc_, size, 0 );
	if ( Q_.empty() )
		Q_ = resize< double >( Q_, size, 0 );
}

void MarkovRateTable::handleVm( double Vm )
{
	Vm_ = Vm;
}

void MarkovRateTable::handleLigandConc( double ligandConc )
{
	ligandConc_ = ligandConc;
}

void MarkovRateTable::setVtChildTable( unsigned int i, unsigned int j, 
																			Id vecTabId, unsigned int ligandFlag )
{
	VectorTable* vecTable = reinterpret_cast< VectorTable* >
																( vecTabId.eref().data() );	

	innerSetVtChildTable( i, j, *vecTable, ligandFlag );
}

void MarkovRateTable::setInt2dChildTable( unsigned int i, unsigned int j,
																					Id int2dTabId ) 
{
	Interpol2D* int2dTable = reinterpret_cast< Interpol2D* >
															( int2dTabId.eref().data() );

	innerSetInt2dChildTable( i, j, *int2dTable );
}

//Using the default 'setTable' function from the VectorTable class is rather
//clumsy when trying to set a constant rate, as one has to set up an entire
//VectorTable object. This function provides a wrapper around this rather
//complex setTable call.
void MarkovRateTable::setConstantRate( unsigned int i, unsigned int j, double rate )
{
	VectorTable vecTable;

	vecTable.setMin( rate );
	vecTable.setMax( rate );
	vecTable.setDiv( 1 );

	vector< double > rateWrap;
	rateWrap.push_back( rate );

	vecTable.setTable( rateWrap );

	innerSetVtChildTable( i, j, vecTable, 0 ); 
}
