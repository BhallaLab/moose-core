#include "header.h"
#include "../shell/Shell.h"

////////////
//Author : Vishaka Datta S, 2011
//
//This is a stress test for the scaling and squaring matrix exponential
//based solver. 
//
//The channel being simulated is the same 4-state NMDA model used in the 
//unit test, but with a few differences. 
//
//The time step has been reduced to 0.00001 seconds from 0.001 seconds.
//Further, the number of divisions in all lookup tables have been increased
//by a factor of 10. 
///////////
void perfTestMarkovSolver( )
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< int > dims( 1, 1 );
	
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims ); 

	Id comptId = shell->doCreate( "Compartment", nid, 	
																"compt", dims );

	Id rateTableId = shell->doCreate( "MarkovRateTable", comptId, 
																 	  "rateTable", dims );

	Id mChanId = shell->doCreate( "MarkovChannel", comptId, 
																"mChan", dims );

	Id solverId = shell->doCreate( "MarkovSolver", comptId, 
	  														 "solver", dims );

	Id int2dTableId = shell->doCreate( "Interpol2D", nid, "int2dTable", dims );
	Id vecTableId = shell->doCreate( "VectorTable", nid, "vecTable", dims );

	vector< double > table1d;
	vector< vector< double > > table2d;

 	MsgId mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
			  ObjId( mChanId ), "channel" );
	assert( mid != Msg::bad );

	mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
			ObjId( solverId ), "channel" );
	assert( mid != Msg::bad );						

	mid = shell->doAddMsg( "Single", ObjId( solverId ), "stateOut", 
			ObjId( mChanId ), "handlestate" );
	assert( mid != Msg::bad );

	Field< double >::set( comptId, "Cm", 0.007854e-6 );
	Field< double >::set( comptId, "Ra", 7639.44e3 ); // does it matter?
	Field< double >::set( comptId, "Rm", 424.4e3 );
	Field< double >::set( comptId, "Em", -0.04 );	
	Field< double >::set( comptId, "inject", 0 );
	Field< double >::set( comptId, "initVm", -0.07 );

	//////////////////
	//Setup of rate tables.
	/////////////////

	//Number of states and open states.
	Field< unsigned int >::set( mChanId, "numstates", 4 );		

	Field< unsigned int >::set( mChanId, "numopenstates", 1 );		

	vector< string > stateLabels;

	//In the MarkovChannel class, the opening states are listed first.
	//This is in line with the convention followed in Chapter 20, Sakmann & 
	//Neher. 
	stateLabels.push_back( "O" );		//State 1.
	stateLabels.push_back( "B1" );	//State 2.
	stateLabels.push_back( "B2" );	//State 3.
	stateLabels.push_back( "C" ); 	//State 4.

	Field< vector< string > >::set( mChanId, "labels", stateLabels );	

	//Setting up conductance value for single open state.	Value chosen
	//is quite arbitrary.
	vector< double > gBar;

	gBar.push_back( 5.431553e-9 );

	Field< vector< double > >::set( mChanId, "gbar", gBar );

	//Initial state of the system. This is really an arbitrary choice.
	vector< double > initState;

	initState.push_back( 0.00 ); 
	initState.push_back( 0.20 ); 
	initState.push_back( 0.80 ); 
	initState.push_back( 0.00 ); 

	Field< vector< double > >::set( mChanId, "initialstate", initState );

	//Initializing MarkovrateTable object.
	double v;
	double conc;

	SetGet1< unsigned int >::set( rateTableId, "init", 4 );

	//Setting up lookup tables for the different rates.		
	//Please note that the rates should be in sec^(-1).  
	
	//Transition from "O" to "B1" i.e. r12 or a1.
	Field< double >::set( vecTableId, "xmin", -0.10 );
	Field< double >::set( vecTableId, "xmax", 0.10 );
	Field< unsigned int >::set( vecTableId, "xdivs", 2000 );

	v = Field< double >::get( vecTableId, "xmin" );
	for ( unsigned int i = 0; i < 2001; ++i )	
	{
		table1d.push_back( 1e3 * exp( -16 * v - 2.91 ) );
		v += 1e-4;
	}

	Field< vector< double > >::set( vecTableId, "table", table1d );

	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 1, 2, vecTableId, 0 );

	table1d.erase( table1d.begin(), table1d.end() );

	//Transition from "B1" back to O i.e. r21 or b1
	v = Field< double >::get( vecTableId, "xmin" );
	for ( unsigned int i = 0; i < 2001; ++i )
	{
		table1d.push_back( 1e3 * exp( 9 * v + 1.22 ) );
		v += 1e-4;
	}

	Field< vector< double > >::set( vecTableId, "table", table1d );
	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 2, 1, vecTableId, 0 );

	table1d.erase( table1d.begin(), table1d.end() );

	//Transition from "O" to "B2" i.e. r13 or a2
	//This is actually a 2D rate. But, there is no change in Mg2+ concentration
	//that occurs. Hence, I create a 2D lookup table anyway but I manually
	//set the concentration on the rate table object anyway.

	Field< double >::set( rateTableId, "ligandconc", 24e-6 );

	Field< double >::set( int2dTableId, "xmin", -0.10 );
	Field< double >::set( int2dTableId, "xmax", 0.10 );
	Field< double >::set( int2dTableId, "ymin", 0 );
	Field< double >::set( int2dTableId, "ymax", 30e-6 );
	Field< unsigned int >::set( int2dTableId, "xdivs", 2000 );
	Field< unsigned int >::set( int2dTableId, "ydivs", 300 );

	table2d.resize( 2001 );
	v = Field< double >::get( int2dTableId, "xmin" );
	for ( unsigned int i = 0; i < 2001; ++i )
	{
		conc = Field< double >::get( int2dTableId, "ymin" );
		for ( unsigned int j = 0; j < 301; ++j )
		{
			table2d[i].push_back( 1e3 * conc * exp( -45 * v - 6.97 ) ); 
			conc += 1e-10;
		}
		v += 1e-4;
	}

	Field< vector< vector< double > > >::set( int2dTableId, "tableVector2D", 
																table2d );

	SetGet3< unsigned int, unsigned int, Id >::set( rateTableId, 
																			"set2d", 1, 3, int2dTableId ); 

	//There is only one 2D rate, so no point manually erasing the elements.
	
	//Transition from "B2" to "O" i.e. r31 or b2
	v = Field< double >::get( vecTableId, "xmin" );
	for ( unsigned int i = 0; i < 2001 ; ++i )
	{
		table1d.push_back( 1e3 * exp( 17 * v + 0.96 ) );
		v += 1e-4;
	}

	Field< vector< double > >::set( vecTableId, "table", table1d );
	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 3, 1, vecTableId, 0 );

	table1d.erase( table1d.begin(), table1d.end() );

	//Transition from "O" to "C" i.e. r14 
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId,	
									"setconst", 1, 4, 1e3 * exp( -2.847 ) ); 
	
	//Transition from "B1" to "C" i.e. r24
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId, 	
									"setconst", 2, 4, 1e3 * exp( -0.693 ) );

	//Transition from "B2" to "C" i.e. r34
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId, 
									"setconst", 3, 4, 1e3* exp( -3.101 ) );

	//Once the rate tables have been set up, we can initialize the 
	//tables in the Markovsolver class.
	SetGet2< Id, double >::set( solverId, "init", rateTableId, 1.0e-5 );
	SetGet1< double >::set( solverId, "ligandconc", 24e-6 );
	Field< vector< double > >::set( solverId, "initialstate", 
																	initState );

	shell->doSetClock( 0, 1.0e-5 );	
	shell->doSetClock( 1, 1.0e-5 );	
	shell->doSetClock( 2, 1.0e-5 );	
	shell->doSetClock( 3, 1.0e-5 );	

	shell->doUseClock( "/n/compt", "init", 0 );
	shell->doUseClock( "/n/compt", "process", 1 );
	shell->doUseClock( "/n/compt/rateTable", "process", 2 );
	shell->doUseClock( "/n/compt/solver", "process", 3 );

	shell->doReinit();
	shell->doReinit();
	shell->doStart( 1.0 );

	shell->doDelete( nid );
	cout << "." << flush;
}
