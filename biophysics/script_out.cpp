/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <math.h>
#include "../element/Wildcard.h"
#include "../basecode/setget.h"
#include "script_out.h"


/**
 * The initscript_outCinfo() function sets up the script_out class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use SharedFinfos are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* initscript_outCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &script_out::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &script_out::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process messages from the scheduler objects. "
			"The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, "
			"which holds lots of information about current time, thread, dt and so on. "
			"The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo." );
			
	static Finfo* parserShared[] = {
		new SrcFinfo( "setVecField", // object, field, value 
			Ftype3< vector< Id >, string, string >::global() )
	};
	/*
	static Finfo* initShared[] =
	{
		new DestFinfo( "init", Ftype1< ProcInfo >::global(),
				RFCAST( &script_out::initFunc ) ),
		new DestFinfo( "initReinit", Ftype1< ProcInfo >::global(),
				RFCAST( &script_out::initReinitFunc ) ),
	};
	static Finfo* init = new SharedFinfo( "init", initShared,
			sizeof( initShared ) / sizeof( Finfo* ) );
	*/
	static Finfo* script_outFinfos[] = {
		new SharedFinfo( "parser", parserShared,
				sizeof( parserShared ) / sizeof( Finfo* ) ),
		new ValueFinfo( "action", ValueFtype1< int >::global(),
			reinterpret_cast< GetFunc >( &script_out::getaction ),
			reinterpret_cast< RecvFunc >( &script_out::setaction )
		),
		new ValueFinfo( "bar_x", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &script_out::getX ),
			reinterpret_cast< RecvFunc >( &script_out::setX )
		),
		new ValueFinfo( "bar_y", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &script_out::getY ),
			reinterpret_cast< RecvFunc >( &script_out::setY )
		),
		new ValueFinfo( "bar_h", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &script_out::getH ),
			reinterpret_cast< RecvFunc >( &script_out::setH )
		),
		new ValueFinfo( "bar_w", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &script_out::getW ),
			reinterpret_cast< RecvFunc >( &script_out::setW )
		),
		new ValueFinfo( "bar_dx", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &script_out::getDX ),
			reinterpret_cast< RecvFunc >( &script_out::setDX )
		),
		new ValueFinfo( "bar_dy", ValueFtype1< double >::global(),
			reinterpret_cast< GetFunc >( &script_out::getDY ),
			reinterpret_cast< RecvFunc >( &script_out::setDY )
		),
		process,
// 		init
	};

	// This sets up two clocks: first a process clock at stage 0, tick 0,
	// then an init clock at stage 0, tick 1.
	static SchedInfo schedInfo[] = { { process, 0, 0 }/*, { init, 0, 1 }*/ };

	static string doc[] =
	{
		"Name", "script_out",
		"Author", "Raamesh Deshpande",
		"Description", "script_out object, for making orient_tut work",
	};	
	static Cinfo script_outCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),				
				initNeutralCinfo(),
				script_outFinfos,
				sizeof( script_outFinfos ) / sizeof( Finfo* ),
				ValueFtype1< script_out >::global(),
				schedInfo, 2
	);

	return &script_outCinfo;
}

static const Cinfo* script_outCinfo = initscript_outCinfo();

static const Slot setVecFieldSlot = 
	initscript_outCinfo()->getSlot( "parser.setVecField" );



//////////////////////////////////////////////////////////////////
// Here we put the script_out class functions.
//////////////////////////////////////////////////////////////////


script_out::script_out(){
	action_ = 0;
	bar_x_ = 0;
	bar_y_ = 0;
	bar_dx_ = 0;
	bar_dy_ = 0;
	bar_h_ = 0;
	bar_w_ = 0;
}

void script_out::setX(const Conn* c, double value){
	static_cast< script_out* >( c->data() )->bar_x_ = value;
}
void script_out::setY(const Conn* c, double value){
	static_cast< script_out* >( c->data() )->bar_y_ = value;
}
void script_out::setH(const Conn* c, double value){
	static_cast< script_out* >( c->data() )->bar_h_ = value;
}
void script_out::setW(const Conn* c, double value){
	static_cast< script_out* >( c->data() )->bar_w_ = value;
}
void script_out::setDX(const Conn* c, double value){
	static_cast< script_out* >( c->data() )->bar_dx_ = value;
}
void script_out::setDY(const Conn* c, double value){
	static_cast< script_out* >( c->data() )->bar_dy_ = value;
}

double script_out::getX( Eref e )
{
	return static_cast< script_out* >( e.data() )->bar_x_;
}
double script_out::getY( Eref e )
{
	return static_cast< script_out* >( e.data() )->bar_y_;
}
double script_out::getH( Eref e )
{
	return static_cast< script_out* >( e.data() )->bar_h_;
}
double script_out::getW( Eref e )
{
	return static_cast< script_out* >( e.data() )->bar_w_;
}
double script_out::getDX( Eref e )
{
	return static_cast< script_out* >( e.data() )->bar_dx_;
}
double script_out::getDY( Eref e )
{
	return static_cast< script_out* >( e.data() )->bar_dy_;
}

void script_out::innerProcessFunc( Eref e, ProcInfo p ){
	if (!action_) return;
	vector <Id> ret;
	simpleWildcardFind("/retina/recplane/rec[]/input", ret);
	string field = "rate";
	string value = "10"; //baserate
	vector< Id >::iterator i;
	for (i = ret.begin(); i != ret.end(); i++){
		assert( !i->bad() ); 
		//Element* e = ( *i )();
		Eref eref( ( *i )(), ( *i ).index() );
		// Appropriate off-node stuff here.
		const Finfo* f = eref.e->findFinfo( field );
		if ( f ) {
			if ( !f->strSet( eref, value ) )
				cout << "script_out process: Error: cannot set field " << i->path() <<
						"." << field << " to " << value << endl;
		} else {
			cout << "script_out process: Error: cannot find field: " << i->path() <<
				"." << field << endl;
		}
	}
	bar_x_ = bar_x_ + bar_dx_;
	bar_y_ = bar_y_ + bar_dy_;
// 	Id id ("/control/bar_firing_rate");
// // 	Eref eref(id(), id.index());
// 	const Finfo* f = eref.e->findFinfo( "value" );
// 	f->strGet( eref, value );
	//bool r = get <string> (eref, "value", v);
	value = "10";
	double x1, x2, y1, y2;
	x1 = bar_x_ - bar_w_/2;
	y1 = bar_y_ - bar_h_/2;
	x2 = bar_x_ + bar_w_/2;
	y2 = bar_y_ + bar_h_/2;
	simpleWildcardFind("/retina/recplane/rec[x>{x1}][y>{y1}][x<{x2}][y<{y2}]/input", ret);
	for (i = ret.begin(); i != ret.end(); i++){
		assert( !i->bad() ); 
		//Element* e = ( *i )();
		Eref eref( ( *i )(), ( *i ).index() );
		// Appropriate off-node stuff here.
		const Finfo* f = eref.e->findFinfo( field );
		if ( f ) {
			if ( !f->strSet( eref, value ) )
				cout << "script_out process: Error: cannot set field " << i->path() <<
						"." << field << " to " << value << endl;
		} else {
			cout << "script_out process: Error: cannot find field: " << i->path() <<
				"." << field << endl;
		}
	}
}

void script_out::processFunc( const Conn* c, ProcInfo p )
{
	static_cast< script_out* >( c->data() )->innerProcessFunc( c->target(), p );
}

void script_out::reinitFunc( const Conn* c, ProcInfo p )
{
	cout << "resetting" << endl;
}

void script_out::initFunc( const Conn* c, ProcInfo p )
{
	cout << "init" << endl;
}

void script_out::initReinitFunc( const Conn* c, ProcInfo p )
{
	cout << "initReinit" << endl;
}

void script_out::setaction( const Conn* c, int action )
{
	static_cast< script_out* >( c->data() )->action_ = action;
}

int script_out::getaction( Eref e )
{
	return static_cast< script_out* >( e.data() )->action_;
}



