/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"
#include "Dinfo.h"

const Cinfo* Shell::initCinfo()
{
	/*
	static Finfo* reacFinfos[] = {
		new Finfo( setKf_ ),
		new Finfo( setKb_ ),
	};
	*/
	static Finfo* shellFinfos[] = {
		new ValueFinfo< Shell, string >( 
			"name",
			"Name of object", 
			&Shell::setName, 
			&Shell::getName ),
		new DestFinfo( "handleGet", 
			"Function to handle returning values for 'get' calls.",
			new RetFunc< Shell >( &Shell::handleGet ) ),
		new SrcFinfo1< FuncId >( "requestGet",
			"Function to request another Element for a value", 0 ),
			
	};

	static Cinfo shellCinfo (
		"Shell",
		0, // No base class.
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		new Dinfo< Shell >()
	);

	return &shellCinfo;
}

static const Cinfo* shellCinfo = Shell::initCinfo();


Shell::Shell()
	: name_( "" )
{
	;
}

void Shell::process( const ProcInfo* p, Eref e )
{
	;
}

void Shell::setName( const string& name )
{
	name_ = name;
}

const string& Shell::getName() const
{
	return name_;
}

void Shell::handleGet( Eref& e, const Qinfo* q, const char* arg )
{
	getBuf_.resize( q->size() );
	memcpy( &getBuf_[0], arg, q->size() );
	// Instead of deleting and recreating the msg, it could be a 
	// permanent msg on this object, reaching out whenever needed
	// to targets.
}

const char* Shell::getBuf() const
{
	if ( getBuf_.size() > 0 )
		return &( getBuf_[0] );
	return 0;
}

/*
bool set( Eref& srce, Eref& dest, const string& destField, const double& val )
{
	Element* src = srce.element();
	SrcFinfo1< double > sf( "set", "dummy", 0 );

	FuncId fid = dest->cinfo()->getOpFuncId( destField );
	const OpFunc* func = dest->cinfo()->getOpFunc( fid );
	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			Msg* m = new OneToOnemsg( src, dest.element() );
			Conn c;
			ConnId setCid = 0;
			unsigned int setFuncIndex = 0;
			c.add( m );
			src->addConn( c, setCid );
			src->addTargetFunc( fid, setFuncIndex );
		}
	}
	sf.send( srce, val );
}
*/

bool set( Eref& srce, Eref& dest, const string& destField, const string& val )
{
	Element* src = srce.element();
	SrcFinfo1< string > sf( "set", "dummy", 0 );

	FuncId fid = dest.element()->cinfo()->getOpFuncId( destField );
	const OpFunc* func = dest.element()->cinfo()->getOpFunc( fid );
	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			Msg* m = new SingleMsg( srce, dest );
			Conn c;
			ConnId setCid = 0;
			unsigned int setFuncIndex = 0;
			c.add( m );
			src->addConn( c, setCid );
			src->addTargetFunc( fid, setFuncIndex );
			sf.send( srce, val );
			// c.clearConn();
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}

bool get( Eref& srce, const Eref& dest, const string& destField )
{
	Element* src = srce.element();
	SrcFinfo1< string > sf( "get", "dummy", 0 );

	FuncId fid = dest.element()->cinfo()->getOpFuncId( destField );
	const OpFunc* func = dest.element()->cinfo()->getOpFunc( fid );

	FuncId retFunc = srce.element()->cinfo()->getOpFuncId( "handleGet" );
	const Finfo* reqFinfo = srce.element()->cinfo()->findFinfo( "requestGet" );
	const SrcFinfo1< FuncId >* rf = 
		dynamic_cast< const SrcFinfo1< FuncId >* >( reqFinfo );
	assert( rf != 0 );

	if ( func ) {
		if ( func->checkFinfo( &sf ) ) {
			Msg* m = new SingleMsg( srce, dest );
			Conn c;
			ConnId setCid = 0;
			unsigned int setFuncIndex = 0;
			c.add( m );
			src->addConn( c, setCid );
			src->addTargetFunc( fid, setFuncIndex );
			rf->send( srce, retFunc );
			// Now, dest has to clearQ, do its stuff, then src has to clearQ
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}

