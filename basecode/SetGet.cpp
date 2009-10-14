/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SetGet.h"
#include "Dinfo.h"

const Cinfo* SetGet::initCinfo()
{
	/*
	static Finfo* reacFinfos[] = {
		new Finfo( setKf_ ),
		new Finfo( setKb_ ),
	};
	*/
	static Finfo* setGetFinfos[] = {
		new ValueFinfo< SetGet, string >( 
			"name",
			"Name of object", 
			&SetGet::setName, 
			&SetGet::getName ),
	};

	static Cinfo setGetCinfo (
		"SetGet",
		0, // No base class.
		setGetFinfos,
		sizeof( setGetFinfos ) / sizeof( Finfo* ),
		new Dinfo< SetGet >()
	);

	return &setGetCinfo;
}

static const Cinfo* setGetCinfo = SetGet::initCinfo();


SetGet::SetGet()
	: name_( "" )
{
	;
}

void SetGet::process( const ProcInfo* p, Eref e )
{
	;
}

void SetGet::setName( const string& name )
{
	name_ = name;
}

const string& SetGet::getName() const
{
	return name_;
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
			return 1;
		} else {
			cout << "set::Type mismatch" << dest << "." << destField << endl;
		}
	} else {
		cout << "set::Failed to find " << dest << "." << destField << endl;
	}
	return 0;
}
