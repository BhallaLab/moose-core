/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Neutral.h"
#include "Dinfo.h"
#include "SetGet.h"
#include "Message.h"

void insertIntoQ( )
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;

	Id i1 = nc->create( "test1", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new SingleMsg( e1, e2 );
	

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[10];
		sprintf( temp, "objname_%d", i );
		string stemp( temp );
		char buf[200];

		unsigned int size = Conv< string >::val2buf( buf, stemp );
		Qinfo qi( 1, i, size + sizeof( unsigned int ), 1 );

		*reinterpret_cast< unsigned int* >( buf + size ) = i;

		e1.element()->addToQ( qi, buf );
	}
	e1.element()->clearQ();

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "objname_%d", i );
		assert( static_cast< Neutral* >(e1.element()->data( i ))->getName()
			== temp );
	}
	cout << "." << flush;

	delete m;
	delete i1();
	delete i2();
}

void testSendMsg()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	FuncId fid = 1;

	Id i1 = nc->create( "test1", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new OneToOneMsg( e1.element(), e2.element() );
	Conn c;
	c.add( m );
	ConnId cid = 0;
	e1.element()->addConn( c, cid );
	
	SrcFinfo1<string> s( "test", "", cid );
	s.registerSrcFuncIndex( 0 );
	e1.element()->addTargetFunc( fid, s.getFuncIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "send_to_e2_%d", i );
		string stemp( temp );
		s.send( Eref( e1.element(), i ), stemp );
	}
	e2.element()->clearQ();

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "send_to_e2_%d", i );
		assert( static_cast< Neutral* >(e2.element()->data( i ))->getName()
			== temp );
	}
	cout << "." << flush;

	delete i1();
	delete i2();
}

void testCreateMsg()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	Id i1 = nc->create( "test1", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	bool ret = add( e1.element(), "child", e2.element(), "parent" );
	
	assert( ret );

	const Finfo* f = nc->findFinfo( "child" );

	for ( unsigned int i = 0; i < size; ++i ) {
		const SrcFinfo0* sf = dynamic_cast< const SrcFinfo0* >( f );
		assert( sf != 0 );
		sf->send( Eref( e1.element(), i ) );
	}
	e2.element()->clearQ();

	/*
	for ( unsigned int i = 0; i < size; ++i )
		cout << i << "	" << static_cast< Neutral* >(e2.element()->data( i ))->getName() << endl;

*/
	delete i1();
	delete i2();
}

void testSet()
{
	const Cinfo* sgc = SetGet::initCinfo(); // This will later be the shell.
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i1 = sgc->create( "set", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();
	
	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "set_e2_%d", i );
		string stemp( temp );
		Eref dest( e2.element(), i );
		set( e1, dest, "setname", stemp );
	}
	e2.element()->clearQ();

	for ( unsigned int i = 0; i < size; ++i )
		cout << i << "	" << static_cast< Neutral* >(e2.element()->data( i ))->getName() << endl;

	delete i1();
	delete i2();
}

void testAsync( )
{
	insertIntoQ();
	testSendMsg();
	testCreateMsg();
	testSet();
}
