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
#include "Shell.h"
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
	// Conn c;
	// c.add( m );
	ConnId cid = 0;
	e1.element()->addMsgToConn( m, cid );
	
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
	cout << "." << flush;
	delete i1();
	delete i2();
}

void testSet()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = nc->create( "test2", size );

	Eref e2 = i2.eref();
	
	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "set_e2_%d", i );
		string stemp( temp );
		Eref dest( e2.element(), i );
		set( dest, "set_name", stemp );
		e2.element()->clearQ();
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "set_e2_%d", i );
		assert( static_cast< Neutral* >(e2.element()->data( i ))->getName()
			== temp );
	}

	cout << "." << flush;

	delete i2();
}

void testGet()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = nc->create( "test2", size );
	Element* shell = Id()();

	Eref e2 = i2.eref();
	
	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "get_e2_%d", i );
		string stemp( temp );
		static_cast< Neutral* >(e2.element()->data( i ))->setName( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		string stemp;
		Eref dest( e2.element(), i );

			// I don't really want an array of SetGet/Shells to originate
			// get requests, but just
			// to test that it works from anywhere...
		if ( get( dest, "get_name" ) ) {
			e2.element()->clearQ(); // Request goes to e2
			shell->clearQ(); // Response comes back to e1

			stemp = ( static_cast< Shell* >(shell->data( 0 )) )->getBuf();
			// cout << i << "	" << stemp << endl;
			char temp[20];
			sprintf( temp, "get_e2_%d", i );
			assert( stemp == temp );
		}
	}

	cout << "." << flush;
	delete i2();
}

void testSetGet()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = nc->create( "test2", size );

	
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		char temp[20];
		sprintf( temp, "sg_e2_%d", i );
		SetGet1< string >::set( e2, "name", temp );
		assert( static_cast< Neutral* >(e2.data())->getName() == temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		char temp[20];
		sprintf( temp, "sg_e2_%d", i );
		string ret = SetGet1< string >::get( e2, "name" );
		assert( ret == temp );
	}

	cout << "." << flush;
	delete i2();
}

void testAsync( )
{
	insertIntoQ();
	testSendMsg();
	testCreateMsg();
	testSet();
	testGet();
//	testSetGet();
}
