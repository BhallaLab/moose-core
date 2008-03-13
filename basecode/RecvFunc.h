/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _RecvFunc_h
#define _RecvFunc_h
class Conn;
class Element;

typedef void ( *RecvFunc )( const Conn* );

#define RFCAST(x) reinterpret_cast< RecvFunc >( x )

typedef double ( *GetFunc )( Eref );

#define GFCAST(x) reinterpret_cast< GetFunc >( x )

typedef std::vector < RecvFunc > FuncList;

void dummyFunc( const Conn* c );

#endif
