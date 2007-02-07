/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SWIG_H
#define _SWIG_H

extern void pwe();
extern void ce( const std::string& dest );
extern void create( const std::string& type, const std::string& path );
extern void destroy( const std::string& path );
extern void le ( const std::string& dest );
extern void le ( );

#endif // _SWIG_H

