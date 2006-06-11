/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _P_ELEMENT_H
#define _P_ELEMENT_H

#include "header.h"

class Neutral: public Element {
	public:
		Neutral(const string& name_arg)
			: Element( name_arg ),
			  childOut_( this ),
			  childSrc_( &childOut_ )
		{ ; }

		~Neutral();

		bool adoptChild(Element* child);

///////////////////////////////////////////////////////////////////////
// These boilerplate functions are used to access the object.
///////////////////////////////////////////////////////////////////////
		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		static Element* create(
			const string& n, Element* pa, const Element* proto);

///////////////////////////////////////////////////////////////////////
// Some child-handling stuff.
///////////////////////////////////////////////////////////////////////
		static Conn* getChildOut( Element* e ) {
			return &( static_cast< Neutral* >( e )->childOut_ );
		}

		RecvFunc getDeleteFunc() {
			return deleteFunc;
		}

		// This function causes the elm and all children to be deleted.
		// It goes depth first.
		static void deleteFunc( Conn* c ) {
			Neutral* e = static_cast< Neutral* >( c->parent() );
			e->childSrc_.send();
			delete e;
		}

		static NMsgSrc* getChildOutSrc( Element* e ) {
			return &( static_cast< Neutral* >( e )->childSrc_ );
		}

		// Override the virtual function for Elements.
		Element* relativeFind( const string& n );

		// Override the virtual function for Elements.
		// This is defined in Wildcard.cpp
		int wildcardRelativeFind(
			const string& n, vector< Element* >& ret, int doublehash) ;

		Element* internalDeepCopy(
			Element* pa, map<const Element*, Element* >& tree) const;

	private:
		MultiConn childOut_;
		NMsgSrc0 childSrc_;
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};

#endif	// _P_ELEMENT_H
