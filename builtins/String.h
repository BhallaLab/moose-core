/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MU_STRING_H
#define _MU_STRING_H

#include "header.h"

class String: public Element {
	public:
		String(const string& name)
			: Element(name), value_("bzing")
		{ ; }

		~String();

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		static Element* create(
			const string& n, Element* pa, const Element* proto);

		bool adoptChild(Element* child) {
			return 0;
		}

		static void set(Conn* c, string s)
		{
			static_cast< String* >( c->parent() )->value_ = s;
		}

		static string get( const Element* e ) {
			return static_cast< const String* >( e )->value_;
		}

		void print() {
			cout << name() << ":	" << value_ << "\n";
		}

	private:
		string value_;

		static string* valFunc(Element* e) {
			return &(static_cast< String* >( e )->value_);
		}

		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};

#endif	// _MU_STRING_H
