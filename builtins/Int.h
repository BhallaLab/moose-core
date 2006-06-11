/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MU_INT_H
#define _MU_INT_H

#include "header.h"

class Int: public Element {
	public:
		Int(const string& name)
			: Element(name), value_(0)
		{ ; }

		~Int();

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		static Element* create(
			const string& n, Element* pa, const Element* proto);

		bool adoptChild(Element* child) {
			return 0;
		}

		static void set(Conn* c, int i)
		{
			static_cast< Int* >( c->parent() )->value_ = i;
		}

		static int get( const Element* e ) {
			return static_cast< const Int* >( e )->value_;
		}


		void print() {
			cout << name() << ":	" << value_ << "\n";
		}

	private:
		int value_;

		static int* valFunc(Element* e) {
			return &(static_cast< Int* >( e )->value_);
		}

		static void setstr(Conn* c, string s) {
			static_cast< Int* >(c->parent())->value_ = atoi(s.c_str());
		}
		static string getstr(const Element* e) {
			char s[40];
			sprintf(s, "%d", static_cast< const Int* >(e)->value_);
			return s;
		}

		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};

#endif	// _MU_INT_H
