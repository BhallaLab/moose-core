
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MU_TESTFIELD_H
#define _MU_TESTFIELD_H

#include "header.h"

class TestField: public Element {
	public:
		TestField(const string& name)
			: Element(name), i_(0), j_(0), s_("")
		{
			v_.resize(100);
		}

		~TestField();

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		static Element* create(
			const string& n, Element* pa, const Element* proto);

		bool adoptChild(Element* child) {
			return 0;
		}

		void print() {
			cout << name() << ":	" << i_ << ", " << j_ << ", " <<
				s_ << "\n";
		}

	///////////////////////////////////////////////////////////////
	// Field handler functions
	///////////////////////////////////////////////////////////////
	static void setI(Conn* c, int i)
	{
		static_cast< TestField* >( c->parent() )->i_ = i;
	}

	static int getI( const Element* e ) {
		return static_cast< const TestField* >( e )->i_;
	}

	static void setJ(Conn* c, int i)
	{
		static_cast< TestField* >( c->parent() )->j_ = i;
	}

	static int getJ( const Element* e ) {
		return static_cast< const TestField* >( e )->j_;
	}

	static void setS(Conn* c, string s)
	{
		static_cast< TestField* >( c->parent() )->s_ = s;
	}

	static string getS( const Element* e ) {
		return static_cast< const TestField* >( e )->s_;
	}

	static void setV(Element* e, unsigned long index, int value )
	{
		TestField* tf = static_cast< TestField* >( e );
		if ( tf->v_.size() > index )
			tf->v_[ index ] = value;
		else
			cout << "Error:TestMsg::setV: Failed to cast into ArrayFinfo\n";
	}

/*
	static void setV(Conn* c, int i)
	{
		cerr << "TestField:: In setV with i = " << i << "\n"; 
		RelayConn* rc = dynamic_cast<RelayConn *>(c);
		if (rc) {
			cerr << "TestField:: In setV rc OK\n";
			TestField* tf = static_cast< TestField* >( c->parent() );
			RelayFinfo1< int >* rf = 
				dynamic_cast< RelayFinfo1< int >* >( rc->finfo() );
			if ( rf ) {
				cerr << "TestField:: In setV rf OK\n";
				ArrayFinfo< int >* af = 
					dynamic_cast< ArrayFinfo< int >* >( rf->innerFinfo() );
				if (af) {
				

					cerr << "TestField:: In setV with index = " << af->index() << "\n"; 
					if ( tf->v_.size() > af->index() )
						tf->v_[ af->index() ] = i;
				}
			}
		}
	}
	*/

	static int getV( const Element* e, unsigned long index ) {
		cerr << "TestField:: In getV with index = " << index << "\n"; 
		const TestField* tf = static_cast< const TestField* >( e );
		if (index < tf->v_.size()) {
			cerr << "TestField:: In getV, returning: " << tf->v_[index] << "\n"; 
			return tf->v_[index];
		}
		return 0;
	}


	private:
		int i_;
		int j_;
		string s_;
		vector<int> v_;

		/*
		static int* valFunc(Element* e) {
			return &(static_cast< TestField* >( e )->value_);
		}

		operator=(const Element* other) {
			const TestField* tf = static_cast< const TestField* >(other);
			i_ = tf->_i;
			j_ = tf->_j;
			s_ = tf->_s;
		}
		*/

		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};

#endif	// _MU_TESTFIELD_H
