
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MU_TESTMSG_H
#define _MU_TESTMSG_H

#include "header.h"

class TestMsg: public Element {
	friend Element* inOneConnLookup( const Conn* );
	friend Element* outOneConnLookup( const Conn* );
	public:
		TestMsg(const string& name)
			: Element(name), i_(0), j_(0), s_(""), 
			outSrc_( &outConn_ ),
			outSrc2_( &outConn2_ ),
			outOneSrc_( &outOneConn_ ),
			inConn_( this ),
			inConn2_( this ),
			outConn_( this ),
			outConn2_( this )
		{
			v_.resize(1);
		}

		~TestMsg();

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
		static_cast< TestMsg* >( c->parent() )->i_ = i;
	}

	static int getI( const Element* e ) {
		return static_cast< const TestMsg* >( e )->i_;
	}

	static void setJ(Conn* c, int i)
	{
		static_cast< TestMsg* >( c->parent() )->j_ = i;
	}

	static int getJ( const Element* e ) {
		return static_cast< const TestMsg* >( e )->j_;
	}

	static void setS(Conn* c, string s)
	{
		static_cast< TestMsg* >( c->parent() )->s_ = s;
	}

	static string getS( const Element* e ) {
		return static_cast< const TestMsg* >( e )->s_;
	}

	// static void setV(Conn* c, int i);
	static void setV( Element* e, unsigned long index, int i );
	static int getV( const Element* e, unsigned long index );

	// Access functions for synaptic weights.
	// static void setW(Conn* c, int i);
	static void setW( Element* e, unsigned long index, int i );
	static int getW( const Element* e, unsigned long index );

	static Conn* getInConn( Element* e ) {
		return &( static_cast< TestMsg* >( e )->inConn_ );
	}

	static Conn* getOneInConn( Element* e ) {
		return &( static_cast< TestMsg* >( e )->inOneConn_ );
	}

	static void inFunc( Conn* c, int i ) {
		TestMsg* t = static_cast< TestMsg * >( c->parent() );
		t->i_ += i;
		t->outSrc_.send( t->i_ );
	}

	static void inOneFunc( Conn* c, int i ) {
		TestMsg* t = static_cast< TestMsg * >( c->parent() );
		++t->j_;
		t->i_ = i + 1;
		t->outOneSrc_.send( t->i_ );
	}

	void testInOneFunc( int i ) {
		++j_;
		i_ = i + 1;
	}

	static vector< Conn* >& getSynConn( Element* e ) {
		return reinterpret_cast< vector< Conn* >& >(
			static_cast< TestMsg* >( e )->synConn_
		);
	}

	static unsigned long newSynConn( Element* e );
	static void synFunc( Conn* c, int i );

	static NMsgSrc* getOut( Element* e ) {
		return &( static_cast< TestMsg* >( e )->outSrc_ );
	}

	static SingleMsgSrc* getOneOut( Element* e ) {
		return &( static_cast< TestMsg* >( e )->outOneSrc_ );
	}

	static Conn* getInConn2( Element* e ) {
		return &( static_cast< TestMsg* >( e )->inConn2_ );
	}

	static void inFunc2( Conn* c, int i ) {
		TestMsg* t = static_cast< TestMsg * >( c->parent() );
		t->j_ += i;
		t->outSrc2_.send( t->j_ );
	}

	static NMsgSrc* getOut2( Element* e ) {
		return &( static_cast< TestMsg* >( e )->outSrc2_ );
	}


	private:
		int i_;
		int j_;
		string s_;
		vector<int> v_;

		NMsgSrc1<int> outSrc_;
		NMsgSrc1<int> outSrc2_;
		SingleMsgSrc1<int> outOneSrc_;

		// The destination conns
		PlainMultiConn inConn_;
		PlainMultiConn inConn2_;
		UniConn< inOneConnLookup > inOneConn_;
		UniConn< outOneConnLookup > outOneConn_;
		// The int holds the local value that the synConn attaches to
		// inputs.
		vector< SynConn< int >* > synConn_;

		// The src conn
		MultiConn outConn_;
		MultiConn outConn2_;

		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};

#endif	// _MU_TESTMSG_H
