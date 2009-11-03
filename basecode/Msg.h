/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MSG_H
#define _MSG_H

/**
 * Manages data flow between two elements. Is always many-to-many, with
 * assorted variants.
 */

typedef unsigned short MsgId;

class Msg
{
	public:
		Msg( Element* e1, Element* e2 );
		virtual ~Msg();

		/**
		 * Call clearQ on e1. Note that it applies at
		 * the Element level, not the index level.
		 */
		virtual void clearQ() const;

		virtual void addToQ( const Element* caller, Qinfo& q,
			const char* arg ) const = 0;

		/**
		 * Calls Process on e1.
		 */
		virtual void process( const ProcInfo *p ) const;

		/**
		 * Execute func( arg ) on target, all relevant indices.
		 * Returns next buf pos.
		 */
		virtual void exec( 
			Element* target, const char* arg 
		) const = 0;
		/*
		virtual const char* exec( 
			Element* target, const OpFunc* func, 
			unsigned int srcIndex, const char* arg 
		) const = 0;
		*/

		/*
		// return pointer to parent connection 1.
		virtual const Conn* parent1() const;

		// return pointer to parent connection 2.
		virtual const Conn* parent2() const;
		*/

		// Something here to set up sync message buffers

		// Something here to print message info
		// virtual void print( string& output ) const;

		// Duplicate message on new Elements.
		// virtual Msg* dup( Element* e1, Element* e2 ) const;

		Element* e1() const {
			return e1_;
		}

		Element* e2() const {
			return e2_;
		}

		MsgId mid1() const {
			return m1_;
		}

		MsgId mid2() const {
			return m2_;
		}

	protected:
		Element* e1_;
		Element* e2_;
		MsgId m1_; // Index of Msg on e1
		MsgId m2_; // Index of Msg on e2
};

class SingleMsg: public Msg
{
	public:
		SingleMsg( Eref e1, Eref e2 );
		~SingleMsg() {;}

		void addToQ( const Element* caller, Qinfo& q, 
			const char* arg ) const;
		void exec( Element* target, const char* arg) const;
		/*
		const char* exec( 
			Element* target, const OpFunc* func, 
			unsigned int srcIndex,  const char* arg
		) const;
		*/

	private:
		DataId i1_;
		DataId i2_;
};

class OneToOneMsg: public Msg
{
	public:
		OneToOneMsg( Element* e1, Element* e2 );
		~OneToOneMsg() {;}

		void addToQ( const Element* caller, Qinfo& q, 
			const char* arg ) const;
		void exec( Element* target, const char* arg) const;
		/*
		const char* exec( 
			Element* target, const OpFunc* func, 
			unsigned int srcIndex,  const char* arg
		) const;
		*/
	private:
		//unsigned int i1_;
		//unsigned int i2_;
};


#endif // _MSG_H
