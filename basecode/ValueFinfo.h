/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _VALUE_FINFO_H
#define _VALUE_FINFO_H

template < class T, class F > class ValueFinfo: public Finfo
{
	public:
		~ValueFinfo() {
			delete set_;
			delete get_;
		}

		ValueFinfo( const string& name, const string& doc, 
			void ( T::*setFunc )( F ),
			F ( T::*getFunc )() const )
			: Finfo( name, doc )
		{
				string setname = "set_" + name;
				set_ = new DestFinfo(
					setname,
					"Assigns field value.",
					new OpFunc1< T, F >( setFunc ) );

				string getname = "get_" + name;
				get_ = new DestFinfo(
					getname,
					"Requests field value. The requesting Element must "
					"provide a handler for the returned value.",
					new GetOpFunc< T, F >( getFunc ) );
		}


		void registerFinfo( Cinfo* c ) {
			c->registerFinfo( set_ );
			c->registerFinfo( get_ );
		}

	private:
		DestFinfo* set_;
		DestFinfo* get_;
		
	//	OpFunc1< T, F >* setOpFunc_;
	//	GetOpFunc< T, F >* getOpFunc_;
};

template < class T, class F > class ReadOnlyValueFinfo: public Finfo
{
	public:
		~ReadOnlyValueFinfo() {
			delete get_;
		}

		ReadOnlyValueFinfo( const string& name, const string& doc, 
			F ( T::*getFunc )() const )
			: Finfo( name, doc )
		{
				string getname = "get_" + name;
				get_ = new DestFinfo(
					getname,
					"Requests field value. The requesting Element must "
					"provide a handler for the returned value.",
					new GetOpFunc< T, F >( getFunc ) );
		}


		void registerFinfo( Cinfo* c ) {
			c->registerFinfo( get_ );
		}

	private:
		DestFinfo* get_;
};


/**
 * Here the value belongs to an array field within class T.
 * This is used when the assignment function for an array field 
 * should also update some information in the parent class T.
 * The function thus does not refer to the class of the array field.
 */
template < class T, class F > class UpValueFinfo: public Finfo
{
	public:
		~UpValueFinfo() {
			delete set_;
			delete get_;
		}

		UpValueFinfo( const string& name, const string& doc, 
			void ( T::*setFunc )( DataId, F ),
			F ( T::*getFunc )( DataId ) const )
			: Finfo( name, doc )
		{
				string setname = "set_" + name;
				set_ = new DestFinfo(
					setname,
					"Assigns field value.",
					new UpFunc1< T, F >( setFunc ) );

				string getname = "get_" + name;
				get_ = new DestFinfo(
					getname,
					"Requests field value. The requesting Element must "
					"provide a handler for the returned value.",
					new GetUpFunc< T, F >( getFunc ) );
		}


		void registerFinfo( Cinfo* c ) {
			c->registerFinfo( set_ );
			c->registerFinfo( get_ );
			// set_->registerFinfo( c );
			// get_->registerFinfo( c );
		}

	private:
		DestFinfo* set_;
		DestFinfo* get_;
		
	//	OpFunc1< T, F >* setOpFunc_;
	//	GetOpFunc< T, F >* getOpFunc_;
};

#endif // _VALUE_FINFO_H
