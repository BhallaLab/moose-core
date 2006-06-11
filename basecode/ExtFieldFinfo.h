/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _EXT_FIELD_FINFO_H
#define _EXT_FIELD_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of Finfo classes that handle extended
// fields, that is, fields that are added at run-time.
///////////////////////////////////////////////////////////////////////

template < class T > class ExtFieldFinfo: public ValueFinfoBase< T >
{
	public:
		ExtFieldFinfo( const string& name, const string& className )
		:	ValueFinfoBase< T >(name, this->set, className )
		{
			;
		}

		static void set( Conn* c, T value ) {
			RelayConn* rc = dynamic_cast< RelayConn* >( c );
			if ( rc ) {
				ExtFieldFinfo< T >* eff = 
					dynamic_cast< ExtFieldFinfo< T >* >( rc->finfo() );
				if ( eff ) {
					eff->value_ = value;
				} else {
					cerr << "Error: ExtFieldFinfo::Set: Failed to cast to ExtFieldFinfo< T >\n";
				}
			} else {
				cerr << "Error: ExtFieldFinfo::Set: Failed to cast to RelayConn\n";
			}
		}

		T value ( const Element* e ) const {
			return value_;
		}

		bool add( Element* e, Field& destfield, bool useSharedConn = 0)
		{
			return valueFinfoAdd(
				this, &newValueRelayFinfo< T >,
				e, destfield, useSharedConn ) ;
		}
		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			 return valueFinfoRespondToAdd(
			 	this, &newValueRelayFinfo< T >, &newRelayFinfo< T >,
				e, sender );
		}

		const Ftype* ftype() const {
			static const Ftype1< T > myFtype_;
			return & myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}

	private:
		T value_;
};

#endif // _EXT_FIELD_FINFO_H
