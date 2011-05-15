/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class sets up fields that should be accessed as independent
 * Elements. These fields are typically complex ones with several sub-
 * fields, and are also typically arranged in an array.
 * Examples are the Ticks for Clock objects, and Synapses on an
 * IntFire Neuron.
 *
 * The creation of this one Finfo sets up three things: 
 * First, the information for the FieldElement that will handle the class.
 * Second, the set and get functions for the size of the array of Fields.
 * Third, an automatic creation of the FieldElement whenever an Element
 * of the Parent class is made. This defaults to a child, but can be moved.
 */

#ifndef _FIELD_ELEMENT_FINFO_H
#define _FIELD_ELEMENT_FINFO_H

bool adopt( Id parent, Id child );

template < class T, class F > class FieldElementFinfo: public Finfo
{
	public:

		FieldElementFinfo( 
			const string& name, 
			const string& doc, 
			const Cinfo* fieldCinfo,
			F* ( T::*lookupField )( unsigned int ),
			void( T::*setNumField )( unsigned int num ),
			unsigned int ( T::*getNumField )() const,
			bool deferCreate = 0
		)
			: 	Finfo( name, doc), 
				fieldCinfo_( fieldCinfo ),
				lookupField_( lookupField ),
				setNumField_( setNumField ),
				getNumField_( getNumField ),
				deferCreate_( deferCreate )
		{
				string setname = "set_num_" + name;
				// setNumField is a tricky operation, because it may require
				// cross-node rescaling of the 
				// FieldDataHandler::fieldDimension. To acheive this we
				// wrap the setNumField in something more interesting
				// than a simple OpFunc.
				setNum_ = new DestFinfo(
					setname,
					"Assigns number of field entries in field array.",
					new OpFunc1< T, unsigned int >( setNumField ) );

				string getname = "get_num_" + name;
				getNum_ = new DestFinfo(
					getname,
					"Requests number of field entries in field array."
					"The requesting Element must "
					"provide a handler for the returned value.",
					new GetOpFunc< T, unsigned int >( getNumField ) );
		}

		~FieldElementFinfo() {
			delete setNum_;
			delete getNum_;
		}

		/**
		 * Virtual function. 
		 */
		void postCreationFunc( Id parent, Element* parentElm ) const
		{
			if ( deferCreate_ )
				return;
			Id kid = Id::nextId();
			new Element(
				kid, fieldCinfo_, name(), 
				new FieldDataHandler< T, F >(
					fieldCinfo_->dinfo(),
					parentElm->dataHandler(),
					lookupField_, getNumField_, setNumField_ )
			);
			adopt( parent, kid );
		}

		void registerFinfo( Cinfo* c ) {
			c->registerFinfo( setNum_ );
			c->registerFinfo( getNum_ );
			c->registerPostCreationFinfo( this );
		}

		bool strSet( const Eref& tgt, const string& field, 
			const string& arg ) const {
			return 0; // always fails
		}

		bool strGet( const Eref& tgt, const string& field, 
			string& returnValue ) const {
			return 0; // always fails
		}

		string rttiType() const {
			return Conv<F>::rttiType();
		}

	private:
		DestFinfo* setNum_;
		DestFinfo* getNum_;
		const Cinfo* fieldCinfo_;
		F* ( T::*lookupField_ )( unsigned int );
		void( T::*setNumField_ )( unsigned int num );
		unsigned int ( T::*getNumField_ )() const;
		bool deferCreate_;
		
	//	OpFunc1< T, F >* setOpFunc_;
	//	GetOpFunc< T, F >* getOpFunc_;
};


#endif // _FIELD_ELEMENT_FINFO_H
