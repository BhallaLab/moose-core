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

class FieldElementFinfoBase: public Finfo
{
	public:
		FieldElementFinfoBase( 
			const string& name, 
			const string& doc, 
			const Cinfo* fieldCinfo,
			unsigned int defaultSize,
			bool deferCreate
		)
			: 	Finfo( name, doc), 
				setNum_( 0 ),
				getNum_( 0 ),
				fieldCinfo_( fieldCinfo ),
				defaultSize_( defaultSize ),
				deferCreate_( deferCreate )
		{;}

		virtual ~FieldElementFinfoBase() {
			if ( setNum_ )
				delete setNum_;
			if ( getNum_ )
				delete getNum_;
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

		// Virtual function to look up type of FieldElementFinfo, not
		// defined here.  
		// virtual string rttiType() const = 0;

	protected:
		DestFinfo* setNum_;
		DestFinfo* getNum_;
		const Cinfo* fieldCinfo_;
		unsigned int defaultSize_;
		bool deferCreate_;
};

template < class T, class F > class FieldElementFinfo: public FieldElementFinfoBase
{
	public:

		FieldElementFinfo( 
			const string& name, 
			const string& doc, 
			const Cinfo* fieldCinfo,
			F* ( T::*lookupField )( unsigned int ),
			void( T::*setNumField )( unsigned int num ),
			unsigned int ( T::*getNumField )() const,
			unsigned int defaultSize,
			bool deferCreate = 0
		)
			: FieldElementFinfoBase( name, doc, fieldCinfo, 
				defaultSize, deferCreate ),
				lookupField_( lookupField ),
				setNumField_( setNumField ),
				getNumField_( getNumField )
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

		/**
		 * Virtual function
		 */
		void postCreationFunc( Id parent, Element* parentElm ) const
		{
			if ( deferCreate_ )
				return;
			Id kid = Id::nextId();
			FieldDataHandlerBase* fdh = 
				new FieldDataHandler< T, F >(
					fieldCinfo_->dinfo(),
					parentElm->dataHandler(),
					defaultSize_,
					lookupField_, getNumField_, setNumField_ );
			new Element( kid, fieldCinfo_, name(), fdh );
			adopt( parent, kid );
			// This function applies during a copy, in which case we
			// already know the size of the array. So we shouldn't have
			// to call this.
			// fdh->setMaxFieldEntries( fdh->biggestFieldArraySize() );
		}

		/// Virtual function to look up type of FieldElementFinfo
		string rttiType() const {
			return Conv<F>::rttiType();
		//	return "";

		}

	private:
		F* ( T::*lookupField_ )( unsigned int );
		void( T::*setNumField_ )( unsigned int num );
		unsigned int ( T::*getNumField_ )() const;
};


#endif // _FIELD_ELEMENT_FINFO_H
