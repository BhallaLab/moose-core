
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _CINFO_H
#define _CINFO_H

/**
 * Class to manage class information for all the other classes.
 */
class Cinfo
{
		public:
			/**
			 * The Cinfo intializer is used for static initialization
			 * of all the MOOSE Cinfos. Each MOOSE class must set up
			 * a function to build its Cinfo. This function must be
			 * called statically in the MOOSE class .cpp file. 
			 * Note how it takes the base *Cinfo as an argument. This
			 * lets us call the base Cinfo initializer when making
			 * each Cinfo class, thus ensuring the correct static
			 * initialization sequence, despite the somewhat loose
			 * semantics for this sequence in most C++ compilers.
			 */
			Cinfo( const std::string& name,
					const std::string& author,
					const std::string& description,
					const Cinfo* baseCinfo,
					Finfo** finfoArray,
					unsigned int nFinfos,
					const Ftype* ftype
			);

			const std::string& name() const {
					return name_;
			}

			static const Cinfo* find( const std::string& name );

			/**
			 * Finds Finfo on an element based on the name of the Finfo.
			 * Checks the match with 
			 * the element first, in case there is something overridden
			 */
			const Finfo* findFinfo( Element* e, const string& name )
					const;

			/**
			 * Finds Finfo on an element based on the connIndex.
			 */
			const Finfo* findFinfo( 
					const Element* e, unsigned int connIndex) const;

			/**
			 * Finds Finfo by name in the list for this class, 
			 * ignoring any element-specific fields.
			 */
			const Finfo* findFinfo( const string& name) const;

			static void initialize();

			bool isA( const Cinfo* other ) const {
				return ftype_->isSameType( other->ftype_ );
			}

			Element* create( const string& name ) const ;
			Element* create( const string& name, void* data ) const;

			void destroy( void* ) const ;

			const Ftype* ftype() const {
					return ftype_;
			}

			void listFinfos( vector< const Finfo* >& flist ) const;

			unsigned int getSlotIndex( const string& name ) const;

		private:
			const std::string name_;
			const std::string author_;
			const std::string description_;
			const Cinfo* baseCinfo_;
			vector< Finfo* > finfos_;
			const Cinfo* base_;
			const Ftype* ftype_;
			Finfo* thisFinfo_;
			unsigned int nSrc_;
			unsigned int nDest_;

			static std::map< std::string, Cinfo* >& lookup();
};

#endif // _CINFO_H
