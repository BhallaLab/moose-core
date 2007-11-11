
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

struct SchedInfo
{
	const Finfo* finfo;
	unsigned int tick;
	unsigned int stage;
};

/**
 * Class to manage class information for all the other classes.
 */
class Cinfo
{
    friend class Class;
    
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
					const Ftype* ftype,
					struct SchedInfo* schedInfo = 0,
					unsigned int nSched = 0
			);

			~Cinfo();

			const std::string& name() const {
					return name_;
			}

			/**
			 * Finds the Cinfo with the specified name.
			 */
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

			/**
			 * Returns true if 'other' is the same class or a base
			 * class of the calling Cinfo.
			 */
			bool isA( const Cinfo* other ) const;

			Element* create( Id id, const string& name ) const ;
			Element* create( Id id, const string& name, 
							void* data, bool noDelFlag = 0 ) const;

			Element* createArray( Id id, const string& name,
				unsigned int numEntries, size_t objectSize ) const ;
			Element* createArray( Id id, const string& name, 
					void* data, unsigned numEntries, size_t objectSize,
					bool noDelFlag = 0 ) const;

			bool schedule( Element* e ) const;
			// void destroy( void* ) const ;

			const Ftype* ftype() const {
					return ftype_;
			}

			void listFinfos( vector< const Finfo* >& flist ) const;

			unsigned int getSlotIndex( const string& name ) const;
			const Finfo* getThisFinfo() const {
				return thisFinfo_;
			}

		private:
			const std::string name_;
			const std::string author_;
			const std::string description_;
			const Cinfo* baseCinfo_;
			vector< Finfo* > finfos_;
			const Ftype* ftype_;
			vector < SchedInfo > scheduling_;
			// const Cinfo* base_;
			/**
			 * These two fields hold the Finfo that refers back to
			 * this Cinfo for class information. The noDelFinfo_
			 * is used where the data field of the Element must
			 * not be freed at delete time. The only difference it
			 * has is the state of the noDeleteFlag.
			 */
			Finfo* thisFinfo_;
			Finfo* noDelFinfo_;
			unsigned int nSrc_;
			unsigned int nDest_;

			static std::map< std::string, Cinfo* >& lookup();
};

#endif // _CINFO_H
