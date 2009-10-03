/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _CINFO_H
#define _CINFO_H

class DinfoBase;

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
					const Cinfo* baseCinfo, // Base class
					Finfo** finfoArray,	// Field information array
					unsigned int nFinfos,
					DinfoBase* d,	// A handle to lots of utility functions for the Data class.
					struct SchedInfo* schedInfo = 0,
					unsigned int nSched = 0
			);

			~Cinfo();

			void init( Finfo** finfoArray, unsigned int nFinfos );

			OpFunc* getOpFunc( FuncId fid ) const;

			// Some dummy funcs
			const std::string& name() const;

			/**
			 * Finds the Cinfo with the specified name.
			 */
			static const Cinfo* find( const std::string& name );

			/**
			 * Finds Finfo by name in the list for this class, 
			 * ignoring any element-specific fields.
			 */
			const Finfo* findFinfo( const string& name) const;

			/**
			 * Finds the funcId by name. Returns 0 on failure.
			 */
			const FuncId findFuncId( const string& name) const;

			/**
			 * Creates a new Element. Assigns a new Id or takes one you give
			 */
			Id create( const string& name, unsigned int numEntries ) const;

			/**
			 * Destroys data on element
			 */
			void destroy( Data* d ) const;

		private:
			const string name_;

			std::map< std::string, std::string > doc_;
			// const std::string author_;
			// const std::string description_;
			const Cinfo* baseCinfo_;
			const DinfoBase* dinfo_;

			map< string, Finfo* > finfoMap_;
			vector< OpFunc* > funcs_;
			map< string, FuncId > opFuncNames_;

			static map< string, Cinfo* >& cinfoMap();

			// Many opfuncs share same FuncId
			// static map< OpFunc*, FuncId >& funcMap();
};

#endif // _CINFO_H
