/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FUNC_VEC_H
#define _FUNC_VEC_H

/**
 * The FuncVec class manages all the functions related to a Finfo,
 * typically a SrcFinfo or a SharedFinfo. The system sets up unique
 * ids for each FuncVec that are independent of setup order of the
 * various FuncVecs.
 * The FuncVec also holds a flag 'isDest' to indicate whether it is a 
 * destination or a source. This is usually true, but if it is empty
 * or if it is part of a shared message on the source side it is false.
 */
class FuncVec
{
	public:
		FuncVec( const string& className, const string& finfoName );

		/// addFunc pushes a new function onto the FuncVec.
		void addFunc( RecvFunc func, const Ftype* ftype );

		unsigned int size() const {
			return static_cast< unsigned int >( func_.size() );
		}
		/**
 		* Makes a trigger FuncVec from the current one. Used by ValueFinfos.
 		*/
		void makeTrig();
		/**
 		* Makes a LookupFinfo FuncVec from the current one. 
		* Used by LookupFinfos.
 		*/
		void makeLookup();

		/**
		 * Looks up the function specified by funcNum
		 */
		RecvFunc func( unsigned int funcNum ) const;

		/**
		 * Comparison operator needed to sort all the funcVecs so
		 * that the indexing can be the same across nodes or even
		 * architectures. 
		 */
		bool operator<( const FuncVec& other );

		/**
		 * parFuncSync returns a destination function that will handle
		 * the identical arguments and package them for sending to a
		 * remote node on a parallel message call.
		 * The Sync means that this func is for synchronous data.
		 */
		RecvFunc parFuncSync( unsigned int funcNum ) const;

		/**
		 * parFuncAsync returns a destination function that will handle
		 * the identical arguments and package them for sending to a
		 * remote node on a parallel message call.
		 * The Async means that this func is for asynchronous data.
		 */
		RecvFunc parFuncAsync( unsigned int funcNum ) const;

		/**
		 * id returns the identifier of the funcVec
		 */
		unsigned int id() const {
			return id_;
		}

		/**
		 * trigId returns the identifier of the trigFuncVec if it exists.
		 */
		unsigned int trigId() const;

		/**
		 * lookupId returns the id of the lookupFuncVec if it exists.
		 */
		unsigned int lookupId() const;

		/**
		 * True if this funcVec handles any destination functions,
		 * or if the SharedFinfo that manages this funcVec
		 * has been set up as a destination.
		 */
		bool isDest() const {
			return isDest_;
		}

		/**
		 * fType returns a vector of Ftypes for the functions
		 */
		const vector< const Ftype* >& fType() const;

		/**
		 * name returns the name string, consisting of className.finfoName
		 */
		const string& name() const;

		void setDest() {
			isDest_ = 1;
		}

	////////////////////////////////////////////////////////////////////
	//  Some static functions here.
	////////////////////////////////////////////////////////////////////

		/**
 		* This function returns the FuncVec belonging to the specified id
 		*/
		static const FuncVec* getFuncVec( unsigned int id );

		/**
		 * The sortFuncVec function sorts all the FuncVecs and assigns
		 * ids to each according to the sort results. 
		 * To be called only at the end of static initialization.
		 */
		static void sortFuncVec();


	private:
		string name_; // className.finfoName
		vector< RecvFunc > func_;
		vector< RecvFunc > parFuncSync_;
		vector< RecvFunc > parFuncAsync_;
		vector< const Ftype* > funcType_;
		unsigned int id_; // Identifier for it across nodes.
		bool isDest_;	// Is the Finfo a destination?
		FuncVec* trigFuncVec_; // Points to trigger FuncVec if it exists.
		FuncVec* lookupFuncVec_; // Points to lookup FuncVec if it exists.
};

#endif // _FUNC_VEC_H
