/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FTYPE_H
#define _FTYPE_H

#include <typeinfo>
class FuncVec;

using namespace std;
class Conn;
class Finfo;

/*
enum FinfoIdentifier { VALUE_SET, VALUE_TRIG, 
		ARRAY_SET, ARRAY_TRIG, 
		NEST_SET, NEST_TRIG };
		*/

/*
 * This typedef is used for functions converting serial data
 * into calls to messages on the target end of a parallel message.
 * The IncomingFunc munches through serial data stream to send data to
 * destination objects. This function is called from 
 * PostMaster on target node. Returns the data pointer
 * incremented with the size of the fields in the Ftype.
 * Index looks up the message slot (MsgSrc) to send from.
typedef const void* ( *IncomingFunc )( 
			const Conn* c, const void* data, RecvFunc rf );
 */

/**
 * Virtual base class for typing information. 
 */
class Ftype
{
		public:

			/**
			 * This initialization function sets up the FuncVecs defined
			 * by the Ftype for parallel message passing.
			 */
			Ftype( const string& name );

			/**
			 * Common destructor. We don't really need to do anything.
			 */
			virtual ~Ftype()
			{;}

			/**
			 * nValues is the number of arguments in the FecvFunc for
			 * this Ftype. It is zero for trigger functions, one for
			 * value functions, and so on.
			 */
			virtual unsigned int nValues() const = 0;

			/**
			 * This virtual function is used to compare two 
			 * instantiated Ftypes. If you just want to check if the
			 * instantiated Ftype is of a given Ftype, use
			 * FtypeN<T>::isA( const Ftype* other );
			 * which is a static function.
			 */
			virtual bool isSameType( const Ftype* other ) const = 0;

			/**
			 * This virtual function returns the base type of the Ftype,
			 * that is, the elementary template specification T1 T2 etc.
			 * For most Ftypes 'this' will do, but in some cases
			 * (ValueFtype) we need to return the simpler variant.
			 */
			virtual const Ftype* baseFtype() const {
				return this;
			}
			
			/**
			 * Size of the data contents of the Ftype. For simple
			 * types like float or double, it is just sizeof, but for
			 * things like vectors we need to specify the total number
			 * of bytes needed to serialize the data.
			 * If there are multiple data types then add all together.
			 */
			virtual size_t size() const = 0;

			virtual RecvFunc recvFunc() const = 0;
			virtual RecvFunc trigFunc() const = 0;

			/**
			 * StrGet extracts the value, converts it to a string,
			 * and returns true if successful
			 */
			virtual bool strGet( Eref e, const Finfo* f,
					string& s ) const {
					s = "";
					return 0;
			}
			
			/**
			 * StrSet takes a string, converts it to the value,
			 * does the assignment and returns true if successful
			 */
			virtual bool strSet( Eref e, const Finfo* f,
					const string& s ) const {
					return 0;
			}

			/**
			 * Returns a void* to allocated instance of converted
			 * string. Returns 0 on failure.
			 * This must be supported by Ftype1, and possibly other
			 * ftypes. In general the conversion doesn't work so we
			 * return 0 by default.
			 */
			virtual void* strToIndexPtr( const string& s ) const {
					return 0;
			}

			/**
			 * create an object of the specified type. Applies of
			 * course only to objects with a single type, ie, Ftype1.
			 */
			virtual void* create( const unsigned int num ) const
			{
				return 0;
			}

			/**
			 * Copy an object of the specified type, possibly an
			 * entire array of it if num > 1. Applies only to Ftype1.
			 * Returns the copy.
			 */
			virtual void* copy( 
					const void* orig, const unsigned int num ) const
			{
				return 0;
			}
			
			/**
			 * Not sure why Raamesh has numCopies here: I thought that 
			 * num was doing the job.
			 * \todo: Fix the numCopies.
			 */
			virtual void* copyIntoArray( 
					const void* orig, const unsigned int num, 
					const unsigned int numCopies ) const
			{
				return 0;
			}

			/**
			 * Free data of the specified type. If isArray, then
			 * the array delete[] is used.
			 * Applies only to objects with a single type.
			 */
			virtual void destroy( void* data, bool isArray ) const
			{;}

			/**
			 * Free index data. Currently used only for LookupFtype.
			 */
			virtual void destroyIndex( void* index ) const
			{;}

			/**
			 * Copy index data. Currently used only for LookupFtype
			 * in the context of DynamicFinfo.
			 */
			virtual void* copyIndex( void* index ) const
			{
				return 0;
			}

			////////////////////////////////////////////////////////
			// Here are some ftype to string conversion functions.
			////////////////////////////////////////////////////////
			// I believe these are used by the Python code.
			
			static std::string full_type(const type_info& type);

			virtual std::string getTemplateParameters() const
    		{
        		return "void";        
    		}

			/**
			 * This is a helper function for val2str and returns a 
			 * string for the Ftype. The function is specialized
			 * when the derived class is 
			 * a SharedFtype where we need to append lots of Ftypes.
			 * The function is used by val2str<Ftype*>
			 */
			virtual std::string typeStr() const
			{
				return typeid( this ).name();
			}

			////////////////////////////////////////////////////////
			// Here are some serialization functions used for
			// parallel message transfer
			////////////////////////////////////////////////////////

			/**
			 * This returns the funcId of a FuncVec whose entries are
			 * filled out by the Ftype to handle synchronous parallel
			 * messages. These function(s) simply take the arguments
			 * from the incoming message, serialize them, and put them
			 * in the buffer at locations specified by the connIndex of
			 * the message. All the target coding etc is handled by
			 * the ordering of the message, which is predefined during
			 * message setup.
			 * The FuncVec is set up by the base class constructor.
			 * We can't store the id directly because it is computed
			 * only after all the static initialization.
			 */
			unsigned int syncFuncId() const;

			/**
			 * The asyncFunc is similar to the syncFunc, but it is for
			 * data that is transmitted sporadically such as action
			 * potentials and cross-node shell calls. It stores
			 * both the value data and the conn index in the buffer.
			 * The target node figures out which target object to call
			 * using the conn index.
			 */
			unsigned int asyncFuncId() const;

			/**
			 * These Funcs calls the send<T>(... ) function in the proxy, 
			 * reading from the specified block of memory. Not really
			 * a bunch of RecvFuncs for FuncVec, but since the management 
			 * stuff is all there...
			 */
			unsigned int proxyFuncId() const;

			/*
			/// Returns the statically defined proxy functions
			virtual void proxyFunc( vector< RecvFunc >& ret ) const = 0;

			/// Returns the statically defined outgoingSync functions
			virtual void syncFunc( vector< RecvFunc >& ret ) const = 0;

			/// Returns the statically defined outgoingAsync functions
			virtual void asyncFunc( vector< RecvFunc >& ret ) const = 0;
			*/
	protected:
			/**
			 * These three functions are called during the Ftype subclass
			 * creation to add their locally generated recvfunc(s) to the 
			 * FuncVecs for handling parallel messaging
			 */
			void addProxyFunc( RecvFunc r );
			void addSyncFunc( RecvFunc r );
			void addAsyncFunc( RecvFunc r );

			/**
			 * These three variants are called for SharedFtype
			 * initialization, otherwise we'll need to tap into the
			 * FuncVecs directly.
			 */
			void addProxyFunc( const Ftype* ft );
			void addSyncFunc( const Ftype* ft );
			void addAsyncFunc( const Ftype* ft );
	private:
		/**
		 * These three fields store the FuncVec ids for the Ftype-generated
		 * functions used in parallel messaging.
		 */
		FuncVec* proxyFuncs_;
		FuncVec* asyncFuncs_;
		FuncVec* syncFuncs_;
};

#endif // _FTYPE_H
