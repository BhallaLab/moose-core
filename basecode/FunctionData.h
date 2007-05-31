/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FUNCTION_DATA_H
#define _FUNCTION_DATA_H

class FunctionData
{
	public:
		FunctionData( RecvFunc func, const Ftype* type, unsigned int index);

		RecvFunc func() const {
			return func_;
		}

		unsigned int index() const {
			return index_;
		}

		const Ftype* funcType() const {
			return type_;
		}

	private:
		RecvFunc func_;
		const Ftype* type_;
		unsigned int index_;

};

class FunctionDataManager
{
	public:
		/// creates a new FunctionData and inserts into the map and vector.
		const FunctionData* add( RecvFunc func, const Ftype* type );
		const FunctionData* find( RecvFunc rf );
		const FunctionData* find( unsigned int index );
		
	private:
		map< RecvFunc, const FunctionData* > funcMap_;
		vector< const FunctionData* > funcVec_;
};

extern FunctionDataManager* getFunctionDataManager();
extern const FunctionData* lookupFunctionData( RecvFunc rf );
extern const FunctionData* lookupFunctionData( unsigned int index );

#endif // _FUNCTION_DATA_H
