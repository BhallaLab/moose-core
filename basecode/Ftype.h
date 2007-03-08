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

/*
enum FinfoIdentifier { VALUE_SET, VALUE_TRIG, 
		ARRAY_SET, ARRAY_TRIG, 
		NEST_SET, NEST_TRIG };
		*/

/**
 * Virtual base class for typing information. 
 */
class Ftype
{
		public:
			Ftype()
			{;}

			virtual ~Ftype()
			{;}

			virtual unsigned int nValues() const = 0;

			/**
			 * This virtual function is used to compare two 
			 * instantiated Ftypes. If you just want to check if the
			 * instantiated Ftype is of a given Ftype, use
			 * FtypeN<T>::isA( const Ftype* other );
			 * which is a static function.
			 */
			virtual bool isSameType( const Ftype* other ) const = 0;
			virtual size_t size() const = 0;

			virtual RecvFunc recvFunc() const = 0;
			virtual RecvFunc trigFunc() const = 0;

			/**
			 * StrGet extracts the value, converts it to a string,
			 * and returns true if successful
			 */
			virtual bool strGet( const Element* e, const Finfo* f,
					string& s ) const {
					s = "";
					return 0;
			}
			
			/**
			 * StrSet takes a string, converts it to the value,
			 * does the assignment and returns true if successful
			 */
			virtual bool strSet( Element* e, const Finfo* f,
					const string& s ) const {
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
};

#endif // _FTYPE_H
