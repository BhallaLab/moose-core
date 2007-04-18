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
#include <map>
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
			    static std::string full_type(std::string type)
    {
        static map < std::string, std::string > type_map;
        if (type_map.find("j") == type_map.end())
        {
            type_map["j"] = "unsigned int";
            type_map["i"] = "int";        
            type_map["f"] = "float";        
            type_map["d"] = "double";        
            type_map["Ss"] = "string";        
            type_map["s"] = "short";
            type_map["b"] = "bool";            
        }
        const map< std::string, std::string >::iterator i = type_map.find(type);
        if (i == type_map.end())
        {
            cout << "Not found - " << type << endl;
            
            return type;
        }
        else 
        {
            return i->second;
        }
    }

virtual std::string getTemplateParameters() const
    {
        return "void";        
    }

};

#endif // _FTYPE_H
