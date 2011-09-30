/*******************************************************************
 * File:            TableIterator.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-04-25 16:52:58
 ********************************************************************/
#ifndef _TABLEITERATOR_H
#define _TABLEITERATOR_H
#include "Interpol.h"
namespace pymoose
{

    /**
       This class is for allowing iteration over the table in an interpol object
       The swig wrapper should extend it with the help of __next__ method to implement the next() method
    */
    class TableIterator
    {
      public:
        TableIterator(Interpol* table):table_(table)
        {
            next_ = 0;            
        }
        
        TableIterator* __iter__()
        {
            return  this;            
        }
        bool __hasNext__()
        {
            if (next_ < table_->__len__())
                return true;
            else
                return false;            
        }
        double __next__()
        {
            return table_->__getitem__(next_++);            
        }
        
        // the next() method is to be implemented 
      private:
        Interpol *table_;
        int next_;        
    };
}

#endif
