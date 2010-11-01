/*******************************************************************
 * File:            PyMooseIterable.h
 * Description:      This class is an attempt to overcome the
 *                      problem of writing different classes for
 *                      each iterable type ( which are not
 *                      present in STL ) members of moose classes.
 *                  For example, the SynChan class has a vector of
 *                      weights and a vector of delays. We should be
 *                      able to use them like in MOOSE -
 *                      mySynChan.weight[5]
 *                      for this weight needs to belong to a class
 *                      that has __setitem__ and __getitem__ methods.
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-12-03 11:14:47
 ********************************************************************/
/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU General Public License version 2
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#ifndef _PYMOOSEITERABLE_H
#define _PYMOOSEITERABLE_H
namespace pymoose
{

    template <typename KeyType, typename ValueType> class PyMooseIterable 
    {
      public:
        void __setitem__(KeyType index, ValueType value)
        {
            __setItem(index, value);        
        }
        ValueType __getitem(KeyType index)
        {
            return __getItem(index);
        }
        void (*__setItem)(KeyType index,ValueType value);
        ValueType (*__getItem)(KeyType index);    
    };

    template <typename OuterType, typename KeyType, typename ValueType> class InnerPyMooseIterable
    {
      public:
        InnerPyMooseIterable ()
        {
            outer_ = 0;
            __setItem = 0;
            __getItem = 0;        
        }
    
        InnerPyMooseIterable ( OuterType* outer ,
                               ValueType (OuterType::*getItem)(KeyType index) const,
                               void (OuterType::*setItem)(KeyType index,ValueType value))
        {
            outer_ = outer;
            __setItem = setItem;
            __getItem = getItem;        
        }
        void __setitem__(KeyType index, ValueType value)
        {
            (outer_->*(__setItem))(index, value);        
        }
        ValueType __getitem__(KeyType index)
        {
            return (outer_->*(__getItem))(index);
        }
      private:
        void (OuterType::*__setItem)(KeyType index,ValueType value);
        ValueType (OuterType::*__getItem)(KeyType index) const;
        OuterType* outer_;
    
    };

    
}

    
#endif
