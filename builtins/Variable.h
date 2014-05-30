// Variable.h --- 
// 
// Filename: Variable.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Fri May 30 19:37:24 2014 (+0530)
// Version: 
// Last-Updated: 
//           By: 
//     Update #: 0
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// Field element for variables in Function class.
// 
// 

// Change log:
// 
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:
#ifndef _VARIABLE_H
#define _VARIABLE_H

template< typename T> class Variable
{
public:
    Variable(){};

    virtual ~Variable(){};

    void setValue( T value)
    {
        _value = value;
    }
    
    T getValue() const
    {
        return _value;
    }

    static const Cinfo * initCinfo(const char * name);

protected:
    T _value;
};

#endif



// 
// Variable.h ends here
