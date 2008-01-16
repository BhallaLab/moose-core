/*******************************************************************
 * File:            Cell.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-24 15:54:26
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

#ifndef _pymoose_Cell_h
#define _pymoose_Cell_h
#include "PyMooseBase.h"
namespace pymoose
{
    
    class Cell : public PyMooseBase
    {
      public:
        static const std::string className;
        Cell(Id id);
        Cell(std::string path);
        Cell(std::string name, Id parentId);
        Cell(std::string name, PyMooseBase* parent);
        ~Cell();
        const std::string& getType();
    };
}// namepsace pymoose
    
#endif
