/*******************************************************************
 * File:            Class.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-10-24 15:58:50
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

#ifndef _pymoose_Class_h
#define _pymoose_Class_h
#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose{

    class Class : public Neutral
    {
      public:
        static const std::string className_;
        Class(Id id);
        Class(std::string path, std::string name);
        Class(std::string name, Id parentId);
        Class(std::string name, PyMooseBase& parent);
        Class(const Class& src,std::string name, PyMooseBase& parent);
        Class(const Class& src,std::string name, Id& parent);
        Class(const Id& src,std::string name, Id& parent);
        Class(const Class& src,std::string path);
        Class(const Id& src,std::string path);
        ~Class();
        const std::string& getType();
        std::string __get_name();
        void __set_name(std::string name);
        const std::string __get_author();
        const std::string __get_description();
        unsigned int __get_tick();
        void __set_tick(unsigned int);
        unsigned int __get_stage();
        void __set_stage(unsigned int);
        std::string __get_clock();
        void setClock(std::string function, std::string clock);    
    };
}

#endif
