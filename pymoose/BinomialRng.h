/*******************************************************************
 * File:            pymoose/BinomialRng.h
 * Description:      SWIG wrapper class for BinomialRng. This class
 *                      deviates from the general pattern of pymoose
 *                      classes. It has been manually modified from
 *                      the generated class definition.
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-11-30 20:20:09
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

#ifndef _pymoose_BinomialRng_h
#define _pymoose_BinomialRng_h
#include "RandGenerator.h"
namespace pymoose
{
    class BinomialRng : public RandGenerator
    {
      public:
        static const std::string className_;
        BinomialRng(Id id);
        BinomialRng(std::string path);
        BinomialRng(std::string name, Id parentId);
        BinomialRng(std::string name, PyMooseBase& parent);
        BinomialRng(const BinomialRng& src,std::string name, PyMooseBase& parent);
        BinomialRng(const BinomialRng& src,std::string name, Id& parent);
        BinomialRng(const Id& src,std::string name, Id& parent);
        BinomialRng(const BinomialRng& src,std::string path);
        BinomialRng(const Id& src,std::string path);
        ~BinomialRng();
        const std::string& getType();
        int __get_n() const;
        void __set_n(int n);
        double __get_p() const;
        void __set_p(double p);
    };
}// namepsace pymoose

#endif
