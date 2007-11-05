/*******************************************************************
 * File:            RandGenerator.cpp
 * Description:     Interface class for MOOSE to access various
 *                  random number generator.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-03 21:48:17
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

#ifndef _RANDGENERATOR_CPP
#define _RANDGENERATOR_CPP
const Cinfo * initRandGeneratorCinfo()
{
    static Finfo* processShared[] = 
        {
            new DestFinfo("process", Ftype1< ProcInfo >::global(),
                          RFCAST( &RandGenerator::processFunc )),
            new DestFinfo("reinit", Ftype1<ProcInfo >::global(),
                          RFCAST( &RandGenerator::reinitFunc)),
        };
    static Finfo* process = new SharedFinfo("process", processShared,
                                            sizeof(processShared)/sizeof(Finfo*));
    
    
}

#endif
