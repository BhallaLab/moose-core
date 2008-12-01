// Helpless.cpp --- 
// 
// Filename: Helpless.cpp
// Description: Contains a set of funny(?) lines to present to the
// user when no other help is available.
// Author: Subhasis Ray
// Maintainer: 
// Created: Mon Nov 24 10:26:16 2008 (+0530)
// Version: 
// Last-Updated: Tue Dec  2 03:43:57 2008 (+0530)
//           By: Subhasis Ray
//     Update #: 16
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
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
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
** Development of this software was supported by
** Biophase Simulations Inc, http://www.bpsims.com
** See the file BIOPHASE.INFO for details.
**********************************************************************/


// Code:
#include <string>
#include "randnum/randnum.h"

const std::string& helpless(void)
{
    static const std::string helpless_[] = 
        {
            "Not documented.",
            "The documentation for this is yet to be written.",
            "Please bug the developers to provide documentation.",
            "Please help us by submitting a bug report for lack of documentation.",
            "Developers too lazy to help with docs.",
            "You are right, developers hate to write docs.",
            "Don't hesitate - send a mail to the developers for documentation. It's there fault!",
            "Lazy programmers need a nerve pinch to output some documentation",
            "Play me online? Well you know that I'll beat you \nIf I ever meet you, I'll Control-Alt-Delete you. - Weird Al Yankovic, It's All About the Pentiums",
            "Remember, if you ever need a helping hand, you'll find one at the end of your arm. - Audrey Hepburn",
            "When you can't have what you want, it's time to start wanting what you have. -Kathleen A. Sutton",
            "When it gets dark enough you can see the stars. -Lee Salk",
            "You may not realize it when it happens, but a kick in the teeth may be the best thing in the world for you. - Walt Disney",
        };
    // Avoid the teasers - they are irritating and clutter the screen.
    return helpless_[0];
#if 0    
    unsigned long item = genrand_int32()%(sizeof(helpless_)/sizeof(std::string));
    return helpless_[item];    
#endif
}


// 
// Helpless.cpp ends here
