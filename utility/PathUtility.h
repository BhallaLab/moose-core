/*******************************************************************
 * File:            PathUtility.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-05 12:28:19
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

#ifndef _PATHUTILITY_H
#define _PATHUTILITY_H
#include <string>
#include <vector>
#include "StringUtil.h"
using namespace std;

class PathUtility
{
  public:
    static const char* const PATH_SEPARATOR;
    static const char* const DIR_SEPARATOR;
    PathUtility(string paths);    
    static bool isAbsolutePath(string path);
    void addPath(string paths);
    string getPath(int index);
    string getAllPaths();    
    string makeFilePath(string fileName, int index);    
    size_t size();
    
  private:
    vector <string> path_;
};

    
#endif
