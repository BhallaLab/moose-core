/*******************************************************************
 * File:            PathUtility.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-05 12:29:37
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

#ifndef _PATHUTILITY_CPP
#define _PATHUTILITY_CPP
#include "PathUtility.h"
#include "StringUtil.h"
#include <iostream>
#include <cstring>

#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__MACH__)

const char* const PathUtility::PATH_SEPARATOR = ":";
const char* const PathUtility::DIR_SEPARATOR = "/";

#elif defined(_WIN32) //if defined WINDOWS

const char* const PathUtility::PATH_SEPARATOR = ";";
const char* const PathUtility::DIR_SEPARATOR = "\\";
#else
#error System type is neither Unix nor Windows. Failing
#endif // UNIX / WINDOWS

PathUtility::PathUtility(string paths)
{
    addPath(paths);        
}
/**
   append the directories in paths string to the list of directories.
*/
void PathUtility::addPath(string paths)
{
    if ( paths.length() == 0 )
    {
        return;
    }
    
    string path;    
    size_t start = 0;
    size_t pos = paths.find(PathUtility::PATH_SEPARATOR, start);        
        
    while ( ( pos != string::npos ))
    {
        path = trim(paths.substr(start, pos-start));
        if ( path.length() > 0 )
        {
            size_t trailing_sep =  path.rfind(DIR_SEPARATOR);
            
            if ( trailing_sep == path.length() - strlen(DIR_SEPARATOR) )
            {
                path = path.substr(0, trailing_sep);
            }
            path_.push_back(path);            
        }
        
        start = pos+1;
        if (start >= paths.length()){
            break;
        }
        pos = paths.find(PATH_SEPARATOR, start);
    }
    path = trim(paths.substr(start));
    if ( path.length() > 0 )
    {
         size_t trailing_sep =  path.rfind(DIR_SEPARATOR);
            
         if ( trailing_sep == path.length() - strlen(DIR_SEPARATOR) )
         {
             path = path.substr(0, trailing_sep);
         }
           
         path_.push_back(path);
    }
}

/**
   Check if given path is an absolute path. On *nix systems check if first
   character is '/'. On windows, check if second character is ':', as in
   "C:\path\to\file".
*/
bool PathUtility::isAbsolutePath(string path)
{
#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__MACH__)
    return path.find( DIR_SEPARATOR ) == 0;
#elif defined(_WIN32) //if defined WINDOWS
    return path.find( ":" ) == 1;
#else
#error System type is neither Unix nor Windows. Failing
#endif // UNIX / WINDOWS
}

/**
   return index-th entry in list of directories in path.
*/
string PathUtility::getPath(int index)
{
    return path_[index];
}
/**
   returns all the paths in list separated by
   PathUtility::PATH_SEPARATOR. This is the inverse of addPath.
   
*/
string PathUtility::getAllPaths()
{
    string path="";
    for ( unsigned int i = 0; i < path_.size(); ++i )
    {
        path.append(path_[i]);
        path.append(PathUtility::PATH_SEPARATOR);        
    }
    if ( path.length() > 1)
    {
        path = path.substr(0,path.length() - 1);
    }
    
    return path;
}

/**
   take the index-th directory in list and append fileName to it to
   make a full file path and return it.
*/
string PathUtility::makeFilePath(string fileName, int index)
{
    return path_[index]+DIR_SEPARATOR+fileName;
}

/**
   The number of entries in path list
*/
size_t PathUtility::size()
{
    return path_.size();
}

#ifdef TEST_PATHUTILITY
#include <iostream>
#include "StringUtil.h"
using namespace std;

int main(int argc, char **argv)
{
    string path = ".:/home/subha:/home:/usr/local/share/:/tmp";
    PathUtility pu(path);
    cout << "Path " << path << " has been divided into: " << endl;
    for ( unsigned i = 0; i < pu.size(); ++i )
    {
        cout << pu.getPath(i) << endl;
    }
    pu.addPath("/mnt/disk : /home/genesis/");
    cout << "Path " << path << " has been divided into: " << endl;
    for ( unsigned i = 0; i < pu.size(); ++i )
    {
        cout << pu.getPath(i) << endl;
    }
    cout << "Making path for file test.c :" <<endl;
    for ( unsigned i = 0; i < pu.size(); ++i )
    {
        cout << pu.makeFilePath("test.c", i) << endl;
    }
    
    return 0;    
}

#endif


#endif
