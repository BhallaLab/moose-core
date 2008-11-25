// help.cpp --- 
// 
// Filename: help.cpp
// Description: Retrieve and display documentation 
// Author: Subhasis Ray
// Maintainer: 
// Created: Tue Nov 25 11:03:34 2008 (+0530)
// Version: 
// Last-Updated: Tue Nov 25 15:01:07 2008 (+0530)
//           By: Subhasis Ray
//     Update #: 75
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
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2008 Upinder S. Bhalla and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// Code:
#include <string>
#include <fstream>

#include "basecode/header.h"
#include "basecode/moose.h"
#include "PathUtility.h"
#include "Property.h"

extern const std::string& helpless();

#if defined(unix) || defined(__unix__) || defined(__unix)
const char START_BOLD[] = {27, '[', '1', 'm', '\0'};
const char END_BOLD[] = {27, '[', '0', 'm', '\0'};
#else
const char START_BOLD[] = {'\0'};
const char END_BOLD[] = {'\0'};
#endif // if defined(unix)

/**
   cinfo - pointer to the Cinfo instance for the class whose
   documentation is to be retrieved.

   fieldName - if empty, the full class documentation ( Name, Author,
   Description and documentation for each field) is
   returned. Otherwise, only the documentation string for the
   specified field is returned.

   Also, for UNIX systems, we add ANSII escape sequences to print the
   field titles in bold. But this assumes the terminal to be VT100
   compatible and is not portable.
 */
const std::string& getCinfoDoc(const Cinfo* cinfo, const std::string& fieldName)
{
    static std::string doc = "";
 
    std::string docstr = "";
    
    doc = ""; // clear the doc string
    
    if (trim(fieldName).empty()) // no field name - get full class documentation
    {
        doc.append("\n").append(START_BOLD).append("Name        :  ").append(END_BOLD).append(cinfo->name()).append("\n");
        doc.append("\n").append(START_BOLD).append("Author      :  ").append(END_BOLD).append(cinfo->author()).append("\n");
        doc.append("\n").append(START_BOLD).append("Description :  ").append(END_BOLD).append(cinfo->description()).append("\n");
        vector <const Finfo* > finfoList;
        cinfo->listFinfos(finfoList);
        for ( vector <const Finfo* >::iterator iter = finfoList.begin();
              iter != finfoList.end();
              ++iter)
        {
            // TODO: it would have been nicer if we could print the data
            // type also - Ftype::fulle_type, Ftype::getTemplateParameters
            // are sued in pymoose code generator. But my experience
            // is that C++ RTTI is unreliable - in particular GCC produces
            // human-unreadable typename
            docstr = (*iter)->doc();
            if (trim(docstr).empty())
            {
                docstr = helpless();
            }

            doc.append("\n").append(START_BOLD).append((*iter)->name()).append(END_BOLD).append(": \n").append(docstr).append("\n");
        }
    }

    else
    {
        const Finfo* finfo = cinfo->findFinfo(fieldName);
        docstr = finfo->doc();
        if (trim(docstr).empty())
        {
            docstr = helpless();
        }
        doc.append("\n").append(START_BOLD).append(fieldName).append(END_BOLD).append(": \n").append(docstr).append("\n");
    }

    doc.append("\r");
    return doc;
}


/**
   return documentation for builtin commands.
   Currently does not do much.
*/
const std::string& getCommandDoc(const std::string& command)
{
    static std::string doc;
    string filename = Property::getProperty(Property::DOCPATH);
    
    filename.append(PathUtility::DIR_SEPARATOR).append(command);
    cout << filename << endl; // for testing
    string line;
    std::ifstream docfile(filename.c_str());
    if (docfile.is_open()){
        doc = string("\n").append(START_BOLD).append(command).append(END_BOLD).append(":\n"); 
        while (! docfile.eof() ){
            std::getline (docfile,line);
            doc.append(line).append("\n");            
        }
        docfile.close();
    }
    else {
        doc = "Help not present for this command.\n";
    }

    return doc;
}


// 
// help.cpp ends here
