// help.cpp --- 
// 
// Filename: help.cpp
// Description: Retrieve and display documentation 
// Author: Subhasis Ray
// Maintainer: 
// Created: Tue Nov 25 11:03:34 2008 (+0530)
// Version: 
// Last-Updated: Wed Nov 26 21:08:39 2008 (+0530)
//           By: Subhasis Ray
//     Update #: 167
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
   documentation is to be retrieved. If NULL, empty string is returned.

   fieldName 
   - if empty, the class documentation ( Name, Author,
   Description) and a list of field names with data type is
   returned. 
   - if "-full", the class documentation and a list of field names
   with data type and available documentation is returned.
   - otherwise, the field name, data type and documentation string for
   the specified field is returned.

   NOTE: for UNIX systems, we add ANSII escape sequences to print the
   field titles in bold. But this assumes the terminal to be VT100
   compatible and is not portable.
 */
const std::string& getCinfoDoc(const Cinfo* cinfo, const std::string& fieldName)
{
    static const Cinfo* previousCinfo = 0; 
    static std::string doc = "";
    static std::string previousFieldName = "-";// initialize to impossible field
    
    static std::vector <const Finfo*> finfoList;

    std::string docstr = "";
    
    // check if arguments are identical to previous call
    if (previousCinfo == cinfo && fieldName == previousFieldName)
    {
        return doc; // return buffered documentation string
    }
    
    doc = ""; // clear the doc string
    if (!cinfo)
    {
        return doc;
    }
    
    previousFieldName = fieldName;
    
    if (previousCinfo != cinfo) // it is a new cinfo, not buffered
    {
        cinfo->listFinfos(finfoList);
        previousCinfo = cinfo;
    }

    if (fieldName.empty() || fieldName == "-full") // get class documentation
    {
        doc.append("\n").append(START_BOLD).append("Name        :  ").append(END_BOLD).append(cinfo->name()).append("\n");
        doc.append("\n").append(START_BOLD).append("Author      :  ").append(END_BOLD).append(cinfo->author()).append("\n");
        doc.append("\n").append(START_BOLD).append("Description :  ").append(END_BOLD).append(cinfo->description()).append("\n");
        doc.append("\n").append(START_BOLD).append("Fields      :  ").append(END_BOLD).append(cinfo->description()).append("\n");
        if (fieldName == "-full")
        {
            for ( vector <const Finfo* >::iterator iter = finfoList.begin();
                  iter != finfoList.end();
                  ++iter)
            {
                docstr = (*iter)->doc();
                if (trim(docstr).empty())
                {
                    docstr = helpless();
                }
                doc.append("\n").
                    append(START_BOLD).
                    append((*iter)->name()).
                    append(END_BOLD).
                    append(": ").
                    append((*iter)->ftype()->getTemplateParameters()).
                    append("\n").
                    append(docstr).
                    append("\n");
            }
        } //! if (fieldName == "-full")
        else 
        {
            for ( vector <const Finfo* >::iterator iter = finfoList.begin();
                  iter != finfoList.end();
                  ++iter)
            {
                doc.append("\n").
                    append(START_BOLD).
                    append((*iter)->name()).
                    append(END_BOLD).
                    append(": ").
                    append((*iter)->ftype()->getTemplateParameters()).
                    append("\n");
            }
        } //! if (fieldName == "-full")
    } //!if (fieldName.empty() || fieldName == "-full")
    else
    {
        const Finfo* finfo = cinfo->findFinfo(fieldName);
        if (!finfo)
        {
            doc.append("\n").
                append(START_BOLD).
                append(cinfo->name()).
                append(".").
                append(fieldName).
                append(END_BOLD).
                append(": No such field\n");
            return doc;
        }
        
        docstr = finfo->doc();
        if (trim(docstr).empty())
        {
            docstr = helpless();
        }
        doc.append("\n").
            append(START_BOLD).
            append(fieldName).
            append(END_BOLD).
            append(": \n").
            append(docstr).
            append("\n");
    } //!if (fieldName.empty() || fieldName == "-full")
    doc.append("\n");
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

const std::string& getClassDoc(const std::string& args)
{
    string target = args;
    string field = "";
    string::size_type field_start = target.find_first_of(".");
    if ( field_start != string::npos)
    {
        // May we need to go recursively?
        // Assume for the time being that only one level of field
        // documentation is displayed. No help for channel.xGate.A
        // kind of stuff.
        field = target.substr(field_start+1); 
        target = target.substr(0, field_start);
    }

    const Cinfo * classInfo = Cinfo::find(target);
    return getCinfoDoc(classInfo, field);
}


// 
// help.cpp ends here
