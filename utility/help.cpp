// help.cpp --- 
// 
// Filename: help.cpp
// Description: Retrieve and display documentation 
// Author: Subhasis Ray
// Maintainer: 
// Created: Tue Nov 25 11:03:34 2008 (+0530)
// Version: 
// Last-Updated: Wed Dec 31 14:27:51 2008 (+0530)
//           By: subhasis ray
//     Update #: 566
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
// 2008-12-01: added terminal support
//
// 2008-12-31 14:20:04 (+0530): Modified getClassDoc to take classname
// and field name. Now the class name and field name are separated in
// the GenesisParserWrapper.
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
static int height = 24;
static int width = 78;

static char START_BOLD[] = {27, '[', '1', 'm', '\0'};
static char END_BOLD[] = {27, '[', '0', 'm', '\0'};

#ifdef USE_CURSES
#include <curses.h>
#include <term.h>


/**
   get the height and width of the screen
*/
int init_size(void)
{
    static bool inited = false;
    if(inited) return 1;
    inited = true;

    char *term;
    if ((term = getenv("TERM")) == NULL )
    {
        return 1;
    }
    int status;
    setupterm(term, 1, &status);
    
    if (status != 1)
    {
        return 1;
    }
    height = lines;
    if (columns < width)
    {
        width = columns;
    }
    if (!enter_bold_mode)
    {
        START_BOLD[0] = '\0';
        END_BOLD[0] = '\0';
    }
    return 0;
}

static const int inited = init_size();

#else
void init_size(void)
{
    static bool inited = false;
    if (inited)
    {
        return;
    }
    inited = true;
    START_BOLD[0] = '\0';
    END_BOLD[0] = '\0';
}

#endif // if defined(unix)


/**
   append the string src to dest with wrapping at position fill_col
   and leaving indent_col spaces in front.  
*/
void append_with_wrap(string& dest, const string& src, size_t indent_col)
{
    size_t length = (size_t)width - indent_col;
    size_t start = 0;
    size_t end_pos;
    
    while (start < src.length())
    {
        string to_append = (start+length >= src.length())? src.substr(start): src.substr(start, length);
        end_pos = to_append.find('\n');//look for embedded new line
        if (end_pos == string::npos)
        {
            if (start + length >= src.length())
            {
                //we have all of it, no need for word wrap
                dest.append(string(indent_col, ' ')).append(to_append.substr(0, end_pos)).append("\n");
                return;
            }                
            // no embedded new line : find a space for word wrap
            end_pos = to_append.find_last_of(" \t"); 
        }
        else 
        {
            start ++; // compensate for one new line char eaten up
        }
        
        if (end_pos == string::npos) // no space found in this string
                                     // (unlikely) - break word at fill-col
        {
            end_pos = (to_append.length() <= length)? string::npos: length;
        }
        else if (to_append[end_pos] != '\n')
        {
            start ++; // compensate for one space char eaten up
        }
        dest.append(string(indent_col, ' ')).append(to_append.substr(0, end_pos)).append("\n");
        if (end_pos != string::npos)
        {
            start += end_pos;
        }
        else
        {
            break;            
        }        
    }
}


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
    std::string tmp;
    
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
        doc.append("\n").append(START_BOLD).append("Description :  ").append(END_BOLD).append("\n");
        append_with_wrap(doc, cinfo->description(), 8);
        doc.append("\n");
        doc.append("\n").append(START_BOLD).append("Fields      :  ").append(END_BOLD).append("\n");
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
                    append(string(4, ' ')).
                    append((*iter)->name()).
                    append(END_BOLD).
                    append(": ").
                    append((*iter)->ftype()->getTemplateParameters())
                    .append("\n");
                append_with_wrap(doc, docstr, 8);
                doc.append("\n");
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
                    append(string(4, ' ')).
                    append((*iter)->name()).
                    append(END_BOLD).
                    append(": ").
                    append((*iter)->ftype()->getTemplateParameters()).
                    append("\n");
            }
        } //! if (fieldName == "-full")
    } //!if (fieldName.empty() || fieldName == "-full")
    else // A specific field documentation has been requested
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
            append(": \n");
        append_with_wrap(doc, docstr, 4);
        doc.append("\n");
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
            append_with_wrap(doc, line, 8);    
        }
        doc.append("\n");
        docfile.close();
    }
    else {
        doc = "Help not present for this command.\n";
    }

    return doc;
}

/**
   Calls getCinfoDoc to retrieve documentation.
*/
const std::string& getClassDoc(const std::string& className, const std::string& fieldName)
{
    const Cinfo * classInfo = Cinfo::find(className);
    return getCinfoDoc(classInfo, fieldName);
}

/**
   print as many lines as visible in terminal and let the user press
   any key to continue.
   TODO: We need to improve this with proper terminal handling. Now
   it does not do much - and output is ugly when the screen rows are
   exhausted in the middle of a field description.
*/
void print_help(const std::string& message)
{
    size_t start = 0;    
    size_t end = 0;
    int line_count = 0;
    
    while (start < message.length())
    {
        end = message.find('\n', start);
        cout << message.substr(start, end - start + 1);
        start = end + 1;        
        ++line_count;        
        
        if (line_count == (height - 2) || end == string::npos)
        {
            if (end < message.length())
            {
                cout <<  "***************** PRESS RETURN TO CONTINUE ****************" << endl;
                char c = getchar();
                
            } // ! if (end != string::npos)
            
            line_count = 0;       
        }
    }    
}

// 
// help.cpp ends here
