/*******************************************************************
 * File:            StringUtil.h
 * Description:     Some common utility functions for strings.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 12:08:00
 ********************************************************************/
#ifndef _STRINGUTIL_H
#define _STRINGUTIL_H

#include <string>
#include <vector>

namespace moose 
{
    /** List of characters considered to be whitespace */
    static const char* const DELIMITERS=" \t\r\n";

    /** Splits given string into tokens */
    void tokenize( const std::string& str,
            const std::string& delimiters,
            std::vector< std::string >& tokens 
            );

    /** trims the leading and trailing white spaces */
    std::string trim(const std::string myString, const std::string& delimiters=" \t\r\n");

    /** Fix the user-given path whenever possible */
    std::string fix(const std::string myString, const std::string& delimiters=" \t\r\n");

    std::string& clean_type_name(std::string& arg);
    bool endswith(const std::string& full, const std::string& ending);

    /**
     * @brief Compares the two strings a and b, ignoring the case of the 
     * characters. It returns an integer less than, equal to, or greater than 
     * zero if a is found, respectively, to be less than, to match, or be greater than s2. 
     *
     * If both strings are not of same
     * size, return false even when smaller string is a proper substring of the
     * bigger one.
     *
     * @param a First string.
     * @param b Second string.
     *
     */
    int strncasecmp( const std::string& a, const std::string& b, size_t n = -1 );


    // TODO: other std::string utilities to add
    // /** Trim leading and trailing whitespace and replace  convert any two or more consecutive whitespace inside the std::string by a single 'blank' character. */
    // std::string fulltrim(std::string& myString) const;
    // /** Convert to uppercase */
    // std::string upcase(std::string& myString) const;
    // /** Convert to lowercase */
    // std::string downcase(std::string & myString);
    // Maybe a implement regular expression search - reinventing wheel - but no standard way without using some bloated library.

}

#endif //_STRINGUTIL_H
