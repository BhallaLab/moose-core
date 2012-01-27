// utility.h --- 
// 
// Filename: utility.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Wed Mar 23 10:25:58 2011 (+0530)
// Version: 
// Last-Updated: Sat Mar 26 22:35:46 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 15
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

// Code:

#ifndef _UTILITY_H

char shortType(std::string type);
char shortFinfo(std::string ftype);

void tokenize(
	const std::string& str,	
	const std::string& delimiters,
	std::vector< std::string >& tokens );
std::string trim(const std::string myString);
const map<std::string, std::string>& getArgMap();

#endif // !_UTILITY_H


// 
// utility.h ends here
