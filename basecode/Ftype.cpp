/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Ftype::Ftype( const string& name )
{

	// Assume that this is called exactly once.
	// The order and ids of FuncVecs are sorted in 
	// FuncVec::sortFuncVec in the mooseInit function which 
	// is called as soon as main begins.
	proxyFuncs_ = 	
		new FuncVec( name + ".proxy", this->typeStr() ); 
	asyncFuncs_ = 
		new FuncVec( name + ".async", this->typeStr() ); 
	syncFuncs_ = 
		new FuncVec( name + ".sync", this->typeStr() ); 

/*

	vector< RecvFunc >::iterator i;
	vector< RecvFunc > ret;
	this->proxyFunc( ret );
	for ( i = ret.begin(); i != ret.end(); ++i )
		proxyFunc->addFunc( *i, this );

	this->asyncFunc( ret );
	for ( i = ret.begin(); i != ret.end(); ++i )
		asyncFunc->addFunc( *i, this );

// 	asyncFunc->addFunc( RFCAST( this->asyncFunc ), this );

	this->syncFunc( ret );
	for ( i = ret.begin(); i != ret.end(); ++i )
		syncFunc->addFunc( *i, this );

//	syncFunc->addFunc( RFCAST( this->syncFunc ), this );
	*/
}

void Ftype::addSyncFunc( RecvFunc r ) {
	syncFuncs_->addFunc( r, this );
	syncFuncs_->setDest();
}

void Ftype::addAsyncFunc( RecvFunc r ) {
	asyncFuncs_->addFunc( r, this );
	asyncFuncs_->setDest();
}

void Ftype::addProxyFunc( RecvFunc r ) {
	proxyFuncs_->addFunc( r, this );
	proxyFuncs_->setDest();
}



void Ftype::addSyncFunc( const Ftype* ft ) {
	assert( ft->syncFuncs_->size() == 1 );
	syncFuncs_->addFunc( ft->syncFuncs_->func( 0 ), ft );
	syncFuncs_->setDest();
}

void Ftype::addAsyncFunc( const Ftype* ft ) {
	assert( ft->asyncFuncs_->size() == 1 );
	asyncFuncs_->addFunc( ft->asyncFuncs_->func( 0 ), ft );
	asyncFuncs_->setDest();
}

void Ftype::addProxyFunc( const Ftype* ft ) {
	assert( ft->proxyFuncs_->size() == 1 );
	proxyFuncs_->addFunc( ft->proxyFuncs_->func( 0 ), ft );
	proxyFuncs_->setDest();
}



unsigned int Ftype::syncFuncId() const {
	return syncFuncs_->id();
}

unsigned int Ftype::asyncFuncId() const {
	return asyncFuncs_->id();
}

unsigned int Ftype::proxyFuncId() const {
	return proxyFuncs_->id();
}
std::string Ftype::full_type(const type_info& paramType)
{
    if (paramType == typeid(unsigned char))
    {
        return "unsigned char";
    }
    else if ( paramType == typeid(char))
    {
        return "char";
    }
    else if (paramType == typeid(short))
    {
        return "short";
    }
    else if (paramType == typeid(unsigned short))
    {
        return "unsigned short";
    }
    else if ( paramType == typeid(unsigned int))
    {
        return "unsigned int";
    }
    else if ( paramType == typeid(int) )
    {
        return "int";
    }
    else if (paramType == typeid(float))
    {
        return "float";
    }
    else if ( paramType == typeid(double) )
    {
        return "double";
    }
    else if (paramType == typeid(string))
    {
        return "string";
    }
    else if (paramType == typeid(bool))
    {
        return "bool";
    }
    else if ( paramType == typeid(vector<bool>))
    {
        return "vector < bool >";
    }
    else if (paramType == typeid(vector <char>))
    {
        return "vector < char >";
    }
    else if (paramType == typeid(unsigned char))
    {
        return "unsigned char";
    }
    else if ( paramType == typeid(vector <short>))
    {
        return "vector < short >";
    }
    else if (paramType == typeid(unsigned short))
    {
        return "unsigned short";
    }
    else if ( paramType == typeid(vector <int>))
    {
        return "vector < int >";
    }
    else if (paramType == typeid(vector <unsigned int>))
    {
        return "vector < unsigned int >";
    }
    else if (paramType == typeid(vector<float>))
    {
        return "vector < float >";
    }
    else if (paramType == typeid(vector <double>))
    {
        return "vector < double >";
    }
    else if (paramType == typeid(vector <string >))
    {
        return "vector < string >";
    }    
    else
    {
        return "none";
    }
}

/*
std::string Ftype::full_type(std::string type)
{
	static map < std::string, std::string > type_map;
	if (type_map.find("j") == type_map.end())
	{
		type_map["j"] = "unsigned int";
		type_map["i"] = "int";        
		type_map["f"] = "float";        
		type_map["d"] = "double";        
		type_map["Ss"] = "string";        
		type_map["s"] = "short";
		type_map["b"] = "bool";            
	}
	const map< std::string, std::string >::iterator i = type_map.find(type);
	if (i == type_map.end())
	{
		cout << "Not found - " << type << endl;

		return type;
	}
	else 
	{
		return i->second;
	}
}
*/
