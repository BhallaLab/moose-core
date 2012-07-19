/*******************************************************************
 * File:            SbmlWriter.cpp
 * Description:      
 * Author:          
 * E-mail:          
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**  copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
//~#include <sstream>
//~ #include <set>
//~ #include <algorithm>
//~ #include <time.h>
//~ #include <ctime>

#include "header.h"
#include <sbml/SBMLTypes.h>
#include "SbmlWriter.h"
#include "../shell/Wildcard.h"
//#include "kinetics/ChemCompt.h"
//#include <sbml/annotation/ModelHistory.h>


/**
*  write a Model after validation
*/

int SbmlWriter::write( string filepath,string target )
{
	cout << "Sbml Writer: " << filepath << " ---- " << target << endl;
	//cout << "use_SBML:" << USE_SBML << endl;
	
	#ifdef USE_SBML
	string::size_type loc;
	while ( ( loc = filepath.find( "\\" ) ) != string::npos ) 
	  {
	    filepath.replace( loc, 1, "/" );
	  }
	if ( filepath[0]== '~' )
	  {
	    cerr << "Error : Replace ~ with absolute path " << endl;
	  }
	string filename = filepath;
	string::size_type last_slashpos = filename.find_last_of("/");
	filename.erase( 0,last_slashpos + 1 );  

	/** Check:  I have to comeback to this and check what to do with file like text.xml2 cases and also shd keep an eye on special char **/
	vector< string > fileextensions;
	fileextensions.push_back( ".xml" );
	fileextensions.push_back( ".zip" );
	fileextensions.push_back( ".bz2" );
	fileextensions.push_back( ".gz" );
	vector< string >::iterator i;
	for( i = fileextensions.begin(); i != fileextensions.end(); i++ ) 
	  {
	    string::size_type loc = filename.find( *i );
	    if ( loc != string::npos ) 
	      {
		int strlen = filename.length(); 
		filename.erase( loc,strlen-loc );
		break;
	      }
	  }
	if ( i == fileextensions.end() && filename.find( "." ) != string::npos )
	  {
	    string::size_type loc;
	    while ( ( loc = filename.find( "." ) ) != string::npos ) 
	      {
		filename.replace( loc, 1, "_" );
	      }
	  }

	if ( i == fileextensions.end() )
	  filepath += ".xml";

	cout << " filepath " << filepath << filename << endl;

	
	SBMLDocument sbmlDoc(2,4);

	createModel(filename,sbmlDoc,target);
	writeModel( &sbmlDoc, filepath );
	/**
	bool SBMLok = false;
	if (SBMLok)
	  writeModel(&sbmlDoc,filepath);
	if( !SBMLok) cerr << "Errors encountered " << endl;
	**/
	#endif     
	return 0;
}

#ifdef USE_SBML
/** Check : first will put a model in according to l2v4 later will see what changes to the latest stable version l3v1 **/

/** Create an SBML model in the SBML Level 2 version 4 specification **/
void SbmlWriter::createModel(string filename,SBMLDocument& sbmlDoc,string path)
{ cremodel_ = sbmlDoc.createModel();
  cremodel_->setId(filename);

  // Unit definition for substance
  UnitDefinition* unitdef;
  Unit* unit;
  unitdef = cremodel_->createUnitDefinition();
  unitdef->setId("substance");
  unit = unitdef->createUnit();
  unit->setKind(UNIT_KIND_MOLE);
  unit->setExponent(1);
  unit->setScale(-6);
 
  //Getting Compartment from moose
  vector< Id > chemCompt;

  wildcardFind(path+"/##[ISA=ChemMesh]",chemCompt);
  //cout << "compts vector size  " << compts.size()<<endl;
  vector< Id >::iterator itr;

  for (itr = chemCompt.begin(); itr != chemCompt.end();itr++)
    {
          vector <unsigned int>dims = Field <vector <unsigned int> > :: get(ObjId(*itr),"objectDimensions");
	  //cout << itr->path() << ": no. of dimensions: " << dims.size();
	  unsigned int dims_size;
	  if (dims.size() == 0){ 
	    dims_size = 1;
	    }
	  if (dims.size()>0){ 
	    dims_size= dims.size();
	    }
	  //cout << " dims_size: " << dims_size << endl;
	  for (unsigned index = 0; index < dims_size; ++index){
	    string comptname = Field<string>::get(ObjId(*itr,index),"name") ;
	    double size = Field<double>::get(ObjId(*itr,index),"size");
	    unsigned int ndim = Field<unsigned int>::get(ObjId(*itr,index),"numDimensions");
	    ostringstream cid;
	    cid << (*itr)  << "_" << index;
	    comptname = nameString(comptname);
	    string comptname_id = changeName(comptname,cid.str());
	    string clean_comptname = idBeginWith(comptname_id);
	    cout <<  "id: "<< clean_comptname << " compatname " << comptname << " size: "<< size << " dimensions: " << ndim << endl;
	    	    
	    ::Compartment* compt = cremodel_->createCompartment();
	    compt->setId(clean_comptname);
	    compt->setName(comptname);
	    compt->setSpatialDimensions(ndim);
	    if(ndim == 3)
	      compt->setSize(size*1e3);
	    else
	      compt->setSize(size);
	    vector< Id > Compt_spe;
	    wildcardFind(itr->path()+"/##[TYPE=ZombiePool]",Compt_spe);
	    vector < Id > :: iterator itr2;
	    vector <int> :: size_type i;
	    i = Compt_spe.size();
	    cout << "size" << i;
	    for (itr2 = Compt_spe.begin();itr2 != Compt_spe.end();itr2++)
	      { string poolname = Field<string> :: get(ObjId(*itr2),"name");
		cout << *itr2 << " path " <<  itr2->path() << " class "<<  poolname << " parent " << comptname << endl;
		
		//string parentName;
		//ostringstream parentId;
		//Id parent = Neutral::getParent(*itr2);
		//parentName = parent()->name();
		//parentId << parent()->id().id() << "_" << parent()->id().index()
		//string parentpool = Field< string > :: get(ObjId(*itr2),"parent");
		/*
		Species *sp = model_->createSpecies();
		string molName = (moleEl)->name();
		mid << moleEl.id().id() << "_" << moleEl.id().index();
		molName = nameString( molName ); 
		string newSpName = changeName( molName,mid.str() );
		newSpName = idBeginWith( newSpName );
		sp->setId( newSpName );
		sp->setCompartment( parentCompt );
		sp->setHasOnlySubstanceUnits( true );
		*/
		//cout << *itr2 << " path " <<  itr2->path() << " class "<<  poolname << " parent " << parent << endl;
	      } 

	    
	  }// for unsigned int
    } // itr=compts
}//createModel
#endif

/** Writes the given SBMLDocument to the given file. **/
 
bool SbmlWriter::writeModel( const SBMLDocument* sbmlDoc, const string& filename )
{
  SBMLWriter sbmlWriter;
  cout << "sbml writer" << filename << sbmlDoc << endl;
  bool result = sbmlWriter.writeSBML( sbmlDoc, filename );
  if ( result )
    {
      cout << "Wrote file \"" << filename << "\"" << endl;
      return true;
    }
  else
    {
      cerr << "Failed to write \"" << filename << "\"" << "  check for path and write permission" << endl;
      return false;
    }
    
}
/* *  removes special characters  **/
string SbmlWriter::nameString( string str )
{ string str1;
  int len = str.length();
  int i= 0;
  do
    {
      switch( str.at(i) )
	{
	case '-':
	  str1 = "_minus_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '+':
	  str1 = "_plus_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '*':
	  str1 = "_star_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '/':
	  str1 = "_slash_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '(':
	  str1 = "_bo_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case ')':
	  str1 = "_bc_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '[':
	  str1 = "_sbo_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case ']':
	  str1 = "_sbc_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '.':
	  str1 = "_dot_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	}//end switch 
      i++;
    }while ( i < len );
  return str;
}
/*   id preceeded with its parent Name   */
string SbmlWriter::changeName( string parent, string child )
{
  string newName = parent + "_" + child + "_";
  return newName;
}
/* *  change id  if it starts with  a numeric  */
string SbmlWriter::idBeginWith( string name )
{
  string changedName = name;
  if ( isdigit(name.at(0)) )
    changedName = "_" + name;
  return changedName;
}
