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
#include "../shell/Neutral.h"
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
  unit->setScale(-3);
 
  //Getting Compartment from moose
  vector< Id > chemCompt;

  //wildcardFind(path+"/##[ISA=ChemMesh]",chemCompt);
  wildcardFind(path+"/##[TYPE=MeshEntry]",chemCompt);
  //cout << "compts vector size  " << chemCompt.size()<<endl;
  vector< Id >::iterator itr;
  for (itr = chemCompt.begin(); itr != chemCompt.end();itr++)
    {
      vector <unsigned int>dims = Field <vector <unsigned int> > :: get(ObjId(*itr),"objectDimensions");
      //cout << "mesh id " << *itr << " " << itr->path() << ": no. of dimensions: " << dims.size() << endl;
      unsigned int dims_size;
      if (dims.size() == 0){
	dims_size = 1;
	}
      if (dims.size()>0){ 
	dims_size= dims.size();
	}
      for (unsigned index = 0; index < dims_size; ++index)
	{ ObjId meshParent = Neutral::parent( itr->eref() );
	  string comptname = Field<string>::get(ObjId(meshParent),"name") ;
	  /* or cout << " name from element: " << meshParent.element()->getName() << endl; */
	  double size = Field<double>::get(ObjId(*itr,index),"size");
	  unsigned int ndim = Field<unsigned int>::get(ObjId(*itr,index),"dimensions");
	  ostringstream cid;
	  cid << (meshParent)  << "_" << index;
	  comptname = nameString(comptname);
	  string comptname_id = changeName(comptname,cid.str());
	  string clean_comptname = idBeginWith(comptname_id);
	  //cout <<  "compartmen id: "<< meshParent << " compartment name "<< clean_comptname <<  " size: "<< size << " dimensions: " << ndim <<endl;
	  
	  ::Compartment* compt = cremodel_->createCompartment();
	  compt->setId(clean_comptname);
	  compt->setName(comptname);
	  compt->setSpatialDimensions(ndim);
	  if(ndim == 3)
	    compt->setSize(size*1e3);
	  else
	    compt->setSize(size);
	  
	   /* All the pools are taken here */
	  vector <Id> :: iterator itrp;
	  vector< Id > Compt_spe = LookupField< string, vector< Id > >::get(*itr, "neighbours", "remesh" );
	  for (itrp = Compt_spe.begin();itrp != Compt_spe.end();itrp++)
	    { string poolclass = Field<string> :: get(ObjId(*itrp),"class");
	      if (poolclass != "GslStoich")
		{
		  string poolname = Field<string> :: get(ObjId(*itrp),"name");
		  ostringstream pid;
		  pid << (*itrp) <<"_"<<index;
		  poolname = nameString(poolname);
		  string pool_id = changeName(poolname,pid.str());
		  string clean_poolname = idBeginWith(pool_id);
		  double initamt = 0.0;
		  initamt = Field<double> :: get(ObjId(*itrp),"nInit");
		  //cout << "poolclass:"<< poolclass << " id: " << *itrp << " path " <<  itrp->path()  << " compartment " << clean_comptname << " initamt " << initamt << endl;
		  Species *sp = cremodel_->createSpecies();
		  sp->setId( clean_poolname );
		  sp->setCompartment( clean_comptname );
		  sp->setInitialAmount( initamt ); 
		  sp->setHasOnlySubstanceUnits( true );
		  /* Buffered Molecule setting BoundaryCondition and constant has to be made true */
		  if (poolclass == "ZBufPool")
		    {sp->setBoundaryCondition(true);
		      sp->setConstant(true);
		    }//zbufpool

		  /* Funpool need to get SumFunPool */
		  if (poolclass == "ZFuncPool")
		    { vector< Id > funpoolChildren = Field< vector< Id > >::get( *itrp, "children" );
		      for ( vector< Id >::iterator itrfunpoolchild = funpoolChildren.begin();  itrfunpoolchild != funpoolChildren.end(); ++itrfunpoolchild )
			{
			  string funpoolclass = Field<string> :: get(ObjId(*itrfunpoolchild),"class");
			  if (funpoolclass == "SumFunc")
			    {vector < Id > sumfunpool = LookupField <string,vector < Id> >::get(*itrfunpoolchild, "neighbours","input");
			      int sumtot_count = sumfunpool.size();
			      if ( sumtot_count > 0 )
				{
				  ostringstream sumtotal_formula;
				  for(vector< Id >::iterator itrsumfunc = sumfunpool.begin();itrsumfunc != sumfunpool.end(); itrsumfunc++)
				    { 
				      ostringstream spId;
				      sumtot_count -= 1;
				      string sfName = Field<string> :: get(ObjId(*itrsumfunc),"name");
				      ostringstream sumFuncid;
				      sumFuncid << (*itrsumfunc) <<"_"<<index;
				      string sumFuncname = nameString(sfName);
				      string sumFunc_id = changeName(sumFuncname,sumFuncid.str());
				      string clean_sumFuncname = idBeginWith(sumFunc_id);
				      if ( sumtot_count == 0 )
					sumtotal_formula << clean_sumFuncname;
				      else
					sumtotal_formula << clean_sumFuncname << "+";
				    }
				  Rule * rule =  cremodel_->createAssignmentRule();
				  rule->setVariable( clean_poolname );
				  rule->setFormula( sumtotal_formula.str() );
				}
			    }
			}
		    } //zfunPool
		} //poolclass != gsl
	    } //itrp
	}//index
    }//itr
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
