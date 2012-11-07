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
#include <set>
#include <sstream>

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

  for ( vector< Id >::iterator itr = chemCompt.begin(); itr != chemCompt.end();itr++)
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
  	  ostringstream cid;
	  cid << (meshParent)  << "_" << index;
	  comptname = nameString(comptname);
	  string comptname_id = changeName(comptname,cid.str());
	  string clean_comptname = idBeginWith(comptname_id);

	  double size = Field<double>::get(ObjId(*itr,index),"size");
	  unsigned int ndim = Field<unsigned int>::get(ObjId(*itr,index),"dimensions");
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
	  vector< Id > Compt_spe = LookupField< string, vector< Id > >::get(*itr, "neighbours", "remesh" );
	  for (vector <Id> :: iterator itrp = Compt_spe.begin();itrp != Compt_spe.end();itrp++)
	    { string poolclass = Field<string> :: get(ObjId(*itrp),"class");
	      if (poolclass != "GslStoich")
		{
		  string clean_poolname = cleanNameId(*itrp,index);
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
				    {  /* Check with Upi: Finds the source pool for a SumTot. It also deals with cases wherthe source is an enz-substrate complex Readkkit.cpp */
				      ostringstream spId;
				      sumtot_count -= 1;
				      string clean_sumFuncname = cleanNameId(*itrsumfunc,index);
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
	  vector< Id > Compt_ReacEnz = LookupField< string, vector< Id > >::get(*itr, "neighbours", "remeshReacs" );
	  for (vector <Id> :: iterator itrRE= Compt_ReacEnz.begin();itrRE != Compt_ReacEnz.end();itrRE++)
	    { string re_enClass = Field<string> :: get(ObjId(*itrRE),"class");
	      if (re_enClass == "ZReac")
		{
		  string clean_reacname = cleanNameId(*itrRE,index);
		  Reaction* reaction;
		  reaction = cremodel_->createReaction(); 
		  
		  SpeciesReference* spr;
		  KineticLaw* kl;
		  Parameter* para; 
		  
		  reaction->setId( clean_reacname);
		  double kf = Field<double>::get(ObjId(*itrRE),"kf");
		  double kb = Field<double>::get(ObjId(*itrRE),"kb");
		  if (kb == 0.0)
		    reaction->setReversible( false );
		  else
		    reaction->setReversible( true );
		  vector < Id > rct = LookupField <string,vector < Id> >::get(*itrRE, "neighbours","sub");
		  std::set < Id > rctUniq;
		  rctUniq.insert(rct.begin(),rct.end());
		  for (std::set < Id> :: iterator rRct = rctUniq.begin();rRct!=rctUniq.end();rRct++)
		    { double rctstoch = count( rct.begin(),rct.end(),*rRct );
		      string clean_rctname = cleanNameId(*rRct,index);
		      spr = reaction->createReactant();
		      spr->setSpecies( clean_rctname );
		      spr->setStoichiometry( rctstoch );
		    } //rRct
		  
		  vector < Id > prd = LookupField <string,vector < Id> >::get(*itrRE, "neighbours","prd");
 		  std::set < Id > prdUniq;
		  prdUniq.insert(prd.begin(),prd.end());
		  for (std::set < Id> :: iterator rPrd = prdUniq.begin();rPrd!=prdUniq.end();rPrd++)
		    { double prdstoch = count( prd.begin(),prd.end(),*rPrd );
		      string clean_prdname = cleanNameId(*rPrd,index);
		      spr = reaction->createProduct();
		      spr->setSpecies( clean_prdname );
		      spr->setStoichiometry( prdstoch );
		    } //rprd
		  
		  ostringstream rate_law,kfparm,kbparm;
		  kfparm << clean_reacname << "_" << "kf";
		  kbparm << clean_reacname << "_" << "kb";
		  rate_law << kfparm.str();
		  double rctstoch = 0.0,rct_order = 0.0,prdstoch = 0.0,prd_order =0.0;
		  for( std::set < Id> :: iterator ri = rctUniq.begin(); ri != rctUniq.end(); ri++ )
		    { string clean_rctname = cleanNameId(*ri,index);
		      rctstoch = count( rct.begin(),rct.end(),*ri );
		      rct_order += rctstoch;
		      if ( rctstoch == 1 )
			rate_law << "*" << clean_rctname;
		      else
			rate_law << "*" <<clean_rctname << "^" << rctstoch;
		    }
		  if ( kb != 0.0 )
		    {
		      rate_law << "-" << kbparm.str();
		      for(std::set < Id> :: iterator pi = prdUniq.begin(); pi != prdUniq.end(); pi++ )
			{ string clean_prdname = cleanNameId(*pi,index);
			  prdstoch = count( prd.begin(),prd.end(),*pi );
			  prd_order +=prdstoch;
			  if ( prdstoch == 1 )
			    rate_law << "*" << clean_prdname;
			  else
			    rate_law << "*" << clean_prdname << "^" << prdstoch;
			} 
		    }
		  kl = reaction->createKineticLaw();
		  kl->setFormula( rate_law.str() );
		  //cout<<"rate law: "<<rate_law.str()<<endl;
		  // Create local Parameter objects inside the KineticLaw object.
		  para = kl->createParameter();
		  para->setId( kfparm.str() );
		  string unit=parmUnit( rct_order-1 );
		  para->setUnits( unit );
		  double rvalue,pvalue;
		  const double m = 1.0;
		  rvalue = kf *(pow(m,rct_order-1));
		  para->setValue( rvalue );
		  if ( kb != 0.0 ){
		    pvalue = kb * (pow(m,prd_order-1));
		    para = kl->createParameter();
		    para->setId( kbparm.str() );
		    string unit=parmUnit( prd_order-1 );
		    para->setUnits( unit );
		    para->setValue( pvalue );
		  }
		}//re_enclass
	    }//itrRE

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
//string SbmlWriter::cleanNameId(vector < Id > const &itr,int index)

string SbmlWriter::cleanNameId(Id itrid,int  index)
{ string objname = Field<string> :: get(ObjId(itrid),"name");
  ostringstream Objid;
  Objid << (itrid) <<"_"<<index;
  objname = nameString(objname);
  string objname_id = changeName(objname,Objid.str());
  string clean_nameid = idBeginWith(objname_id);
  return clean_nameid ;
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
string SbmlWriter::parmUnit( double rct_order )
{
  ostringstream unit_stream;
  int order = ( int) rct_order;
  switch( order )
    {
      case 0:
	unit_stream<<"per_second";
	break;
      case 1:
	unit_stream<<"per_uMole_per_second";
	break;
      case 2:
	unit_stream<<"per_uMole_sq_per_second";
	break;
      default:
	unit_stream<<"per_uMole_"<<rct_order<<"_per_second";
	break;
    }
  ListOfUnitDefinitions * lud =cremodel_->getListOfUnitDefinitions();
  bool flag = false;
  for ( unsigned int i=0;i<lud->size();i++ )
    {
      UnitDefinition * ud = lud->get(i);
      if ( ud->getId() == unit_stream.str() ){
	flag = true;
	break;
	}
    }
  if ( !flag ){
    UnitDefinition* unitdef;
    Unit* unit;
    unitdef = cremodel_->createUnitDefinition();
    unitdef->setId( unit_stream.str() );
    // Create individual unit objects that will be put inside the UnitDefinition .
    unit = unitdef->createUnit();
    unit->setKind( UNIT_KIND_MOLE );
    unit->setExponent( -order );
    unit->setScale( -6 );
    unit = unitdef->createUnit();
    unit->setKind( UNIT_KIND_SECOND );
    unit->setExponent( -1 );
  }
  return unit_stream.str();
}
