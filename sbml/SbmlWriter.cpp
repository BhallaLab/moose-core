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
#include "header.h"
#include <sbml/SBMLTypes.h>
#include "SbmlWriter.h"
#include "../shell/Wildcard.h"
#include "../shell/Neutral.h"
#include <set>
#include <sstream>
#include "../shell/Shell.h"
//#include "../manager/SimManager.h"
/**
*  write a Model after validation
*/
/* ToDo: Tables should be implemented
 */
int SbmlWriter::write( string filepath,string target )
{
  //cout << "Sbml Writer: " << filepath << " ---- " << target << endl;
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
  
  //* Check:  I have to comeback to this and check what to do with file like text.xml2 cases and also shd keep an eye on special char **/
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
  
  //cout << " filepath " << filepath << " " << filename << endl;
  
  SBMLDocument sbmlDoc = 0;
  bool SBMLok = false;
  createModel( filename,sbmlDoc,target ); 
  SBMLok  = validateModel( &sbmlDoc );
  if ( SBMLok ) 
    writeModel( &sbmlDoc, filepath );
  //delete sbmlDoc;
  if ( !SBMLok ) {
    cerr << "Errors encountered " << endl;
    return 1;
  }
  
#endif     
  return 0;
}
#ifdef USE_SBML

//* Check : first will put a model in according to l2v4 later will see what changes to the latest stable version l3v1 **/

//* Create an SBML model in the SBML Level 2 version 4 specification **/
void SbmlWriter::createModel(string filename,SBMLDocument& sbmlDoc,string path)
{ 
  XMLNamespaces  xmlns;
  xmlns.add("http://www.sbml.org/sbml/level2/version4");
  xmlns.add("http://www.moose.ncbs.res.in","moose");
  xmlns.add("http://www.w3.org/1999/xhtml","xhtml");
  sbmlDoc.setNamespaces(&xmlns);

  cremodel_ = sbmlDoc.createModel();
  cremodel_->setId(filename);
  Id baseId(path);
  /*
  SimManager* sm = reinterpret_cast< SimManager* >(baseId.eref().data());
  double runtime = sm->getRunTime();
  double simdt = sm->getSimDt();
  double plotdt = sm->getPlotDt();
  */
  vector< ObjId > graphs;
  string plots;
  wildcardFind(path+"/##[TYPE=Table]",graphs);
  for ( vector< ObjId >::iterator itrgrp = graphs.begin(); itrgrp != graphs.end();itrgrp++)
    { 
      vector< Id > graphsrc =LookupField< string, vector< Id > >::get(*itrgrp, "neighbors", "requestOut" );
      for (vector <Id> :: iterator itrsrc = graphsrc.begin();itrsrc != graphsrc.end();itrsrc++)
	{  string species = nameString(Field<string> :: get(*itrsrc,"name"));
	  //TODO: Check For graph how to I get compartment that it belongs
	  /*vector< Id > comptplot = LookupField<string, vector< Id > > ::get(*itrsrc,"neighbours","requestVolume");
	  for (vector <Id> :: iterator comptp = comptplot.begin();comptp != comptplot.end();comptp++)
	    { 
	      ObjId meshParent = Neutral::parent( comptp->eref() );
	      cout << " if " << meshParent;
	      string speciesCompt = nameString(Field<string>::get(ObjId(meshParent),"name") );
	      plots += "/"+speciesCompt+"/"+species+";";
	      cout << "plots "<<plots<<endl;
	    }
	  */
	}
    }
  /*
  ostringstream modelAnno;
  modelAnno << "<moose:ModelAnnotation>\n";
  modelAnno << "<moose:runTime> " << runtime << " </moose:runTime>\n";
  modelAnno << "<moose:simdt> " << simdt << " </moose:simdt>\n";
  modelAnno << "<moose:plotdt> " << plotdt << " </moose:plotdt>\n";
  modelAnno << "<moose:plots> "<< plots<< "</moose:plots>\n";
  modelAnno << "</moose:ModelAnnotation>";
  XMLNode* xnode =XMLNode::convertStringToXMLNode( modelAnno.str() ,&xmlns);
  cremodel_->setAnnotation( xnode );	
  */
  UnitDefinition *ud1 = cremodel_->createUnitDefinition();
  ud1->setId("volume");
  Unit * u = ud1->createUnit();
  u->setKind(UNIT_KIND_LITRE);
  u->setMultiplier(1.0);
  u->setExponent(1.0);
  
  UnitDefinition * unitdef;
  Unit* unit;
  unitdef = cremodel_->createUnitDefinition();
  unitdef->setId("substance");
  unit = unitdef->createUnit();
  unit->setKind( UNIT_KIND_ITEM );
  unit->setMultiplier(1);

  cout << "Writing to SBML started "<<endl;
//Getting Compartment from moose
  vector< ObjId > chemCompt;
  wildcardFind(path+"/##[ISA=ChemCompt]",chemCompt);
  for ( vector< ObjId >::iterator itr = chemCompt.begin(); itr != chemCompt.end();itr++)
    { cout << "itr " << *itr << "name " << itr->eref() <<endl;
      //TODO: CHECK need to check how do I get ObjectDimensions
      vector < unsigned int>dims;
      unsigned int dims_size;
      /*vector <unsigned int>dims = Field <vector <unsigned int> > :: get(ObjId(*itr),"objectDimensions");

      if (dims.size() == 0){
     	dims_size = 1;
	}
      if (dims.size()>0){ 
	dims_size= dims.size();
	}
      */
      dims_size = 1;
      
      for (unsigned index = 0; index < dims_size; ++index)
	{ 
	  string comptName = Field<string>::get(*itr,"name");
	  string comptPath = Field<string>::get(*itr,"path");
  	  ostringstream cid;
	  cid << *itr  << "_" << index;
	  comptName = nameString(comptName);
	  string comptname_id = comptName + "_" + cid.str() + "_";
	  //changeName(comptname,cid.str());
	  string clean_comptname = idBeginWith(comptname_id);

	  double size = Field<double>::get(ObjId(*itr,index),"Volume");
	  unsigned int ndim = Field<unsigned int>::get(ObjId(*itr,index),"NumDimensions");
	  
	  ::Compartment* compt = cremodel_->createCompartment();
	  compt->setId(clean_comptname);
	  compt->setName(comptName);
	  compt->setSpatialDimensions(ndim);
	  compt->setUnits("volume");
	  if(ndim == 3)
	    // Unit for Compartment in moose is cubic meter
	    //   Unit for Compartment in SBML is lts 
	    //   so multiple by 1000                  
	    compt->setSize(size*1e3);
	  else
	    compt->setSize(size);
	  
	   // All the pools are taken here 
	  vector< ObjId > Compt_spe;
	  wildcardFind(comptPath+"/##[ISA=PoolBase]",Compt_spe);

	  //vector< Id > Compt_spe = LookupField< string, vector< Id > >::get(*itr, "neighbours", "remesh" );
	  int species_size = 1;
	  string objname;
	  for (vector <ObjId> :: iterator itrp = Compt_spe.begin();itrp != Compt_spe.end();itrp++)
	    { string objclass = Field<string> :: get(*itrp,"className");
	      string clean_poolname = cleanNameId(*itrp,index);
	      double initAmt = Field<double> :: get(*itrp,"nInit");
	      Species *sp = cremodel_->createSpecies();
	      sp->setId( clean_poolname );
	      string poolname = Field<string> :: get(*itrp,"name");
	      std::size_t found = poolname.find("cplx");
	      //If Species name has cplx then assuming its cplx molecule, since genesis there can be same cplx
	      // name in number place, as its build under site, but in SBML this is not possible
	      // so adding enzsite_enzname_cplxname
	      if (found!=std::string::npos)
		{vector < Id > rct = LookupField <string,vector < Id> >::get(*itrp, "neighbors","reacDest");
		  std::set < Id > rctprd;
		  rctprd.insert(rct.begin(),rct.end());
		  for (std::set < Id> :: iterator rRctPrd = rctprd.begin();rRctPrd!=rctprd.end();rRctPrd++)
		    { 
		      string enz = Field<string> :: get(*rRctPrd,"name");
		      ObjId meshParent = Neutral::parent( rRctPrd->eref() );
		      string enzPoolsite = Field<string>::get(ObjId(meshParent),"name") ;
		      objname = nameString(enzPoolsite)+"_"+nameString(enz)+"_"+nameString(poolname);
		    }
		}
	      else
		objname = nameString(poolname);
	      
	      sp->setName( objname);
	      sp->setCompartment( clean_comptname );
	      // AS of 12-6-2013
	      //   Units in moose for  pool : is milli molar,    In sbml for pool: default unit is mole.
	      //   so 'conc' shd be converted to from milli molar to mole
	      //   molar (M) = moles/ltrs
	      //   mole =  Molar*1e-3*Vol*1000*Avogadro constant (1e-3 milli, 1000 is for converting cubicmeter to lts)
	      //   As of 2-7-2013
	      //   At this point I am using item which is # number so directly using nInit
	      //
	      sp->setInitialAmount( initAmt ); 
	      string path = Field<string> :: get(*itrp,"path");
	      Id annotaId( path+"/info");
	      string noteClass = Field<string> :: get(annotaId,"className");
	      string notes;
	      if (noteClass =="Annotator")
		{ string notes = Field <string> :: get(annotaId,"notes");
		  if (notes != "")
		    { string cleanNotes = nameString1(notes);
		      string notesString = "<body xmlns=\"http://www.w3.org/1999/xhtml\">\n \t \t"+
			cleanNotes + "\n\t </body>"; 
		      sp->setNotes(notesString);
		    }
		}
	      sp->setUnits("substance");
	      species_size = species_size+1;
	      // true if species is amount, false: if species is concentration 
	      sp->setHasOnlySubstanceUnits( true );
	      
	      // Buffered Molecule setting BoundaryCondition and constant has to be made true 
	      if (objclass == "BufPool")
		{sp->setBoundaryCondition(true);
		  sp->setConstant(true);
		}//zbufpool
	      
	      // Funpool need to get SumFunPool 
	      
	      if (objclass == "FuncPool")
		{ 
		  vector< Id > funpoolChildren = Field< vector< Id > >::get( *itrp, "children" );
		  for ( vector< Id >::iterator itrfunpoolchild = funpoolChildren.begin();  itrfunpoolchild != funpoolChildren.end(); ++itrfunpoolchild )
		    { 
		      string funpoolclass = Field<string> :: get(*itrfunpoolchild,"className");
		      if (funpoolclass == "SumFunc")
			{vector < Id > sumfunpool = LookupField <string,vector < Id> >::get(*itrfunpoolchild, "neighbors","input");
			  int sumtot_count = sumfunpool.size();
			  if ( sumtot_count > 0 )
			    { // For sumfunc pool boundingcondition has to made true,
			      // if its acting in react or product for a reaction/enzyme and also funcpool
			      vector < Id > connectreac = LookupField <string,vector < Id> >::get(*itrp, "neighbors","reac");
			      if( connectreac.size() > 0)
				sp->setBoundaryCondition(true);
			      ostringstream sumtotal_formula;
			      for(vector< Id >::iterator itrsumfunc = sumfunpool.begin();itrsumfunc != sumfunpool.end(); itrsumfunc++)
				{  // Check with Upi: Finds the source pool for a SumTot. It also deals with cases whether source is an enz-substrate complex Readkkit.cpp
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
	    } //itrp
	  vector< ObjId > Compt_Reac;
	  wildcardFind(comptPath+"/##[ISA=ReacBase]",Compt_Reac);
	  //vector< Id > Compt_Reac = LookupField< string, vector< Id > >::get(*itr, "neighbours", "remeshReacs" );
	  for (vector <ObjId> :: iterator itrR= Compt_Reac.begin();itrR != Compt_Reac.end();itrR++)
	    { string cleanReacname = cleanNameId(*itrR,index);
	      string recClass = Field<string> :: get(*itrR,"className");
	      string pathR = Field<string> :: get(*itrR,"path");
	      Id annotaIdR(pathR+"/info");
	      string noteClassR = Field<string> :: get(annotaIdR,"className");
	      Reaction* reaction;
	      reaction = cremodel_->createReaction(); 
	      string notesR;
	      if (noteClassR =="Annotator")
		notesR = Field <string> :: get(annotaIdR,"notes");
	      if (notesR != ""){
		string cleanNotesR = nameString1(notesR);
		string notesStringR = "<body xmlns=\"http://www.w3.org/1999/xhtml\">\n \t \t"+
		  cleanNotesR + "\n\t </body>"; 
		reaction->setNotes(notesStringR);
	      }
	      string objname = Field<string> :: get(*itrR,"name");
	      objname = nameString(objname);

	      KineticLaw* kl;
	      // Reaction 
	      reaction->setId( cleanReacname);
	      reaction->setName( objname);
	      double Kf = Field<double>::get(*itrR,"kf");
	      double Kb = Field<double>::get(*itrR,"kb");
	      
	      if (Kb == 0.0)
		reaction->setReversible( false );
	      else
		reaction->setReversible( true );
	      
	      // Reaction's Reactant are Written 
	      ostringstream rate_law,kfparm,kbparm;
	      double rct_order = 0.0;
	      kfparm << cleanReacname << "_" << "Kf";
	      rate_law << kfparm.str();
	      
	      // This function print out reactants and update rate_law string 
	      getSubPrd(reaction,"sub","",*itrR,index,rate_law,rct_order,true,recClass);
	      double prd_order =0.0;
	      kbparm << cleanReacname << "_" << "Kb";
	      
	      // This function print out product and update rate_law string  if kb != 0 
	      if ( Kb != 0.0 ){
		rate_law << "-" << kbparm.str();
		getSubPrd(reaction,"prd","",*itrR,index,rate_law,prd_order,true,recClass);
	      }
	      else
		getSubPrd(reaction,"prd","",*itrR,index,rate_law,prd_order,false,recClass);
	      
	      kl = reaction->createKineticLaw();
	      kl->setFormula( rate_law.str() );
	      double rvalue,pvalue;
	      rvalue = Kf;
	      string unit=parmUnit( rct_order-1 );
	      printParameters( kl,kfparm.str(),rvalue,unit ); 
	      if ( Kb != 0.0 ){
		pvalue = Kb;
		string unit=parmUnit( prd_order-1 );
		printParameters( kl,kbparm.str(),pvalue,unit ); 
	      }
	    }//itrR
	  //     Reaction End 
	      
	  // Enzyme start
	  vector< ObjId > Compt_Enz;
	  wildcardFind(comptPath+"/##[ISA=EnzBase]",Compt_Enz);
	  for (vector <ObjId> :: iterator itrE=Compt_Enz.begin();itrE != Compt_Enz.end();itrE++)
	    { string enzClass = Field<string>::get(*itrE,"className");
	      string cleanEnzname = cleanNameId(*itrE,index);
	      string pathE = Field < string > :: get(*itrE,"Path");
	      Id annotaIdE(pathE+"/info");
	      string noteClassE = Field < string > :: get(annotaIdE,"className");
	      Reaction* reaction;
	      reaction = cremodel_->createReaction();
	      string notesE;
	      if (noteClassE == "Annotator")
		notesE = Field < string > ::get(annotaIdE,"Notes");
	      if (notesE != "")
		{ string cleanNotesE = nameString1(notesE);
		  string notesStringE = "<body xmlns=\"http://www.w3.org/1999/xhtml\">\n \t \t"+cleanNotesE + "\n\t </body>"; 
		  reaction->setNotes(notesStringE);
		}
	      string objname = Field < string> :: get(*itrE,"name");
	      objname = nameString(objname);
	      KineticLaw* kl;
	      if (enzClass == "Enz")
		{// Complex Formation S+E -> SE*;
		   reaction->setId( cleanEnzname);
		   reaction->setName( objname);
		   string enzname = Field<string> :: get(*itrE,"name");
		   ostringstream enzid;
		   enzid << (*itrE) <<"_"<<index;
		   enzname = nameString(enzname);
		   string enzNameAnno = enzname;
		   ostringstream Objid;
		   Objid << (*itrE) <<"_"<<index <<"_";
		   string enzName = enzname + "_" + Objid.str();
		   enzName = idBeginWith( enzName );
		   string enzAnno = "<moose:EnzymaticReaction>";
		   double k1 = Field<double>::get(*itrE,"k1");
		   double k2 = Field<double>::get(*itrE,"k2");
		   
		   ostringstream rate_law;
		   double rct_order = 0.0,prd_order=0.0;
		   rate_law << "k1";
		   getSubPrd(reaction,"enzOut","sub",*itrE,index,rate_law,rct_order,true,enzClass);
		   for(unsigned int i =0;i<nameList_.size();i++)
		     enzAnno += "<moose:enzyme>"+nameList_[i]+"</moose:enzyme>\n";
		   
		   getSubPrd(reaction,"sub","",*itrE,index,rate_law,rct_order,true,enzClass);
		   for (unsigned int i =0;i<nameList_.size();i++)
		     enzAnno += "<moose:substrates>"+nameList_[i]+"</moose:substrates>\n";
		   
		   // product 
		   rate_law << "-" << "k2";
		   getSubPrd(reaction,"cplxDest","prd",*itrE,index,rate_law,prd_order,true,enzClass);
		   for(unsigned int i =0;i<nameList_.size();i++)
		     enzAnno += "<moose:product>"+nameList_[i]+"</moose:product>\n";
		   enzAnno += "<moose:groupName>"+enzName+"</moose:groupName>\n";
		   enzAnno += "<moose:stage>1</moose:stage> \n";
		   enzAnno += "</moose:EnzymaticReaction>";
		   
		   XMLNode* xnode =XMLNode::convertStringToXMLNode( enzAnno ,&xmlns);
		   reaction->setAnnotation( xnode );	
		   kl = reaction->createKineticLaw();
		   kl->setFormula( rate_law.str() );
		   string unit=parmUnit( rct_order-1 );
		   printParameters( kl,"k1",k1,unit ); 
		   string punit=parmUnit( prd_order-1 );
		   printParameters( kl,"k2",k2,punit ); 
		   // 2 Stage SE* -> E+P  
		   
		   Objid << "Product_formation";
		   string enzName1 = enzname + "_" + Objid.str() + "_";
		   enzName1 = idBeginWith( enzName1 );
		   //Reaction* reaction;
		   reaction = cremodel_->createReaction(); 
		   reaction->setId( enzName1 );
		   reaction->setName( objname);
		   if (notesE != ""){
		     string cleanNotesE = nameString1(notesE);
		     string notesStringE = "<body xmlns=\"http://www.w3.org/1999/xhtml\">\n \t \t"+
		       cleanNotesE + "\n\t </body>"; 
		     reaction->setNotes(notesStringE);
		   }
		   reaction->setReversible( false );
		   double k3 = Field<double>::get(*itrE,"k3");
		   double erct_order = 0.0,eprd_order = 0.0;
		   ostringstream enzrate_law;
		   enzrate_law << "k3";
		   string enzAnno2 = "<moose:EnzymaticReaction>";
		   getSubPrd(reaction,"cplxDest","sub",*itrE,index,enzrate_law,erct_order,true,enzClass);
		   for(unsigned int i =0;i<nameList_.size();i++)
		     enzAnno2 +=  "<moose:complex>"+nameList_[i]+"</moose:complex>\n";
		   
		   getSubPrd(reaction,"enzOut","prd",*itrE,index,enzrate_law,eprd_order,false,enzClass);
		   for(unsigned int i =0;i<nameList_.size();i++)
		     enzAnno2 += "<moose:enzyme>"+nameList_[i]+"</moose:enzyme>\n";
		   getSubPrd(reaction,"prd","",*itrE,index,enzrate_law,eprd_order,false,enzClass);
		   for(unsigned int i =0;i<nameList_.size();i++)
		     enzAnno2 += "<moose:product>"+nameList_[i]+"</moose:product>\n";
		   
		   enzAnno2 += "<moose:groupName>"+enzName+"</moose:groupName>\n";
		   enzAnno2 += "<moose:stage>2</moose:stage> \n";
		   enzAnno2 += "</moose:EnzymaticReaction>";
		   XMLNode* xnode2 =XMLNode::convertStringToXMLNode( enzAnno2 ,&xmlns);
		   reaction->setAnnotation( xnode2 );	
		   
		   kl = reaction->createKineticLaw();
		   kl->setFormula( enzrate_law.str() );
		   printParameters( kl,"k3",k3,"per_second" );
		} //enzclass = Enz
	      else if (enzClass == "MMenz")
		{ reaction->setId( cleanEnzname);
		  reaction->setName( objname);
		  double Km = Field<double>::get(*itrE,"numKm");
		  double kcat = Field<double>::get(*itrE,"kcat");
		  reaction->setReversible( false );
		  // Substrate 
		  ostringstream rate_law,sRate_law,fRate_law;
		  double rct_order = 0.0,prd_order=0.0;
		  
		  getSubPrd(reaction,"sub","",*itrE,index,rate_law,rct_order,true,enzClass);
		  sRate_law << rate_law.str();
		  
		  // Modifier 
		  getSubPrd(reaction,"enzDest","",*itrE,index,rate_law,rct_order,true,enzClass);
		  
		   // product 
		  getSubPrd(reaction,"prd","",*itrE,index,rate_law,prd_order,false,enzClass);
		  
		  kl = reaction->createKineticLaw();
		  
		   string s = sRate_law.str();
		   if(!s.empty()) {
		     s = s.substr(1); 
		   } 
		   fRate_law << "kcat" << rate_law.str() << "/" << "(" << "Km" << " +" << s << ")"<<endl;
		   kl->setFormula( fRate_law.str() );
		   kl->setNotes("<body xmlns=\"http://www.w3.org/1999/xhtml\">\n\t\t" + fRate_law.str() + "\n\t </body>");
		   printParameters( kl,"Km",Km,"substance" ); 
		   string kcatUnit = parmUnit( 0 );
		   printParameters( kl,"kcat",kcat,kcatUnit );
		}
	    } //itrE
	  // Enzyme End
	}//index
    }//itr
  //return sbmlDoc;
}//createModel
#endif

/** Writes the given SBMLDocument to the given file. **/
 
bool SbmlWriter::writeModel( const SBMLDocument* sbmlDoc, const string& filename )
{
  SBMLWriter sbmlWriter;
  //cout << "sbml writer" << filename << sbmlDoc << endl;
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
{ string objname = Field<string> :: get(itrid,"name");
  string objclass = Field<string> :: get(itrid,"className");
  ostringstream Objid;
  Objid << (itrid) <<"_"<<index;
  objname = nameString(objname);
  string objname_id = objname + "_" + Objid.str() + "_";
    //changeName(objname,Objid.str());
  if (objclass == "MMenz")
    { string objname_id_n = objname_id + "_MM_Reaction_";
      //changeName(objname_id,"MM_Reaction" );
      objname_id = objname_id_n;
    }
  else if (objclass == "Enz")
    { string objname_id_n = objname_id + "Complex_formation_";
	//changeName(objname_id,"Complex_formation" );
      objname_id = objname_id_n;
    }
  string clean_nameid = idBeginWith(objname_id);
  return clean_nameid ;
}
void SbmlWriter::getSubPrd(Reaction* rec,string type,string enztype,Id itrRE, int index,ostringstream& rate_law,double &rct_order,bool w,string re_enClass)
{
  nameList_.clear();
  SpeciesReference* spr;
  ModifierSpeciesReference * mspr;
  vector < Id > rct = LookupField <string,vector < Id> >::get(itrRE, "neighbors",type);
  std::set < Id > rctprdUniq;
  rctprdUniq.insert(rct.begin(),rct.end());
  for (std::set < Id> :: iterator rRctPrd = rctprdUniq.begin();rRctPrd!=rctprdUniq.end();rRctPrd++)
    { double stoch = count( rct.begin(),rct.end(),*rRctPrd );
      string objname = Field<string> :: get(*rRctPrd,"name");
      string cleanObjname = nameString(objname);
      string clean_name = cleanNameId(*rRctPrd,index);
      string fname;
      if (type == "sub" or (type == "enzOut" and enztype == "sub" ) or (type == "cplxDest" and enztype == "sub")) 
	{ spr = rec->createReactant();
	  spr->setSpecies(clean_name);
	  spr->setStoichiometry( stoch );
	}
      else if(type == "prd" or (type == "enzOut" and enztype == "prd" ) or (type == "cplxDest" and enztype == "prd"))
	{
	  spr = rec->createProduct();
	  spr->setSpecies( clean_name );
	  spr->setStoichiometry( stoch );
	}
      else if(type == "enzDest")
	{
	  mspr = rec->createModifier();
	  mspr->setSpecies(clean_name);
	}
      /* Updating list of object for annotation for Enzymatic reaction */
      if (re_enClass =="Enz")
	nameList_.push_back(clean_name);

      /* Rate law is also updated in rate_law string */
      //std::size_t found = clean_name.find("cplx");
      //cout << " stoch" << stoch << " c name " << clean_name;
      if (w)
	{
	  rct_order += stoch;
	  if ( stoch == 1 )
	    rate_law << "*" << clean_name;
	  else
	    rate_law << "*" <<clean_name << "^" << stoch;
	}
    } //rRct
  //return rctprdUniq ;
}

void SbmlWriter::getModifier(ModifierSpeciesReference* mspr,vector < Id> mod, int index,ostringstream& rate_law,double &rct_order,bool w)
{ 
  std::set < Id > modifierUniq;
  modifierUniq.insert(mod.begin(),mod.end());
  for (std::set < Id> :: iterator rmod = modifierUniq.begin();rmod!=modifierUniq.end();rmod++)
    { double stoch = count( mod.begin(),mod.end(),*rmod );
      string clean_name = cleanNameId(*rmod,index);
      mspr->setSpecies( clean_name );
      /* Rate law is also updated in rate_law string */
      if (w)
	{
	  rct_order += stoch;
	  if ( stoch == 1 )
	    rate_law << "*" << clean_name;
	  else
	    rate_law << "*" <<clean_name << "^" << stoch;
	}
    } //rRct
  //return modUniq ;
}
/* *  removes special characters  **/

string SbmlWriter::nameString1( string str )
{ string str1;
  int len = str.length();
  int i= 0;
  do
    {
      switch( str.at(i) )
	{
	  case '&':
	    str1 = "_and_";
	    str.replace( i,1,str1 );
	    len += str1.length()-1;
	    break;
	  case '<':
	    str1 = "_lessthan_";
	    str.replace( i,1,str1 );
	    len += str1.length()-1;
	    break; 
	case '>':
	    str1 = "_greaterthan_";
	    str.replace( i,1,str1 );
	    len += str1.length()-1;
	    break; 
	case '':
	    str1 = "&#176;";
	    str.replace( i,1,str1 );
	    len += str1.length()-1;
	    break; 
	}
    i++;
    }while ( i < len );
  return str;

}
string SbmlWriter::nameString( string str )
{ string str1;

  int len = str.length();
  int i= 0;
  do
    {
      switch( str.at(i) )
	{
	case '-':
	  str1 = "_dash_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '\'':
	  str1 = "_prime_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;

	case '+':
	  str1 = "_plus_";
	  str.replace( i,1,str1 );
	  len += str1.length()-1;
	  break;
	case '&':
	  str1 = "_and_";
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
	case ' ':
	  str1 = "_";
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
{string newName = parent + "_" + child + "_";
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
	unit_stream<<"per_item_per_second";
	break;
      case 2:
	unit_stream<<"per_item_sq_per_second";
	break;
      default:
	unit_stream<<"per_item_"<<rct_order<<"_per_second";
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
    //cout << "order:" << order << endl;
    unitdef = cremodel_->createUnitDefinition();
    unitdef->setId( unit_stream.str() );
    
    // Create individual unit objects that will be put inside the UnitDefinition .
    if (order != 0)
      { //cout << "order is != 0: " << order << endl;
	unit = unitdef->createUnit();
	unit->setKind( UNIT_KIND_ITEM );
	unit->setExponent( -order );
	unit->setMultiplier(1);
      }

    unit = unitdef->createUnit();
    unit->setKind( UNIT_KIND_SECOND );
    unit->setExponent( -1 );
  }
  return unit_stream.str();
}
void SbmlWriter::printParameters( KineticLaw* kl,string k,double kvalue,string unit )
{
  Parameter* para = kl->createParameter();
  para->setId( k );
  para->setValue( kvalue );
  para->setUnits( unit );
}

string SbmlWriter::findNotes(Id itr)
{ string path = Field<string> :: get(itr,"path");
  Id annotaId( path+"/info");
  string noteClass = Field<string> :: get(annotaId,"className");
  string notes;
  if (noteClass =="Annotator")
    string notes = Field <string> :: get(annotaId,"notes");

  return notes;
}

/* *  validate a model before writing */

bool SbmlWriter::validateModel( SBMLDocument* sbmlDoc )
{
  if ( !sbmlDoc )
    {cerr << "validateModel: given a null SBML Document" << endl;
      return false;
    }

  string consistencyMessages;
  string validationMessages;
  bool noProblems                     = true;
  unsigned int numCheckFailures       = 0;
  unsigned int numConsistencyErrors   = 0;
  unsigned int numConsistencyWarnings = 0;
  unsigned int numValidationErrors    = 0;
  unsigned int numValidationWarnings  = 0;
  // Once the whole model is done and before it gets written out, 
  // it's important to check that the whole model is in fact complete, consistent and valid.
  numCheckFailures = sbmlDoc->checkInternalConsistency();
  if ( numCheckFailures > 0 )
    {
      noProblems = false;
      for ( unsigned int i = 0; i < numCheckFailures; i++ )
	{
	  const SBMLError* sbmlErr = sbmlDoc->getError(i);
	  if ( sbmlErr->isFatal() || sbmlErr->isError() )
	    {
	      ++numConsistencyErrors;
	    }
	  else
	    {
	      ++numConsistencyWarnings;
	    }
	  } 
      ostringstream oss;
      //cout << "here in l 695 line" << endl;
      sbmlDoc->printErrors(oss);
      consistencyMessages = oss.str();
      }
  // If the internal checks fail, it makes little sense to attempt
  // further validation, because the model may be too compromised to
  // be properly interpreted.
  if ( numConsistencyErrors > 0 )
    {
      consistencyMessages += "Further validation aborted.";
     }
  else
    {
      numCheckFailures = sbmlDoc->checkConsistency();
      if ( numCheckFailures > 0 )
	{
	  noProblems = false;
	  for ( unsigned int i = 0; i < numCheckFailures; i++ )
	    {
	      const SBMLError* sbmlErr = sbmlDoc->getError(i);
	      if ( sbmlErr->isFatal() || sbmlErr->isError() )
		{
		  ++numValidationErrors;
		}
	      else
		{
		  ++numValidationWarnings;
		}      
	      }
	  ostringstream oss;
	  sbmlDoc->printErrors( oss );
	  validationMessages = oss.str();
	  }
	  }
  if ( noProblems )
    return true;
  else
    {
      if ( numConsistencyErrors > 0 )
	{
	  cout << "ERROR: encountered " << numConsistencyErrors
	       << " consistency error" << ( numConsistencyErrors == 1 ? "" : "s" )
	       << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
	  }
      if ( numConsistencyWarnings > 0 )
	{
	  cout << "Notice: encountered " << numConsistencyWarnings
	       << " consistency warning" << ( numConsistencyWarnings == 1 ? "" : "s" )
	       << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
	  }
      cout << endl << consistencyMessages;
      if ( numValidationErrors > 0 )
	{
	  cout << "ERROR: encountered " << numValidationErrors
	       << " validation error" << ( numValidationErrors == 1 ? "" : "s" )
	       << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
	}
      if ( numValidationWarnings > 0 )
	{
	  cout << "Notice: encountered " << numValidationWarnings
	       << " validation warning" << ( numValidationWarnings == 1 ? "" : "s" )
	       << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
	  }
      cout << endl << validationMessages;
      return ( numConsistencyErrors == 0 && numValidationErrors == 0 );
      }
}
