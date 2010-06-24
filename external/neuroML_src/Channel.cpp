#include <map>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
#include "NCell.h"
#include "NBase.h"
#include "Channel.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/xmlreader.h>
#include <fstream>

map< std::string,ChannelDefinition >Channel::lookup_;
/**
 * Returns xmlXPathObjectPtr to channelML node for the given path.
 * The xmlPathObjectPtr returned by the function contains a set of nodes and 
 * other information needed to iterate through the set and act on the results.
*/
xmlXPathObjectPtr ChannelDefinition::getnodeset (xmlChar *xpath) 
{
       xmlXPathObjectPtr result;
        
        if (cxtptr == NULL) {
                cout << "Error in xmlXPathNewContext" <<endl;
                return NULL;
        }
	result = xmlXPathEvalExpression(xpath, cxtptr);
        if (result == NULL) {
                cout << "Error in xmlXPathEvalExpression" <<endl;
                return NULL;
        }
        /*if(xmlXPathNodeSetIsEmpty(result->nodesetval)){
                xmlXPathFreeObject(result);
                cout << "No result" <<endl;
                return NULL;
        }*/
        return result;
}
/**
 * Register the namespaces used in channelML file and returns 0 if success.
 */
int ChannelDefinition::register_channelNamespaces() 
{
    
     int ret,r;
     if (rdrptr != NULL) {
     	ret = xmlTextReaderRead(rdrptr);
	if (ret == 1) {
	   int tot = xmlTextReaderAttributeCount(rdrptr);
	   for( int  i = 0; i < tot-1; i++ )
	   {
	 	if (xmlTextReaderMoveToAttributeNo(rdrptr,i)){
		   const xmlChar * readername = xmlTextReaderName(rdrptr);
		   const xmlChar * value = xmlTextReaderValue(rdrptr);
		   const xmlChar * prefix = xmlTextReaderPrefix(rdrptr);
		   xmlNodePtr root = xmlDocGetRootElement(docptr);
                   if (prefix == NULL){ 
			if ((!xmlStrcmp(root->name, (const xmlChar *) "channelml")))
				readername = (xmlChar * )" channelml" ;
			prefix = (xmlChar *)"";
		   }
		   int	namelen = xmlStrlen(readername);
		   int	pfxlen = xmlStrlen(prefix);
		   const xmlChar * name = xmlStrsub(readername,pfxlen+1,namelen); 
		   r = xmlXPathRegisterNs(cxtptr,name,value);
	           if ( r != 0 ){
		      cerr << "Error: unable to register NS with prefix= " 
				 << name << " and href= " << value;    
		      return(-1);
		   }
		   else setchlnamespaces = true;
	    	}
	    }
	    
	  }
	     
	}
	return(r);
}
/**
 *  Reads in the definition for the given channelML file. 
 */
void ChannelDefinition::readDefinition( std::string& filename )
{
        docptr = xmlParseFile(&filename[0]);
  	cxtptr = xmlXPathNewContext(docptr);
  	rdrptr = xmlReaderForFile(filename.c_str(), NULL, 0);
	int r = register_channelNamespaces();
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
	xmlNodePtr cur,root;
	xmlChar *density;
	int size,psize,grpsize;
	char *endp, *conc_name, *conc_ion, *depen_charge, *variable_name, *smin_conc, *smax_conc;
	char * type, *chlname, *stvalue, *claw, *d_gmax, *ion, *d_erev, *scharge;
	if (r == 0){	
	   result = getnodeset((xmlChar *)"//channelml:channel_type");
   	   nodeset = result->nodesetval;
	   unsigned int num_types = (nodeset) ? nodeset->nodeNr : 0;
   	   for(unsigned int i = 0; i < num_types; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
        	   cur = nodeset->nodeTab[i];  
        	    if(cur->ns) { 
          		chlname = (char *) xmlGetProp(cur,(const xmlChar *) "name");
	  	        setChannel_type( chlname );
          		density = (xmlChar *) xmlGetProp(cur,(const xmlChar *) "density");
	  		if ((!xmlStrcmp(density, (const xmlChar *) "yes")))
				setDensity( true );
	  		else 
				setDensity( false );
			xmlFree( chlname );
			xmlFree( density );
	  	    }
		/* result = getnodeset((xmlChar *)"//channelml:status");
       		nodeset = result->nodesetval;
		size = (nodeset) ? nodeset->nodeNr : 0;
		for(int v = 0; v < size; v++) {
           	    assert(nodeset->nodeTab[v]);
           	    if(nodeset->nodeTab[v]->type == XML_ELEMENT_NODE) {
             	            cur = nodeset->nodeTab[v];  
		            if(cur->ns){ 
		        	stvalue = (char *) xmlGetProp(cur,(const xmlChar *) "value");
		     	    }
	             }
     		}*/
      	     	}
         }
        result = getnodeset((xmlChar *)"//channelml:current_voltage_relation");
        nodeset = result->nodesetval;
        size = (nodeset) ? nodeset->nodeNr : 0;
        for(int i = 0; i < size; i++) {
           assert(nodeset->nodeTab[i]);
           if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              cur = nodeset->nodeTab[i];  
              if(cur->ns){ 
                claw = (char *) xmlGetProp(cur,(const xmlChar *) "cond_law");
		ion = (char *) xmlGetProp(cur,(const xmlChar *) "ion");
		d_erev = (char *) xmlGetProp(cur,(const xmlChar *) "default_erev");
		d_gmax = (char *) xmlGetProp(cur,(const xmlChar *) "default_gmax");
		scharge = (char *) xmlGetProp(cur,(const xmlChar *) "charge");
		double erev = strtod( d_erev,&endp );
		double gmax = strtod( d_gmax, &endp );
		if ( scharge != '\0' ){
			double charge = strtod( scharge,&endp );
			setVolt_Charge( charge );
			setCharge = true; 
		}
		setCond_law( claw );
		setIon( ion );
		setDefault_erev( erev );
		setDefault_gmax( gmax );
		xmlFree(claw);
		xmlFree(ion);
		xmlFree(d_erev);
		xmlFree(d_gmax);
		xmlFree(scharge);
	   }
           }
       }
       result = getnodeset((xmlChar *)"//channelml:conc_dependence");
       nodeset = result->nodesetval;
       size = (nodeset) ? nodeset->nodeNr : 0;
       if( size == 0 ){
	  result = getnodeset((xmlChar *)"//channelml:conc_factor");
       	  nodeset = result->nodesetval;
          size = (nodeset) ? nodeset->nodeNr : 0;
       }
       for(int i = 0; i < size; i++) {
           assert(nodeset->nodeTab[i]);
           if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              cur = nodeset->nodeTab[i];  
              if(cur->ns){ 
               // conc_name = (char *) xmlGetProp(cur,(const xmlChar *) "name");
		conc_ion = (char *) xmlGetProp(cur,(const xmlChar *) "ion");
		depen_charge = (char *) xmlGetProp(cur,(const xmlChar *) "charge");
		variable_name = (char *) xmlGetProp(cur,(const xmlChar *) "variable_name");
		smin_conc = (char *) xmlGetProp(cur,(const xmlChar *) "min_conc");
		smax_conc = (char *) xmlGetProp(cur,(const xmlChar *) "max_conc");
		double charge = strtod( depen_charge,&endp );
		double min_conc = strtod( smin_conc, &endp );
		double max_conc = strtod( smax_conc, &endp );
		setconc_dependence = true;
		//setConc_name( conc_name );
		setConc_ion( conc_ion );
		setVariable_name( variable_name );
		setDepen_Charge( charge );
		setMin_conc( min_conc );
		setMax_conc( max_conc );
		
	   }
           }
       	}
       	result = getnodeset((xmlChar *)"//channelml:impl_prefs/table_settings");
       	nodeset = result->nodesetval;
       	size = (nodeset) ? nodeset->nodeNr : 0;
	char *endp, *max_v, *min_v, *divs;
	for(int i = 0; i < size; i++) {
           assert(nodeset->nodeTab[i]);
           if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              cur = nodeset->nodeTab[i];  
              if(cur->ns){ 
                max_v = (char *) xmlGetProp(cur,(const xmlChar *) "max_v");
		min_v = (char *) xmlGetProp(cur,(const xmlChar *) "min_v");
		divs = (char *) xmlGetProp(cur,(const xmlChar *) "table_divisions");
		double max = strtod( max_v,&endp );
		double min = strtod( min_v,&endp );
		double div = strtod( divs,&endp );
		setMax_v( max );
		setMin_v( min );
		setDivs( div );
       	      }
	   }
	}

   }
}
/**
 * Returns the ChannelDefinition object corresponding to the channel.
 * For now it is assumed that filename is channelname.xml 
 */
ChannelDefinition* Channel::lookupDefinition( const std::string& name )
{	std::string filename = name + ".xml";
        ifstream fin;
	string path;
	const vector< string >& paths = NBase::getPaths();
	for ( unsigned int i = 0; i < paths.size(); i++ ) {
		path = paths[ i ] + "/" + filename;
		cout << "path : " << path << endl; 
		fin.open( path.c_str() );
		if ( fin ){
			fin.close();
			cout << "The channel "<< name << " is loaded from "<< path << endl;
			break;
		}
		/*else{
			fin.clear();
			cerr<< "The channel " << name << " is not found in any of the paths" << endl ;
			return 0;
		}*/
	}

	static map< string,ChannelDefinition >::iterator iter;
	iter = lookup_.find( name );
	if( iter == lookup_.end() ){
		lookup_[name].readDefinition( path ); 
	}
	return &lookup_[name];
	
	
}
/**
 * Sets the value of the "name" attribute of this Channel. 
 */
void Channel::setName( const std::string& value )
{
	name = value;
	definition_ = lookupDefinition( value );
	
}
/**
 * Unsets the value of the "name" attribute of this Channel. 
 */
void Channel::unsetName( )
{
	name.erase();
}
/**
 * Predicate returning true or false depending on whether this Channel's "name"
 * attribute has been set. 
 */
bool Channel::isSetName () const
{
   return (name.empty() == false);
}
/**
 * Returns the value of the "name" attribute of this Channel. 
 * 
 */
const std::string& Channel::getName() const
{
	return name;
}

double ChannelDefinition::getMax_v()
{
	return max_v;
}

void ChannelDefinition::setMax_v( double value )
{
	max_v = value;
}

double ChannelDefinition::getMin_v()
{
	return min_v;
}
void ChannelDefinition::setMin_v( double value )
{
	min_v = value;
}

double ChannelDefinition::getDivs()
{
	return divs;
}

void ChannelDefinition::setDivs( double value )
{
	divs = value;
}
/**
 * Returns the value of the "max_v" attribute of this Channel.
 * 
 */
double Channel::getMax_v()
{
	return definition_->getMax_v();
}
/**
 * Sets the value of the "max_v" attribute of this Channel. 
 * 
 */
void Channel::setMax_v( double value )
{
	definition_->setMax_v( value );
}
/**
 * Returns the value of the "min_v" attribute of this Channel.
 * 
 */
double Channel::getMin_v()
{
	return definition_->getMin_v();
}
/**
 * Sets the value of the "min_v" attribute of this Channel. 
 * 
 */
void Channel::setMin_v( double value )
{
	definition_->setMin_v( value );
}
/**
 * Returns the value of the "divs" attribute of this Channel.
 * 
 */
double Channel::getDivs()
{
	return definition_->getDivs();
}
/**
 * Sets the value of the "divs" attribute of this Channel. 
 * 
 */
void Channel::setDivs( double value )
{
	definition_->setDivs( value );
}
/**
 * Sets the value of the "gmax" attribute of this Channel. 
 * 
 */
void Channel::setGmax( double value )
{
	gmax = value;
}
/**
 * Returns the value of the "gmax" attribute of this Channel.
 * 
 */
const double Channel::getGmax() const
{
	return gmax;
}
/**
 * Returns the value of the "segGroups" attribute of this Channel.
 * 
 */
vector< string > Channel::getSegGroups() const
{
	return segGroups;
}
/**
 * Sets the value of the "segGroups" attribute of this Channel. 
 * 
 */
void Channel::setSegGroups( string seg )
{
	segGroups.push_back( seg );
}
/**
 * 
 * Unsets the value of the "segGroups" attribute of this Channel. 
 */
void Channel::unsetSegGroups()
{
	segGroups.clear();
}
/**
 * Sets the value of the "groups" attribute of this Channel. 
 * 
 */
void Channel::setGroups( string value )
{
	groups.push_back( value );
}
/**
 * Returns the value of the "groups" attribute of this Channel.
 * 
 */
vector < string > Channel::getGroups() const
{
	return groups;
}
/**
 * Sets the value of the "passivecond" attribute of this Channel. 
 * 
 */
void Channel::setPassivecond(bool value)
{
 	passivecond = value;
}
/**
 * Returns the value of the "passivecond" attribute of this Channel.
 * 
 */
bool Channel::getPassivecond() const
{
	return passivecond;
}
/**
 * Returns the value of the "channel_type" attribute of this Channel.
 * 
 */
const std::string& Channel::getChannel_type()
{
	return definition_->getChannel_type();
}
/**
 * Sets the value of the "channel_type" attribute of this Channel. 
 * 
 */
void Channel::setChannel_type(const std::string& value )
{
	definition_->setChannel_type( value );
}
/**
 * Returns the value of the "density" attribute of this Channel.
 * 
 */
bool Channel::getDensity()
{
	return definition_->getDensity();
}
/**
 * Sets the value of the "density" attribute of this Channel. 
 * 
 */
void Channel::setDensity( bool value )
{
	definition_->setDensity( value );
}
/**
 * Returns the value of the "claw" attribute of this Channel.
 * 
 */
const std::string& Channel::getCond_law()
{
	return definition_->getCond_law();
}
/**
 * Sets the value of the "claw" attribute of this Channel. 
 * 
 */
void Channel::setCond_law(const std::string& value )
{
	definition_->setCond_law( value );
}
/**
 * Returns the value of the "ion" attribute of this Channel.
 * 
 */
const std::string& Channel::getIon()
{
	return definition_->getIon();
}
/**
 * Sets the value of the "ion" attribute of this Channel. 
 * 
 */
void Channel::setIon(const std::string& value )
{
	definition_->setIon( value );
}
/**
 * Returns the value of the "erev" attribute of this Channel.
 * 
 */
double Channel::getDefault_erev() 
{
	return definition_->getDefault_erev();
}
/**
 * Sets the value of the "erev" attribute of this Channel. 
 * 
 */
void Channel::setDefault_erev(double value )
{
	definition_->setDefault_erev( value );
}
/**
 * Returns the value of the "gmax" attribute of this Channel.
 * 
 */
double Channel::getDefault_gmax()
{
	return definition_->getDefault_gmax();
}
/**
 * Sets the value of the "gmax" attribute of this Channel. 
 * 
 */
void Channel::setDefault_gmax(double value )
{
	definition_->setDefault_gmax( value );
}

const std::string& ChannelDefinition::getChannel_type()
{
	return channel_type;
}

void ChannelDefinition::setChannel_type(const std::string& value )
{
	channel_type = value;
}

bool ChannelDefinition::getDensity()
{
	return density;
}

void ChannelDefinition::setDensity( bool value )
{
	density = value;
}

const std::string& ChannelDefinition::getCond_law()
{
	return claw;
}

void ChannelDefinition::setCond_law(const std::string& value )
{
	claw = value;
}

const std::string& ChannelDefinition::getIon()
{
	return ion;
}
void ChannelDefinition::setIon(const std::string& value )
{
	ion = value;
}

double ChannelDefinition::getDefault_erev() 
{
	return erev;
}

void ChannelDefinition::setDefault_erev(double value )
{
	erev = value;
}

double ChannelDefinition::getDefault_gmax()
{
	return gmax;
}
void ChannelDefinition::setDefault_gmax(double value )
{
	gmax = value;
}

const std::string& Gate::getX_variable()
{
	return x_variable;
}
void Gate::setX_variable(const std::string& value )
{
	x_variable = value;
}
/**
 * Returns the value of the "name" attribute of this Channel.
 * 
 */
const std::string& Gate::getY_variable()
{
	return y_variable;
}
/**
 * Sets the value of the "name" attribute of this Channel. 
 * 
 */
void Gate::setY_variable(const std::string& value )
{
	y_variable = value;
}

const std::string& Gate::getGateName()
{
	return gateName;
}

void Gate::setGateName(const std::string& value )
{
	gateName = value;
}

double Gate::getInstances()
{
	return instances;
}

void Gate::setInstances(double value )
{
	instances = value;
}

const std::string& Gate::getClosestateId()
{
	return closeId;
}

void Gate::setClosestateId(const std::string& value )
{
	closeId = value;
}

const std::string& Gate::getOpenstateId()
{
	return openId;
}

void Gate::setOpenstateId(const std::string& value )
{
	openId = value;
}
/*const std::string& ChannelDefinition::getConc_name()
{
	return conc_name;
}
void ChannelDefinition::setConc_name(const std::string& value )
{
	conc_name = value;
}*/

const std::string& ChannelDefinition::getConc_ion()
{
	return conc_ion;
}

void ChannelDefinition::setConc_ion(const std::string& value )
{
	conc_ion = value;
}

const std::string& ChannelDefinition::getVariable_name()
{
	return variable_name;
}

void ChannelDefinition::setVariable_name(const std::string& value )
{
	variable_name = value;
}

double ChannelDefinition::getVolt_Charge()
{
	return charge;
}

void ChannelDefinition::setVolt_Charge(double value )
{
	charge = value;
}

double ChannelDefinition::getDepen_Charge()
{
	return depen_charge;
}

void ChannelDefinition::setDepen_Charge(double value )
{
	depen_charge = value;
}

double ChannelDefinition::getMin_conc()
{
	return min_conc;
}

void ChannelDefinition::setMin_conc(double value )
{
	min_conc = value;
}

double ChannelDefinition::getMax_conc()
{
	return max_conc;
}

void ChannelDefinition::setMax_conc(double value )
{
	max_conc = value;
}

bool ChannelDefinition::isSetConc_dependence()
{
	return setconc_dependence;
}

bool ChannelDefinition::isSetCharge()
{
 	return setCharge;
}

bool ChannelDefinition::getFixed_erev()
{
	return fixed_erev;
}

void ChannelDefinition::setFixed_erev( bool value )
{
	fixed_erev = value;
}

unsigned int ChannelDefinition::getNumGates() 
{
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
  	if (! issetChannelNamespaces() ){	
	   if (register_channelNamespaces() < 0){	
    	    	cerr << "Error: unable to register Namespaces" << endl;
     		xmlXPathFreeContext(cxtptr); 
     		xmlFreeDoc(docptr); 
  	   }
	}
	result = getnodeset((xmlChar *)"//channelml:gate");
        nodeset = result->nodesetval;
        unsigned int numGates = (nodeset) ? nodeset->nodeNr : 0;
	return numGates;
}

bool ChannelDefinition::issetChannelNamespaces() 
{
	return setchlnamespaces;
}

Gate* ChannelDefinition::getGate(int n)
{
  xmlXPathObjectPtr result;
  xmlNodeSetPtr nodeset,tab_nodeset,entry_nodeset;
  xmlNodePtr cur,cur1;
  if (! issetChannelNamespaces() ){	
	   if (register_channelNamespaces() < 0){	
    	    	cerr << "Error: unable to register Namespaces" << endl;
     		xmlXPathFreeContext(cxtptr); 
     		xmlFreeDoc(docptr);
  	   }
  }
  ostringstream xpath;
  xpath<<"//channelml:gate["<<n<<"]";
  result = getnodeset((xmlChar *)(xpath.str().c_str()));
  nodeset = result->nodesetval;
  int size = (nodeset) ? nodeset->nodeNr : 0;
  for(int i = 0; i < size; ++i) {
     assert(nodeset->nodeTab[i]);
     if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
       cur = nodeset->nodeTab[i];  
       if(cur->ns) { 
         char *instances, *gtname, *x_variable, *y_variable, *endp;
	 gtname = (char *) xmlGetProp(cur,(const xmlChar *) "name");
	 gate.setGateName( gtname );
	 instances = (char *) xmlGetProp(cur,(const xmlChar *) "instances");
	 double inst = strtod( instances,&endp );
	 gate.setInstances( inst );
	 x_variable = (char *) xmlGetProp(cur,(const xmlChar *) "x_variable");
	 if ( x_variable != '\0' )
	     gate.setX_variable( x_variable );
	else 
		gate.setX_variable( "");
	 y_variable = (char *) xmlGetProp(cur,(const xmlChar *) "y_variable");
	 if ( y_variable != '\0' )
	     gate.setY_variable( y_variable );
	 xpath.str( "" ); 
	 xpath<<"//channelml:gate["<<n<<"]/channelml:closed_state";	   
	 result = getnodeset((xmlChar *)xpath.str().c_str());
	 nodeset = result->nodesetval;
	 int size = (nodeset) ? nodeset->nodeNr : 0;
	 for(int i = 0; i < size; ++i) {
             assert(nodeset->nodeTab[i]);
             if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 cur = nodeset->nodeTab[i];  
                 if(cur->ns) { 
                 	char *id;
	     	 	id = (char *) xmlGetProp(cur,(const xmlChar *) "id");
			gate.setClosestateId( id );
		 }
	     }
	 }
	 xpath.str( "" ); 
	 xpath<<"//channelml:gate["<<n<<"]/channelml:open_state";	   
	 result = getnodeset((xmlChar *)xpath.str().c_str());
	 nodeset = result->nodesetval;
	 size = (nodeset) ? nodeset->nodeNr : 0;
	 for(int i = 0; i < size; ++i) {
             assert(nodeset->nodeTab[i]);
             if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 cur = nodeset->nodeTab[i];  
                 if(cur->ns) { 
                 	char *id;
	     	 	id = (char *) xmlGetProp(cur,(const xmlChar *) "id");
			gate.setOpenstateId( id );
		 }
	     }
	 }
	 xpath.str( "" ); 
	 xpath<<"//channelml:gate["<<n<<"]/channelml:transition";	   
	 result = getnodeset((xmlChar *)xpath.str().c_str());
	 nodeset = result->nodesetval;
	 size = (nodeset) ? nodeset->nodeNr : 0;
	 for(int i = 0; i < size; ++i) {
             assert(nodeset->nodeTab[i]);
             if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 cur = nodeset->nodeTab[i];  
                 if(cur->ns) { 
			xmlChar *name, *expr_form;
                 	char *from, *to, *rate, *scale, *midpoint, *expr;
			double drate,dscale,mpoint;
	     	 	name = (xmlChar *) xmlGetProp(cur,(const xmlChar *) "name");
			from = (char *) xmlGetProp(cur,(const xmlChar *) "from");
			to = (char *) xmlGetProp(cur,(const xmlChar *) "to");
			expr_form = (xmlChar *) xmlGetProp(cur,(const xmlChar *) "expr_form");
			rate = (char *) xmlGetProp(cur,(const xmlChar *) "rate");
			scale = (char *) xmlGetProp(cur,(const xmlChar *) "scale");
			midpoint = (char *) xmlGetProp(cur,(const xmlChar *) "midpoint");
			expr = (char *) xmlGetProp(cur,(const xmlChar *) "expr");
			if ((!xmlStrcmp(name, (const xmlChar *) "alpha"))){
				if ((!xmlStrcmp(expr_form, (const xmlChar *) "tabulated"))){
			     	    xpath.str( "" ); 
	 		     	    xpath<<"//channelml:gate["<<n<<"]/channelml:transition[1]/channelml:table";
				    result = getnodeset((xmlChar *)xpath.str().c_str());
	 			    tab_nodeset = result->nodesetval;
				    
	 			    int tabsize = (tab_nodeset) ? tab_nodeset->nodeNr : 0;
	 			    for(int i = 0; i < tabsize; ++i) {
             				assert(tab_nodeset->nodeTab[i]);
             				if(tab_nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 			   cur = tab_nodeset->nodeTab[i];  
                 			   if(cur->ns) { 
					     char *min, *max;
				    	     double xmin, xmax;
					     min = (char *) xmlGetProp(cur,(const xmlChar *) "xmin");
					     max = (char *) xmlGetProp(cur,(const xmlChar *) "xmax");
					     if ( min != '\0' ){
						xmin = strtod( min,&endp );
						gate.alpha.xmin = xmin;
					     }
					     if ( max != '\0' ){
						xmax = strtod( max,&endp );
						gate.alpha.xmax = xmax;
					     }
					   }
					}
					xpath.str( "" ); 
					xpath<<"//channelml:gate["<<n<<"]/channelml:transition[1]/channelml:table/channelml:entry";
				        result = getnodeset((xmlChar *)xpath.str().c_str());
	 			        entry_nodeset = result->nodesetval;
				        int entrysize = (entry_nodeset) ? entry_nodeset->nodeNr : 0;
					gate.alpha.tableEntry.clear();
	 			    	for(int i = 0; i < entrysize; ++i) {
             				    assert(entry_nodeset->nodeTab[i]);
             				    if(entry_nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 			       cur1 = entry_nodeset->nodeTab[i];  
                 			       if(cur1->ns) { 
					          char *entry;
				    	          double value;
					          entry = (char *) xmlGetProp(cur1,(const xmlChar *) "value");
						  if ( entry != '\0' ){
							value = strtod( entry,&endp );
							//cout << "value" << value << endl;
							gate.alpha.tableEntry.push_back( value );
					     	  }
						}
					    }
					}
				     }
				}
				gate.alpha.name = "alpha";
				gate.alpha.from = from;
				gate.alpha.to = to;
				std::string expform((char *) expr_form);
				gate.alpha.expr_form = expform;
				if ( rate != '\0' ){
					drate = strtod( rate,&endp );
				gate.alpha.rate = drate;
				}
				if ( scale != '\0' ){
					dscale = strtod( scale,&endp );
				gate.alpha.scale = dscale;
				}
				if ( midpoint != '\0' ){
					mpoint = strtod( midpoint,&endp );
				gate.alpha.midpoint = mpoint;
				}
				if( expr != '\0' )
					gate.alpha.expr = expr;
			}
			else if ((!xmlStrcmp(name, (const xmlChar *) "beta"))){
				if ((!xmlStrcmp(expr_form, (const xmlChar *) "tabulated"))){
			     	    xpath.str( "" ); 
	 		     	    xpath<<"//channelml:gate["<<n<<"]/channelml:transition[2]/channelml:table";	
				    result = getnodeset((xmlChar *)xpath.str().c_str());
	 			    tab_nodeset = result->nodesetval;
				    int tabsize = (tab_nodeset) ? tab_nodeset->nodeNr : 0;
	 			    for(int i = 0; i < tabsize; ++i) {
             				assert(tab_nodeset->nodeTab[i]);
             				if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 			   cur = tab_nodeset->nodeTab[i];  
                 			   if(cur->ns) { 
					     char *min, *max;
				    	     double xmin, xmax;
					     min = (char *) xmlGetProp(cur,(const xmlChar *) "xmin");
					     max = (char *) xmlGetProp(cur,(const xmlChar *) "xmax");
					     if ( min != '\0' ){
						xmin = strtod( min,&endp );
						gate.beta.xmin = xmin;
					     }
					     if ( max != '\0' ){
						xmax = strtod( max,&endp );
						gate.beta.xmax = xmax;
					     }
					   }
					}
					xpath.str( "" ); 
	 		     	        xpath<<"//channelml:gate["<<n<<"]/channelml:transition[2]/channelml:table/channelml:entry";
				        result = getnodeset((xmlChar *)xpath.str().c_str());
	 			        entry_nodeset = result->nodesetval;
					gate.beta.tableEntry.clear();
				        int entrysize = (entry_nodeset) ? entry_nodeset->nodeNr : 0;
	 			    	for(int i = 0; i < entrysize; ++i) {
             				    assert(entry_nodeset->nodeTab[i]);
             				    if(entry_nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
                 			       cur1 = entry_nodeset->nodeTab[i];  
                 			       if(cur1->ns) { 
					          char *entry;
				    	          double value;
					          entry = (char *) xmlGetProp(cur1,(const xmlChar *) "value");
						  if ( entry != '\0' ){
							value = strtod( entry,&endp );
							//cout << "beta value" << value << endl;
							gate.beta.tableEntry.push_back( value );
					     	  }
						}
					    }
					}
				     }
				}			
				
				gate.beta.name = "beta";
				gate.beta.from = from;
				gate.beta.to = to;
				std::string expform((char *) expr_form);
				gate.beta.expr_form = expform;
				if ( rate != '\0' ){
					drate = strtod( rate,&endp );
				gate.beta.rate = drate;
				}
				if ( scale != '\0' ){
					dscale = strtod( scale,&endp );
				gate.beta.scale = dscale;
				}
				if ( midpoint != '\0' ){
					mpoint = strtod( midpoint,&endp );
				gate.beta.midpoint = mpoint;
				}
				if( expr != '\0' )
					gate.beta.expr = expr;
		 	  }
	     	   }
	    }
	 }
	
	}
     }
   }
   return &gate;	
}
/**
 * Returns the value of the "x_variable" attribute of this Channel.
 * 
 */
const std::string& Channel::getX_variable()
{
	return definition_->gate.getX_variable();
}
/**
 * Sets the value of the "x_variable" attribute of this Channel. 
 * 
 */
void Channel::setX_variable(const std::string& value )
{
	definition_->gate.setX_variable( value );
}
/**
 * Returns the value of the "y_variable" attribute of this Channel.
 * 
 */
const std::string& Channel::getY_variable()
{
	return definition_->gate.getY_variable();
}
/**
 * 
 * Sets the value of the "y_variable" attribute of this Channel. 
 */
void Channel::setY_variable(const std::string& value )
{
	definition_->gate.setY_variable( value );
}
/**
 * Returns the value of the "gateName" attribute of this Channel.
 * 
 */
const std::string& Channel::getGateName()
{
	return definition_->gate.getGateName();
}
/**
 * Sets the value of the "gateName" attribute of this Channel. 
 * 
 */
void Channel::setGateName(const std::string& value )
{
	definition_->gate.setGateName( value );
}
/**
 * Returns the value of the "instances" attribute of this Channel.
 * 
 */
double Channel::getInstances()
{
	return definition_->gate.getInstances();
}
/**
 * 
 * Sets the value of the "instances" attribute of this Channel. 
 */
void Channel::setInstances(double value )
{
	definition_->gate.setInstances( value );
}
/**
 * Returns the value of the "closeId" attribute of this Channel.
 * 
 */
const std::string& Channel::getClosestateId()
{
	return definition_->gate.getClosestateId();
}
/**
 * Sets the value of the "closeId" attribute of this Channel. 
 * 
 */
void Channel::setClosestateId(const std::string& value )
{
	definition_->gate.setClosestateId( value );
}
/**
 * Returns the value of the "openId" attribute of this Channel.
 * 
 */
const std::string& Channel::getOpenstateId()
{
	return definition_->gate.getOpenstateId();
}
/**
 * Sets the value of the "openId" attribute of this Channel. 
 * 
 */
void Channel::setOpenstateId(const std::string& value )
{
	definition_->gate.setOpenstateId( value );
}
/**
 * Returns an object of Gate attribute of this Channel.
 * 
 */
Gate* Channel::getGate(int n)
{
	return definition_->getGate(n);
}
/**
 * Returns the number of gates in the Channel.
 * 
 */
unsigned int Channel::getNumGates() 
{
	return definition_->getNumGates();
}
/*const std::string& Channel::getConc_name()
{
	return definition_->getConc_name();
}
void Channel::setConc_name(const std::string& value )
{
	definition_->setConc_name( value );
}*/
/**
 * Returns the value of the "conc_ion" attribute of this Channel.
 * 
 */
const std::string& Channel::getConc_ion()
{
	return definition_->getConc_ion();
}
/**
 * Sets the value of the "conc_ion" attribute of this Channel. 
 * 
 */
void Channel::setConc_ion(const std::string& value )
{
	definition_->setConc_ion( value );
}
/**
 * Returns the value of the "variable_name" attribute of this Channel.
 * 
 */
const std::string& Channel::getVariable_name()
{
	return definition_->getVariable_name();
}
/**
 * Sets the value of the "variable_name" attribute of this Channel. 
 * 
 */
void Channel::setVariable_name(const std::string& value )
{
	definition_->setVariable_name( value );
}
/**
 * Returns the value of the "charge" attribute of this Channel.
 * 
 */
double Channel::getVolt_Charge()
{
	return definition_->getVolt_Charge();
}
/**
 * Sets the value of the "charge" attribute of this Channel. 
 * 
 */
void Channel::setVolt_Charge(double value )
{
	definition_->setVolt_Charge( value );
}
/**
 * Returns the value of the "min_conc" attribute of this Channel.
 * 
 */
double Channel::getMin_conc()
{
	return definition_->getMin_conc();
}
/**
 * Sets the value of the "min_conc" attribute of this Channel. 
 * 
 */
void Channel::setMin_conc(double value )
{
	definition_->setMin_conc( value );
}
/**
 * Returns the value of the "max_conc" attribute of this Channel.
 * 
 */
double Channel::getMax_conc()
{
	return definition_->getMax_conc();
}
/**
 * Sets the value of the "max_conc" attribute of this Channel. 
 * 
 */
void Channel::setMax_conc(double value )
{
	definition_->setMax_conc( value );
}
/**
 * Predicate returning true or false depending on whether this Channel's 
 * "setconc_dependence" attribute has been set. 
 * 
 */
bool Channel::isSetConc_dependence()
{
	return definition_->isSetConc_dependence();
}
/**
 * Returns the value of the "fixed_erev" attribute of this Channel.
 * 
 */
bool Channel::getFixed_erev()
{
	return definition_->getFixed_erev();
}
/**
 * Sets the value of the "fixed_erev" attribute of this Channel. 
 * 
 */
void Channel::setFixed_erev( bool value )
{
	definition_->setFixed_erev(value);
}
/**
 * Returns the value of the "depen_charge" attribute of this Channel.
 * 
 */
double Channel::getDepen_Charge()
{
	return definition_->getDepen_Charge();
}
/**
 * Sets the value of the "depen_charge" attribute of this Channel. 
 * 
 */
void Channel::setDepen_Charge(double value )
{
	definition_->setDepen_Charge(value);
}
/**
 * Predicate returning true or false depending on whether this Channel's 
 * "setCharge" attribute has been set. 
 * 
 */
bool Channel::isSetCharge()
{
 	return definition_->isSetCharge();
}
/**
 * 
 * Unsets the value of the "groups" attribute of this Channel. 
 */
void Channel::unsetGroups()
{
	groups.clear();
}
