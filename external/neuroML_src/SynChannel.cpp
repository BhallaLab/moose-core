#include <map>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "NCell.h"
#include "SynChannel.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/xmlreader.h>

map< std::string,SynChannelDefinition >SynChannel::synlookup_  ;
/**
 * Returns xmlXPathObjectPtr to channelML node for the given path.
 * The xmlPathObjectPtr returned by the function contains a set of nodes and 
 * other information needed to iterate through the set and act on the results.
*/
xmlXPathObjectPtr SynChannelDefinition::getnodeset (xmlChar *xpath) 
{
       xmlXPathObjectPtr result;
        
        if (syncxtptr == NULL) {
                cout << "Error in xmlXPathNewContext" <<endl;
                return NULL;
        }
	result = xmlXPathEvalExpression(xpath, syncxtptr);
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
int SynChannelDefinition::register_synchannelNamespaces() 
{
    
     int ret,r;
     if (synrdrptr != NULL) {
     	ret = xmlTextReaderRead(synrdrptr);
	if (ret == 1) {
	   int tot = xmlTextReaderAttributeCount(synrdrptr);
	   for( int  i = 0; i < tot-1; i++ )
	   {
	 	if (xmlTextReaderMoveToAttributeNo(synrdrptr,i)){
		   const xmlChar * readername = xmlTextReaderName(synrdrptr);
		   const xmlChar * value = xmlTextReaderValue(synrdrptr);
		   const xmlChar * prefix = xmlTextReaderPrefix(synrdrptr);
		   xmlNodePtr root = xmlDocGetRootElement(syndocptr);
                   if (prefix == NULL){ 
			if ((!xmlStrcmp(root->name, (const xmlChar *) "channelml")))
				readername = (xmlChar * )" channelml" ;
			prefix = (xmlChar *)"";
		   }
		   int	namelen = xmlStrlen(readername);
		   int	pfxlen = xmlStrlen(prefix);
		   const xmlChar * name = xmlStrsub(readername,pfxlen+1,namelen); 
		   r = xmlXPathRegisterNs(syncxtptr,name,value);
	           if ( r != 0 ){
		      cerr << "Error: unable to register NS with prefix= " 
				 << name << " and href= " << value;    
		      return(-1);
		   }
		   else setsynnamespaces = true;
	    	}
	    }
	    
	  }
	     
	}
	return(r);
}
/**
 *  Reads in the synaptic channel definition for the given channelML file. 
 */
void SynChannelDefinition::readSynChlDefinition( std::string& filename )
{
        
        syndocptr = xmlParseFile(&filename[0]);
  	syncxtptr = xmlXPathNewContext(syndocptr);
  	synrdrptr = xmlReaderForFile(filename.c_str(), NULL, 0);
	int r = register_synchannelNamespaces();
	xmlXPathObjectPtr result;
  	xmlNodeSetPtr nodeset;
	xmlNodePtr cur,root;
	int size;
	char *endp,*conc,*eta,*gamma;
	char *chlname, *stvalue, *max_cond, *rise_time, *decay_time, *rev_poten;
	isblock = false;
	if (r == 0){	
	   result = getnodeset((xmlChar *)"//channelml:synapse_type");
   	   nodeset = result->nodesetval;
	   unsigned int num_types = (nodeset) ? nodeset->nodeNr : 0;
   	   for(int i = 0; i < num_types; ++i) {
     		assert(nodeset->nodeTab[i]);
     		if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
        	   cur = nodeset->nodeTab[i];  
        	    if(cur->ns) { 
          		chlname = (char *) xmlGetProp(cur,(const xmlChar *) "name");
	  	        setChannel_type( chlname );
          	    }
		result = getnodeset((xmlChar *)"//channelml:status");
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
     		}
      	     }
        }
        result = getnodeset((xmlChar *)"//channelml:doub_exp_syn");
        nodeset = result->nodesetval;
        size = (nodeset) ? nodeset->nodeNr : 0;
	if ( size == 0 ){
	    result = getnodeset((xmlChar *)"//channelml:blocking_syn");
            nodeset = result->nodesetval;
            size = (nodeset) ? nodeset->nodeNr : 0;
	    if (size != 0 ) isblock = true;
	}
        for(int i = 0; i < size; i++) {
           assert(nodeset->nodeTab[i]);
           if(nodeset->nodeTab[i]->type == XML_ELEMENT_NODE) {
              cur = nodeset->nodeTab[i];  
              if(cur->ns){ 
                max_cond = (char *) xmlGetProp(cur,(const xmlChar *) "max_conductance");
		rise_time = (char *) xmlGetProp(cur,(const xmlChar *) "rise_time");
		decay_time = (char *) xmlGetProp(cur,(const xmlChar *) "decay_time");
		rev_poten = (char *) xmlGetProp(cur,(const xmlChar *) "reversal_potential");
		double d_max = strtod( max_cond,&endp );
		double d_riset = strtod( rise_time, &endp );
		double d_decayt = strtod( decay_time, &endp );
		double d_revpoten = strtod( rev_poten, &endp );
		setMax_Conductance( d_max );
		setRise_Time( d_riset );
		setDecay_Time( d_decayt );
		setReversal_Potential( d_revpoten );
	     }
          }
       }
       if ( isblock == true ){
	   result = getnodeset((xmlChar *)"//channelml:block");
           nodeset = result->nodesetval;
           size = (nodeset) ? nodeset->nodeNr : 0;
	   for(int b = 0; b < size; b++) {
           	assert(nodeset->nodeTab[b]);
           	if(nodeset->nodeTab[b]->type == XML_ELEMENT_NODE) {
              	  cur = nodeset->nodeTab[b];  
              	  if(cur->ns){ 
                   conc = (char *) xmlGetProp(cur,(const xmlChar *) "conc");
		   eta = (char *) xmlGetProp(cur,(const xmlChar *) "eta");
		   gamma = (char *) xmlGetProp(cur,(const xmlChar *) "gamma");
		   double d_conc = strtod( conc,&endp );
	           double d_eta = strtod( eta,&endp );
		   double d_gamma = strtod( gamma,&endp );
		   setMgConc( d_conc );
		   setEta ( d_eta );
		   setGamma( d_gamma );
		   
	         }
               }
	    }
        }

   }
}
/**
 * Returns the SynChannelDefinition object corresponding to the synaptic channel.
 * For now it is assumed that filename is channelname.xml 
 */
SynChannelDefinition* SynChannel::synlookupDefinition( const std::string& name )
{	std::string filename = name + ".xml";
	static map< string,SynChannelDefinition >::iterator iter;
	iter = synlookup_.find( name );
	if( iter == synlookup_.end() ){
		synlookup_[name].readSynChlDefinition( filename ); 
	}
	return &synlookup_[name];
}

const std::string& SynChannelDefinition::getChannel_type()
{
	return synapse_type;
}
void SynChannelDefinition::setChannel_type(const std::string& value )
{
	synapse_type = value;
}

bool SynChannelDefinition::issetsynChannelNamespaces()
{
	return setsynnamespaces;
}

double SynChannelDefinition::getMax_Conductance()
{
	return max_conductance;
}

void SynChannelDefinition::setMax_Conductance( double value )
{
	max_conductance = value;
}

double SynChannelDefinition::getRise_Time()
{
	return rise_time;
}

void SynChannelDefinition::setRise_Time( double value )
{
	rise_time = value;
}

double SynChannelDefinition::getDecay_Time()
{
	return decay_time;
}

void SynChannelDefinition::setDecay_Time( double value )
{
	decay_time = value;
}

double SynChannelDefinition::getReversal_Potential()
{
	return reversal_potential;
}

void SynChannelDefinition::setReversal_Potential( double value )
{
	reversal_potential = value;
}
double SynChannelDefinition::getMgConc()
{
	return conc;
}

void SynChannelDefinition::setMgConc( double value )
{
	conc = value;
}
double SynChannelDefinition::getEta()
{
	return eta;
}

void SynChannelDefinition::setEta( double value )
{
	eta = value;
}
double SynChannelDefinition::getGamma()
{
	return gamma;
}

void SynChannelDefinition::setGamma( double value )
{
	gamma = value;
}
/**
 * Sets the value of the "type" attribute of this SynChannel. 
 * 
 */
void SynChannel::setSynType(string value)
{
	type = value;
        syndefinition_ = synlookupDefinition( value );
}
/**
 * Returns the value of the "type" attribute of this SynChannel.
 * 
 */
const std::string& SynChannel::getSynType()const
{
	return type;
}
/**
 * Sets the value of the "groups" attribute of this SynChannel. 
 * 
 */
void SynChannel::setGroups( string value )
{
	groups.push_back( value );
}
/**
 * Returns the value of the "groups" attribute of this SynChannel.
 * 
 */
vector < string > SynChannel::getGroups() const
{
	return groups;
}
/**
 * Unsets the value of the "groups" attribute of this SynChannel. 
 */
void SynChannel::unsetGroups()
{
	groups.clear();
}
/**
 * Returns the value of the "synapse_type" attribute of this SynChannel.
 * 
 */
const std::string& SynChannel::getChannel_type()
{
	return syndefinition_->getChannel_type();
}
/**
 * Sets the value of the "synapse_type" attribute of this SynChannel. 
 * 
 */
void SynChannel::setChannel_type(const std::string& value )
{
	syndefinition_->setChannel_type( value );
}
/**
 * Returns the value of the "max_conductance" attribute of this SynChannel.
 * 
 */
double SynChannel::getMax_Conductance()
{
	return syndefinition_->getMax_Conductance();
}
/**
 * Sets the value of the "max_conductance" attribute of this SynChannel. 
 * 
 */
void SynChannel::setMax_Conductance( double value )
{
	syndefinition_->setMax_Conductance( value );
}
/**
 * Returns the value of the "rise_time" attribute of this SynChannel.
 * 
 */
double SynChannel::getRise_Time()
{
	return syndefinition_->getRise_Time();
}
/**
 * Sets the value of the "rise_time" attribute of this SynChannel. 
 * 
 */
void SynChannel::setRise_Time( double value )
{
	syndefinition_->setRise_Time(value);
}
/**
 * Returns the value of the "decay_time" attribute of this SynChannel.
 * 
 */
double SynChannel::getDecay_Time()
{
	return syndefinition_->getDecay_Time();
}
/**
 * Sets the value of the "decay_time" attribute of this SynChannel. 
 * 
 */
void SynChannel::setDecay_Time( double value )
{
	syndefinition_->setDecay_Time(value);
}
/**
 * Returns the value of the "reversal_potential" attribute of this SynChannel.
 * 
 */
double SynChannel::getReversal_Potential()
{
	return syndefinition_->getReversal_Potential();
}
/**
 * Sets the value of the "reversal_potential" attribute of this SynChannel. 
 * 
 */
void SynChannel::setReversal_Potential( double value )
{
	syndefinition_->setReversal_Potential(value);
}
double SynChannel::getMgConc()
{
	return syndefinition_->getMgConc();
}

void SynChannel::setMgConc( double value )
{
	syndefinition_->setMgConc(value);
}
double SynChannel::getEta()
{
	return syndefinition_->getEta();
}

void SynChannel::setEta( double value )
{
	syndefinition_->setEta(value);
}
double SynChannel::getGamma()
{
	return syndefinition_->getGamma();
}

void SynChannel::setGamma( double value )
{
	syndefinition_->setGamma(value);
}
bool SynChannel::isMgblock()
{
	return syndefinition_->isblock;
}
