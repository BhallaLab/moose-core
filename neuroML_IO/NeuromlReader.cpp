#include "moose.h"
#include "element/Neutral.h"
#include <iostream>
#include "neuroML_IO/Cable.h"
#include <neuroML/NCell.h>
#include <neuroML/NBase.h>
#include "Cable.h"
#include "NeuromlReader.h"
using namespace std;
const double NeuromlReader::PI = 3.14159;
void NeuromlReader::readModel( string filename,Id location )
{
   static const Cinfo* compartmentCinfo = initCompartmentCinfo();
   static const Finfo* lengthFinfo = compartmentCinfo->findFinfo( "length" );
   static const Finfo* diameterFinfo = compartmentCinfo->findFinfo( "diameter" );
   static const Finfo* x0Finfo = compartmentCinfo->findFinfo( "x0" );
   static const Finfo* y0Finfo = compartmentCinfo->findFinfo( "y0" );
   static const Finfo* z0Finfo = compartmentCinfo->findFinfo( "z0" );
   static const Finfo* xFinfo = compartmentCinfo->findFinfo( "x" );
   static const Finfo* yFinfo = compartmentCinfo->findFinfo( "y" );
   static const Finfo* zFinfo = compartmentCinfo->findFinfo( "z" );
   static const Finfo* RaFinfo = compartmentCinfo->findFinfo( "Ra" );
   static const Finfo* initVmFinfo = compartmentCinfo->findFinfo( "initVm" );
   static const Finfo* CmFinfo = compartmentCinfo->findFinfo( "Cm" );
   static const Finfo* EmFinfo = compartmentCinfo->findFinfo( "Em" );
   static const Finfo* RmFinfo = compartmentCinfo->findFinfo( "Rm" );
   static const Cinfo* cableCinfo = initCableCinfo();
   NBase nb;   
   NCell* c;
   c = nb.readNeuroML (filename);
   c->register_namespaces();
   double x,y,z,x1,y1,z1,diam1,diam2,diameter,length,initVm,r,ca,Cm,Ra;
   initVm = c->getInit_memb_potential();
   ca = c->getSpec_capacitance();
   r = c->getSpec_axial_resistance();
   static const Finfo* nameFinfo = cableCinfo->findFinfo( "name" );
   static const Finfo* groupsFinfo = cableCinfo->findFinfo( "groups" );
   unsigned int num_cables = c->getNumCables();
   unsigned int num_segments = c->getNumSegments();
   Id loc("/cables");  
   if ( loc.good() ) 
      cerr << "Warning: cables already exists. You are trying to overwrite it" << endl;	
   Element* locE = Neutral::create( "Neutral", "cables", Id(), Id::scratchId() );
   loc = locE->id();
   map< string,vector<string> > groupcableMap; 
   map< string,Id > cabMap;	
   for( int cb = 1; cb <= num_cables; cb++ )
   {
	const NCable * cab;
	cab = c->getCable(cb);
	std::string cid = "", cname = "";
	cid = cab->getId();
	cname = cab->getName(); 
	cout << "id " << cid << " name " << cname << endl;
	cable_ = Neutral::create( "Cable",cid,loc,Id::scratchId() ); 
	cabMap[cid] = cable_->id();
	set< string >( cable_,nameFinfo,cid );
	std::vector< std::string > groups;
	groups = cab->getGroups();
	set< vector< string > >( cable_,groupsFinfo,groups );
	for ( int itr = 0; itr < groups.size(); itr++ )
	     groupcableMap[groups[itr]].push_back(cid);
   }
   map< string,vector< string > > cablesegMap;
   map< string,Id > segMap;	
   //vector<Id> segmentId;
  // map< Id,string > parentMap;
  // map< Id,string >::iterator iter;
   for( int i = 1; i<= num_segments; i ++ )
   {
		
	const Segment * seg ;
	seg = c->getSegment(i);
	std::string id = "", name = "", cable = "", parent = ""; 
	id = seg->getId();
	name = seg->getName();   
	cable = seg->getCable();
	parent = seg->getParent();
	cout << "id " << id << " name " << name << " cable " << cable << endl; 
	compt_ = Neutral::create( "Compartment",id,location,Id::scratchId() );
	segMap[id] = compt_->id(); 
	if (parent != "")
		Eref(compt_).add("raxial",segMap[parent](),"axial",ConnTainer::Default ); 
	Eref(cabMap[cable]()).add("compartment",Eref(compt_),"cable",ConnTainer::Default ); 
	//parentMap[compt_->id()] = parent;
	//Eref(reaction_).add( subFinfo->msg(),elmtMap_[sp],reacFinfo->msg(),ConnTainer::Default );
	//Eref S = elmtMap_.find(sp)->second;
	//Eref(enzyme_).add( "sub",S,"reac",ConnTainer::Default ); 
	cablesegMap[cable].push_back(id);
	x = seg->proximal.getX();
	y = seg->proximal.getY();
	z = seg->proximal.getZ();
	x1 = seg->distal.getX();
	y1 = seg->distal.getY();
	z1 = seg->distal.getZ();
	diam1 = seg->proximal.getDiameter();
	diam2 = seg->distal.getDiameter();
    	set< double >( compt_, x0Finfo, x );
	set< double >( compt_, y0Finfo, y );
 	set< double >( compt_, z0Finfo, z );
 	set< double >( compt_, xFinfo, x1  );
	set< double >( compt_, yFinfo, y1 );
	set< double >( compt_, zFinfo, z1 );
	diameter = (diam1 + diam2 )/2;
	length = sqrt(pow((x1-x),2) + pow((y1-y),2) + pow((z1-z),2));
	set< double >( compt_, diameterFinfo, diameter );
	set< double >( compt_, lengthFinfo, length );
        Cm = ca * length * PI * diameter;
        Ra = (r * length )/(PI * diameter * diameter/4);
        set< double >( compt_, CmFinfo, Cm );
        set< double >( compt_, RaFinfo, Ra ); 
        set< double >( compt_, initVmFinfo, initVm );
	set< double >( compt_, EmFinfo, initVm );
    }
    
    // xmlDocPtr xmlDoc = c.getDocument(filename);
  /* const Channel * chl ;
   chl = c.getChannelML();
   unsigned int numgates = c.getNumGates();
   for (int i=1;i<=numgates;i++)
   {

   	c.getGate(i);
   }*/
   //c.getSegment("1");
    static const Cinfo* channelCinfo = initHHChannelCinfo();
    static const Finfo* gbarFinfo = channelCinfo->findFinfo( "Gbar" );
    static const Finfo* ekFinfo = channelCinfo->findFinfo( "Ek" );
    static const Finfo* xpowerFinfo = channelCinfo->findFinfo( "Xpower" );
    static const Finfo* ypowerFinfo = channelCinfo->findFinfo( "Ypower" );
    static const Finfo* zpowerFinfo = channelCinfo->findFinfo( "Zpower" );
    unsigned int num_channels = c->getNumChannels();
    map< string,vector<string> >::iterator cmap_iter;
    map< string,vector<string> >::iterator smap_iter; 
    for( int ch = 1; ch <= num_channels; ch++ )
    {
    	Channel * chl ;
	chl = c->getChannel( ch );
	string name = "", group = "";
	double gmax,ek;
	Id loc( "/library" );
	group = chl->getGroup();
	cout<< " group: " << group ;
	vector< string > cId ;
	vector< string > sId;
	cmap_iter = groupcableMap.find( group );
	if ( cmap_iter != groupcableMap.end() )
	    cId = cmap_iter->second;
	else {
	   cout << "Error: CableId is empty" << endl;
	   return;
	}
	for( int i = 0; i < cId.size(); i++ )
	{
	   //cout << " cable id:  " << cId[i];
	    smap_iter = cablesegMap.find( cId[i] );
	    if ( smap_iter != cablesegMap.end() )
		sId = smap_iter->second;
	    else{
		cout << "Error:SegmentId is empty" << endl;
		return;
	    }
	    for( int j = 0; j < sId.size(); j++ )
		cout << " seg Id:  " << sId[j] << endl; 
         }
	 name = chl->getName();
 	 channel_ = Neutral::create( "HHChannel",name,loc,Id::scratchId() );
	 gmax = chl->getParameterValue();
	 set< double >( channel_, gbarFinfo, gmax );
	 ek = chl->getDefault_erev();
	 set< double >( channel_, ekFinfo, ek );
	 unsigned int  num = chl->getNumGates();
	 for( int i=1;i<=num;i++ )
	 {	
		Gate* gat ;		
		gat = chl->getGate(i);
		string gatename = gat->getGateName();
		double power = gat->getInstances();
		if ( i == 1 )
			set< double >( channel_, xpowerFinfo, power );
		else if ( i == 2 )
			set< double >( channel_, ypowerFinfo, power );
		else if ( i == 3 )
			set< double >( channel_, zpowerFinfo, power );
		string closeid = gat->getClosestateId();
		string openid = gat->getOpenstateId();
		string alname = gat->alpha.name;
		string from = gat->alpha.from;
		string to = gat->alpha.to;
		string expr_form = gat->alpha.expr_form;
		double rate = gat->alpha.rate;
		double scale = gat->alpha.scale;
		double midpoint = gat->alpha.midpoint;
		//cout << "gatename " << gatename << "  closeid " << closeid << " openid " << openid << " alname " << alname << " from " << from << " to " << to << 
		//" expr_form " << expr_form << " rate " << rate << " scale " << scale << " midpoint " << midpoint <<  endl;
	 }
	 string path = Eref(channel_).id().path();
	 string Apath = path + "/xGate/A";
	 string Bpath = path + "/yGate/B";
	 cout << "Apath " << Apath << endl;
         Id Aid(Apath);
	 if ( Aid.good() ) 
		cout << "A good";
	
	 Id Bid(Bpath);
	 if ( Bid.good() ) 
		cout << "B good";
	 bool passive = chl->getPassivecond();
	 cout << "name: "<< name << " passive " << passive<< endl;
	 if (passive){
	   double Rm = 1/gmax;
	   set< double >( compt_, RmFinfo, Rm );
	 }
     }
   
}
double NeuromlReader::solve (string expr_form,double r,double s,double m)
{
	double result ;
	if( expr_form == "exp_linear"
		result = (r*(v-m)/s)/(1-(1e-1*((v-m)/s))
	return result;
}



  

  

