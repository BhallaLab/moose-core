#include "moose.h"
#include "element/Neutral.h"
#include <iostream>
#include <set>
#include <libneuroml/NCable.h>
#include <libneuroml/NCell.h>
#include <libneuroml/NBase.h>
#include "Cable.h"
#include "NeuromlReader.h"
using namespace std;
const double NeuromlReader::PI = M_PI;
//void setupSegments();
//void setupCables();
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
   static const Cinfo* cableCinfo = initCableCinfo();
   NBase nb;   
   ncl_= nb.readNeuroML (filename);
   ncl_->register_namespaces();
   string lengthunits = ncl_->getLengthUnits();
   double x,y,z,x1,y1,z1,diam1,diam2,diameter,length,initVm,r,ca,Cm,Ra;
   initVm = ncl_->getInit_memb_potential();
   ca = ncl_->getSpec_capacitance();
   r = ncl_->getSpec_axial_resistance();
  // ca /= 100; //si unit
  // r *= 10; //si
   static const Finfo* nameFinfo = cableCinfo->findFinfo( "name" );
   static const Finfo* groupsFinfo = cableCinfo->findFinfo( "groups" );
   unsigned int num_cables = ncl_->getNumCables();
   unsigned int num_segments = ncl_->getNumSegments();
   Id loc("/cables");  
   if ( loc.good() ) 
      cerr << "Warning: cables already exists. You are trying to overwrite it" << endl;	
   Element* locE = Neutral::create( "Neutral", "cables", Id(), Id::scratchId() );
   loc = locE->id();
   map< string,vector<string> > groupcableMap; 
   map< string,Id > cabMap;	
   map< string,string >NMcabMap;
   for( unsigned int cb = 1; cb <= num_cables; cb++ )
   {
	const NCable * cab;
	cab =ncl_->getCable(cb);
	std::string cid = "", cname = "", name = "";
	cid = cab->getId();
	cname = cab->getName(); 
	name = cname + "_" + cid;
	NMcabMap[cid] = name; //Neuroml Moose Map
	//cout << "id " << cid << " cname " << cname << "name " << name << endl;
	cable_ = Neutral::create( "Cable",name,loc,Id::scratchId() ); 
	cabMap[name] = cable_->id();
	::set< string >( cable_,nameFinfo,name );
	std::vector< std::string > groups;
	groups = cab->getGroups();
	::set< vector< string > >( cable_,groupsFinfo,groups );
	for (unsigned int itr = 0; itr < groups.size(); itr++ )
	     groupcableMap[groups[itr]].push_back(name);
   }
   map< string,vector< string > > cablesegMap;
   for(unsigned int i = 1; i<= num_segments; i ++ )
   {
		
	const Segment * seg ;
	seg =ncl_->getSegment(i);
	std::string id = "", sname = "", cable = "", sparent = "", name = "", parent =""; 
	id = seg->getId();
	sname = seg->getName();   
	cable = seg->getCable();
	sparent = seg->getParent();
	if (sparent != "")
	    parent = NMsegMap_[sparent];
	name = sname + "_" + id ;
	NMsegMap_[id] = name; //Neuroml Moose Map
	string mooseCable = NMcabMap[cable];
	//cout << "id " << id << "name " << name << " parent " << parent << endl; 
	compt_ = Neutral::create( "Compartment",name,location,Id::scratchId() );
	segMap_[name] = compt_->id(); 
	if (parent != "")
		Eref(compt_).add("raxial",segMap_[parent](),"axial",ConnTainer::Default ); 
	Eref(cabMap[mooseCable]()).add("compartment",Eref(compt_),"cable",ConnTainer::Default ); 
	cablesegMap[mooseCable].push_back(name);
	bool proxm = seg->isSetProximal();
        /* if proximal is not given then proximal is the  distal of parent compt */
	if ( proxm ){
		x = seg->proximal.getX();
		y = seg->proximal.getY();
		z = seg->proximal.getZ();
		x *= 1e-6;
		y *= 1e-6;
		z *= 1e-6;
		
	}
	else {
		get< double >( segMap_[parent].eref(), "x",x );
		get< double >( segMap_[parent].eref(), "y",y );
		get< double >( segMap_[parent].eref(), "z",z );
	}
	x1 = seg->distal.getX();
	y1 = seg->distal.getY();
	z1 = seg->distal.getZ();
	x1 *= 1e-6;
	y1 *= 1e-6;
	z1 *= 1e-6;
	
	length = sqrt(pow((x1-x),2) + pow((y1-y),2) + pow((z1-z),2));
	if ( proxm )
		diam1 = seg->proximal.getDiameter();
	else 
		diam1 = seg->distal.getDiameter();
	diam2 = seg->distal.getDiameter();
	diameter = (diam1 + diam2 )/2;
	diameter *= 1e-6; //physiological unit
	//length *= 1e-6;//si
	//diameter *= 1e-6;//si
	::set< double >( compt_, x0Finfo, x );
	::set< double >( compt_, y0Finfo, y );
 	::set< double >( compt_, z0Finfo, z );
 	::set< double >( compt_, xFinfo, x1  );
	::set< double >( compt_, yFinfo, y1 );
	::set< double >( compt_, zFinfo, z1 );
	::set< double >( compt_, diameterFinfo, diameter );
	::set< double >( compt_, lengthFinfo, length );
	if ( length == 0 ){
		Cm = ca * PI * diameter * diameter;
		Ra = 13.50 * r /( diameter * PI );
	}
	else{
        	Cm = ca * length * PI * diameter;
        	Ra = (r * length )/(PI * diameter * diameter/4);
	}
	::set< double >( compt_, CmFinfo, Cm );
        ::set< double >( compt_, RaFinfo, Ra ); 
	::set< double >( compt_, initVmFinfo, initVm );
        //set< double >( compt_, initVmFinfo, initVm/1000);//si unit 
	
    }
   
   setupChannels(groupcableMap,cablesegMap);
   setupPools(groupcableMap,cablesegMap);
 // setupSynChannels(groupcableMap,cablesegMap);
  
}
double NeuromlReader::calcSurfaceArea(double length,double diameter)
{
	double sa ;
	if( length == 0 )
		sa = PI * diameter * diameter;
	else
		sa = PI * diameter * length;
	return sa;
}

double NeuromlReader::calcVolume(double length,double diameter)
{
	double v ;
	double r = diameter / 2 ;
	if( length == 0 )
		
		v = 4 / 3 * PI * r * r * r;
	else
		v = PI * r * r * length;
	return v;
}
void NeuromlReader::setupPools(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap)
{
	static const Cinfo* CaConcCinfo = initCaConcCinfo();
	static const Finfo* Ca_baseFinfo = CaConcCinfo->findFinfo( "Ca_base" );
	static const Finfo* tauFinfo = CaConcCinfo->findFinfo( "tau" );
	static const Finfo* BFinfo = CaConcCinfo->findFinfo( "B" );
	static const Finfo* thickFinfo = CaConcCinfo->findFinfo( "thick" );
	unsigned int num_pools = ncl_->getNumPools(); 
	if ( num_pools != 0 ) {  		
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	vector< string > pool_vec;
	pool_vec.clear();
	vector< string >::iterator it;
	for( unsigned int i = 0; i < num_pools; i++ )
    	{
    		IonPool* pool;
		pool = ncl_->getPool( i );
		Id loc( "/library" );
		string name = pool->getName();
		//cout << "Pool is : "<< name << endl;
		it = find(pool_vec.begin(), pool_vec.end(), name);
                if( it == pool_vec.end() ){
		    pool_vec.push_back(name);
		    ionPool_ = Neutral::create( "CaConc",name,loc,Id::scratchId() );
		}
	}

    	for( unsigned int i = 0; i < num_pools; i++ )
    	{
    		IonPool* pool;
		pool = ncl_->getPool( i );
		string name = "", group = "",scaling;
		double Ca_base, tau, B, thick;
		//Id loc( "/library" );
		name = pool->getName();
		cout << "Pool is : "<< name << endl;
		string path = "/library/"+name;
		Id poolId(path);
		
		
		//ionPool_ = Neutral::create( "CaConc",name,loc,Id::scratchId() );
		//path = Eref(ionPool_).id().path();
		Ca_base = pool->getResting_conc();
		tau = pool->getDecay_constant();
		thick = pool->getShell_thickness();
		scaling = pool->getScaling();	
		double B_ = pool->getB();
		::set< double >( Eref(poolId()), Ca_baseFinfo, Ca_base );
		::set< double >( Eref(poolId()), tauFinfo, tau );
		::set< double >( Eref(poolId()), thickFinfo, thick );
		std::vector< std::string > groups;
		groups = pool->getGroups();
		std::vector< std::string > segs;
		segs = pool->getSegGroups();
		vector< Eref > channels;
		for(unsigned int i = 0; i < segs.size(); i++ )
		{
			string segName = NMsegMap_[segs[i]];
			Id comptEl = segMap_[segName];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double V = calcVolume( len,dia );
			//cout << "volume : " << V << "  of  " << comptEl()->name()<< endl;
			if( scaling == "off" )
				B = B_;
			else if( scaling == "shell" ){
				if ( thick <= 0 )
				    B = B_ / V ; //use compt vol or use absolute value
				else B = B_ / thick;
			}
			else if( scaling == "volume" )
				B = B_ / V;
			::set< double >( Eref(poolId()), BFinfo, B );
		 	Element* copyEl = Eref(poolId())->copy(comptEl(),ionPool_->name());
			//cout << comptEl()->name() << " " << tau << " " << B << " " << Ca_base << endl;
			channels.clear();
			targets( comptEl(),"channel",channels );
			unsigned int numchls = channels.size();
			for ( unsigned int i=0; i<numchls; i++ ){
				string name = channels[ i ].name();
				if ( name == "CaConductance" )
				    channels[ i ].add("IkSrc",Eref(copyEl),"current",ConnTainer::Default ); 
				int use_conc;
				get< int >( channels[ i ], "useConcentration", use_conc );
				if( use_conc == 1 )
					Eref(copyEl).add("concSrc",channels[ i ],"concen",ConnTainer::Default ); 
			}
		  }
		vector< string > cId ;
		vector< string > sId;
		std::set< string > cableId;
		for(unsigned int gr = 0; gr < groups.size(); gr ++ )
		{
			//cout << "groups in vector : " << groups[gr] << endl;			
			cmap_iter = groupcableMap.find( groups[gr] );
			if ( cmap_iter != groupcableMap.end() ){
			   cId = cmap_iter->second;
			   cableId.insert( cId.begin(),cId.end() );
			}
		}
		std::set< string >::iterator itr;
		for( itr = cableId.begin(); itr != cableId.end(); itr++ )
		{
		    smap_iter = cablesegMap.find( (*itr) );
		    if ( smap_iter != cablesegMap.end() ){
			sId = smap_iter->second;
		    }
		    for(unsigned int j = 0; j < sId.size(); j++ ){
			//cout << " seg Id:  " << sId[j] << endl; 
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double V = calcVolume( len,dia );
			//cout << "volume : " << V << "  of  " << comptEl()->name()<< endl;
			if( scaling == "off" )
				B = B_;
			else if( scaling == "shell" ){
				if ( thick <= 0 )
				    B = B_ / V ; //use compt vol or use absolute value
				else B = B_ / thick;
			}
			else if( scaling == "volume" )
				B = B_ / V;
			::set< double >( Eref(poolId()), BFinfo, B );
		 	Element* copyEl = Eref(poolId())->copy(comptEl(),ionPool_->name());
			channels.clear();
			targets( comptEl(),"channel",channels );
			unsigned int numchls = channels.size();
			for ( unsigned int i=0; i<numchls; i++ ){
				string name = channels[ i ].name();
				if ( name == "CaConductance" )
				    channels[ i ].add("IkSrc",Eref(copyEl),"current",ConnTainer::Default ); 
				int use_conc;
				get< int >( channels[ i ], "useConcentration", use_conc );
				if( use_conc == 1 )
					Eref(copyEl).add("concSrc",channels[ i ],"concen",ConnTainer::Default ); 
			}
		    }
           	}
	 
     	    }
	}
}

void NeuromlReader::setupChannels(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap)
{
	//NCell* c;
	static const Cinfo* compartmentCinfo = initCompartmentCinfo();
	static const Finfo* EmFinfo = compartmentCinfo->findFinfo( "Em" );
        static const Finfo* RmFinfo = compartmentCinfo->findFinfo( "Rm" );	
	static const Cinfo* channelCinfo = initHHChannelCinfo();
	static const Finfo* gbarFinfo = channelCinfo->findFinfo( "Gbar" );
	static const Finfo* ekFinfo = channelCinfo->findFinfo( "Ek" );
	static const Finfo* xpowerFinfo = channelCinfo->findFinfo( "Xpower" );
	static const Finfo* ypowerFinfo = channelCinfo->findFinfo( "Ypower" );
	static const Finfo* zpowerFinfo = channelCinfo->findFinfo( "Zpower" );
	static const Finfo* useConcFinfo = channelCinfo->findFinfo( "useConcentration" );
	static const Cinfo* gateCinfo = initHHGateCinfo();
	static const Cinfo* leakCinfo = initLeakageCinfo();
	static const Finfo* leakekFinfo = leakCinfo->findFinfo( "Ek" );
	static const Finfo* setupAlphaFinfo = gateCinfo->findFinfo( "setupAlpha" );
	static const Cinfo* interpolCinfo = initInterpolCinfo();
	static const Finfo* tableVectorFinfo = interpolCinfo->findFinfo( "tableVector" );
	unsigned int num_channels = 0;
	num_channels = ncl_->getNumChannels();
	if ( num_channels != 0 ) {  		
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	vector< string > channel_vec;
	channel_vec.clear();
	vector< string >::iterator it;
	for(unsigned int ch = 0; ch < num_channels; ch++ )
	{
		Channel * chl ;	        
		chl = ncl_->getChannel( ch );
		double ek,gmax;
		Id loc( "/library" );
		string name = chl->getName();
		//cout << "channel is : "<< name << endl;
		// iterator to vector element:
  	        it = find(channel_vec.begin(), channel_vec.end(), name);
                if( it == channel_vec.end() ){
		    channel_vec.push_back(name);
		    //bool is2DChannel = chl->isSetConc_dependence();
		    bool is2DChannel  = false;
		    bool passive = chl->getPassivecond();
		    ek = chl->getDefault_erev();
		    gmax = chl->getDefault_gmax();
		    //string path;
		    //bool use_conc = false;
		    if ( passive ){
			leak_ = Neutral::create( "Leakage",name,loc,Id::scratchId() ); 
			::set< double >( leak_, leakekFinfo, ek );
			//path = Eref(leak_).id().path();
		    }
		    else if ( is2DChannel ){
			channel_ = Neutral::create( "HHChannel2D",name,loc,Id::scratchId() );	
			::set< double >( channel_, ekFinfo, ek );
			 ::set< double > ( channel_, gbarFinfo, gmax );
			//path = Eref(channel_).id().path();
		   }
		   else{	
 	 		channel_ = Neutral::create( "HHChannel",name,loc,Id::scratchId() );
			::set< double >( channel_, ekFinfo, ek );
			 ::set< double > ( channel_, gbarFinfo, gmax );
			//path = Eref(channel_).id().path();
		   }
	      }
        }
    	for(unsigned int ch = 0; ch < num_channels; ch++ )
    	{
    		Channel * chl ;
		chl =ncl_->getChannel( ch );
		string name = "", group = "";
		double gmax,ek;
		//Id loc( "/library" );
		name = chl->getName();
		cout << "channel is : "<< name << endl;
		string path = "/library/"+name;
		Id chlId(path);
		bool passive = chl->getPassivecond();
		ek = chl->getDefault_erev();
		bool use_conc = false;
		/*if ( passive ){
			leak_ = Neutral::create( "Leakage",name,loc,Id::scratchId() ); 
			::set< double >( leak_, leakekFinfo, ek );
			path = Eref(leak_).id().path();
		}
		else if ( is2DChannel ){
			channel_ = Neutral::create( "HHChannel2D",name,loc,Id::scratchId() );	
			::set< double >( channel_, ekFinfo, ek );
			path = Eref(channel_).id().path();
		}
		else{	
 	 		channel_ = Neutral::create( "HHChannel",name,loc,Id::scratchId() );
			::set< double >( channel_, ekFinfo, ek );
			path = Eref(channel_).id().path();
		}*/
		if ( passive )
			::set< double >( Eref(chlId()), leakekFinfo, ek );
		else ::set< double >( Eref(chlId()), ekFinfo, ek );
		if ( !passive ){
			use_conc = chl->isSetConc_dependence();
			if( use_conc )
				::set< int >( Eref(chlId()), useConcFinfo, 1 );	
			else 
				::set< int >( Eref(chlId()), useConcFinfo, 0 );		
		}
	 	gmax = chl->getGmax();
	 	//gmax *= 10;//si
	 	
	 	double xmin,xmax;
	 	int xdivs;
		double min,max;
	 	string biophysicsunits = ncl_->getBiophysicsUnits();
		//cout << " unit " << biophysicsunits << endl;
	 	if ( biophysicsunits == "Physiological Units" ){
	 		xmin = -100;
	 		xmax = 50;
		}
	 	else if ( biophysicsunits == "SI Units" ){
	 		xmin = -0.1;
	 		xmax = 0.05;
		}
		xdivs = 3000;
		
		//string path = Eref(channel_).id().path();
		vector< string > gatename;
		gatename.push_back("/xGate");
		gatename.push_back("/yGate");
		gatename.push_back("/zGate");
		string gatepath, tablepath;
		unsigned int  num = chl->getNumGates();
		if ( num > 3 ) 
		    	cout << "Error: More than 3 gates is not alllowed" << endl;
		int counter = 0;
	 	for( unsigned int i=1;i<=num;i++ )
	 	{	
			Gate* gat ;	
			bool flag = false;	
			gat = chl->getGate(i);
			string name = gat->getGateName();
			//cout << "gate name " << name << endl;
			double power = gat->getInstances();
			string x_variable = gat->getX_variable();
			//cout << "x_variable " << x_variable << endl;
			if( (x_variable == "concentration") || (use_conc == true && num == 1 ) ) {
			    if( flag == true )
				cout << "Error: already encountered Z gate" << endl;
			    else{
			        ::set< double >( Eref(chlId()), zpowerFinfo, power );
			        gatepath = path + gatename[2];
			        flag = true;
			    }
			}
			string y_variable = gat->getY_variable();
			if ( flag == false ){
			   counter++;
			   if ( counter == 1 )
			       ::set< double >( Eref(chlId()), xpowerFinfo, power );
			   else if ( counter == 2 )
				::set< double >( Eref(chlId()), ypowerFinfo, power );
			   else if ( counter == 3 )
				::set< double >( Eref(chlId()), zpowerFinfo, power );
			   gatepath = path + gatename[counter-1];
			}
			//cout << " gate path : " << gatepath << endl;
		 	Id gateid(gatepath);
			vector< double > result;
			vector< double > alphaTable;
			vector< double > betaTable;
			vector< double > tableEntry;
			result.clear();
			alphaTable.clear();
			betaTable.clear();
			tableEntry.clear();
			string Aexpr_form = gat->alpha.expr_form; 
			if( Aexpr_form == "tabulated" ){
				tablepath = gatepath + "/A";
				Id tableid(tablepath);
				min = gat->alpha.xmin;
				max = gat->alpha.xmax;
				alphaTable = gat->alpha.tableEntry;
				::set< double > ( Eref(tableid()),"xmin",min );
				::set< double > ( Eref(tableid()),"xmax",max );
				::set< std::vector< double > >( Eref(tableid()),tableVectorFinfo,alphaTable );
			}
			else{
				double Arate = gat->alpha.rate;
				//Arate *= 1000;
				double Ascale = gat->alpha.scale;
				//Ascale /= 1000;
				double Amidpoint = gat->alpha.midpoint;
				//Amidpoint /= 1000;
				pushtoVector(result,Aexpr_form,Arate,Ascale,Amidpoint);
				
			}
			string Bexpr_form = gat->beta.expr_form;
			if( Bexpr_form == "tabulated" ){
				tablepath = gatepath + "/B";
				Id tableid(tablepath);
				min = gat->beta.xmin;
				max = gat->beta.xmax;
				alphaTable = gat->alpha.tableEntry;
				tableEntry = gat->beta.tableEntry;
				int asize = alphaTable.size();
				int bsize = tableEntry.size();
				//cout << "asize : " << asize << " bsize:  " << bsize << endl; 
				if ( asize == bsize ){
				   for(int i = 0; i < asize; i++ )
					betaTable.push_back(alphaTable[i]+tableEntry[i]);
				   ::set< double > ( Eref(tableid()),"xmin",min );
				   ::set< double > ( Eref(tableid()),"xmax",max );
				   ::set< std::vector< double > >( Eref(tableid()),tableVectorFinfo,betaTable );						
				   //::set< std::vector< double > >( Eref(tableid()),"tableVector",betaTable );
				}
				else cout << "Error: two table values should be of same size " << endl; 
			}
			else{
				double Brate = gat->beta.rate;
				//Brate *= 1000;
				double Bscale = gat->beta.scale;
				//Bscale /= 1000;
				double Bmidpoint = gat->beta.midpoint;
				//Bmidpoint /= 1000;
				pushtoVector(result,Bexpr_form,Brate,Bscale,Bmidpoint);
				result.push_back(xdivs);
				result.push_back(xmin);
				result.push_back(xmax);
				::set< std::vector< double > >( Eref(gateid()),setupAlphaFinfo,result );
			}
			
	 	}
		
		//group = chl->getGroups();
		std::vector< std::string > groups;
		groups = chl->getGroups();
		std::vector< std::string > segs;
		segs = chl->getSegGroups();
		for(unsigned int i = 0; i < segs.size(); i++ )
		{
			string segName = NMsegMap_[segs[i]];
			Id comptEl = segMap_[segName];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			//cout << "surface area : " << sa << "  of  " << comptEl()->name()<< endl;
		 	double gbar = gmax * sa;
		 	
			if (passive){
				double Rm = 1/(gmax*sa);
		   		::set< double >( comptEl(), RmFinfo, Rm );
				::set< double >( comptEl(), EmFinfo, ek );
		 	}
			else{
				::set< double >( Eref(chlId()), gbarFinfo, gbar );  
				/*if (chlId()->name() == "K_CConductance"){
				   cout << comptEl()->name() << " " << gbar << " " << ek << " " << endl;
				   showmsg chlId()->path() ;
				}*/
				Element* copyEl = Eref(chlId())->copy(comptEl(),Eref(chlId())->name());
				Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
			}
		  }
		  vector< string > cId ;
		  vector< string > sId;
		  std::set< string > cableId;
		  for(unsigned int gr = 0; gr < groups.size(); gr ++ )
		  {
			//cout << "groups in vector : " << groups[gr] << endl;			
			cmap_iter = groupcableMap.find( groups[gr] );
			if ( cmap_iter != groupcableMap.end() ){
			   cId = cmap_iter->second;
			   cableId.insert( cId.begin(),cId.end() );
			}
		  }
		  std::set< string >::iterator itr;
		  for( itr = cableId.begin(); itr != cableId.end(); itr++ )
		  {
		    smap_iter = cablesegMap.find( (*itr) );
		    if ( smap_iter != cablesegMap.end() ){
			sId = smap_iter->second;
		    }
		    for( unsigned int j = 0; j < sId.size(); j++ ){
			//cout << " seg Id:  " << sId[j] << endl; 
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			//cout << "surface area : " << sa << "  of  " << comptEl()->name()<< endl;
		 	double gbar = gmax * sa;
		 	
			if (passive){
				double Rm = 1/(gmax*sa);
		   		::set< double >( comptEl(), RmFinfo, Rm );
				::set< double >( comptEl(), EmFinfo, ek );
		 	}
			else{
				::set< double >( Eref(chlId()), gbarFinfo, gbar );  
				
				Element* copyEl = Eref(chlId())->copy(comptEl(),Eref(chlId())->name());
				Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
			}
           	    }
	          }
	  }  
	}
}

void NeuromlReader::setupSynChannels(map< string,vector<string> > &groupcableMap,map< string,vector< string > > &cablesegMap)
{
	//NCell* c;	
	unsigned int numsynchans =ncl_->getNumSynChannels(); 
	static const Cinfo* synchanCinfo = initSynChanCinfo();
    	static const Finfo* synGbarFinfo = synchanCinfo->findFinfo( "Gbar" );
	static const Finfo* synEkFinfo = synchanCinfo->findFinfo( "Ek" );
	static const Finfo* synTau1Finfo = synchanCinfo->findFinfo( "tau1" );
	static const Finfo* synTau2Finfo = synchanCinfo->findFinfo( "tau2" );
	map< string,vector<string> >::iterator cmap_iter;
    	map< string,vector<string> >::iterator smap_iter; 
	if ( numsynchans != 0 ){
	for( unsigned int i = 1; i <= numsynchans; i ++ )
	{
		SynChannel* synchl;	
		Id loc( "/library" );
		synchl = ncl_->getSynChannel(i);
		string name = synchl->getSynType();
 	 	synchannel_ = Neutral::create( "SynChan",name,loc,Id::scratchId() );
		double gmax = synchl->getMax_Conductance();
		double tau1 = synchl->getRise_Time();
		::set< double >( synchannel_, synTau1Finfo, tau1 );
		double tau2 = synchl->getDecay_Time();
		::set< double >( synchannel_, synTau2Finfo, tau2 );
		double ek = synchl->getReversal_Potential();
		::set< double >( synchannel_, synEkFinfo, ek );
		std::vector< std::string > groups;
		groups = synchl->getGroups();
		vector< string > cId ;
	 	vector< string > sId;
		std::set< string > cableId;
		for( unsigned int gr = 0; gr < groups.size(); gr ++ )
		{
			//cout << "groups in vector : " << groups[gr] << endl;			
			cmap_iter = groupcableMap.find( groups[gr] );
			if ( cmap_iter != groupcableMap.end() ){
			   cId = cmap_iter->second;
			   cableId.insert( cId.begin(),cId.end() );
			}
		}
		std::set< string >::iterator itr;
		for( itr = cableId.begin(); itr != cableId.end(); itr++ )
		{
		    smap_iter = cablesegMap.find( (*itr) );
		    if ( smap_iter != cablesegMap.end() ){
			sId = smap_iter->second;
		    }
	    	    for( unsigned int j = 0; j < sId.size(); j++ ){
			Id comptEl = segMap_[sId[j]];
			double len,dia;
			get< double >( comptEl.eref(), "length",len );
			get< double >( comptEl.eref(), "diameter",dia );
			double sa = calcSurfaceArea( len,dia );
			//cout << " len "<< len << " dia " << dia << " sa " << sa << endl; 
			double gbar = gmax * sa;
			::set< double >( synchannel_,synGbarFinfo,gbar );
			Element* copyEl = synchannel_->copy(comptEl(),synchannel_->name());
			Eref(comptEl()).add("channel",copyEl,"channel",ConnTainer::Default ); 
		    }
		}
	}
	}
}

void NeuromlReader::pushtoVector(vector< double > &result,string expr_form,double r,double s,double m)
{
	double A,B,C,D,F;
	if( expr_form == "exp_linear" ){
		A = r*m/s;
		B = -r/s;
		C = -1;
		D = -m;
		F = -s;	
	}
	if( expr_form == "sigmoid" ){
		A = r;
		B = 0;
		C = 1;
		D = -m;
		F = s;	
	}
	if( expr_form == "exponential" ){
		A = r;
		B = 0;
		C = 0;
		D = -m;
		F = -s;
	}
	result.push_back(A);
	result.push_back(B);
	result.push_back(C);
	result.push_back(D);
	result.push_back(F);	
}
int NeuromlReader::targets( Eref object,const string& msg,vector< Eref >& target,const string& type )
{
	unsigned int oldSize = target.size();
	Eref found;
	Conn* i = object->targets( msg, 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target();
		if ( type != "" && !isType( found, type ) )	// speed this up
			continue;
		
		target.push_back( found );		
	}
	delete i;
	
	return target.size() - oldSize;
}

bool NeuromlReader::isType( Eref object, const string& type )
{
	return object->cinfo()->isA( Cinfo::find( type ) );
}


  

  

