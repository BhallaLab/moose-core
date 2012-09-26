/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <time.h>
#include <iostream>
#include <fstream>
#include "header.h"
#include "PoolBase.h"
#include "../shell/Wildcard.h"
#include "EnzBase.h"
#include "CplxEnzBase.h"
#include "ReacBase.h"
#include "../builtins/TableBase.h"
#include "../builtins/Table.h"
#include "../builtins/StimulusTable.h"
#include "Pool.h"
#include "FuncPool.h"
#include "SumFunc.h"

void writeHeader( ofstream& fout, 
		double simdt, double plotdt, double maxtime, double defaultVol)
{
	time_t rawtime;
	time( &rawtime );

	fout << 
	"//genesis\n"
	"// kkit Version 11 flat dumpfile\n\n";
	fout << "// Saved on " << ctime( &rawtime ) << endl;
	fout << "include kkit {argv 1}\n";
	fout << "FASTDT = " << simdt << endl;
	fout << "SIMDT = " << simdt << endl;
	fout << "CONTROLDT = " << plotdt << endl;
	fout << "PLOTDT = " << plotdt << endl;
	fout << "MAXTIME = " << maxtime << endl;
	fout << "TRANSIENT_TIME = 2\n"
	"VARIABLE_DT_FLAG = 0\n";
	fout << "DEFAULT_VOL = " << defaultVol << endl;
	fout << "VERSION = 11.0\n"
	"setfield /file/modpath value ~/scripts/modules\n"
	"kparms\n\n";

	fout << 
	"initdump -version 3 -ignoreorphans 1\n"
	"simobjdump table input output alloced step_mode stepsize x y z\n"
	"simobjdump xtree path script namemode sizescale\n"
	"simobjdump xcoredraw xmin xmax ymin ymax\n"
	"simobjdump xtext editable\n"
	"simobjdump xgraph xmin xmax ymin ymax overlay\n"
	"simobjdump xplot pixflags script fg ysquish do_slope wy\n"
	"simobjdump group xtree_fg_req xtree_textfg_req plotfield expanded movealone \\\n"
  	"  link savename file version md5sum mod_save_flag x y z\n"
	"simobjdump geometry size dim shape outside xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump kpool DiffConst CoInit Co n nInit mwt nMin vol slave_enable \\\n"
  	"  geomname xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump kreac kf kb notes xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol k1 k2 k3 \\\n"
  	"  keepconc usecomplex notes xtree_fg_req xtree_textfg_req link x y z\n"
	"simobjdump stim level1 width1 delay1 level2 width2 delay2 baselevel trig_time \\\n"
  	"  trig_mode notes xtree_fg_req xtree_textfg_req is_running x y z\n"
	"simobjdump xtab input output alloced step_mode stepsize notes editfunc \\\n"
  	"  xtree_fg_req xtree_textfg_req baselevel last_x last_y is_running x y z\n"
	"simobjdump kchan perm gmax Vm is_active use_nernst notes xtree_fg_req \\\n"
  	"  xtree_textfg_req x y z\n"
	"simobjdump transport input output alloced step_mode stepsize dt delay clock \\\n"
  	"  kf xtree_fg_req xtree_textfg_req x y z\n"
	"simobjdump proto x y z\n"
	"simundump geometry /kinetics/geometry 0 1.6667e-19 3 sphere \"\" white black 0 0 0\n\n";
}


void writeReac( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	double kf = Field< double >::get( id, "kf" );
	double kb = Field< double >::get( id, "kb" );

	fout << "simundump kreac " << path << " 0 " << 
			kf << " " << kb << " \"\" " << 
			colour << " " << textcolour << " " << x << " " << y << " 0\n";
}

unsigned int getSlaveEnable( Id id )
{
	static const Finfo* setNinitFinfo = 
			PoolBase::initCinfo()->findFinfo( "set_nInit" );
	static const Finfo* setConcInitFinfo = 
			PoolBase::initCinfo()->findFinfo( "set_concInit" );
	unsigned int ret = 0;
	vector< Id > src;
	if ( id.element()->cinfo()->isA( "BufPool" ) ) {
		if ( id.element()->getNeighbours( src, setConcInitFinfo ) > 0 ) {
				ret = 2;
		} else if ( id.element()->getNeighbours( src, setNinitFinfo ) > 0 ){
				ret = 4;
		}
	} else {
		return 0;
	}
	if ( ret == 0 )
			return 4; // Just simple buffered molecule
	if ( src[0].element()->cinfo()->isA( "StimulusTable" ) )
			return ret; // Following a table, this is fine.
	
	// Fallback: I have no idea what sent it the input, assume it is legit.
	return ret;
}

void writePool( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	double diffConst = Field< double >::get( id, "diffConst" );
	double concInit = Field< double >::get( id, "concInit" );
	double conc = Field< double >::get( id, "conc" );
	double nInit = Field< double >::get( id, "nInit" );
	double n = Field< double >::get( id, "n" );
	double size = Field< double >::get( id, "size" );
	unsigned int slave_enable = getSlaveEnable( id );

	fout << "simundump kpool " << path << " 0 " << 
			diffConst << " " <<
			concInit << " " << 
			conc << " " <<
			n << " " <<
			nInit << " " <<
			0 << " " << 0 << " " << // mwt, nMin
			size * NA * 1e-3  << " " << // volscale
			slave_enable << // GENEISIS field here.
			" /kinetics/geometry " << 
			colour << " " << textcolour << " " << x << " " << y << " 0\n";
}

Id getEnzMol( Id id )
{
	static const Finfo* enzFinfo = 
			EnzBase::initCinfo()->findFinfo( "enzDest" );
	vector< Id > ret;
	if ( id.element()->getNeighbours( ret, enzFinfo ) > 0 ) 
		return ret[0];
	return Id();
}

Id getEnzCplx( Id id )
{
	static const Finfo* cplxFinfo = 
			CplxEnzBase::initCinfo()->findFinfo( "cplxDest" );
	vector< Id > ret;
	if ( id.element()->getNeighbours( ret, cplxFinfo ) > 0 )
		return ret[0];
	return Id();
}

void writeEnz( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	double k1 = 0;
	double k2 = 0;
	double k3 = 0;
	double nInit = 0;
	double concInit = 0;
	double n = 0;
	double conc = 0;
	Id enzMol = getEnzMol( id );
	assert( enzMol != Id() );
	double vol = Field< double >::get( enzMol, "size" ) * NA * 1e-3; 
	unsigned int isMichaelisMenten = 0;
	if ( id.element()->cinfo()->isA( "CplxEnzBase" ) ) {
		k1 = Field< double >::get( id, "k1" );
		k2 = Field< double >::get( id, "k2" );
		k3 = Field< double >::get( id, "k3" );
		Id cplx = getEnzCplx( id );
		assert( cplx != Id() );
		nInit = Field< double >::get( cplx, "nInit" );
		n = Field< double >::get( cplx, "n" );
		concInit = Field< double >::get( cplx, "concInit" );
		conc = Field< double >::get( cplx, "conc" );
	} else {
		k1 = Field< double >::get( id, "numKm" );
		k3 = Field< double >::get( id, "kcat" );
		k2 = 4.0 * k3;
		k1 = (k2 + k3) / k1;
		isMichaelisMenten = 1;
	}

	fout << "simundump kenz " << path << " 0 " << 
			concInit << " " <<
			conc << " " << 
			nInit << " " <<
			n << " " <<
			vol << " " <<
			k1 << " " <<
			k2 << " " <<
			k3 << " " <<
			0 << " " <<
			isMichaelisMenten << " " <<
			"\"\"" << " " << 
			colour << " " << textcolour << " \"\"" << 
			" " << x << " " << y << " 0\n";
}

void writeGroup( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	if ( pos == string::npos ) // Might be finding unrelated neutrals
			return;
	path = path.substr( pos );
	fout << "simundump group " << path << " 0 " << 
			colour << " " << textcolour << " x 0 0 \"\" defaultfile \\\n";
	fout << "  defaultfile.g 0 0 0 " << x << " " << y << " 0\n";
}

void writeStimulusTable( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/kinetics" );
	path = path.substr( pos );
	unsigned int stepMode = Field< bool >::get( id, "doLoop" );
	if ( stepMode == 0 )
			stepMode = 2; // TAB_ONCE in GENESIS/kkit terms.
	double stepSize = Field< double >::get( id, "stepSize" );

	fout << "simundump xtab " << path << " 0 0 0 1 " << stepMode <<
			" " << stepSize << " \"\" edit_xtab \"\" " << 
			colour << " " << textcolour << " " <<
			"0 0 0 1 " << x << " " << y << " 0\n";

	vector< double > vec = Field < vector< double > >::get( id, "vec" );
	double startTime = Field< double >::get( id, "startTime" );
	double stopTime = Field< double >::get( id, "stopTime" );

	fout << "loadtab " << path << " table 1 " << vec.size() - 1 << " " <<
			startTime << " " << stopTime << "\\\n";
	for ( unsigned int i = 0; i < vec.size(); ++i ) {
			fout << " " << vec[i] * 1000;
			if ( i % 10 == 9 )
					fout << "\\\n";
	}
	fout << "\n";

	double dx = ( stopTime - startTime ) / ( vec.size() - 1 );
	fout << "setfield " << path << " table->dx " << dx << endl;
	fout << "setfield " << path << " table->invdx " << 1.0/dx << endl;
}

void writePlot( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
	string path = id.path();
	size_t pos = path.find( "/graphs" );
	if ( pos == string::npos ) 
		pos = path.find( "/moregraphs" );
		if ( pos == string::npos ) 
			return;
	path = path.substr( pos );
	fout << "simundump xplot " << path << " 3 524288 \\\n" << 
	"\"delete_plot.w <s> <d>; edit_plot.D <w>\" " << textcolour << " 0 0 1\n";
}

void writeLookupTable( ofstream& fout, Id id,
				string colour, string textcolour,
			 	double x, double y )
{
}

void writeGui( ofstream& fout )
{
	fout << "simundump xgraph /graphs/conc1 0 0 99 0.001 0.999 0\n"
	"simundump xgraph /graphs/conc2 0 0 100 0 1 0\n"
	"simundump xgraph /moregraphs/conc3 0 0 100 0 1 0\n"
	"simundump xgraph /moregraphs/conc4 0 0 100 0 1 0\n"
	"simundump xcoredraw /edit/draw 0 -6 4 -2 6\n"
	"simundump xtree /edit/draw/tree 0 \\\n"
	"  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] \"edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>\" auto 0.6\n"
	"simundump xtext /file/notes 0 1\n";
}

void writeFooter( ofstream& fout )
{
	fout << "\nenddump\n";
	fout << "complete_loading\n";
}

Id findInfo( Id id )
{
	vector< Id > kids;
	Neutral::children( id.eref(), kids );

	for ( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i ) {
		Element* e = i->element();
		if ( e->getName() == "info" && e->cinfo()->isA( "Annotator" ) )
			return *i;
	}
	return Id();
}

void getInfoFields( Id id, string& bg, string& fg, 
				double& x, double& y, double side, double dx )
{
	Id info = findInfo( id );
	if ( info != Id() ) {
		bg = Field< string >::get( info, "color" );
		fg = Field< string >::get( info, "textColor" );
		x = Field< double >::get( info, "x" );
		y = Field< double >::get( info, "y" );
	} else {
		bg = "cyan";
		fg = "black";
		x += dx;
		if ( x > side ) {
				x = 0;
				y += dx;
		}
	}
}


string trimPath( const string& path )
{
	size_t pos = path.find( "/kinetics" );
	if ( pos == string::npos ) // Might be finding unrelated neutrals
			return "";
	return path.substr( pos );
}

void storeReacMsgs( Id reac, vector< string >& msgs )
{
	// const Finfo* reacFinfo = PoolBase::initCinfo()->findFinfo( "reacDest" );
	// const Finfo* noutFinfo = PoolBase::initCinfo()->findFinfo( "nOut" );
	static const Finfo* subFinfo = 
			ReacBase::initCinfo()->findFinfo( "toSub" );
	static const Finfo* prdFinfo = 
			ReacBase::initCinfo()->findFinfo( "toPrd" );
	vector< Id > targets;
	
	reac.element()->getNeighbours( targets, subFinfo );
	string reacPath = trimPath( reac.path() );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string s = "addmsg " + trimPath( i->path() ) + " " + reacPath +
			   	" SUBSTRATE n";
		msgs.push_back( s );
		s = "addmsg " + reacPath + " " + trimPath( i->path() ) + 
				" REAC A B";
		msgs.push_back( s );
	}

	targets.resize( 0 );
	reac.element()->getNeighbours( targets, prdFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string s = "addmsg " + trimPath( i->path() ) + " " + reacPath +
			   	" PRODUCT n";
		msgs.push_back( s );
		s = "addmsg " + reacPath + " " + trimPath( i->path() ) + 
				" REAC B A";
		msgs.push_back( s );
	}
}

void storeMMenzMsgs( Id enz, vector< string >& msgs )
{
	static const Finfo* subFinfo = 
			EnzBase::initCinfo()->findFinfo( "toSub" );
	static const Finfo* prdFinfo = 
			EnzBase::initCinfo()->findFinfo( "toPrd" );
	static const Finfo* enzFinfo = 
			EnzBase::initCinfo()->findFinfo( "enzDest" );
	vector< Id > targets;
	
	string enzPath = trimPath( enz.path() );
	enz.element()->getNeighbours( targets, subFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + tgtPath + " " + enzPath + " SUBSTRATE n";
		msgs.push_back( s );
		s = "addmsg " + enzPath + " " + tgtPath + " REAC sA B";
		msgs.push_back( s );
	}

	targets.resize( 0 );
	enz.element()->getNeighbours( targets, prdFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + enzPath + " " + tgtPath + " MM_PRD pA";
		msgs.push_back( s );
	}

	targets.resize( 0 );
	enz.element()->getNeighbours( targets, enzFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + tgtPath + " " + enzPath + " ENZYME n";
		msgs.push_back( s );
	}
}

void storeCplxEnzMsgs( Id enz, vector< string >& msgs )
{
	static const Finfo* subFinfo = 
			EnzBase::initCinfo()->findFinfo( "toSub" );
	static const Finfo* prdFinfo = 
			EnzBase::initCinfo()->findFinfo( "toPrd" );
	static const Finfo* enzFinfo = 
			CplxEnzBase::initCinfo()->findFinfo( "toEnz" );
	// In GENESIS we don't need to explicitly connect up the enz cplx, so
	// no need to deal with the toCplx msg.
	vector< Id > targets;
	
	string enzPath = trimPath( enz.path() );
	enz.element()->getNeighbours( targets, subFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + tgtPath + " " + enzPath + " SUBSTRATE n";
		msgs.push_back( s );
		s = "addmsg " + enzPath + " " + tgtPath + " REAC sA B";
		msgs.push_back( s );
	}

	targets.resize( 0 );
	enz.element()->getNeighbours( targets, prdFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + enzPath + " " + tgtPath + " MM_PRD pA";
		msgs.push_back( s );
	}

	targets.resize( 0 );
	enz.element()->getNeighbours( targets, enzFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + tgtPath + " " + enzPath + " ENZYME n";
		msgs.push_back( s );
		s = "addmsg " + enzPath + " " + tgtPath + " REAC eA B";
		msgs.push_back( s );
	}
}

void storeEnzMsgs( Id enz, vector< string >& msgs )
{
	if ( enz.element()->cinfo()->isA( "CplxEnzBase" ) ) 
		storeCplxEnzMsgs( enz, msgs );
	else
		storeMMenzMsgs( enz, msgs );
}

void writeMsgs( ofstream& fout, const vector< string >& msgs )
{
	for ( vector< string >::const_iterator i = msgs.begin();
					i != msgs.end(); ++i )
			fout << *i << endl;
}

void storeFuncPoolMsgs( Id pool, vector< string >& msgs )
{
	// Find the child SumFunc by following the input msg.
	static const Finfo* poolInputFinfo = 
			FuncPool::initCinfo()->findFinfo( "input" );

	static const Finfo* funcInputFinfo = 
			SumFunc::initCinfo()->findFinfo( "input" );

	assert( poolInputFinfo );
	assert( funcInputFinfo );
	vector< Id > funcs;
	pool.element()->getNeighbours( funcs, poolInputFinfo );
	assert( funcs.size() == 1 );
	
	// Get the msg sources into this SumFunc.
	vector< Id > src;
	funcs[0].element()->getNeighbours( src, funcInputFinfo );
	assert( src.size() > 0 );

	string poolPath = trimPath( pool.path() );

	// Write them out as msgs.
	for ( vector< Id >::iterator i = src.begin(); i != src.end(); ++i ) {
		string srcPath = trimPath( i->path() );
		string s = "addmsg " + srcPath + " " + poolPath + 
			" SUMTOTAL n nInit";
		msgs.push_back( s );
	}
}

void storeStimulusTableMsgs( Id tab, vector< string >& msgs )
{
	static const Finfo* outputFinfo = 
			StimulusTable::initCinfo()->findFinfo( "output" );
	// In GENESIS we don't need to explicitly connect up the enz cplx, so
	// no need to deal with the toCplx msg.
	vector< Id > targets;
	
	string tabPath = trimPath( tab.path() );
	tab.element()->getNeighbours( targets, outputFinfo );
	for ( vector< Id >::iterator i = targets.begin(); i != targets.end(); ++i ) {
		string tgtPath = trimPath( i->path() );
		string s = "addmsg " + tabPath + " " + tgtPath + " SLAVE output";
		msgs.push_back( s );
	}
}

void storePlotMsgs( Id tab, vector< string >& msgs )
{
	static const Finfo* plotFinfo = 
			Table::initCinfo()->findFinfo( "requestData" );
	vector< Id > pools;
	
	tab.element()->getNeighbours( pools, plotFinfo );
	assert( pools.size() == 1 );
	string bg;
	string fg;
	double x;
	double y;
	getInfoFields( pools[0], bg, fg, x, y, 1, 1 );
	string tabPath = tab.path(); 

	size_t pos = tabPath.find( "/graphs" );
	if ( pos == string::npos ) 
		pos = tabPath.find( "/moregraphs" );
		assert( pos != string::npos );
	tabPath = tabPath.substr( pos );
	string s = "addmsg " + trimPath( pools[0].path() ) + " " + tabPath + 
			" PLOT Co *" + pools[0].element()->getName() + " *" + bg;
	msgs.push_back( s );
}

/**
 * A bunch of heuristics to find good SimTimes to use for kkit. 
 * Returns runTime.
 */
double estimateSimTimes( double& simDt, double& plotDt )
{
		double runTime = Field< double >::get( Id( 1 ), "runTime" );
		if ( runTime <= 0 )
				runTime = 100.0;
		vector< double > dts = 
				Field< vector< double> >::get( Id( 1 ), "dts" );
		simDt = dts[6];
		plotDt = dts[8];
		if ( plotDt <= 0 )
				plotDt = runTime / 200.0;
		if ( simDt == 0 )
				simDt = 0.01;
		if ( simDt > plotDt )
				simDt = plotDt / 100;

		return runTime;
}

/// Returns an estimate of the default volume used in the model.
double estimateDefaultVol( Id model )
{
		vector< Id > children = 
				Field< vector< Id > >::get( model, "children" );
		vector< double > vols;
		double maxVol = 0;
		for ( vector< Id >::iterator i = children.begin(); 
						i != children.end(); ++i ) {
				if ( i->element()->cinfo()->isA( "ChemMesh" ) ) {
						double v = Field< double >::get( *i, "size" );
						if ( i->element()->getName() == "kinetics" )
								return v;
						vols.push_back( v );
						if ( maxVol < v ) 
								maxVol = v;
				}
		}
		if ( maxVol > 0 )
				return maxVol;
		return 1.0e-15;
}

void writeKkit( Id model, const string& fname )
{
		vector< Id > ids;
		vector< string > msgs;
		unsigned int num = simpleWildcardFind( model.path() + "/##", ids );
		if ( num == 0 ) {
			cout << "Warning: writeKkit:: No model found on " << model << 
					endl;
			return;
		}
		ofstream fout( fname.c_str(), ios::out );
		
		double simDt;
		double plotDt;
		double runTime = estimateSimTimes( simDt, plotDt );
		double defaultVol = estimateDefaultVol( model );
		writeHeader( fout, simDt, plotDt, runTime, defaultVol );
		writeGui( fout );

		string bg = "cyan";
		string fg = "black";
		double x = 0;
		double y = 0;
		double side = floor( 1.0 + sqrt( static_cast< double >( num ) ) );
		double dx = side / num;
		for( vector< Id >::iterator i = ids.begin(); i != ids.end(); ++i ) {
			getInfoFields( *i, bg, fg, x, y , side, dx );
			if ( i->element()->cinfo()->isA( "PoolBase" ) ) {
				ObjId pa = Neutral::parent( i->eref() );
				// Check that it isn't an enz cplx.
				if ( !pa.element()->cinfo()->isA( "CplxEnzBase" ) ) {
					writePool( fout, *i, bg, fg, x, y );
				}
				if ( i->element()->cinfo()->isA( "FuncPool" ) ) {
					storeFuncPoolMsgs( *i, msgs );
				}
			} else if ( i->element()->cinfo()->isA( "ReacBase" ) ) {
				writeReac( fout, *i, bg, fg, x, y );
				storeReacMsgs( *i, msgs );
			} else if ( i->element()->cinfo()->isA( "EnzBase" ) ) {
				writeEnz( fout, *i, bg, fg, x, y );
				storeEnzMsgs( *i, msgs );
			} else if ( i->element()->cinfo()->name() == "Neutral" ) {
				writeGroup( fout, *i, bg, fg, x, y );
			} else if ( i->element()->cinfo()->isA( "StimulusTable" ) ) {
				writeStimulusTable( fout, *i, bg, fg, x, y );
				storeStimulusTableMsgs( *i, msgs );
			} else if ( i->element()->cinfo()->isA( "Table" ) ) {
				writePlot( fout, *i, bg, fg, x, y );
				storePlotMsgs( *i, msgs );
			}
		}
		writeMsgs( fout, msgs );
		writeFooter( fout );
}
