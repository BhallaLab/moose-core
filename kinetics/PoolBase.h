/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _POOL_BASE_H
#define _POOL_BASE_H

typedef unsigned int SpeciesId;
extern const SpeciesId DefaultSpeciesId;


/**
 * The PoolBase class is the base class for molecular pools.
 * A pool is a set of molecules of a
 * given species, in a uniform chemical context. Note that the same
 * species might be present in other compartments, or be handled by
 * other solvers.
 * PoolBase is the base class for mass-action, single particle
 * and other numerical variants of pools.
 * Note that in this version it only acts as an interface for the solvers.
 */
class PoolBase
{
    friend void testSyncArray( unsigned int size, unsigned int numThreads,
                               unsigned int method );
    friend void checkVal( double time, const PoolBase* m, unsigned int size );
    friend void forceCheckVal( double time, Element* e, unsigned int size );

public:
    PoolBase();
    ~PoolBase();

    //////////////////////////////////////////////////////////////////
    // Field assignment stuff: Interface for the Cinfo, hence regular
    // funcs. These internally call the virtual funcs that do the real
    // work.
    //////////////////////////////////////////////////////////////////
    void setN( const Eref& e, double v );
    double getN( const Eref& e ) const;
    void setNinit( const Eref& e, double v );
    double getNinit( const Eref& e ) const;
    void setDiffConst( const Eref& e, double v );
    double getDiffConst( const Eref& e ) const;
    void setMotorConst( const Eref& e, double v );
    double getMotorConst( const Eref& e ) const;

    void setConc( const Eref& e, double v );
    double getConc( const Eref& e ) const;
    void setConcInit( const Eref& e, double v );
    double getConcInit( const Eref& e ) const;

    /**
     * Volume is usually volume, but we also permit areal density
     * This is obtained by looking up the corresponding spatial mesh
     * entry in the parent compartment. If the message isn't set then
     * it defaults to 1.0.
     */
    void setVolume( const Eref& e, double v );
    double getVolume( const Eref& e ) const;

    void setSpecies( const Eref& e, SpeciesId v );
    SpeciesId getSpecies( const Eref& e ) const;
    /**
     * Functions to examine and change class between Pool and BufPool.
     */
    void setIsBuffered( const Eref& e, bool v );
    bool getIsBuffered( const Eref& e ) const;
    //////////////////////////////////////////////////////////////////
    // Dest funcs
    //////////////////////////////////////////////////////////////////
    void process( const Eref& e, ProcPtr p );
    void reinit( const Eref& e, ProcPtr p );
    void reac( double A, double B );
    void handleMolWt( const Eref& e, double v );
    void increment( double val );
    void decrement( double val );
    void nIn( double val );

	void notifyDestroy( const Eref& e );

	void notifyCreate( const Eref& e, ObjId parent );
	void notifyMove( const Eref& e, ObjId newParent );
	void notifyAddMsgSrc( const Eref& e, ObjId msgId );
	void notifyAddMsgDest( const Eref& e, ObjId msgId );


    static const Cinfo* initPoolBaseCinfo();
    static const Cinfo* initPoolCinfo();
    static const Cinfo* initBufPoolCinfo();
protected:
    /**
     * The KsolveBase pointers hold the solvers for the
     * PoolBase. At least one must be assigned. Field assignments
     * propagate from the pool to whichever is assigned. Field
     * lookups first check the dsolve, then the ksolve.
     * The PoolBase may be managed by the diffusion solver without
     * the involvement of the Stoich class at all. So instead of
     * routing the zombie operations through the Stoich, we have
     * pointers directly into the Dsolve and Ksolve.
     */
    KsolveBase* dsolve_;
    KsolveBase* ksolve_;
};

#endif	// _POOL_BASE_H
