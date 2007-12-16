/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _InterHubFlux_h
#define _InterHubFlux_h

/**
 * FluxTerm manages an individual molecule's flux between two hubs
 */
class FluxTerm
{
	public:
		double* map_; /// Points to entries in S_.
		double effluxRate_; /// Rates for efflux. effluxRate_ = area * D
		double efflux_; /// Value of efflux. Needed only for diagnostics.
};

/**
 * the InterHubFlux class manages exchange of molecules with other
 * solvers, typically diffusive exchange at specified junctions
 * between the spatial domains of the respective solvers. This
 * is designed to be compatible with stochastic and deterministic solvers.
 */
class InterHubFlux
{
	public:
		/// Name of target hub. Just for convenience
		string name_;

		/// One FluxTerm for each molecular species exchanged between hubs.
		vector< FluxTerm > term_;

		bool individualParticlesFlag_; // Is target hub stoch?
};


#endif // _InterHubFlux_h
