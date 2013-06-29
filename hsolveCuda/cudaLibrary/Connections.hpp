/*
 * Connections.h
 *
 *  Created on: 10/08/2009
 *      Author: rcamargo
 */

#ifndef CONNECTIONS_H_
#define CONNECTIONS_H_

#include "Definitions.hpp"
//#include "HinesStruct.hpp"
//#include "ThreadInfo.hpp"
#include <vector>

//#define __aligned__ ignored
//#include <tr1/unordered_map>
//#undef __aligned__

#include <map>


#define CONN_NEURON_TYPE 1000000

/**
 * Contains all the connections to the neurons of a given neuron group.
 * There are nGroups neuron groups
 */
struct ConnGpu {

	int *srcDevice;
	int *srcHost;

	int *destDevice;
	int *destHost;

	ucomp *synapseDevice;
	ucomp *synapseHost;

	ftype *weightDevice;
	ftype *weightHost;

	ftype *delayDevice;
	ftype *delayHost;

	//int *neuronPosDevice;
	//int *neuronPosHost;

	int nConnectionsTotal;
	int nNeuronsGroup;
	int nNeuronsInPreviousGroups;
};

struct Conn {
	int dest;   // neuron identifier
	ucomp synapse;
	ftype weigth;
	ftype delay;
};

struct ConnectionInfo{
	int *source;
	int *dest;
	ucomp *synapse;
	ftype *weigth;
	ftype *delay;
	int nConnections;

};

class Connections {
public:

	//std::tr1::unordered_map< int, std::vector<Conn> > connMap;
	std::map< int, std::vector<Conn> > connMap;

	//int *getConnArray (int source);
	//int getConnArraySize (int source);
	std::vector<Conn> & getConnArray (int source);

	Connections();
	virtual ~Connections();

	ConnectionInfo *getConnectionInfo ();
	void clearMPIConnections(ConnectionInfo *connInfo);

	void setPositionsPlanar (struct ThreadInfo *tInfo, ucomp type, ftype totalLength);

	int connectTypeToTypeRandom(
			struct ThreadInfo *tInfo, int sourceType, int destType, ucomp synapse,
			ftype connRatio, ftype baseW, ftype randW, ftype baseD, ftype randD);

	int connectTypeToTypeOneToOne( struct ThreadInfo *tInfo,
			int sourceType, int destType, ucomp synapse, ftype baseW, ftype baseD);

	int connectAssociativeFromFile (char *filename);
	int connectRandom ( struct ThreadInfo *tInfo );
	int connectRandom2( struct ThreadInfo *tInfo );
	int createTestConnections ();
private:
    int transformToCombinedConnection(struct ThreadInfo *tInfo, int destType, int neuron);
};

#endif /* CONNECTIONS_H_ */
