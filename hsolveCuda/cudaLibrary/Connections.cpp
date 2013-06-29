/*
 * Connections.cpp
 *
 *  Created on: 10/08/2009
 *      Author: rcamargo
 */

#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "ThreadInfo.hpp"
#include "SharedNeuronGpuData.hpp"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>

Connections::Connections() {
}

Connections::~Connections() {
	// TODO: Remove connections
}

std::vector<Conn> & Connections::getConnArray (int source) {
	return connMap[source];
}

ConnectionInfo *Connections::getConnectionInfo () {

	int countTotal = 0;
	ConnectionInfo *connInfo = new ConnectionInfo;

	// Counts the total number of connections
	std::map< int, std::vector<Conn> >::iterator p;
	for(p = connMap.begin(); p != connMap.end(); p++)
		countTotal += p->second.size();

	connInfo->nConnections = countTotal;
	connInfo->source  = new int[countTotal];
	connInfo->dest 	 = new int[countTotal];
	connInfo->synapse = new ucomp[countTotal];
	connInfo->weigth  = new ftype[countTotal];
	connInfo->delay   = new ftype[countTotal];

	int infoPos = 0;
	for(p = connMap.begin(); p != connMap.end(); p++) {

		int source = p->first;
		std::vector<Conn> & conn = p->second;

		std::vector<Conn>::iterator p;
		for(p = conn.begin(); p != conn.end(); p++) {
			connInfo->source [infoPos] = source;
			connInfo->dest   [infoPos] = (*p).dest;
			connInfo->synapse[infoPos] = (*p).synapse;
			connInfo->weigth [infoPos] = (*p).weigth;
			connInfo->delay  [infoPos] = (*p).delay;
			infoPos++;
		}
	}

	return connInfo;
}

void Connections::setPositionsPlanar (ThreadInfo *tInfo, ucomp targetType, ftype totalLength) {

	HinesMatrix **matrizList = tInfo->sharedData->matrixList;

	// Finds the total number of neurons of the selected type
	int typeNeurons = 0;
	for (int sType=0; sType < tInfo->totalTypes; sType++)
		if (tInfo->sharedData->typeList[sType] == targetType)
			typeNeurons += tInfo->nNeurons[sType];

	int nNeuronsDim = (int) sqrt (typeNeurons);
	if (typeNeurons % nNeuronsDim != 0) nNeuronsDim++;

	// Checks the number of the type neurons in the previous processes
	int neuronsPrevProc = 0;
	ftype dx = totalLength/nNeuronsDim;
	int x = neuronsPrevProc % nNeuronsDim;
	int y = neuronsPrevProc / nNeuronsDim;

	// Updates the positions
	for (int type=0; type < tInfo->totalTypes; type++) {
		if (tInfo->sharedData->typeList[type] != type) continue;

		for (int neuron=0; neuron < tInfo->nNeurons[type]; neuron++) {
			matrizList[type][neuron].posx = dx * x;
			matrizList[type][neuron].posy = dx * y;
			// reached the end of line
			if ( ++x % nNeuronsDim == 0) { x = 0; y++; }
		}
	}

}

int Connections::createTestConnections () {

	int source;

	Conn conn1;
	source = CONN_NEURON_TYPE*0 + 0;
	conn1.dest   = CONN_NEURON_TYPE*3 + 1;
	conn1.synapse = 0;
	conn1.weigth = 1;
	conn1.delay = 10;
	connMap[source].push_back(conn1);

	Conn conn2;
	source = CONN_NEURON_TYPE*0 + 0;
	conn2.dest   = CONN_NEURON_TYPE*1 + 1;
	conn2.synapse = 0;
	conn2.weigth = 1;
	conn2.delay = 10;
	connMap[source].push_back(conn2);

	Conn conn3;
	source = CONN_NEURON_TYPE*3 + 1;
	conn3.dest   = CONN_NEURON_TYPE*1 + 0;
	conn3.synapse = 1;
	conn3.weigth = 1;
	conn3.delay = 10;
	connMap[source].push_back(conn3);

	return 0;
}

/**
 * type [0][1][2][3] pyramidal
 * type [4][5][6][7] inhibitory
 */
int Connections::connectAssociativeFromFile (char *filename) {
	return 0;
}

int Connections::transformToCombinedConnection(ThreadInfo *tInfo, int destType, int neuron)
{
	// Transform the connection to the type CONN_NEURON_TYPE*dType + neuron
	int typeNeuron;
	int count = 0;
	for (int dType=0; dType < tInfo->totalTypes; dType++) {

		if (tInfo->sharedData->typeList[dType] != destType) continue;

		if (neuron < count + tInfo->nNeurons[dType]) {
			typeNeuron = CONN_NEURON_TYPE*dType + (neuron-count);
			break;
		}
		count += tInfo->nNeurons[dType];
	}

	return typeNeuron;
}

int Connections::connectTypeToTypeRandom( ThreadInfo *tInfo,
		int sourceType, int destType, ucomp synapse, ftype connRatio,
		ftype baseW, ftype randW, ftype baseD, ftype randD) {

	int nConnTotal = 0;

	// Finds total number of destType neurons
	int nDestTypeNeurons = tInfo->nNeuronsTotalType[destType];

	random_data *randBuff = tInfo->sharedData->randBuf[tInfo->threadNumber];
	int32_t randVal;

	/**
	 * Connects the pyramidal-pyramidal cells
	 */
	int *randomConnectionList = (int *)malloc( sizeof(int) * nDestTypeNeurons );

	for (int type=0; type < tInfo->totalTypes; type++) {

		if (tInfo->sharedData->typeList[type] != sourceType) continue;

		for (int sNeuron=0; sNeuron < tInfo->nNeurons[type]; sNeuron++) {

			for (int i =0; i < nDestTypeNeurons; i++ )
				randomConnectionList[i] = 0;

			/**
			 * Connects to a random neuron nDestTypeNeurons*connRatio times
			 */
			for (int c=0; c < nDestTypeNeurons * connRatio; c++) {

				Conn conn1;
				conn1.synapse = synapse;

				random_r( randBuff, &randVal );
				conn1.weigth = baseW * (randW + ((ftype)randVal)/RAND_MAX);

				random_r( randBuff, &randVal );
				conn1.delay = baseD + randD * ((ftype)randVal)/RAND_MAX;

				// Finds a target that did not receive any connection
				random_r( randBuff, &randVal );
				conn1.dest = randVal % nDestTypeNeurons;
				while(randomConnectionList[conn1.dest] != 0){
					random_r( randBuff, &randVal );
					conn1.dest = randVal % nDestTypeNeurons;
				}
				randomConnectionList[conn1.dest] = 1;

				assert (0 <= conn1.dest && conn1.dest < nDestTypeNeurons);

				conn1.dest = transformToCombinedConnection(tInfo, destType, conn1.dest);

				connMap[CONN_NEURON_TYPE*type + sNeuron].push_back(conn1);
				nConnTotal++;
			}
		}
	}
	free (randomConnectionList);

	return nConnTotal;
}

int Connections::connectTypeToTypeOneToOne( ThreadInfo *tInfo,
		int sourceType, int destType, ucomp synapse, ftype baseW, ftype baseD) {

	int nConnTotal = 0;

	// Checks the number of inhibitory neurons in the previous processes
	int destNeuron = 0;
	if (tInfo->sharedData->pyrInhConnRatio > 0) {
		for (int sType=0; sType < tInfo->totalTypes; sType++) {

			if (tInfo->sharedData->typeList[sType] != sourceType) continue;

			for (int sNeuron=0; sNeuron < tInfo->nNeurons[sType]; sNeuron++, destNeuron++) {

				Conn conn1;
				conn1.synapse = synapse; // inhibitory
				conn1.weigth = baseW;
				conn1.delay  = baseD;
				conn1.dest 	 = transformToCombinedConnection(tInfo, destType, destNeuron);

				connMap[CONN_NEURON_TYPE*sType + sNeuron].push_back(conn1);
				nConnTotal++;
			}
		}
	}

	return nConnTotal;
}

int Connections::connectRandom ( struct ThreadInfo *tInfo ) {

	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	setPositionsPlanar(tInfo, PYRAMIDAL_CELL,  10e-3);
	setPositionsPlanar(tInfo, INHIBITORY_CELL, 10e-3);

	int nConnTotal = 0;

	/**
	 * Connects the pyramidal-pyramidal cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, PYRAMIDAL_CELL, 0, sharedData->pyrPyrConnRatio,
			sharedData->excWeight, 0.5, 10, 10);

	/**
	 * Connects the pyramidal-inhibitory cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, INHIBITORY_CELL, 0, sharedData->pyrInhConnRatio/10,
			sharedData->pyrInhWeight*4, 0.5, 10, 10);


	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, BASKET_CELL, 0, sharedData->pyrInhConnRatio,
			sharedData->pyrInhWeight, 0.5, 10, 10);

	/**
	 * Connects the inhibitory-pyramidal cells
	 * Each inhibitory cell connects to a single pyramidal neuron
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, INHIBITORY_CELL, PYRAMIDAL_CELL, 1, sharedData->pyrInhConnRatio/10,
			sharedData->inhPyrWeight/20, 0.5, 10, 10);

	nConnTotal += connectTypeToTypeOneToOne(
			tInfo, BASKET_CELL, PYRAMIDAL_CELL, 1,
			tInfo->sharedData->inhPyrWeight/2, 10);

	//printf("Total number of connections = %d.\n", nConnTotal);

	return 0;
}

int Connections::connectRandom2 ( struct ThreadInfo *tInfo ) {

	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	setPositionsPlanar(tInfo, PYRAMIDAL_CELL,  10e-3);
	setPositionsPlanar(tInfo, INHIBITORY_CELL, 10e-3);

	int nConnTotal = 0;

	/**
	 * Connects the pyramidal-pyramidal cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, PYRAMIDAL_CELL, 0, sharedData->pyrPyrConnRatio,
			sharedData->excWeight, 0.5, 10, 10);

	/**
	 * Connects the pyramidal-inhibitory cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, PYRAMIDAL_CELL, INHIBITORY_CELL, 0, sharedData->pyrInhConnRatio,
			sharedData->pyrInhWeight, 0.5, 10, 10);

	/**
	 * Connects the inhibitory-pyramidal cells
	 */
	nConnTotal += connectTypeToTypeRandom(
			tInfo, INHIBITORY_CELL, PYRAMIDAL_CELL, 1, sharedData->inhPyrConnRatio,
			sharedData->inhPyrWeight, 0.5, 10, 10);

	return 0;
}
