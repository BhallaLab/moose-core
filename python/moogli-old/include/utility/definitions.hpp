#ifndef __DEFINITIONS_HPP__
#define __DEFINITIONS_HPP__

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

using namespace std;

using namespace osg;

//Forward declaration of Compartment class
class Compartment;

//Forward declaration of Neuron class
class Neuron;

typedef string moose_id_t;

typedef string ontology_t;

typedef unordered_map<moose_id_t, Compartment *>  compartment_map_t;

typedef unordered_map<moose_id_t, Neuron * >      neuron_map_t;

typedef unordered_set<Compartment *>        compartment_collection_t;

/* Ontology collection follows operations akin to sets.
*  One can add ontology, remove ontology, check for ontology existence
*  and duplicated ontologies don't make sense.
*/
typedef unordered_set<ontology_t>           ontology_collection_t;

typedef unordered_set<moose_id_t>           moose_id_collection_t;


typedef LOD neuron_node_t;

#endif /* __DEFINITIONS_HPP__ */
