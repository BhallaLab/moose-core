#ifndef __NEURON_HPP__
#define __NEURON_HPP__

#include "includes.hpp"

class Neuron
{
public:
    const string                id;
    ref_ptr<neuron_node_t>      node;
    ontology_collection_t       ontology_collection;
    compartment_collection_t    compartment_collection;

    explicit
    Neuron( const string& id
          , const neuron_node_t * node = new neuron_node_t()
          , const ontology_collection_t& ontology_collection = ontology_collection_t()
          , const compartment_collection_t& compartment_collection = compartment_collection_t()
          ) : id(id)
            , node(neuron_node_t)
            , ontology_collection(ontology_collection)
            , compartment_collection(compartment_collection)
    {
        node -> setName(id);
    }

    explicit
    Neuron(const Neuron& neuron
          ) : Neuron( neuron.id
                    , neuron.node.get()
                    , neuron.ontology_collection
                    , neuron.compartment_collection
                    )
    { }

    void
    attach(Compartment * compartment);

    void
    create_view( unsigned int lod_resolution
               , float        lod_distance_delta
               , StateSet  *  state_set
               );

    void
    clear();
};

#endif /* __NEURON_HPP__ */
