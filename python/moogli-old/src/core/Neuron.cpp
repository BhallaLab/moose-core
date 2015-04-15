#include "core/Neuron.hpp"

void
Neuron::create_view( unsigned int lod_resolution
                   , float        lod_distance_delta
                   , StateSet   * state_set
                   )
{
    neuron_node -> setStateSet(state_set.get());

    float range = 0.0f;

    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        Geode * geode = new Geode();
        // geode -> setStateSet(_state_set.get());
        node -> addChild( geode
                        , range
                        , range + lod_distance_delta
                        );
        range += lod_distance_delta;
    }

    // Model visibility should be across the entire range of floats.
    // The geode added in the end should be the worst quality and visible
    // no matter how far the model is from the viewer's eyes.
    // Hence we need to reset the max range of this geode.
    node -> setRange( lod_resolution - 1
                    , range - lod_distance_delta
                    , FLT_MAX
                    );

    for(Compartment * compartment : compartment_collection)
    {
        for(unsigned int i = 0; i < lod_resolution; ++i)
        {
            Geode * geode = (Geode *)(neuron_node -> getChild(i));
            geode -> addDrawable(compartment -> geometries[i].get());
        }
    }
}

void
Neuron::_detach_geometry(Compartment * compartment)
{
    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        Geode * geode = (Geode *)(neuron_node -> getChild(i));
        geode -> removeDrawable(compartment -> geometries[i].get());
    }
}

void
Neuron::_attach_geometry(Compartment * compartment)
{
    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        Geode * geode = (Geode *)(neuron_node -> getChild(i));
        geode -> addDrawable(compartment -> geometries[i].get());
    }
}

void
Neuron::hide(Compartment * compartment)
{
    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        compartment -> geometries[i] -> setNodeMask(0);
    }
}

void
Neuron::show(Compartment * compartment)
{
    for(unsigned int i = 0; i < lod_resolution; ++i)
    {
        compartment -> geometries[i] -> setNodeMask(~0);
    }
}

void
Neuron::add(Compartment * compartment)
{
    compartment_collection.insert(compartment);

    _attach_geometry(compartment);
}

void
Neuron::remove(Compartment * compartment)
{
    compartment_collection.insert(compartment);
    _detach_geometry(compartment);
}

void
Neuron::clear()
{
    node(new neuron_node_t());
    ontology_collection.clear();
    compartment_collection.clear();
}
