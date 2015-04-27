%Module morphology

%Import QtGui/QtGuimod.sip

class Morphology
{

%TypeHeaderCode
#include "includes.hpp"
#include "core/Compartment.hpp"
%End

public:

    Morphology( const char *        name
              , const unsigned int  lod_resolution     = 3
              , const float         lod_distance_delta = 50.0f
              , const unsigned int  min_points         = 10
              , const unsigned int  points_delta       = 2
              );

    int
    add_compartment( const char *    compartment_id
                   , const char *    neuron_id
                   , double          proximal_x
                   , double          proximal_y
                   , double          proximal_z
                   , double          proximal_d
                   , double          distal_x
                   , double          distal_y
                   , double          distal_z
                   , double          distal_d
                   );
};
