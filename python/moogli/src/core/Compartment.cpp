#include "core/Compartment.hpp"

using namespace osg;
using namespace std;

Compartment::Compartment( const string & id
                        , const string & neuron
                        ) : id(id)
                          , neuron_id(neuron)
{ }

void
Compartment::set_proximal_parameters( const Vec3d & center
                                    , double d
                                    )
{
    _proximal.set(center);
    _proximal_d = d;
}

pair<const Vec3d &, double>
Compartment::get_proximal_parameters()
{
    return pair<const Vec3d &, double>(_proximal, _proximal_d);
}

void
Compartment::set_proximal_parameters( double x
                                    , double y
                                    , double z
                                    , double d
                                    )
{
    _proximal.set(x, y, z);
    _proximal_d = d;
}

void
Compartment::set_distal_parameters( const Vec3d & center
                                  , double d
                                  )
{
    _distal.set(center);
    _distal_d = d;
}

pair<const Vec3d &, double>
Compartment::get_distal_parameters()
{
    return pair<const Vec3d &, double>(_distal, _distal_d);
}

void
Compartment::set_distal_parameters( double x
                                  , double y
                                  , double z
                                  , double d
                                  )
{
    _distal.set(x, y, z);
    _distal_d = d;
}

void
Compartment::create_geometry( unsigned int lod_resolution
                            , float lod_distance_delta
                            , unsigned int min_points
                            , unsigned int points_delta
                            , StateSet * state_set
                            )
{
    geometries.assign(lod_resolution, ref_ptr<Geometry>());
    _center          = (_distal + _proximal) / 2.0;
    _direction       = _distal - _proximal;
    _height          = _direction.normalize();
    _radius          = (_proximal_d + _distal_d) / 4.0;

    unsigned int points = lod_resolution * points_delta + min_points - points_delta;

    //Floating point comparison is being done with compare_double function.
    if(  (compare_double(_height, _radius, 0.01f) == 0)
      || (compare_double(_height, 0.0f, 0.01f) == 0)
      )
    {
        for(unsigned int i = 0; i < lod_resolution; ++i)
        {
            geometries[i] = sphere( _center
                                  , _radius
                                  , points
                                  );
            // geometries[i] -> setNodeMask(0xffffffff);

            geometries[i] -> setName(id);
            points        -= points_delta;
            // geometries[i] -> getOrCreateStateSet();
        }
    }
    else
    {
        for(unsigned int i = 0; i < lod_resolution; ++i)
        {
            geometries[i] = cylinder( _center
                                    , _distal_d   / 2.0
                                    , _proximal_d / 2.0
                                   , _height
                                   , _direction
                                   , points
                                   );
            // geometries[i] -> setNodeMask(0xffffffff);
            geometries[i] -> setName(id);
            points -= points_delta;
            // geometries[i] -> getOrCreateStateSet();

        }
    }
}

void
Compartment::set_color( double value
                      , double base_value
                      , double peak_value
                      , Vec4f& base_color
                      , Vec4f& peak_color
                      )
{
    _set_chromostat(value, base_value, peak_value);
    // RECORD_INFO("Chromostat : " + to_string(_chromostat));
    Vec4f color = base_color + (peak_color - base_color) * _chromostat;
    Vec4Array * colors = new Vec4Array();
    colors -> push_back(color);
    for(auto & geometry : geometries)
    {
        geometry -> setColorArray(colors);
        geometry -> setColorBinding( osg::Geometry::BIND_OVERALL );
    }
}

void
Compartment::_set_chromostat( double value
                            , double base_value
                            , double peak_value
                            )
{
    _chromostat = (value - base_value) / (peak_value - base_value);
    if(_chromostat > 1.0)         { _chromostat = 1.0; }
    else if(_chromostat < 0.0)    { _chromostat = 0.0; }
}

void
Compartment::set_membrane_voltage( double vm
                                 , double base_vm
                                 , double peak_vm
                                 , Vec4f& initial_color
                                 , Vec4f& final_color
                                 )
{
    _set_chromostat(vm, base_vm, peak_vm);
    // RECORD_INFO("Chromostat : " + to_string(_chromostat));
    Vec4f color = initial_color + (final_color - initial_color) * _chromostat;
    Vec4Array * colors = new Vec4Array();
    colors -> push_back(color);
    for(auto & geometry : geometries)
    {
        geometry -> setColorArray(colors);
        geometry -> setColorBinding( osg::Geometry::BIND_OVERALL );
    }
}



