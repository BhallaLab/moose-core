/*
 * File:            UniformRng.cpp
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-01 11:30:20
 */

#include "../basecode/header.h"
#include "randnum.h"
#include "UniformRng.h"

const Cinfo* UniformRng::initCinfo()
{
    static ValueFinfo< UniformRng, double > min(
        "min",
        "The lower bound on the numbers generated ",
        &UniformRng::setMin,
        &UniformRng::getMin);
    static ValueFinfo< UniformRng, double > max(
        "max",
        "The upper bound on the numbers generated",
        &UniformRng::setMax,
        &UniformRng::getMax);

    static Finfo * uniformRngFinfos[] =
    {
        &min,
        &max,
    };

    static string doc[] =
    {
        "Name", "UniformRng",
        "Author", "Subhasis Ray",
        "Description", "Generates pseudorandom number from a unform distribution.",
    };

    static Dinfo< UniformRng > dinfo;

    static Cinfo uniformRngCinfo(
        "UniformRng",
        Neutral::initCinfo(),
        uniformRngFinfos,
        sizeof( uniformRngFinfos ) / sizeof( Finfo* ),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string )
    );

    return &uniformRngCinfo;
}

static const Cinfo* uniformRngCinfo = UniformRng::initCinfo();

UniformRng::UniformRng( )
{
}

UniformRng& UniformRng::operator=( const UniformRng& r )
{
    return *this;
}

double UniformRng::getMin()const
{
    return dist_.min();
}

double UniformRng::getMax() const
{
    return dist_.max();
}

void UniformRng::setMin(double min)
{
    min_ = min;
    dist_ = moose::MOOSE_UNIFORM_DISTRIBUTION(min_, max_);
}

void UniformRng::setMax(double max)
{
    max_ = max;
    dist_ = moose::MOOSE_UNIFORM_DISTRIBUTION(min_, max_);
}

void UniformRng::reinit(const Eref& e, ProcPtr p)
{
    dist_ = moose::MOOSE_UNIFORM_DISTRIBUTION(min_, max_);
}
