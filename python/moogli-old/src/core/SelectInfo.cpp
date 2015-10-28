#include "core/SelectInfo.hpp"

SelectInfo::SelectInfo()
{
    _id = "";
    _event_type = 0;
}

const char *
SelectInfo::get_id()
{
    return _id;
}

int
SelectInfo::get_event_type()
{
    return _event_type;
}

void
SelectInfo::set_id(const char * id)
{
    _id = id;
}

void
SelectInfo::set_event_type(int event_type)
{
    _event_type = event_type;
}
