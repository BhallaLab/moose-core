#ifndef __SELECT_INFO_HPP__
#define __SELECT_INFO_HPP__

class SelectInfo
{
private:
    const char * _id;
    int          _event_type;

public:
    SelectInfo();
    const char * get_id();
    int get_event_type();
    void set_id(const char * id);
    void set_event_type(int event_type);
};

#endif /* __SELECT_INFO_HPP__ */
