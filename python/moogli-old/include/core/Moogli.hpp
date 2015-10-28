    PyObject *
    add_compartments( PyObject * compartment_id_sequence
                    , PyObject * neuron_id_sequence
                    , PyObject * proximal_sequence
                    , PyObject * distal_sequence
                    );

    PyObject *
    add_neuron( const char * neuron_id
              , PyObject   * compartment_id_sequence
              , PyObject   * proximal_sequence
              , PyObject   * distal_sequence
              );

    void
    remove_compartments(PyObject * compartment_id_sequence);

    int
    remove_neurons(PyObject * neuron_id_sequence);

    PyObject *
    hide_compartments(PyObject * compartment_id_sequence);

    void
    hide_neurons(PyObject * neuron_id_sequence);

    void
    show_compartments(PyObject * compartment_id_sequence);


    void
    show_neurons(PyObject * neuron_id_sequence);

void
Morphology::hide_neurons(PyObject * neuron_id_sequence)
{
    for(size_t i = 0; i < PySequence_Size(neuron_id_sequence); ++i)
    {
        hide_neuron(
            PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(neuron_id_sequence, i))
              );
    }
}

void
Morphology::show_neurons(PyObject * neuron_id_sequence)
{
    for(size_t i = 0; i < PySequence_Size(neuron_id_sequence); ++i)
    {
        show_neuron(
            PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(neuron_id_sequence, i))
              );
    }
}

void
Morphology::hide_compartments(PyObject * compartment_id_sequence)
{
    for(size_t i = 0; i < PySequence_Size(neuron_id_sequence); ++i)
    {
        hide_compartment(
            PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(neuron_id_sequence, i)
                              )
                   );
    }
}

void
Morphology::show_compartments(PyObject * compartment_id_sequence)
{
    for(size_t i = 0; i < PySequence_Size(neuron_id_sequence); ++i)
    {
        show_compartment(
            PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(neuron_id_sequence, i)
                              )
                   );
    }
}


PyObject *
remove_compartments( PyObject * compartment_id_sequence)
{
    chrono::high_resolution_clock::time_point t1;
    t1 = chrono::high_resolution_clock::now();

    bool result = true;

    for(size_t i = 0; i < PySequence_Size(compartment_id_sequence); ++i)
    {
        result = result &&
        remove_compartment(
                PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(compartment_id_sequence, i))
              );
    }

    chrono::high_resolution_clock::time_point t2;
    t2 = chrono::high_resolution_clock::now();

    if(result)
    {
        record( INFO
              , __FILE__
              , __FUNCTION__
              , __LINE__
              , "All compartments removed successfully."
              );
    }

    record( INFO
          , __FILE__
          , __FUNCTION__
          , __LINE__
          , "Executed in "
          + to_string(chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count())
          + "."
          );

    return PyInt_FromSize_t(_compartments.size());
}

PyObject *
Morphology::add_neuron( const char * neuron_id
                      , PyObject   * compartment_id_sequence
                      , PyObject   * proximal_sequence
                      , PyObject   * distal_sequence
                      )
{
    chrono::high_resolution_clock::time_point t1;
    t1 = chrono::high_resolution_clock::now();

    bool result = true;

    for(size_t i = 0; i < PySequence_Size(compartment_id_sequence); ++i)
    {
        result = result &&
        create_compartment(
                PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(compartment_id_sequence, i))
              , neuron_id
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 0)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 1)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 2)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 3)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 0)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 1)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 2)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 3)))
              );
    }

    chrono::high_resolution_clock::time_point t2;
    t2 = chrono::high_resolution_clock::now();

    if(result)
    {
        record( INFO
              , __FILE__
              , __FUNCTION__
              , __LINE__
              , "All items inserted successfully."
              );
    }
    else
    {
        record( ERROR
              , __FILE__
              , __FUNCTION__
              , __LINE__
              , "Some elements were not inserted!"
                "Please check that compartment ids are unique."
              );
    }

    record( INFO
          , __FILE__
          , __FUNCTION__
          , __LINE__
          , "Executed in "
          + to_string(chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count())
          + "."
          );

    return PyInt_FromSize_t(_compartments.size());
}

PyObject *
Morphology::add_compartments( PyObject * compartment_id_sequence
                            , PyObject * neuron_id_sequence
                            , PyObject * proximal_sequence
                            , PyObject * distal_sequence
                            )
{

    chrono::high_resolution_clock::time_point t1;
    t1 = chrono::high_resolution_clock::now();

    bool result = true;

    for(size_t i = 0; i < PySequence_Size(compartment_id_sequence); ++i)
    {
        result = result &&
        create_compartment(
                PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(compartment_id_sequence, i))
              , PyString_AS_STRING(
                    PySequence_Fast_GET_ITEM(neuron_id_sequence, i))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 0)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 1)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 2)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(proximal_sequence, i, 3)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 0)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 1)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 2)))
              , *(static_cast<double*> (
                    PyArray_GETPTR2(distal_sequence, i, 3)))
              );
    }

    chrono::high_resolution_clock::time_point t2;
    t2 = chrono::high_resolution_clock::now();

    if(result)
    {
        record( INFO
              , __FILE__
              , __FUNCTION__
              , __LINE__
              , "All items inserted successfully."
              );
    }
    else
    {
        record( ERROR
              , __FILE__
              , __FUNCTION__
              , __LINE__
              , "Some elements were not inserted!"
                "Please check that compartment ids are unique."
              );
    }

    record( INFO
          , __FILE__
          , __FUNCTION__
          , __LINE__
          , "Executed in "
          + to_string(chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count())
          + "."
          );

    return PyInt_FromSize_t(_compartments.size());
}
