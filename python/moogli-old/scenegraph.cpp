#include "utility/csv.hpp"
#include "utility/fileutils.hpp"
#include "core/PickHandler.hpp"
#include <string>
#include <chrono>
#include <iostream>
#include "core/Morphology.hpp"
#include "core/MorphologyViewer.hpp"
#include "utility/conversions.hpp"
#include <iostream>

#include <osg/ArgumentParser>
#include <osg/Group>
#include <osgDB/ReadFile>
#include <osgFX/Outline>
#include <osgViewer/Viewer>
#include <osg/Camera>
#include <osgText/Font>
#include <osgText/Text>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgUtil/Simplifier>
#include <osg/MatrixTransform>
#include <osgViewer/Viewer>
#include <osg/ref_ptr>
#include <locale>
#include <iomanip>
#include <climits>
#include <osgFX/BumpMapping>
#include <osgFX/Outline>
#include <osgFX/Scribe>
#include <osgFX/Cartoon>
#include <osgGA/GUIEventHandler>
#include <osgFX/SpecularHighlights>
#include <osgFX/AnisotropicLighting>
#include <osgViewer/ViewerEventHandlers>
#include <osg/GraphicsContext>
// #include "core/MorphologyViewer.hpp"


using namespace std;
using namespace osg;

void
read_morphology( ArgumentParser & arguments
               , Csv            & csv
               )
{
    for(int pos=1;pos<arguments.argc();++pos)
    {
        if (!arguments.isOption(pos))
        {
            read_csv_with_stats(arguments[pos], csv);
        }
    }
}



void
create_morphology( ArgumentParser & arguments
                 , Morphology     & morphology
                 )
{
    Csv csv;
    chrono::high_resolution_clock::time_point t1, t2;
    unsigned long long time_difference;
    unsigned int rows = UINT_MAX;

    while(arguments.read("--segment-count", rows));

    read_morphology(arguments, csv);


    const unsigned int NEURON_ID_INDEX      =  0;
    const unsigned int COMPARTMENT_ID_INDEX =  1;
    const unsigned int PROXIMAL_X_INDEX     =  2;
    const unsigned int PROXIMAL_Y_INDEX     =  3;
    const unsigned int PROXIMAL_Z_INDEX     =  4;
    const unsigned int PROXIMAL_D_INDEX     =  5;
    const unsigned int DISTAL_X_INDEX       =  6;
    const unsigned int DISTAL_Y_INDEX       =  7;
    const unsigned int DISTAL_Z_INDEX       =  8;
    const unsigned int DISTAL_D_INDEX       =  9;

    string          neuron_id;
    string          compartment_id;
    float           proximal_x;
    float           proximal_y;
    float           proximal_z;
    float           proximal_d;
    float           distal_x;
    float           distal_y;
    float           distal_z;
    float           distal_d;
    unsigned int    row_index = 0;

    for(auto iter = csv.cbegin(); iter != csv.cend() && row_index < rows; ++iter)
    {
        const CsvRow& row = *iter;

        if(row.size() < 10)
        {
            cerr << "[Format Error] Incomplete number of parameters on line number "
                 << row_index + 1
                 << endl;
            exit(0);
        }

        neuron_id       = row[NEURON_ID_INDEX];
        compartment_id  = row[COMPARTMENT_ID_INDEX];

        proximal_x  = stof(row[PROXIMAL_X_INDEX]);
        proximal_y  = stof(row[PROXIMAL_Y_INDEX]);
        proximal_z  = stof(row[PROXIMAL_Z_INDEX]);
        proximal_d  = stof(row[PROXIMAL_D_INDEX]);

        distal_x    = stof(row[DISTAL_X_INDEX]);
        distal_y    = stof(row[DISTAL_Y_INDEX]);
        distal_z    = stof(row[DISTAL_Z_INDEX]);
        distal_d    = stof(row[DISTAL_D_INDEX]);


        // cerr << "Cell Id        = " << cell_id      << endl;
        // cerr << "Segment Id     = " << segment_id   << endl;
        // cerr << "Proximal X     = " << proximal_x   << endl;
        // cerr << "Proximal Y     = " << proximal_y   << endl;
        // cerr << "Proximal Z     = " << proximal_z   << endl;
        // cerr << "Proximal D     = " << proximal_d   << endl;
        // cerr << "Distal X       = " << distal_x     << endl;
        // cerr << "Distal Y       = " << distal_y     << endl;
        // cerr << "Distal Z       = " << distal_z     << endl;
        // cerr << "Distal D       = " << distal_d     << endl;
        // cerr << "Cell Name      = " << cell_name    << endl;
        // cerr << "Segment Name   = " << segment_name << endl;
        // cerr << "Population     = " << population   << endl;

        // cerr << endl << "--------------------------------------" << endl << endl;

        t1 = chrono::high_resolution_clock::now();

        morphology.add_compartment( neuron_id + "_" + compartment_id
                                  , neuron_id
                                  , proximal_x
                                  , proximal_y
                                  , proximal_z
                                  , proximal_d
                                  , distal_x
                                  , distal_y
                                  , distal_z
                                  , distal_d
                                  );
        t2 = chrono::high_resolution_clock::now();
        time_difference += chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        ++row_index;
    }


    RECORD_INFO( "Created scene graph in "
               + to_string(time_difference)
               + " nanoseconds"
               );
}



//http://stackoverflow.com/questions/7257956/how-to-insert-spaces-in-a-big-number-to-make-it-more-readable
template<typename CharT>
struct Separator : public std::numpunct<CharT>
{
    virtual string do_grouping()      const   {return "\003";}
    virtual CharT  do_thousands_sep() const   {return ',';}
};

void
initialize_output_streams()
{
    cerr.imbue(locale(cerr.getloc(), new Separator<char>()));
}

void
create_arguments(ArgumentParser& arguments)
{

    const string& application_name = arguments.getApplicationName();

    // arguments.getApplicationUsage() ->
    //     setApplicationName(application_name);

    arguments.getApplicationUsage() ->
        setDescription( application_name
                      + "This test creates the scene graph from a csv file "
                      + "and performs some basic benchmarking."
                      );

    arguments.getApplicationUsage() ->
        setCommandLineUsage( application_name
                           + " [options] <morphology_file>"
                           );

    arguments.getApplicationUsage() ->
        addCommandLineOption( "--segment-count <count>"
                            , "number of segments to read from the csv file"
                            );
}

// osgViewer::Viewer *
// create_viewer( ArgumentParser& arguments
//              , Morphology
//              )
// {

//     osgViewer::View* view1 = createView( 50,  50, 320, 240, morphology._matrix_transform.get());
//     osgViewer::View* view2 = createView(370,  50, 320, 240, morphology._matrix_transform.get());
//     osgViewer::View* view3 = createView(185, 310, 320, 240, morphology._matrix_transform.get());

//     ref_ptr<osgViewer::Viewer> viewer(new osgViewer::Viewer(arguments));

//     viewer -> setThreadingModel( osgViewer::Viewer::SingleThreaded );
//     viewer -> addEventHandler(new osgViewer::WindowSizeHandler());
//     viewer -> addEventHandler(new osgViewer::HelpHandler());
//     return viewer.release();
// }


void
check_for_help(ArgumentParser& arguments)
{
    unsigned int helpType = 0;

    if ((helpType = arguments.readHelpType()))
    {
        arguments.getApplicationUsage()
            ->write(cout, helpType);
        exit(1);
    }

    if(arguments.errors())
    {
        arguments.writeErrorMessages(cout);
        exit(1);
    }

    if (arguments.argc() <= 1)
    {
        arguments.getApplicationUsage() ->
            write( cout
                 , ApplicationUsage::COMMAND_LINE_OPTION
                 );
        exit(1);
    }
}

// int
// run( osgViewer::Viewer * viewer
//    , Morphology        & morphology
//    )
// {
//     chrono::high_resolution_clock::time_point t1, t2;
//     t1 = chrono::high_resolution_clock::now();
//     viewer -> setSceneData( morphology._matrix_transform.get() );
//     t2 = chrono::high_resolution_clock::now();
//     cerr << "Attached scene graph to viewer in "
//          << chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
//          << " nanoseconds"
//          << endl;

//     return viewer -> run();
// }


// osgViewer::View *
// createView(int x, int y, int w, int h, osg::Node* scene )
// {
//     osg::ref_ptr<osgViewer::View> view = new osgViewer::View;
//     view -> setSceneData( scene );
//     view -> setUpViewInWindow( x, y, w, h );
//     return view.release();
// }


// Camera *
// create_camera_test(osg::GraphicsContext::Traits * traits)
// {
    // osg::DisplaySettings* ds = osg::DisplaySettings::instance().get();
    // osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits();

    // // traits->windowName      = window_name;
    // // traits->windowDecoration= window_decoration;
    // traits->x               = 0;
    // traits->y               = 0;
    // traits->width           = 400;
    // traits->height          = 400;
    // traits->doubleBuffer    = true;
    // traits->alpha           = ds -> getMinimumNumAlphaBits();
    // traits->stencil         = ds -> getMinimumNumStencilBits();
    // traits->sampleBuffers   = ds -> getMultiSamples();
    // traits->samples         = ds -> getNumMultiSamples();

    // unsigned int major, minor;

    // traits -> getContextVersion(major, minor);

    // RECORD_INFO("Using OpenGL Version - " + to_string(major) + "." + to_string(minor));
/*
    Camera * camera = new Camera();
    camera -> setGraphicsContext(new osg::GraphicsContext(traits));
    // camera->set

    camera->setClearColor(osg::Vec4f(1.0f, 0.0f, 0.0f, 1.0f));
    camera->setViewport( new osg::Viewport( 0
                                          , 0
                                          , traits->width
                                          , traits->height
                                          )
                        );
    camera->setProjectionMatrixAsPerspective( 30.0f
                                            , static_cast<double>(traits->width)/static_cast<double>(traits->height)
                                            , 1.0
                                            , 10000.0
                                            );

    // camera->setCullingMode( CullSettings::NEAR_PLANE_CULLING
    //                       | CullSettings::FAR_PLANE_CULLING
    //                       | CullSettings::VIEW_FRUSTUM_CULLING
    //                       | CullSettings::SMALL_FEATURE_CULLING
    //                       );
    return camera;
}
*/

Group *
create_cylinder()
{
    osg::Group* root = new osg::Group();
    osg::Geode* pyramidGeode = new osg::Geode();
    osg::Geometry* pyramidGeometry = cylinder( Vec3f(0.0,0.0,0.0)
                                   , 2.0
                                   , 10.0
                                   , Vec3f(1.0, 0.0, 0.0)
                                   , 20
                                   );
   pyramidGeode->addDrawable(pyramidGeometry);
   root->addChild(pyramidGeode);
   return root;
}

Group *
create_pyramid()
{
    osg::Group* root = new osg::Group();
    osg::Geode* pyramidGeode = new osg::Geode();
    osg::Geometry* pyramidGeometry = new osg::Geometry();

   pyramidGeode->addDrawable(pyramidGeometry);
   root->addChild(pyramidGeode);

   osg::Vec3Array* pyramidVertices = new osg::Vec3Array;
   pyramidVertices->push_back( osg::Vec3( 0, 0, 0) ); // front left
   pyramidVertices->push_back( osg::Vec3(10, 0, 0) ); // front right
   pyramidVertices->push_back( osg::Vec3(10,10, 0) ); // back right
   pyramidVertices->push_back( osg::Vec3( 0,10, 0) ); // back left
   pyramidVertices->push_back( osg::Vec3( 5, 5,10) ); // peak

   pyramidGeometry->setVertexArray( pyramidVertices );

   osg::DrawElementsUInt* pyramidBase =
      new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
   pyramidBase->push_back(3);
   pyramidBase->push_back(2);
   pyramidBase->push_back(1);
   pyramidBase->push_back(0);
   pyramidGeometry->addPrimitiveSet(pyramidBase);

   osg::DrawElementsUInt* pyramidFaceOne =
      new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
   pyramidFaceOne->push_back(0);
   pyramidFaceOne->push_back(1);
   pyramidFaceOne->push_back(4);
   pyramidGeometry->addPrimitiveSet(pyramidFaceOne);

   osg::DrawElementsUInt* pyramidFaceTwo =
      new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
   pyramidFaceTwo->push_back(1);
   pyramidFaceTwo->push_back(2);
   pyramidFaceTwo->push_back(4);
   pyramidGeometry->addPrimitiveSet(pyramidFaceTwo);

   osg::DrawElementsUInt* pyramidFaceThree =
      new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
   pyramidFaceThree->push_back(2);
   pyramidFaceThree->push_back(3);
   pyramidFaceThree->push_back(4);
   pyramidGeometry->addPrimitiveSet(pyramidFaceThree);

   osg::DrawElementsUInt* pyramidFaceFour =
      new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, 0);
   pyramidFaceFour->push_back(3);
   pyramidFaceFour->push_back(0);
   pyramidFaceFour->push_back(4);
   pyramidGeometry->addPrimitiveSet(pyramidFaceFour);

   osg::Vec4Array* colors = new osg::Vec4Array;
   colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f) ); //index 0 red
   colors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f) ); //index 1 green
   colors->push_back(osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f) ); //index 2 blue
   colors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f) ); //index 3 white
   colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f) ); //index 4 red

   pyramidGeometry->setColorArray(colors);
   pyramidGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
   return root;
}

int
main(int argc, char * argv[])
{
    ArgumentParser arguments(&argc,argv);
    Morphology morphology("cortical_column", 1);
    Csv        * csv;

    initialize_output_streams();
    create_arguments(arguments);

    // viewer = create_viewer(arguments);
    check_for_help(arguments);
    create_morphology(arguments, morphology);
    // QApplication app(argc, argv);

    // MorphologyViewer* viewer = new MorphologyViewer(&morphology,1500,1000);

    // viewer -> setGeometry( 0, 0, 800, 600 );
    // viewer -> create_view();
    osgViewer::Viewer viewer;
    // std::vector<Camera *> cameras;
    // viewer.getCameras(cameras);
    // viewer.setCamera(create_camera_test(cameras[0] -> getGraphicsContext() -> getTraits()));
    viewer.setThreadingModel(osgViewer::Viewer::SingleThreaded);
    viewer.addEventHandler(new PickHandler());
    cerr << "\nChild Count " << morphology._matrix_transform -> getNumChildren();
    cerr << "\n Grand Children Count" << morphology._matrix_transform -> getNumChildren();
    viewer.setSceneData( morphology._matrix_transform.get());//morphology._matrix_transform.get());//create_pyramid()); //morphology.get_scene_graph().get());
    return viewer.run();


    // viewer -> show();
    // return app.exec();

    // osg::ref_ptr<PickHandler> picker = new PickHandler(morphology);
    // viewer -> addEventHandler( picker.get() );
    // return run(viewer.get(), morphology);
}


