/****************************************************************************
** Meta object code from reading C++ file 'MorphologyViewer.hpp'
**
** Created: Thu Feb 26 16:43:14 2015
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "include/core/MorphologyViewer.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MorphologyViewer.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MorphologyViewer[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      33,   18,   17,   17, 0x05,

 // slots: signature, parameters, type, tag, flags
      62,   17,   17,   17, 0x08,
      76,   17,   17,   17, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MorphologyViewer[] = {
    "MorphologyViewer\0\0compartment_id\0"
    "compartment_dragged(QString)\0toggle_mode()\0"
    "handle_translate_positive_x()\0"
};

void MorphologyViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MorphologyViewer *_t = static_cast<MorphologyViewer *>(_o);
        switch (_id) {
        case 0: _t->compartment_dragged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 1: _t->toggle_mode(); break;
        case 2: _t->handle_translate_positive_x(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MorphologyViewer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MorphologyViewer::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_MorphologyViewer,
      qt_meta_data_MorphologyViewer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MorphologyViewer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MorphologyViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MorphologyViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MorphologyViewer))
        return static_cast<void*>(const_cast< MorphologyViewer*>(this));
    return QWidget::qt_metacast(_clname);
}

int MorphologyViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}

// SIGNAL 0
void MorphologyViewer::compartment_dragged(const QString & _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
