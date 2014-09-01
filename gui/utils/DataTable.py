import moose
import re
       

class DataTable():
    
    def __init__(self, dataRoot='/data'):
        self._recordDict = {}
        self._reverseDict = {}
        self._dataRoot = dataRoot

    #Harsha: Moving this fun from default to utils 
    #def createRecordingTable(element, field, _recordDict, _reverseDict,dataRoot='/data'):
    def create(self, element,field):
        """Create table to record `field` from element `element`

        Tables are created under `dataRoot`, the names are generally
        created by removing `/model` in the beginning of `elementPath`
        and replacing `/` with `_`. If this conflicts with an existing
        table, the id value of the target element (elementPath) is
        appended to the name.

        """
        
        if len(field) == 0 or ((element, field) in self._recordDict):            
            return
        # The table path is not foolproof - conflict is
        # possible: e.g. /model/test_object and
        # /model/test/object will map to same table. So we
        # check for existing table without element field
        # path in recording dict.
        relativePath = element.path.partition('/model[0]/')[-1]
        if relativePath.startswith('/'):
            relativePath = relativePath[1:]
        #Convert to camelcase
        if field == "concInit":
            field = "ConcInit"
        elif field == "conc":
            field = "Conc"
        elif field == "nInit":
            field = "NInit"
        elif field == "n":
            field = "N"
        elif field == "volume":
            field = "Volume"
        elif field == "diffConst":
            field ="DiffConst"
        tablePath =  relativePath.replace('/', '_') + '.' + field
        tablePath = re.sub('.', lambda m: {'[':'_', ']':'_'}.get(m.group(), m.group()),tablePath)
        tablePath = self._dataRoot + '/' +tablePath
        if moose.exists(tablePath):
            tablePath = '%s_%d' % (tablePath, element.getId().value)
        if not moose.exists(tablePath):
            #table = moose.Table(tablePath)
            #harsha: Instead of fullpath, just poolname is sent
            table = moose.Table(element.name)
            print 'Created', table.path, 'for plotting', '%s.%s' % (element.path, field)
            target = element
            moose.connect(table, 'requestOut', target, 'get%s' % (field))
            self._recordDict[(target, field)] = table
            self._reverseDict[table] = (target, field)
        return 