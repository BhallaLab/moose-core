from suds.client import Client

biomodelsWSDL = 'http://www.ebi.ac.uk/biomodels-main/services/BioModelsWebServices?wsdl'
biomodelsClient = Client(biomodelsWSDL)
print biomodelsClient
result = biomodelsClient.service.getAllCuratedModelsId()
for value in result:
    name = biomodelsClient.service.getModelNameById(value)
    print name
result = biomodelsClient.service.getSimpleModelById('BIOMD0000000001')
print result
