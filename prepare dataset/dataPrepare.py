import os

folderIN = '01-Data/'
folderInList = (os.listdir(folderIN))
folderOut = '03-Data/'

datasetName = 'ORUNADA'
features = ''
linhaInicio = ''

for folderYear in folderInList:
    folderYearsDay = (os.listdir(folderIN + folderYear))
    meses = ['01','02','03','04','05','06','07','08','09','10','11','12',]
    print(folderYear)
    os.system("mkdir -p " + folderOut + datasetName + '/' + folderYear)
    for mes in meses:
        os.system("mkdir " + folderOut + datasetName + '/' + folderYear + '/' + str(mes))
        foldermesDay = (os.listdir(folderIN + folderYear + '/' + mes))
        print(mes)
        linhaTotal = ''
        foldermesDay.sort()
        for arquivoIN in foldermesDay:
            print(arquivoIN)
            arq_write = open(folderOut + datasetName + '/' + folderYear + '/' + str(mes) + '/' + arquivoIN + '.csv', 'w')
            arq_write.write(features)
            with open(folderIN + folderYear + '/' + mes + '/' + arquivoIN) as f:
                for i, linha in enumerate(f):
                    if i >= linhaInicio:
                        linha = linha.replace('normal', '0')
                        linha = linha.replace('attack', '1')
                        linhaTotal = linhaTotal + linha

            arq_write.write(linhaTotal)
            arq_write.close()
            linhaTotal =''