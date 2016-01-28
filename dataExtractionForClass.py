#january 21th 2016
#this script constructs a Q matrix from a DNS log
#then it computes QQT, QTQ, Heuristic
#and format output for classification purposes


import time
import numpy, scipy.io
import sys
import datetime
import os
#import Image



def lookForIP(stringIP,ipDict,ipIndex):
    index = 0
   
    #global ipIndex, ipDict
    if stringIP not in ipDict:
        ipDict[stringIP] = ipIndex
        index = ipIndex
        ipIndex+=1
        #print "Sumando"
        #print ipIndex
        #exit()
    else:
        index = ipDict[stringIP]
    return index,ipIndex

def lookForDomain(stringDomain,domainDict,domainIndex):
    index = 0
    #global domainIndex, domainIndex
    if stringDomain not in domainDict:
        domainDict[stringDomain] = domainIndex
        index = domainIndex
        domainIndex+=1
    else:
        index = domainDict[stringDomain]
    return index,domainIndex

def segmentation(matrix,ipIndex,domainIndex):
    totalRows = ipIndex
    totalColumns = domainIndex
    
    squareCells = []
    reducedImage = zeroMatrix(ipIndex/2,domainIndex/2)

    for i in range(0,len(reducedImage)*2,2):
        for j in range(0,len(reducedImage[0])*2,2):
            #print "pivot"
            #print i,",",j
            squareCells.append([[i,j],[i,j+1],[i+1,j+1],[i+1,j]])

    tempList = []
    for squareCell in squareCells:
        squareAdd = 0
        for cellIndexes in squareCell:
            #print matrix[cellIndexes[0],cellIndexes[1]]
            #print matrix[cellIndexes[0]][cellIndexes[1]]
            if matrix[cellIndexes[0]][cellIndexes[1]] > 0:
                squareAdd+=1
        if squareAdd >= 3: # if square is 50% of ones flag square as 1
            #print "group"
            tempList.append(1)
        else:
            #print "not a group"
            tempList.append(0)

    tempIndex = 0
    for i in range(0,len(reducedImage)):
        for j in range(0,len(reducedImage[0])):
            reducedImage[i][j] = tempList[tempIndex]
            tempIndex+=1

    #printMatrix(reducedImage)
    vectorOfGroups = {'2,2':0,'2,4':0,'4,2':0,'4,4':0,'6,2':0,'6,4':0,'EE':0}
    if len(reducedImage) > 1 and len(reducedImage[0]) > 1:
        # reducedImage =  [[0,1,0,0,1,0],
        #                  [0,1,1,1,1,0],
        #                  [0,0,1,1,0,0],
        #                  [1,1,1,1,1,0],
        #                  [0,0,0,0,1,0],
        #                  [0,1,0,0,0,0]]
        # reducedImage =  [[0,1,1],
        #                  [1,0,0],
        #                  [0,1,0]]
        # reducedImage =  [[1,1,1,1,1,1],
        #                  [1,0,0,0,0,1],
        #                  [1,0,0,0,0,1],
        #                  [1,0,1,0,0,1],
        #                  [1,0,0,0,0,1],
        #                  [1,1,1,1,1,1]]
        # reducedImage =  [[0,0,0,0,0],
        #                  [0,0,0,0,1],
        #                  [0,1,0,0,1],
        #                  [0,0,0,0,1],
        #                  [0,0,0,0,0]]
        # reducedImage =  [[1,0,0,0,0,1],
        #                 [0,1,0,0,1,0],
        #                 [0,0,1,1,0,0],
        #                 [0,0,1,1,0,0],
        #                 [0,1,0,0,1,0],
        #                 [1,0,0,0,0,1]]
        # reducedImage =  [[1,0,0,0,0,0],
        #                 [0,0,0,0,0,0],
        #                 [0,0,0,0,0,0],
        #                 [0,0,0,0,0,0],
        #                 [0,0,0,0,0,0],
        #                 [0,0,0,0,0,0]]
        # reducedImage =  [[1,1,1],
        #                 [1,1,1],
        #                 [1,1,1]]

        # reducedImage =  [[1,0,0,0,0,1],
        #                 [0,0,0,0,0,0],
        #                 [0,1,1,1,1,0],
        #                 [0,1,0,0,1,1],
        #                 [0,1,0,0,1,0],
        #                 [0,1,1,1,1,0]]

        #printMatrix(reducedImage)
        transposedImg = copyImage(reducedImage) ##COPIA DE LISTA PENDIENTE
        transposedImg = numpy.transpose(reducedImage)
        
        #printMatrix(transposedImg)
        #exit()

        exitCondition = 0
        while exitCondition ==0:
            s = 0
            for i in range(len(reducedImage)):
                for j in range(len(reducedImage[0])):
                    s+=reducedImage[i][j]
            if s ==0:
                exitCondition = 1
                ####print "<<<<<<<<< NO MORE GROUPS >>>>>"
            else:
                splitRegions(reducedImage,vectorOfGroups)
        exitCondition = 0
        while exitCondition ==0:
            s = 0
            for i in range(len(transposedImg)):
                for j in range(len(transposedImg[0])):
                    s+=transposedImg[i][j]
            if s ==0:
                exitCondition = 1
                ####print "<<<<<<<<< NO MORE GROUPS >>>>>"
            else:
                splitRegions(transposedImg,vectorOfGroups,1)
    else:
        ###print "skipping window... "
        pass
#    printMatrix(reducedImage)
    return vectorOfGroups

def copyImage(img):
    if img == []:
        return []
    imgCopy = []
    for row in img:
        imgCopy.append(list(row))
    return imgCopy

def splitRegions(img,vectorOfGroups,op=0):
    ###print "Spliting regions..."  
    groupsInImg = []
    i = 0
    equalPoints = 0
    pastPoints = [],[]
    formingGroup = []
    #vectorOfGroups = {'2,2':0,'2,4':0,'4,2':0,'4,4':0,'6,2':0,'6,4':0,'EE':0}
    while i < len(img):
        ###print "Input-->"
        ###printMatrix(img)
        ###print "i=    ", i 
        currPoints = lookForFirstPoints(img,i,0)
        ###print pastPoints
        ###print currPoints
        ###print equalPoints
        ###print "."
        ###print "-***-*-*-*-*-*-**"
        ###print currPoints
        ###print "-***-*-*-*-*-*-**"
        if currPoints == ([],[]):
            break
        i = currPoints[0][0]

        if equalPoints == 1:
            #past point ([0,3],[0,3]) 
            #curr point ([1,0],[1,1])
            if currPoints[0][1] == pastPoints[0][1] and currPoints[1][1] == pastPoints[1][1]:
                if abs(currPoints[0][0]-pastPoints[0][0])>1:
                    ###print "Nope, almost but..not same group"
                    groupDimension(formingGroup,vectorOfGroups,op)
                    formingGroup = []
                else:
                    ###print "SStill the same group"
                    pass
                formingGroup.append([currPoints[0],currPoints[1]])
                pastPoints = currPoints
            else:
                #not the same group
                tempPoints = lookForFirstPoints(img,i,0)
                ###print tempPoints
                if tempPoints == ([],[]):
                    #formingGroup.append([currPoints[0],currPoints[1]])
                    groupDimension(formingGroup,vectorOfGroups,op)
                    formingGroup = []
                    formingGroup.append([currPoints[0],currPoints[1]])
                    break
                if tempPoints[0][1] == pastPoints[0][1] and tempPoints[1][1] == pastPoints[1][1]:
                    #volver a calcular puntos a partir de abajo
                    #si no es el mismo grupo
                    #rellenar los puntos de abajo,
                    #el temporal
                    #agregar curr y past como dos grupos 
                    #difrentes
                    ###print "-->Still the same group"
                    ###print "<<<<<<<<<<<<<<<",tempPoints
                    ###print "<<<<<<<<<<<<<<<",pastPoints
                    formingGroup.append([currPoints[0],currPoints[1]])
                    pastPoints = tempPoints
                    i = tempPoints[0][0]+1
                    ###print "******",i,"*******"
                    #fill again the image...
                    fillPoints(img,currPoints)
                    #img[currPoints[0]][currPoints[1]] = 1
                    #
                    continue #to start againt the iteration
                else:
                    ###print "not the same temporal points"
                    #so i have to fill the image again...
                    fillPoints(img,tempPoints)

                ###print "Not the same groups as previous>>"
                ###print "previous group",formingGroup
                equalPoints = 0
                #i = 0 #reseting group search
                #determine size and weight and add it to some list
                groupDimension(formingGroup,vectorOfGroups,op)
                formingGroup = []
                #formingGroup.append([currPoints[0],currPoints[1]])
                pastPoints = currPoints
        if currPoints[0] == currPoints[1] and currPoints[0] != [] and equalPoints ==0:
            if pastPoints[0] != [] and pastPoints[1] !=[] and formingGroup != []:
                if pastPoints[0][1] != currPoints [0][1] or pastPoints[1][1] != currPoints[1][1]:
                    ###print ">>not the same groups as previous"
                    ###print "previous group",formingGroup
                    #i = 0 #resiting group search
                    #determine size and weight and add it to some list
                    groupDimension(formingGroup,vectorOfGroups,op)
                    formingGroup = []
                    #formingGroup.append([currPoints[0],currPoints[1]])
            equalPoints = 1
            formingGroup.append([currPoints[0],currPoints[1]])
            #formingGroup.append([currPoints[0],currPoints[1]])
            pastPoints = currPoints

        if equalPoints == 0 and currPoints != ([],[]):
            ###print "theres is a group somewhere..."
            ###print currPoints
            ###print pastPoints
            if pastPoints != ([],[]):
                #print "No previous point...so perhaps "
                #print "This is first row"
                if pastPoints[0][1] == currPoints [0][1] and pastPoints[1][1] == currPoints[1][1]:
                    ###print "Still the same group..."    
                    formingGroup.append([currPoints[0],currPoints[1]])
                    ###print "----------"
                    ###print formingGroup
                    ###print i
                    ###print "**********"
                else:
                    ###print "No longer the same group"
                    #i = 0 #resiting group search
                    #determine size and weight and add it to some list

                    groupDimension(formingGroup,vectorOfGroups,op)
                    formingGroup = []
                    #
                    #fillPoints(img,pastPoints)
                    #fillPoints(img,currPoints)

                    formingGroup.append([currPoints[0],currPoints[1]])
            elif currPoints != ([],[]): #must be the first row
                formingGroup.append([currPoints[0],currPoints[1]])
            pastPoints = currPoints
        i+=1
    if formingGroup != []:
        ###print "At the end of the analysis there still a group..."
        groupDimension(formingGroup,vectorOfGroups,op)
        formingGroup = []


    ###print "------\n"
    ###print "Analysis concluded... groups in the window are: "
    ###for key in vectorOfGroups:
    ###    print key,vectorOfGroups[key]

    #print vectorOfGroups
def fillPoints(img,points):
    ###print "filling..."
    ###print points
    ###print img
    if points[0] == points[1]:
        img[points[0][0]][points[0][1]] = 1
    else:
        for i in range(points[0][0],points[1][0]+1):
            for j in range(points[0][1],points[1][1]+1):
                img[i][j] = 1
    ###print img
    #exit()
    
def groupDimension(g,v,op=0):
    ###print "Now determining structure of group...", g
    ###print "\n"
    ###print "******Size-->", abs(g[0][0][1]-g[0][1][1])*2+2
    ###print "******Weight-->", len(g)*2
    ###print "\n"
    if op==0:
        key = str(abs(g[0][0][1]-g[0][1][1])*2+2)+","+str(len(g)*2)
    else:
        key = str(len(g)*2)+","+str(abs(g[0][0][1]-g[0][1][1])*2+2)


    if key not in v:
        v["EE"]+=1
    else:
        v[key]+=1

def lookForFirstPoints(img,start,columnStart):
    firstPoint = []
    lastPoint = []  
    i=start
    while i < len(img) and firstPoint == []:
        j=columnStart
        while j < len(img[0]):
            if img[i][j] !=0 and firstPoint == []:
                firstPoint = [i,j]
                ###print "Found first point at: ", firstPoint
                lastPoint = [i,j]
                img[i][j] = 0 # putting black in img
            elif firstPoint != [] and img[i][j] ==0:
                ###print "End of the group..."
                break
            elif img[i][j] != 0:
                img[i][j] = 0
                lastPoint = [i,j]
            j+=1
        i+=1
    if firstPoint == []: # means that no info of grp found
        ###print "No groups in that part of image..."
        return [],[]
    else:
        return firstPoint,lastPoint
def printMatrix(m):
    out = ""
    for i in range(0,len(m)):
        for j in range(0,len(m[0])):
            out = out+str(m[i][j]) + "   "
        out = out+"\n"
    print out

def zeroMatrix(rows,columns):
    matrix = []
    tempRow = []
    for i in range(0,rows):
        for j in range(0,columns):
            tempRow.append(0)
        matrix.append(tempRow)
        tempRow=[]
    return matrix





def initProgram(logFile,windowSize,typeOfLog,directory,windowCounter):
    #opening file
    ipIndex = 0 # number of unique IP addresses
    domainIndex = 0 #number of unique domain addresses
    _output = ""

    ipDict = {}
    domainDict = {}
    activity = []
    binaryActivity = [] # binary activity (1 -> visited) (0-> not visited)
    sortedActivity = []
    matVarIndex = 0
    matVarDict = {}


    probIP = {}
    probDomain = {}

    f = open(logFile,'r')
    #f = open("/Volumes/HD2/Scripts/MatricesQ/out16feb50.txt",'r')
    #directory = "outputWindowSize"+str(windowSize)
    directory+=str(windowSize)
    if typeOfLog == '-1':
        directory = "AttackWindowSize"+str(windowSize)
    if not os.path.exists(directory):
        os.makedirs(directory)

    outputFile = open(directory+"/_out"+logFile.split("/")[-1][:-4]+str(windowSize)+".out","w")
    packetCounter = 0

    for line in f:
        #print line
        _output = ""
        packetCounter+=1
        data = str(line).split(' ') # splitting a log line
        
        hour = datetime.datetime.strptime(str(data[1])[:-4], '%H:%M:%S').time()
        minHour = datetime.datetime.strptime('09:00:00', '%H:%M:%S').time()
        maxHour = datetime.datetime.strptime('21:00:00', '%H:%M:%S').time()
        
        if not(hour >=minHour and hour <=maxHour):
            if hour >= maxHour:
                break
            continue
        ipPort = str(data[3]).split('#') #splitting port and ip

        ip,ipIndex = lookForIP(ipPort[0],ipDict,ipIndex) #getting ip address
        dominio,domainIndex = lookForDomain(data[7],domainDict,domainIndex) #getting domain name
        
        #print ip
        #print dominio

        if ip not in probIP:
            probIP[ip] = 1.0
        else:
            probIP[ip] += 1.0

        if dominio not in probDomain:
            probDomain[dominio] = 1.0
        else:
            probDomain[dominio] +=1.0

        activity.append([ip,dominio]) #activity without sorting
        #activity.sort()
        if [ip,dominio] not in binaryActivity:
            binaryActivity.append([ip,dominio])

        if(windowCounter % windowSize == 0):
            #some process
            array = numpy.zeros(shape=(ipIndex,domainIndex))
            binDict = {}
            sortedRows = []        
            newIndexList = []
            newRowIndex = 0

            #creating a binary Q matrix
            for query in binaryActivity: # query is a relation <IP,domain>
                if query[0] not in binDict:
                    binDict[query[0]]=1
                else:
                    binDict[query[0]]+=1
            

            #sorting matrix row index by activity
            for w in sorted(binDict, key=binDict.get, reverse=True):
                #print w, binDict[w]
                sortedRows.append(w) #appending sorted row index into a list
                newIndexList.append(newRowIndex)
                newRowIndex+=1

            #adding sorted rows to a new list
            for r in activity:
                count =0
                for rowIndex in sortedRows:
                    if r[0] == rowIndex:
                        
                        rCopy = r
                        rCopy[0] = count#rowIndex
                        sortedActivity.append(rCopy)
                        break
                    else:
                        count+=1

            #sorting columns according to popularity
            binDict = {}
            sortedColumns = []        
            newIndexList = []
            newColumnIndex = 0

            sortedActivityTemp = [] #sorted Activity wrt columns and rows
            #creating a binary Q matrix
            for query in binaryActivity:
                #print query[1]
                if query[1] not in binDict:
                    binDict[query[1]]=1
                else:
                    binDict[query[1]]+=1
            
            #sorting matrix row index by activity
            for w in sorted(binDict, key=binDict.get, reverse=True):
                #print w, binDict[w]
                sortedColumns.append(w) #appending sorted row index into a list
                #newIndexList.append(newColumnIndex)
                #print w, "ahora es", newColumnIndex
                newColumnIndex+=1
                
            #adding sorted rows to a new list

            for r in sortedActivity:
                count =0
                for columnIndex in sortedColumns:
                    if r[1] == columnIndex: #if r column index equals to sorted col indexes 
                        
                        rCopy = r
                        rCopy[1] = count#columnIndex
                        sortedActivityTemp.append(rCopy)
                        break
                    else:
                        count+=1

            #filling activity of sorted Q matrxix

            for query in sortedActivityTemp:
                #array[query[0]][query[1]]+=1 #### This is important !! since in principle this  operation
                #puts together ALL the DNS request in the sorted binary matrix
                #so, even if activity of agt1 is 100 (to a single objet) and agt2 queries two objects
                # agt2 has more activity than agt1. This is because we are talking about a 'binary sorting'

                array[query[0]][query[1]]=1
            ## HEURISTIC
            totalGroups = segmentation(array,ipIndex,domainIndex)
            ###WINDOW FEATURES
            H_obj = 0
            H_agt = 0

            for k in probDomain:
                H_obj+=(probDomain[k]/windowSize)*numpy.log2((probDomain[k]/windowSize))
            
            for k in probIP:
                H_agt+=(probIP[k]/windowSize)*numpy.log2((probIP[k])/windowSize)
            H_obj*=-1
            H_agt*=-1
         
            _output+=str(packetCounter)+","+str(ipIndex)+","+str(domainIndex)+","+str(H_agt)+","+str(H_obj)+","+str(len(binaryActivity))+","+str((ipIndex*domainIndex)/float(windowSize))+","+str(float(windowSize)/(ipIndex*domainIndex))+","

                   
            QQT = numpy.dot(array,numpy.transpose(array))

            weight_two = 0
            topActAgt = 0
            maximalSize=1
            
            for i in range(1,len(array)): # sliding through diagonals
                weight_two+=numpy.count_nonzero(QQT.diagonal(i) >= 2) #total elements nonzero from diagonal(i)
                if QQT.diagonal(i).max() > maximalSize:
                    maximalSize =QQT.diagonal(i).max()
                    
            topActAgt = QQT.diagonal(0).max()
           
            ##Second QTQ
            QTQ = numpy.dot(numpy.transpose(array),array)
            
            size_two = 0
            topActObj = 0
            maximalWeight=1
            for i in range(1,len(array[0])): # sliding through diagonals
                
                size_two+=numpy.count_nonzero(QTQ.diagonal(i) >= 2) #total elements nonzero from diagonal(i)
                if QTQ.diagonal(i).max() > maximalWeight:
                    maximalWeight =QTQ.diagonal(i).max()
                    
            topActObj = QTQ.diagonal(0).max()

            _output+= str(topActAgt)+","+str(topActObj)+","+str(maximalSize)+","+str(maximalWeight)+","+str(weight_two)+","+str(size_two)+","

            _output+= str(totalGroups['2,2'])+","+str(totalGroups['2,4'])+","+str(totalGroups['4,2'])+","+str(totalGroups['4,4'])+","+str(totalGroups['6,2'])+","+str(totalGroups['6,4'])+","+str(totalGroups['EE'])

            #print _output
            outputFile.write(_output+","+typeOfLog)
            outputFile.write("\n")

            #exit()

            #variables reset
            array=0
            ipIndex = 0
            domainIndex = 0
            ipDict = {}
            domainDict = {}
            activity = []
            sortedActivity = []
            binaryActivity = []
            probDomain = {}
            probIP = {}
            _output = ""

        windowCounter+=1

    f.close()
    outputFile.close()



if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit('Usage: %s    log-file    windowSize      Attack(-1)/Normal(1)    output-directory' % sys.argv[0])

    logFile = str(sys.argv[1])
    windowSize = int(sys.argv[2])
    typeOfLog = str(sys.argv[3])
    directory = str(sys.argv[4])
    windowCounter = 1

    initProgram(logFile,windowSize,typeOfLog,directory,windowCounter)


