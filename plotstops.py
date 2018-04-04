import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from pathlib import Path
import csv
import math

options = sys.argv[1:]

printProgress = False
plotTram = False
plotBus = False
plotTrain = False
showVis = False
addLabels = True
doExport = True
shortRun = False
shortRunLength = 300
showCBDBounds = False

decPl = 3
distTol = 0.003
routeSplitter = " • "
mpl = 6 #max routes to show per line in label

bgColor = '#151313'


colourDict = {"bus": "#f58220", "tram": "#72bf44", "train": "#008dd0"}
dt = np.dtype("(2,2)f8")

cbdBounds = ((144.943244,-37.805101),(144.976522,-37.822414))

terminusInfo = np.dtype("i2,U256,f8,f8,U128")
terminusLabels = np.zeros(5, dtype=terminusInfo)

if '--help' in options or len(options) == 0 or np.in1d(options,['-t','-lr','-b']).all():
  addLabels = False
  doExport = False
  if np.in1d(options,['-t','-lr','-b']).all():
    print("You must specify a transit type to plot.")
  else:
    print("Expected usage:")
  f = Path('guide.txt')
  if f.is_file():
    f = open('guide.txt','r') 
    print(f.read())
if '-prog' in options:
  printProgress = True
  options.remove('-prog')
  
if '-t' in options:
  plotTrain = True
  #options.remove('-t')
  
if '-lr' in options:
  plotTram = True
  #options.remove('-lr')
  
if '-b' in options:
  plotBus = True
  #options.remove('-b')
  
if '-v' in options:
  showVis = True
  #options.remove('-v')
  
if '-nl' in options:
  addLabels = False
  #options.remove('-nl')
  
if '-ne' in options:
  doExport = False
  #options.remove('-ne')

if '-sr' in options:
  shortRun = True
  #options.remove('-sr')

if '-scb' in options:
  showCBDBounds = True
  #options.remove('-scb')
  
tramfile = "data/tramstops.txt"
trainfile = "data/trainstops.txt"
tramshapefile = "data/tramshapes.txt"
trainshapefile = "data/trainshapes.txt"
busshapefile = "data/busshapes.txt"
busshapefile2 = "data/busshapes2.txt"

def printProg(input, doReturn=True):
  if printProgress:
    if doReturn:
      print(input)
    else:
      print(input, end='', flush=True)

def removeQuotes(input):
  trimmedString = list(str(input))[3:-2]
  return ''.join(trimmedString)

def parseAsFloat(input):
  return float(removeQuotes(input))

def parseAsInt(input):
  return int(parseAsFloat(input))

def dist(pointAx, pointAy, pointBx, pointBy):
  return math.hypot(pointAx - pointBx, pointAy - pointBy)

def readStationName(input):
  returnString = ''.join(list(str(input))[2:-2])
  returnString = returnString.split(" Railway")
  returnString = returnString[:-1]
  return ''.join(returnString).upper()
def getRoute(input):
  return str(input.split('-')[1])

def getTransitType(input):
  return str(input.split('-')[0])


def addNewlines(str):
  splitStr = str.split(routeSplitter)
  outputStr = ''
  
  for i in range(0,len(splitStr)):
    if i != 0:
      if len(splitStr)%mpl == 1 and len(splitStr) <= mpl*(mpl-2)+1: #e.g. if mpl=5, we get no more than 15+1 items, which is 3 rows plus one more
        if i%(mpl+1) == 0:
          outputStr = outputStr + "\n"
        else:
          outputStr = outputStr + routeSplitter
      else:
        if i%mpl == 0:
          outputStr = outputStr + "\n"
        else:
          outputStr = outputStr + routeSplitter

    outputStr = outputStr + splitStr[i]
  return outputStr


#terminusLabels[2] = np.array((2, '', 144.951994338999, -37.8186224519219, ''), dtype=terminusInfo,ndmin=1)[0]
#terminusLabels[3] = np.array((2, '', 144.966964346166, -37.8183051340585, ''), dtype=terminusInfo,ndmin=1)[0]
#terminusLabels[4] = np.array((2, '', 144.972910916416, -37.8110540555305, ''), dtype=terminusInfo,ndmin=1)[0]
terminusLabels[2] = np.array((2, '', 144.0,-37.0, ''), dtype=terminusInfo,ndmin=1)[0]
terminusLabels[3] = np.array((2, '', 144.0,-37.0, ''), dtype=terminusInfo,ndmin=1)[0]
terminusLabels[4] = np.array((2, '', 144.0,-37.0, ''), dtype=terminusInfo,ndmin=1)[0]

def submitLabel(ttype, routename, lon, lat, info):
  candidate = np.array((ttype, routename, lon, lat, info), dtype=terminusInfo,ndmin=1)[0]
  preexistingCoords = False
  preexistingLabel = False
  checkCount = 0
  
  #define a modifier for labels in the CBD, beacuse sometimes need to reduce decPlc or distTol (i.e. for high density areas)
  #this will reset every call
  cbdMod = 1.0
  if candidate[2] > cbdBounds[0][0] and candidate[2] < cbdBounds[1][0] and candidate[3] < cbdBounds[0][1] and candidate[3] > cbdBounds[1][1]:
    candidate[4] = "cbd"
    cbdMod = 5
  
  global terminusLabels
  while preexistingCoords == False and preexistingLabel == False and checkCount < len(terminusLabels):
    point = terminusLabels[checkCount]
    
    if point[0] == candidate[0]:
      localDecPl = int(decPl*cbdMod)
      if round(point[2],localDecPl) == round(candidate[2],localDecPl) and round(point[3],localDecPl) == round(candidate[3],localDecPl):
        if point[1] == candidate[1]:
          preexistingLabel = True
        elif candidate[1] not in terminusLabels[checkCount][1].split(routeSplitter):
          terminusLabels[checkCount][1] = terminusLabels[checkCount][1] + routeSplitter + candidate[1] if terminusLabels[checkCount][1] != "" else ""
        preexistingCoords = True
      #if the lat and long for both existing point and current candidate are the same

      if dist(point[2],point[3],candidate[2],candidate[3]) < distTol/cbdMod:
        #else they're not the same, check if the the candidate's route is already specified in the current label's route list
        if candidate[1] not in terminusLabels[checkCount][1].split(" • "):
          terminusLabels[checkCount][1] = terminusLabels[checkCount][1] + " • " + candidate[1] if terminusLabels[checkCount][1] != "" else ""
        preexistingCoords = True
    
    checkCount += 1
      
  if not preexistingCoords and not preexistingLabel:
    terminusLabels = np.append(terminusLabels, candidate)

fig, ax = plt.subplots(figsize=(70,70),dpi=300)

if showCBDBounds:
  cbdBoundsLat = [144.943244, 144.976522, 144.976522, 144.943244,144.943244]
  cbdBoundsLon = [-37.805101,-37.805101,-37.822414,-37.822414,-37.805101]
  ax.plot(cbdBoundsLat,cbdBoundsLon,linewidth=0.25,color='#FFFFFF',linestyle='--')

if plotBus:
  printProg('Reading bus data... ', False)
  busshapedata = np.loadtxt(busshapefile, 
                  skiprows=1,
                  delimiter=',',
                  converters={0:removeQuotes,1:parseAsFloat,2:parseAsFloat,3:parseAsInt},
                  usecols=(0,1,2,3),
                  dtype={"names": ("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"), 
                         "formats": ('U64', 'f8', 'f8', 'i2')})
  busshapedata2 = np.loadtxt(busshapefile2, 
                  skiprows=1,
                  delimiter=',',
                  converters={0:removeQuotes,1:parseAsFloat,2:parseAsFloat,3:parseAsInt},
                  usecols=(0,1,2,3),
                  dtype={"names": ("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"), 
                         "formats": ('U64', 'f8', 'f8', 'i2')})
  busshapedata = np.append(busshapedata, busshapedata2)
  
  printProg('plotting bus... ', False)
  subsetStart = 0; subsetFinish = 0
  numPtSeq = np.where(busshapedata[:]['shape_pt_sequence'] == 1)[0]

  runLength = len(numPtSeq)
  
  if shortRun:
    runLength = shortRunLength
  
  for k in range(0,runLength):
    subSetStart = numPtSeq[k]
    subSetFinish = numPtSeq[k+1] if k != len(numPtSeq)-1 else len(busshapedata)
    #print(i, ": ", subSetStart, ",", subSetFinish)

    lonSubSet = busshapedata[subSetStart:subSetFinish]['shape_pt_lon']
    latSubSet = busshapedata[subSetStart:subSetFinish]['shape_pt_lat']
    ax.plot(lonSubSet,latSubSet,linewidth=0.25,color=colourDict['bus'],linestyle='solid')
    
    if addLabels:
      submitLabel(getTransitType(busshapedata[subSetStart]['shape_id']), 
                  getRoute(busshapedata[subSetStart]['shape_id']), 
                  busshapedata[subSetStart]['shape_pt_lon'],
                  busshapedata[subSetStart]['shape_pt_lat'],"")
      submitLabel(getTransitType(busshapedata[subSetFinish-1]['shape_id']), 
                  getRoute(busshapedata[subSetFinish-1]['shape_id']), 
                  busshapedata[subSetFinish-1]['shape_pt_lon'],
                  busshapedata[subSetFinish-1]['shape_pt_lat'],"")

  printProg("done")
  

  
if plotTram:
  printProg('Reading tram data... ', False)
  tramshapedata = np.loadtxt(tramshapefile, 
                  skiprows=1,
                  delimiter=',',
                  converters={0:removeQuotes,1:parseAsFloat,2:parseAsFloat,3:parseAsInt},
                  usecols=(0,1,2,3),
                  dtype={"names": ("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"), 
                         "formats": ('U64', 'f8', 'f8', 'i2')})
  
  printProg('plotting tram... ', False)

  subsetStart = 0; subsetFinish = 0
  numPtSeq = np.where(tramshapedata[:]['shape_pt_sequence'] == 1)[0]

  runLength = len(numPtSeq)
  
  if shortRun:
    runLength = shortRunLength
  
  for j in range(0,runLength):
    subSetStart = numPtSeq[j]
    subSetFinish = numPtSeq[j+1] if j != len(numPtSeq)-1 else len(tramshapedata)
    #print(i, ": ", subSetStart, ",", subSetFinish)

    lonSubSet = tramshapedata[subSetStart:subSetFinish]['shape_pt_lon']
    latSubSet = tramshapedata[subSetStart:subSetFinish]['shape_pt_lat']
    ax.plot(lonSubSet,latSubSet,linewidth=0.5,color=colourDict['tram'],linestyle='solid')
    
    if addLabels:
        submitLabel(getTransitType(tramshapedata[subSetStart]['shape_id']), 
                    getRoute(tramshapedata[subSetStart]['shape_id']), 
                    tramshapedata[subSetStart]['shape_pt_lon'],
                    tramshapedata[subSetStart]['shape_pt_lat'],"")
        submitLabel(getTransitType(tramshapedata[subSetFinish-1]['shape_id']), 
                    getRoute(tramshapedata[subSetFinish-1]['shape_id']), 
                    tramshapedata[subSetFinish-1]['shape_pt_lon'],
                    tramshapedata[subSetFinish-1]['shape_pt_lat'],"")

  printProg("done")

  
if plotTrain:
  printProg('Reading train data... ', False)
  traindata = np.loadtxt(trainfile, 
                  skiprows=1,
                  delimiter=',',
                  converters={1:readStationName},
                  dtype={"names": ("stop_id","stop_name","stop_lat","stop_lon"), 
                         "formats": ('i4', 'U64', 'f8', 'f8')})

  trainshapedata = np.loadtxt(trainshapefile, 
                  skiprows=1,
                  delimiter=',',
                  converters={0:removeQuotes,1:parseAsFloat,2:parseAsFloat,3:parseAsInt},
                  usecols=(0,1,2,3),
                  dtype={"names": ("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"), 
                         "formats": ('U64', 'f8', 'f8', 'i2')})
  
  printProg('plotting train... ', False)

  subsetStart = 0; subsetFinish = 0
  numPtSeq = np.where(trainshapedata[:]['shape_pt_sequence'] == 1)[0]
  runLength = len(numPtSeq)
  
  if shortRun:
    runLength = shortRunLength
  
  for i in range(0,runLength):
    subSetStart = numPtSeq[i]
    subSetFinish = numPtSeq[i+1] if i != len(numPtSeq)-1 else len(trainshapedata)
    #print(i, ": ", subSetStart, ",", subSetFinish)

    lonSubSet = trainshapedata[subSetStart:subSetFinish]['shape_pt_lon']
    latSubSet = trainshapedata[subSetStart:subSetFinish]['shape_pt_lat']
    ax.plot(lonSubSet,latSubSet,linewidth=0.75,color=colourDict['train'],linestyle='solid')
    
    if addLabels:
      submitLabel(getTransitType(trainshapedata[subSetStart]['shape_id']), 
                  getRoute(trainshapedata[subSetStart]['shape_id']), 
                  trainshapedata[subSetStart]['shape_pt_lon'],
                  trainshapedata[subSetStart]['shape_pt_lat'],"")
      submitLabel(getTransitType(trainshapedata[subSetFinish-1]['shape_id']), 
                  getRoute(trainshapedata[subSetFinish-1]['shape_id']), 
                  trainshapedata[subSetFinish-1]['shape_pt_lon'],
                  trainshapedata[subSetFinish-1]['shape_pt_lat'],"")

  ax.plot(traindata['stop_lon'], traindata['stop_lat'], markersize='4.0',color=bgColor,marker='.',ls='none',zorder=11.0)
  ax.plot(traindata['stop_lon'], traindata['stop_lat'], markersize='3.0',color=colourDict['train'],marker='.',ls='none',zorder=12.0)

  printProg("done")
  
#ax.plot(tramdata['stop_lon'], tramdata['stop_lat'], 'g.', markersize='2.0')

ax.set(xlabel='long', ylabel='lat',title='Tram/Train Network')
ax.set_aspect('equal')
ax.set_facecolor(bgColor)
#ax.invert_yaxis()
if addLabels:
  printProg("Adding labels... ", False)
  terminusLabels.sort(order=['f3'])
  printProg("count: " + str(len(terminusLabels)) + "...", False)
  
  #z orders:
    #2 for train labels
    #4 for tram labels
    #6 for bus labels
    #8 for bus points
    #10 for tram points
    #12 for train points
    
  for l in range(0,len(terminusLabels)):
    terminusLabels[l]['f1'] = addNewlines(terminusLabels[l]['f1'])
    
  ttypeLabels = terminusLabels
  
  if plotTrain:
    
    for station in traindata:
      ax.annotate(station[1],xy=(station['stop_lon'], station['stop_lat']),xytext=[2,-4.0],textcoords='offset pixels',color=colourDict['train'], bbox=dict(boxstyle='square,pad=0.4', fc=bgColor, ec=colourDict['train'], lw=0.25),size=2,zorder=2.0)    
    
    ttypeLabels = terminusLabels[terminusLabels['f0'] == 2]
    
    for label in ttypeLabels:
      ax.annotate(label[1],xy=(label[2],label[3]),xytext=[1.75,-7],textcoords='offset pixels',color=bgColor, bbox=dict(boxstyle='square,pad=0.4', fc=colourDict['train'], ec='none'),size=1.5,zorder=2.0)
  
  if plotTram:
    ttypeLabels = terminusLabels[terminusLabels['f0'] == 3]
    for label in ttypeLabels:
      if label[4] == 'cbd':
        ax.annotate(label[1],xy=(label[2],label[3]),xytext=[1.5,1.5],textcoords='offset pixels',color=bgColor, bbox=dict(boxstyle='square,pad=0.4', fc=colourDict['tram'], ec='none'),size=1.5,zorder=4.0)
      else:
        ax.annotate(label[1],xy=(label[2],label[3]),xytext=[2.0,2.0],textcoords='offset pixels',color=bgColor, bbox=dict(boxstyle='square,pad=0.4', fc=colourDict['tram'], ec='none'),size=2,zorder=4.0)
    ax.plot(ttypeLabels['f2'],ttypeLabels['f3'], markersize='3',color=bgColor,marker='.',ls='none',zorder=9.0)
    ax.plot(ttypeLabels['f2'],ttypeLabels['f3'], markersize='2.0',color=colourDict['tram'],marker='.',ls='none',zorder=10.0)
  
  if plotBus:
    ttypeLabels = terminusLabels[terminusLabels['f0'] == 4]
    for label in ttypeLabels:
      if label[4] == 'cbd':
        ax.annotate(label[1],xy=(label[2],label[3]),xytext=[-1.5,1.5],textcoords='offset pixels',color=bgColor, bbox=dict(boxstyle='square,pad=0.4', fc=colourDict['bus'], ec='none'),size=1.5,ha='right',zorder=6.0)
      else:
        ax.annotate(label[1],xy=(label[2],label[3]),xytext=[-2.0,2.0],textcoords='offset pixels',color=bgColor, bbox=dict(boxstyle='square,pad=0.4', fc=colourDict['bus'], ec='none'),size=2,ha='right',zorder=6.0)
    
    ax.plot(ttypeLabels['f2'],ttypeLabels['f3'], markersize='1.1',color=bgColor,marker='o',ls='none',zorder=7.0)
    ax.plot(ttypeLabels['f2'],ttypeLabels['f3'], markersize='1',color=colourDict['bus'],marker='.',ls='none',zorder=8.0)
  
  
#  for label in terminusLabels:
#    if label[0] == 2: #train
#      ax.annotate(label[1],xy=(label[2],label[3]),xytext=[3.0,-5.0],textcoords='offset pixels',color=bgColor, backgroundColor=colourDict['train'],size=3,zorder=8.0)
#    if label[0] == 3: #tram
#      ax.annotate(label[1],xy=(label[2],label[3]),xytext=[0.0,0.0],textcoords='offset pixels',color=bgColor, backgroundColor=colourDict['tram'],size=2,zorder=9.0)
##      ax.plot(label[2],label[3], markersize='4',color=bgColor,marker='o',ls='none',zorder=9.0)
##      ax.plot(label[2],label[3], markersize='2.0',color=colourDict['tram'],marker='.',ls='none',zorder=10.0)
#    if label[0] == 4: #bus
#      ax.annotate(label[1],xy=(label[2],label[3]),xytext=[-2.0,2.0],textcoords='offset pixels',color=bgColor, backgroundColor=colourDict['bus'],size=2,ha='right',zorder=10.0)
##      ax.plot(label[2],label[3], markersize='2.25',color=bgColor,marker='o',ls='none',zorder=7.0)
#      ax.plot(label[2],label[3], markersize='0.75',color=colourDict['bus'],marker='.',ls='none',zorder=8.0)
      
  printProg("done")

if doExport:
  printProg('Exporting diagram... ', False)
  
  options.sort()
  
  fig.savefig("exports/test" + ''.join(options) + ".svg", format='svg', bbox_inches='tight')
#  printProg('test.svg done... ', False)
#  fig.savefig("exports/test.png", format='png')
  printProg("done")
  
  np.savetxt("exports/labels" + ''.join(options) + ".csv", terminusLabels, delimiter=",", fmt='%i,%s,%f,%f,%s')

if showVis:
  printProg("Opening visualisation...", False)
  plt.show()
  printProg("done")
