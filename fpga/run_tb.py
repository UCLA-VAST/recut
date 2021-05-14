import sys
import os

layerIndex = int(sys.argv[1])

instFile = open('test.insts', 'r')
lines = instFile.readlines()

insts = []

for i in range(0,layerIndex):
  inst = []
  for line in range(0,6):
    instLine = lines[i*7+line].split()
    for num in instLine:
      inst.append(int(num))
  insts.append(inst)

instDicts = []

for inst in insts:
  instDict = {}
  instDict['IN_NUM_HW'	      ] = inst[0 ]
  instDict['OUT_NUM_HW'	      ] = inst[1 ]
  instDict['IN_H_HW'		      ] = inst[2 ]
  instDict['IN_W_HW'		      ] = inst[3 ]
  instDict['OUT_H_HW'		      ] = inst[4 ]
  instDict['OUT_W_HW'		      ] = inst[5 ]
  instDict['IN_NUM'			      ] = inst[6 ]
  instDict['OUT_NUM'		      ] = inst[7 ]
  instDict['IN_H'				      ] = inst[8 ]
  instDict['IN_W'				      ] = inst[9 ]
  instDict['OUT_H'			      ] = inst[10]
  instDict['OUT_W'			      ] = inst[11]
  instDict['CIN_OFFSET'	      ] = inst[12]
  instDict['WEIGHT_OFFSET'  	] = inst[13]
  instDict['BIAS_OFFSET'		  ] = inst[14]
  instDict['COUT_OFFSET'		  ] = inst[15]
  instDict['FILTER_S1'	    	] = inst[16]
  instDict['FILTER_S2'		    ] = inst[17]
  instDict['STRIDE'			      ] = inst[18]
  instDict['EN'				        ] = inst[19]
  instDict['PREV_CIN_OFFSET'	] = inst[20]
  instDict['IN_NUM_T'			    ] = inst[21]
  instDict['OUT_NUM_T'		    ] = inst[22]
  instDict['IN_H_T'			      ] = inst[23]
  instDict['IN_W_T'			      ] = inst[24]
  instDict['BATCH_NUM'		    ] = inst[25]
  instDict['TASK_NUM1'		    ] = inst[26]
  instDict['TASK_NUM2'		    ] = inst[27]
  instDict['LOCAL_ACCUM_NUM'	] = inst[28]
  instDict['LOCAL_REG_NUM'	  ] = inst[29]
  instDict['ROW_IL_FACTOR'	  ] = inst[30]
  instDict['COL_IL_FACTOR'	  ] = inst[31]
  instDict['CONV_TYPE'		    ] = inst[32]
  instDict['FILTER_D0'		    ] = inst[33]
  instDict['FILTER_D1'		    ] = inst[34]
  instDict['DILATION_RATE'	  ] = inst[35]
  instDict['TCONV_STRIDE'		  ] = inst[36]
  instDict['K_NUM'			      ] = inst[37]
  instDict['KH_KW'			      ] = inst[38]
  instDicts.append(instDict)

# for instDict in instDicts:
#   for key in instDict:
#     print(key)

instDict = instDicts[layerIndex-1]
# print(inst)
inFile = open("cnn_sw.h", "r")
outFile = open("temp.h", "w")
for line in inFile:
    values = line.split()
    if(len(values)!=0):
      if(values[0]=="#define" and values[1]=="LAYER"):
        outFile.writelines("#define LAYER "+str(layerIndex)+"\n")
      elif(values[0]=="#define" and values[1]=="OUTFILE"):
        outFile.writelines("#define OUTFILE \"/data/L"+str(layerIndex)+"_outputs.dat\"\n")
      elif(values[0]=="#define" and values[1]=="OUT_OFFSET1"):
        offset = instDict['COUT_OFFSET']-((instDict['OUT_W_HW']+1)*instDict['OUT_NUM_HW'])
        outFile.writelines("#define OUT_OFFSET1 "+str(offset)+"\n")
      elif(values[0]=="#define" and values[1]=="OUT_OFFSET2"):
        outFile.writelines("#define OUT_OFFSET2 "+str(instDict['COUT_OFFSET'])+"\n")
      elif(values[0]=="#define" and values[1]=="CHANGE_LAYOUT"):
        if(((instDict['OUT_W_HW'] == instDict['OUT_W']) or (instDict['OUT_W_HW'] == instDict['IN_W_T'])) and ((instDict['OUT_H_HW'] == instDict['OUT_H']) or (instDict['OUT_H_HW'] == instDict['IN_H_T']))):
          outFile.writelines("#define CHANGE_LAYOUT "+str(1)+"\n")
        else:
          outFile.writelines("#define CHANGE_LAYOUT "+str(0)+"\n")
      elif(values[0]=="#define"):
        for key in instDicts[0]:
          if(values[1]==key):
            outFile.writelines(values[0] + " " + values[1] + " " + str(instDict[values[1]])+"\n")
      else:
        outFile.writelines(line)
outFile.close()
inFile.close()
inFile = open("temp.h", "r")
outFile = open("cnn_sw.h", "w")
for line in inFile:
  outFile.writelines(line)
outFile.close()
inFile.close()

os.system("conda activate tf")
os.system("D:/Programs/Anaconda/envs/tf/python.exe d:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/UNET_tf/unet.py "+str(layerIndex))
os.system("UNET.bat sim")
