#from ROOT import gROOT, TCanvas, TF1, TFile, TTree, gRandom, TH1F, TLorentzVector
from array import array
import math
#from ROOT import TPad, TPaveText, TLegend, THStack
#from ROOT import gBenchmark, gStyle, gROOT, gPad
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.legend_handler import HandlerLine2D

#gROOT.SetStyle("Plain")   # set plain TStyle
#gStyle.SetOptStat(0);
#gROOT.SetBatch(1)         #This was to suppress the canvas outputs
#gStyle.SetOptStat(111111) # draw statistics on plots,
                            # (0) for no output
#gStyle.SetPadTickY(1)
#gStyle.SetOptFit(1111)    # draw fit results on plot,
                            # (0) for no ouput
#gStyle.SetPalette(57)     # set color map
#gStyle.SetOptTitle(0)     # suppress title box

fileDir = "/scratch/murtazas/CS231N-data/inputArraysVSmallCR/"
#   /scratch/murtazas/CS231N-data/inputArraysVSmallCR
#   /nfs/slac/g/atlas/u02/murtazas/PileUPJetID/inputArraysVSmallCR/
plotDir = "/home/murtazas/CS231N-PileUp-Jet-ID/AllPlots/"
#   /home/murtazas/CS231N-PileUp-Jet-ID/AllPlots
#   /nfs/slac/g/atlas/u02/murtazas/PileUPJetID/Plots/

events_of_interest = 500000
#294573

train_isPU = np.load(fileDir + "train_isPU_j0_EM.npy") 
CV_isPU    = np.load(fileDir + "CV_isPU_j0_EM.npy") 
test_isPU  = np.load(fileDir + "test_isPU_j0_EM.npy")  

#isHS
train_isHS = np.load(fileDir + "train_isHS_j0_EM.npy") 
CV_isHS    = np.load(fileDir + "CV_isHS_j0_EM.npy") 
test_isHS  = np.load(fileDir + "test_isHS_j0_EM.npy")

#jet width
train_jet_width = np.load(fileDir + "train_jet_width_j0_EM.npy") 
CV_jet_width    = np.load(fileDir + "CV_jet_width_j0_EM.npy") 
test_jet_width  = np.load(fileDir + "test_jet_width_j0_EM.npy")

train_jet_Rpt = np.load(fileDir + "train_jet_Rpt_j0_EM.npy") 
CV_jet_Rpt    = np.load(fileDir + "CV_jet_Rpt_j0_EM.npy") 
test_jet_Rpt  = np.load(fileDir + "test_jet_Rpt_j0_EM.npy")

train_j0pt    = np.load(fileDir + "train_recopts_j0_EM.npy") 
CV_j0pt       = np.load(fileDir + "CV_recopts_j0_EM.npy") 
test_j0pt     = np.load(fileDir + "test_recopts_j0_EM.npy") 

train_rawEventWeight = np.load(fileDir + "train_rawEventWeights_j0_EM.npy")
CV_rawEventWeight    = np.load(fileDir + "CV_rawEventWeights_j0_EM.npy")
test_rawEventWeight  = np.load(fileDir + "test_rawEventWeights_j0_EM.npy")

whole_pred = np.load(fileDir + "classPredictions_whole_baselineNN.npy")*1000
CV_pred    = np.load(fileDir + "classPredictions_CV_baselineNN.npy")*1000

whole_pred_CNN = np.load(fileDir + "classPredictions_whole_simpleCNN.npy")*1000
CV_pred_CNN    = np.load(fileDir + "classPredictions_CV_simpleCNN.npy")*1000

whole_jet_Rpt = np.concatenate((train_jet_Rpt, CV_jet_Rpt), axis=0)
whole_jet_Rpt = np.concatenate((whole_jet_Rpt, test_jet_Rpt), axis=0)
whole_j0pt = np.concatenate((train_j0pt, CV_j0pt), axis=0)
whole_j0pt = np.concatenate((whole_j0pt, test_j0pt), axis=0)
whole_isPU = np.concatenate((train_isPU, CV_isPU), axis=0)
whole_isPU = np.concatenate((whole_isPU, test_isPU), axis=0)

print("test")
if (train_isHS[train_isHS==False].size != train_isPU[train_isPU==True].size):
	print("	error in bools")

whole_jet_Rpt_PU  = whole_jet_Rpt[whole_isPU == True ]
whole_jet_Rpt_nPU = whole_jet_Rpt[whole_isPU == False]
whole_j0pt_PU     = whole_j0pt[whole_isPU == True ]
whole_j0pt_nPU    = whole_j0pt[whole_isPU == False]
whole_pred_PU     = whole_pred[whole_isPU == True ]
whole_pred_nPU    = whole_pred[whole_isPU == False]
whole_pred_CNN_PU     = whole_pred_CNN[whole_isPU == True ]
whole_pred_CNN_nPU    = whole_pred_CNN[whole_isPU == False]
whole_Rpt_20_30_PU = []
whole_Rpt_30_40_PU = []
whole_Rpt_40_50_PU = []
whole_Rpt_50_60_PU = []
whole_Rpt_20_30_nPU = []
whole_Rpt_30_40_nPU = []
whole_Rpt_40_50_nPU = []
whole_Rpt_50_60_nPU = []
for i,ip in enumerate(whole_isPU):
    if ip==True:
      if whole_j0pt[i]>=8.18  and whole_j0pt[i]<15.13 :
        whole_Rpt_20_30_PU.append(whole_jet_Rpt[i])
      if whole_j0pt[i]>=15.13 and whole_j0pt[i]<22.15 :
        whole_Rpt_30_40_PU.append(whole_jet_Rpt[i])
      if whole_j0pt[i]>=22.15 and whole_j0pt[i]<29.12 :
        whole_Rpt_40_50_PU.append(whole_jet_Rpt[i])
      if whole_j0pt[i]>=29.12 and whole_j0pt[i]<=36.03:
        whole_Rpt_50_60_PU.append(whole_jet_Rpt[i])
    if ip==False:
      if whole_j0pt[i]>=8.18  and whole_j0pt[i]<15.13 :
        whole_Rpt_20_30_nPU.append(whole_jet_Rpt[i])
      if whole_j0pt[i]>=15.13 and whole_j0pt[i]<22.15 :
        whole_Rpt_30_40_nPU.append(whole_jet_Rpt[i])
      if whole_j0pt[i]>=22.15 and whole_j0pt[i]<29.12 :
        whole_Rpt_40_50_nPU.append(whole_jet_Rpt[i])
      if whole_j0pt[i]>=29.12 and whole_j0pt[i]<=36.03:
        whole_Rpt_50_60_nPU.append(whole_jet_Rpt[i])
whole_Rpt_20_30_PU = np.array(whole_Rpt_20_30_PU)
whole_Rpt_30_40_PU = np.array(whole_Rpt_30_40_PU)
whole_Rpt_40_50_PU = np.array(whole_Rpt_40_50_PU)
whole_Rpt_50_60_PU = np.array(whole_Rpt_50_60_PU)
whole_Rpt_20_30_nPU = np.array(whole_Rpt_20_30_nPU)
whole_Rpt_30_40_nPU = np.array(whole_Rpt_30_40_nPU)
whole_Rpt_40_50_nPU = np.array(whole_Rpt_40_50_nPU)
whole_Rpt_50_60_nPU = np.array(whole_Rpt_50_60_nPU)
#############################
#fig, axs = plt.subplots(nrows=2, ncols=2)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.hist(whole_jet_Rpt_PU, bins=100,range=(0,3),alpha=0.5, label='PU')
ax1.hist(whole_jet_Rpt_nPU, bins=100,range=(0,3),alpha=0.5, label='HS')
ax1.legend(loc='upper right')
ax1.set_xlabel("Jet Rpt")
ax1.set_ylabel("Frequency")
plt.suptitle('PU vs NON PU ')
plt.savefig(plotDir + "RptDist.png")
plt.close()
#weights=whole_rawEventWeight[whole_isPU == True]
#alpha=0.5
#############################
fig1 = plt.figure()
ax11 = fig1.add_subplot(111)
ax11.hist(whole_j0pt_PU, bins=100,range=(0,40),alpha=0.5, label='PU')
ax11.hist(whole_j0pt_nPU, bins=100,range=(0,40),alpha=0.5, label='HS')
ax11.legend(loc='upper right')
ax11.set_xlabel("Jet j0pT")
ax11.set_ylabel("Frequency")
plt.suptitle('PU vs NON PU ')
plt.savefig(plotDir+ "j0ptDist.png")
plt.close()
#############################
fig11 = plt.figure()
ax111 = fig11.add_subplot(111)
ax111.hist(whole_pred_PU, bins=100,range=(0,1000),alpha=0.5, label='PU baseline')
ax111.hist(whole_pred_nPU, bins=100,range=(0,1000),alpha=0.5, label='HS baseline')
ax111.hist(whole_pred_CNN_PU, bins=100,range=(0,1000),alpha=0.5, label='PU CNN')
ax111.hist(whole_pred_CNN_nPU, bins=100,range=(0,1000),alpha=0.5, label='HS CNN')
ax111.legend(loc='upper right')
ax111.set_xlabel("Jet Pred")
ax111.set_ylabel("Frequency")
plt.suptitle('PU vs NON PU ')
plt.savefig(plotDir+ "predDist.png")
plt.close()
#############################
effPU = []
effHS = []
effPU2= []
effHS2= []
effPU3 = []
effHS3 = []
effPU4 = []
effHS4 = []
effPU_pt = []
effHS_pt = []
for w in np.linspace(0.0, 3.0, num=1000):
    effHS.append(whole_Rpt_20_30_nPU[whole_Rpt_20_30_nPU >= w].size / np.float32(whole_Rpt_20_30_nPU.size))
    effPU.append(whole_Rpt_20_30_PU[whole_Rpt_20_30_PU >= w].size / np.float32(events_of_interest))
    effHS2.append(whole_Rpt_30_40_nPU[whole_Rpt_30_40_nPU>= w].size / np.float32(whole_Rpt_30_40_nPU.size))
    effPU2.append(whole_Rpt_30_40_PU[whole_Rpt_30_40_PU >= w].size / np.float32(events_of_interest))
    effHS3.append(whole_Rpt_40_50_nPU[whole_Rpt_40_50_nPU>= w].size / np.float32(whole_Rpt_40_50_nPU.size))
    effPU3.append(whole_Rpt_40_50_PU[whole_Rpt_40_50_PU >= w].size / np.float32(events_of_interest))
    effHS4.append(whole_Rpt_50_60_nPU[whole_Rpt_50_60_nPU>= w].size / np.float32(whole_Rpt_50_60_nPU.size))
    effPU4.append(whole_Rpt_50_60_PU[whole_Rpt_50_60_PU >= w].size / np.float32(events_of_interest))

for pt in np.linspace(8.18, 36.03, num=1000):
    effHS_pt.append(whole_j0pt_nPU[whole_j0pt_nPU >= pt].size / np.float32(whole_j0pt_nPU.size))
    effPU_pt.append(whole_j0pt_PU[whole_j0pt_PU >= pt].size / np.float32(events_of_interest))

effHS_pred = []
effPU_pred  = []
effHS_CNN_pred = []
effPU_CNN_pred  = []
for pred in np.linspace(0.0,1000.0, num=1000):
    effHS_pred.append(whole_pred_nPU[whole_pred_nPU <= pred].size / np.float32(whole_pred_nPU.size))
    effPU_pred.append(whole_pred_PU[whole_pred_PU <= pred].size / np.float32(events_of_interest))
    effHS_CNN_pred.append(whole_pred_CNN_nPU[whole_pred_CNN_nPU <= pred].size / np.float32(whole_pred_CNN_nPU.size))
    effPU_CNN_pred.append(whole_pred_CNN_PU[whole_pred_CNN_PU <= pred].size / np.float32(events_of_interest))
#############################
#print(effPU, effHS)
#fig, axs = plt.subplots(nrows=2, ncols=2)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
#ax2.set_xscale("log", nonposx='clip')
#ax2.set_yscale('log', nonposy='clip')
l1, = ax2.plot(effHS_pt, effPU_pt, 'r.:', label='j0pt ROC using j0pt > j0pt0')
l2, = ax2.plot(effHS, effPU, 'b.:', label='Rpt ROC, j0pt 20-30')
l3, = ax2.plot(effHS2, effPU2, 'g.:', label='Rpt ROC, j0pt 30-40')
l4, = ax2.plot(effHS3, effPU3, 'y.:', label='Rpt ROC, j0pt 40-50')
l5, = ax2.plot(effHS4, effPU4, 'm.:', label='Rpt ROC, j0pt 50-60')
l6, = ax2.plot(effHS_pred, effPU_pred, 'k.:', label='Baseline NN ROC, P < P0')
l7, = ax2.plot(effHS_CNN_pred, effPU_CNN_pred, 'c.:', label='CNN ROC, P < P0')
ax2.legend(loc='lower right')
#my_handler = HandlerLine2D(numpoints=4)
#ax2.legend(loc='upper left', handler_map={l1:my_handler,l2:my_handler,l3:my_handler,l4:my_handler,l5:my_handler})
ax2.set_xlabel("effeciency Hard Scatter")
ax2.set_ylabel("Fake Rate")
plt.yscale('log', nonposy='clip')
plt.suptitle('ROC curves')
plt.savefig(plotDir + "ROC.png")
plt.close()
#weights=train_rawEventWeight[train_isPU == True]
#alpha=0.5
#############################
# locations
# upper right	1
# upper left	2
# lower left	3
# lower right	4
# right	5
# center left	6
# center right	7
# lower center	8
# upper center	9
# center	10

# colors
# 'b'	blue
# 'g'	green
# 'r'	red
# 'c'	cyan
# 'm'	magenta
# 'y'	yellow
# 'k'	black
# 'w'	white

# markers
# '-'	solid line style
# '--'	dashed line style
# '-.'	dash-dot line style
# ':'	dotted line style
# '.'	point marker
# ','	pixel marker
# 'o'	circle marker
# 'v'	triangle_down marker
# '^'	triangle_up marker
# '<'	triangle_left marker
# '>'	triangle_right marker
# '1'	tri_down marker
# '2'	tri_up marker
# '3'	tri_left marker
# '4'	tri_right marker
# 's'	square marker
# 'p'	pentagon marker
# '*'	star marker
# 'h'	hexagon1 marker
# 'H'	hexagon2 marker
# '+'	plus marker
# 'x'	x marker
# 'D'	diamond marker
# 'd'	thin_diamond marker
# '|'	vline marker
# '_'	hline marker