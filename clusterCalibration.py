import math
import numpy as np
#import matplotlib.pyplot as plt
'''
Data that I want to divide into training, CV, and test sets:
(I.e, the outputs of my file)

truth_Pt
reco_Pt
revised_weights
npvs
pixel_image
dR_disks

'''
# Apply weights to make the spectrum of truth pT flat.
def weight_Pt():

   numJets = 0
   badEta = 0
   badPhi = 0
   badEtaT = 0
   badPhiT = 0

   fileDir   = "/nfs/slac/g/atlas/u02/murtazas/PileUPJetID/inputArraysVSmallCR/"
   Npixel    = 10 # so delta_eta = delta_phi = 0.1
   endValue  = 0.5
   fullValue = 1.0

   truth_Pt     = []
   event_weight = []
   pixel_image          = []
   pixel_image_onlyPT   = []
   pixel_image_onlyTrks = []
   pixel_image_pT_trks  = []
   #pixel_image_areaSub  = []
   reco_pt        = []
   reco_pt_noarea = []
   sum_pt       = []
   npvs         = []
   mus          = []
   Ats          = []

   isPU         = []
   isHS         = []
   jet_width    = []

   jet_Rpt = []
   jet_JVF = []
   jet_corrJVF = []

   sum_evt_wts_PU    = 0.0
   sum_evt_wts_notPU = 0.0

   ids   = ["jz1_EM"]
   for id in ids:

     # Load the files in the given pT range
     ex_truth_Pt     = np.load(fileDir + "truepts_j0_"    + id + ".npy") 
     ex_event_weight = np.load(fileDir + "weights_j0_"    + id + ".npy") 
     ex_npv          = np.load(fileDir + "npvs_j0_"    + id + ".npy") 
     ex_mu           = np.load(fileDir + "mus_j0_"         + id + ".npy")
     ex_At           = np.load(fileDir + "At_j0_"         + id + ".npy")

     ex_recopts_j0   = np.load(fileDir + "recopts_j0_" + id + ".npy") 
     ex_recopts_jnoarea0   = np.load(fileDir + "recopts_jnoarea0_" + id + ".npy") 
     ex_recoeta_j0   = np.load(fileDir + "recoeta_j0_" + id + ".npy") 
     ex_recophi_j0   = np.load(fileDir + "recophi_j0_" + id + ".npy") 

     ex_sumpt        = np.load(fileDir + "j0_sumpt_"      + id + ".npy")
     ex_cl_pts       = np.load(fileDir + "clstr_pts_j0_"  + id + ".npy")
     ex_cl_etas      = np.load(fileDir + "clstr_etas_j0_" + id + ".npy")
     ex_cl_phis      = np.load(fileDir + "clstr_phis_j0_" + id + ".npy")
     ex_cl_lns       = np.load(fileDir + "clstr_widths_lon_j0_" + id + ".npy")
     ex_cl_lats      = np.load(fileDir + "clstr_widths_lat_j0_" + id + ".npy")

     ex_isPU         = np.load(fileDir + "isPU_j0_" + id + ".npy")
     ex_isHS         = np.load(fileDir + "isHS_j0_" + id + ".npy")

     ex_Rpt          = np.load(fileDir+'Rpt_j0_'        + id + ".npy")
     ex_JVF          = np.load(fileDir+'JVF_j0_'        + id + ".npy")
     ex_corrJVF      = np.load(fileDir+'corrJVF_j0_'    + id + ".npy")

     ex_trkpts       = np.load(fileDir+'trk_pts_j0_'    + id + ".npy")
     ex_trketas      = np.load(fileDir+'trk_etas_j0_'   + id + ".npy")
     ex_trkphis      = np.load(fileDir+'trk_phis_j0_'   + id + ".npy")
     ex_trkisHSs     = np.load(fileDir+'trk_isHS_j0_'   + id + ".npy")

     
     for tpt,evt_wt,cl_pt,cl_eta,cl_phi,j_pt,jnoarea_pt,j_eta,j_phi,dR_pts,npv,mu,At,isP,cl_ln,cl_lt,isH,Rpt,JVF,corrJVF,trkpt,trketa,trkphi,trkisHS in zip(ex_truth_Pt,ex_event_weight,ex_cl_pts,
                                                                                            ex_cl_etas,ex_cl_phis,ex_recopts_j0,ex_recopts_jnoarea0,
                                                                                            ex_recoeta_j0,ex_recophi_j0,
                                                                                            ex_sumpt,ex_npv,ex_mu,ex_At,ex_isPU,ex_cl_lns,ex_cl_lats,ex_isHS,
                                                                                            ex_Rpt,ex_JVF,ex_corrJVF,ex_trkpts,ex_trketas,
                                                                                            ex_trkphis,ex_trkisHSs):
       numJets += 1

       if numJets % 10000 == 0:
         print("Jet {0}".format(numJets))

       truth_Pt.append(tpt)                                   
       event_weight.append(evt_wt)                             
       reco_pt.append(j_pt)
       reco_pt_noarea.append(jnoarea_pt)
       sum_pt.append(dR_pts)
       npvs.append(npv)
       mus.append(mu)
       Ats.append(At)

       isPU.append(isP)
       isHS.append(isH)
       if (isP == True):
           sum_evt_wts_PU    += evt_wt
       else:
           sum_evt_wts_notPU += evt_wt
       jet_Rpt.append(Rpt)
       jet_JVF.append(JVF)
       jet_corrJVF.append(corrJVF)  
       
       # Go ahead and fill the arrays for the truth pt and pT disks since you have all the info you need  
       R          = np.zeros((Npixel,Npixel))
       R_latwidth = np.zeros((Npixel,Npixel))
       R_lonwidth = np.zeros((Npixel,Npixel))
       R_areaSub  = np.zeros((Npixel,Npixel))
       R_PUtrk    = np.zeros((Npixel,Npixel))
       R_HStrk    = np.zeros((Npixel,Npixel))
       curr_width = 0.0
       curr_pt    = 0.0
       
       num_cl = len(cl_pt)
       
       for i_pt,i_eta,i_phi,i_ln,i_lt in zip(cl_pt,cl_eta,cl_phi,cl_ln,cl_lt):
         deta = 0.0
         dphi = 0.0
         x = np.floor((i_eta-j_eta + endValue) * Npixel / fullValue)
         deta = abs(i_eta - j_eta) 
         cutoff = np.pi / 2 
         #if abs(i_phi-j_phi) <= cutoff:  
         if abs(i_phi-j_phi) <= endValue:
           dphi = abs(i_phi-j_phi)
           y = np.floor((i_phi-j_phi + endValue) * Npixel / fullValue) 
         elif abs(i_phi-j_phi-2*np.pi) <= endValue: 
           dphi = abs(i_phi-j_phi-2*np.pi)
           y = np.floor((i_phi-j_phi-2*np.pi + endValue) * Npixel / fullValue) 
         elif abs(i_phi-j_phi+2*np.pi) <= endValue:  
           dphi = abs(i_phi-j_phi+2*np.pi)
           y = np.floor((i_phi-j_phi+2*np.pi + endValue) * Npixel / fullValue) 
         else:
           #print("None of these options worked, j_phi = {0}, cl_phi ={1}".format(j_phi,i_phi))
           dphi = -999
           y = -10

         # Make it ok for the cluster to be w/in 0.1 outside of the bounds, and just put it in the last bin anyways
         if x < 0:
           x = 0
           badEta += 1
         if x >= Npixel:
           x = Npixel -1
           badEta += 1

         if y < 0:
           y = 0
           badPhi += 1
         if y >= Npixel:
           y = Npixel -1
           badPhi += 1
         if dphi != -999:
           curr_width += np.sqrt(deta**2 + dphi**2)*i_pt
           curr_pt    += i_pt
         R[x,y]          += i_pt
         R_latwidth[x,y] += i_lt
         R_lonwidth[x,y] += i_ln
         R_areaSub[x,y]  += i_pt  - (jnoarea_pt - j_pt)/num_cl


       for t_pt,t_eta,t_phi,t_hs in zip(trkpt,trketa,trkphi,trkisHS):
         x = np.floor((t_eta-j_eta + endValue) * Npixel / fullValue) 
         if abs(t_phi-j_phi) <= endValue:
           y = np.floor((t_phi-j_phi + endValue) * Npixel / fullValue) 
         elif abs(t_phi-j_phi-2*np.pi) <= endValue: 
           y = np.floor((t_phi-j_phi-2*np.pi + endValue) * Npixel / fullValue) 
         elif abs(t_phi-j_phi+2*np.pi) <= endValue:  
           y = np.floor((t_phi-j_phi+2*np.pi + endValue) * Npixel / fullValue) 
         else:
           #print("None of these options worked, j_phi = {0}, trk_phi ={1}".format(j_phi,t_phi))
           y = -10
         if x < 0:
           x = 0
           badEtaT += 1
         if x >= Npixel:
           x = Npixel -1
           badEtaT += 1
         if y < 0:
           y = 0
           badPhiT += 1
         if y >= Npixel:
           y = Npixel -1
           badPhiT += 1
         if t_hs:
           R_HStrk[x,y]    += t_pt
         else:
           R_PUtrk[x,y]    += t_pt

       R_PUtrk = R_PUtrk/1000.0
       R_HStrk = R_HStrk/1000.0
       pixel_image.append(np.array([R,R_lonwidth,R_latwidth]))
       pixel_image_onlyPT.append(R)
       jet_width.append(curr_width/curr_pt)
       pixel_image_onlyTrks.append(np.array([R_HStrk,R_PUtrk]))
       pixel_image_pT_trks.append(np.array([R,R_HStrk,R_PUtrk])) #todo maybe divide by 100 to get trk in GeV?
       #pixel_image.append(R)
       #pixel_image_areaSub.append(R_areaSub)
                                                            

       # M also wants to run linear regression on the clusters, so put the matrix into a 64 long array
       #linear_matrix.append()

   # type cast these to numpy arrays                                  
   truth_Pt             = np.array(truth_Pt)
   event_weight         = np.array(event_weight)
   pixel_image          = np.array(pixel_image)
   pixel_image_onlyPT   = np.array(pixel_image_onlyPT)
   pixel_image_onlyTrks = np.array(pixel_image_onlyTrks)
   pixel_image_pT_trks  = np.array(pixel_image_pT_trks)
   reco_pt              = np.array(reco_pt)
   reco_pt_noarea       = np.array(reco_pt_noarea)
   npvs                 = np.array(npvs)
   mus                  = np.array(mus)
   Ats                  = np.array(Ats)
   jet_width            = np.array(jet_width)
   jet_Rpt              = np.array(jet_Rpt)
   jet_JVF              = np.array(jet_JVF)
   jet_corrJVF          = np.array(jet_corrJVF)

   n = truth_Pt.size

   # Divide the data into train, CV and test samples
   train_idx = np.array([x for x in np.arange(n) if x % 10 < 8]) 
   CV_idx    = np.arange(8,n,10) 
   test_idx  = np.arange(9,n,10) 

   n_train = train_idx.size
   n_CV    =    CV_idx.size
   n_test  =  test_idx.size

   if truth_Pt.size != n_train + n_CV + n_test:
     print("Error, wrong size")

   # train_revisedEventWeight = np.zeros(n_train)
   # CV_revisedEventWeight    = np.zeros(n_CV)
   # test_revisedEventWeight  = np.zeros(n_test)

   train_revisedEventWeight = []
   CV_revisedEventWeight    = []
   test_revisedEventWeight  = []

   sum_evt_wts_notPU_train  = 0.0
   sum_evt_wts_PU_train     = 0.0
   sum_evt_wts_notPU_test   = 0.0
   sum_evt_wts_PU_test      = 0.0
   sum_evt_wts_notPU_CV     = 0.0
   sum_evt_wts_PU_CV        = 0.0

   #weights for the training data
   for i,s in zip(np.arange(n_train),train_idx):
      if isPU[s] == False and isHS[s] == True :
          sum_evt_wts_notPU_train += event_weight[s]
      elif isPU[s] == True and isHS[s] == False:
          sum_evt_wts_PU_train    += event_weight[s]
   # Weights for the CV data
   for i,s in zip(np.arange(n_CV),CV_idx):
      if isPU[s] == False and isHS[s] == True :
          sum_evt_wts_notPU_CV += event_weight[s]
      elif isPU[s] == True and isHS[s] == False:
          sum_evt_wts_PU_CV    += event_weight[s]
   # Weights for test
   for i,s in zip(np.arange(n_test),test_idx):
      if isPU[s] == False and isHS[s] == True :
          sum_evt_wts_notPU_test += event_weight[s]
      elif isPU[s] == True and isHS[s] == False:
          sum_evt_wts_PU_test    += event_weight[s]

   for i,s in zip(np.arange(n_train),train_idx):
      if isPU[s] == False and isHS[s] == True :
          #train_revisedEventWeight[i] = event_weight[s]/(2*sum_evt_wts_notPU_train)
          train_revisedEventWeight.append(event_weight[s]/(2*sum_evt_wts_notPU_train))
      elif isPU[s] == True and isHS[s] == False:
          #train_revisedEventWeight[i] = event_weight[s]/(2*sum_evt_wts_PU_train)
          train_revisedEventWeight.append(event_weight[s]/(2*sum_evt_wts_PU_train))
   for i,s in zip(np.arange(n_CV),CV_idx):
      if isPU[s] == False and isHS[s] == True :
          #CV_revisedEventWeight[i] = event_weight[s]/(2*sum_evt_wts_notPU_CV)
          CV_revisedEventWeight.append(event_weight[s]/(2*sum_evt_wts_notPU_CV))
      elif isPU[s] == True and isHS[s] == False:
          #CV_revisedEventWeight[i] = event_weight[s]/(2*sum_evt_wts_PU_CV)
          CV_revisedEventWeight.append(event_weight[s]/(2*sum_evt_wts_PU_CV))
   for i,s in zip(np.arange(n_test),test_idx):
      if isPU[s] == False and isHS[s] == True :
          #test_revisedEventWeight[i] = event_weight[s]/(2*sum_evt_wts_notPU_test)
          test_revisedEventWeight.append(event_weight[s]/(2*sum_evt_wts_notPU_test))
      elif isPU[s] == True and isHS[s] == False:
          #test_revisedEventWeight[i] = event_weight[s]/(2*sum_evt_wts_PU_test)
          test_revisedEventWeight.append(event_weight[s]/(2*sum_evt_wts_PU_test))

   # Put the histogram in 0.5 GeV bins
   # histRange = (5,160)
   # for Nbins in [155,78,39]: # (histRange[1] - histRange[0])

   #   train_hist, bin_edges = np.histogram(truth_Pt[train_idx], bins=Nbins,range=histRange,weights=event_weight[train_idx])
   #   CV_hist,    bin_edges2= np.histogram(truth_Pt[CV_idx],    bins=Nbins,range=histRange,weights=event_weight[CV_idx])
   #   test_hist,  bin_edges3= np.histogram(truth_Pt[test_idx],  bins=Nbins,range=histRange,weights=event_weight[test_idx])
   #   indices = np.digitize(truth_Pt, bin_edges)

   #   # weights for the training data
   #   for i,s in zip(np.arange(n_train),train_idx):
   #     train_revisedEventWeight[i] = event_weight[s]/train_hist[indices[s]-1]

   #   # Weights for the CV data
   #   for i,s in zip(np.arange(n_CV),CV_idx):
   #     CV_revisedEventWeight[i] = event_weight[s]/CV_hist[indices[s]-1]

   #   # Weights for test
   #   for i,s in zip(np.arange(n_test),test_idx):
   #     test_revisedEventWeight[i] = event_weight[s]/test_hist[indices[s]-1]

   train_sum_pt = []
   test_sum_pt = []
   CV_sum_pt = []

   train_truth_Pt = []
   test_truth_Pt  = []
   CV_truth_Pt    = []

   train_reco_Pt = []
   test_reco_Pt  = []
   CV_reco_Pt    = []

   train_reco_noarea_Pt = []
   test_reco_noarea_Pt  = []
   CV_reco_noarea_Pt    = []

   train_npvs = []
   test_npvs  = []
   CV_npvs    = []
   train_mus  = []
   test_mus   = []
   CV_mus     = []
   train_Ats  = []
   test_Ats   = []
   CV_Ats     = []

   train_isPU = []
   test_isPU = []
   CV_isPU = []
   train_isHS = []
   test_isHS = []
   CV_isHS = []

   train_pixel_image = []
   test_pixel_image  = []
   CV_pixel_image    = []


   train_pixel_image_onlyPt = []
   test_pixel_image_onlyPt  = []
   CV_pixel_image_onlyPt    = []

   train_pixel_image_onlyTrks = []
   test_pixel_image_onlyTrks  = []
   CV_pixel_image_onlyTrks    = []

   train_pixel_image_pT_trks = []
   test_pixel_image_pT_trks  = []
   CV_pixel_image_pT_trks    = []

   train_rawEventWeight = []
   test_rawEventWeight  = []
   CV_rawEventWeight    = []

   train_jet_width = []
   test_jet_width  = []
   CV_jet_width    = []

   train_jet_Rpt = []
   test_jet_Rpt  = []
   CV_jet_Rpt    = []
   train_jet_JVF = []
   test_jet_JVF  = []
   CV_jet_JVF    = []
   train_jet_corrJVF = []
   test_jet_corrJVF  = []
   CV_jet_corrJVF    = []

   # weights for the training data
   for s in train_idx:
     if (isPU[s] == False and isHS[s] == False) or (isPU[s] == True and isHS[s] == True) :
       continue
     train_sum_pt.append(sum_pt[s])
     train_pixel_image.append(pixel_image[s])
     train_pixel_image_onlyPt.append(pixel_image_onlyPT[s])
     train_isPU.append(isPU[s])
     train_isHS.append(isHS[s])
     train_rawEventWeight.append(event_weight[s])
     train_truth_Pt.append(truth_Pt[s])
     train_reco_Pt.append(reco_pt[s])
     train_reco_noarea_Pt.append(reco_pt_noarea[s])
     train_npvs.append(npvs[s])
     train_mus.append(mus[s])
     train_Ats.append(Ats[s])
     train_jet_width.append(jet_width[s])
     train_jet_Rpt.append(jet_Rpt[s])
     train_jet_JVF.append(jet_JVF[s])
     train_jet_corrJVF.append(jet_corrJVF[s])
     train_pixel_image_onlyTrks.append(pixel_image_onlyTrks[s])
     train_pixel_image_pT_trks.append(pixel_image_pT_trks[s])
     #train_pixel_image_areaSub.append(pixel_image_areaSub[s])

   # Weights for the CV data
   for s in CV_idx:
     if (isPU[s] == False and isHS[s] == False) or (isPU[s] == True and isHS[s] == True) :
       continue
     CV_sum_pt.append(sum_pt[s])
     CV_pixel_image.append(pixel_image[s])
     CV_pixel_image_onlyPt.append(pixel_image_onlyPT[s])
     CV_isPU.append(isPU[s])
     CV_isHS.append(isHS[s])
     CV_rawEventWeight.append(event_weight[s])
     CV_truth_Pt.append(truth_Pt[s])
     CV_reco_Pt.append(reco_pt[s])
     CV_reco_noarea_Pt.append(reco_pt_noarea[s])
     CV_npvs.append(npvs[s])
     CV_mus.append(mus[s])
     CV_Ats.append(Ats[s])
     CV_jet_width.append(jet_width[s])
     CV_jet_Rpt.append(jet_Rpt[s])
     CV_jet_JVF.append(jet_JVF[s])
     CV_jet_corrJVF.append(jet_corrJVF[s])
     CV_pixel_image_onlyTrks.append(pixel_image_onlyTrks[s])
     CV_pixel_image_pT_trks.append(pixel_image_pT_trks[s])
     #CV_pixel_image_areaSub.append(pixel_image_areaSub[s])

   # Weights for test
   for s in test_idx:
     if (isPU[s] == False and isHS[s] == False) or (isPU[s] == True and isHS[s] == True) :
       continue
     test_sum_pt.append(sum_pt[s])
     test_pixel_image.append(pixel_image[s])
     test_isPU.append(isPU[s])
     test_isHS.append(isHS[s])
     test_pixel_image_onlyPt.append(pixel_image_onlyPT[s])
     test_rawEventWeight.append(event_weight[s])
     test_truth_Pt.append(truth_Pt[s])
     test_reco_Pt.append(reco_pt[s])
     test_reco_noarea_Pt.append(reco_pt_noarea[s])
     test_npvs.append(npvs[s])
     test_mus.append(mus[s])
     test_Ats.append(Ats[s])
     test_jet_width.append(jet_width[s])
     test_jet_Rpt.append(jet_Rpt[s])
     test_jet_JVF.append(jet_JVF[s])
     test_jet_corrJVF.append(jet_corrJVF[s])
     test_pixel_image_onlyTrks.append(pixel_image_onlyTrks[s])
     test_pixel_image_pT_trks.append(pixel_image_pT_trks[s])
     #test_pixel_image_areaSub.append(pixel_image_areaSub[s])

   # revised weights
   np.save(fileDir + "train_revisedWeights_j0_EM.npy", train_revisedEventWeight)
   np.save(fileDir + "CV_revisedWeights_j0_EM.npy",    CV_revisedEventWeight)
   np.save(fileDir + "test_revisedWeights_j0_EM.npy",  test_revisedEventWeight)
   np.save(fileDir + "train_rawEventWeights_j0_EM.npy", train_rawEventWeight)
   np.save(fileDir + "CV_rawEventWeights_j0_EM.npy",    CV_rawEventWeight)
   np.save(fileDir + "test_rawEventWeights_j0_EM.npy",  test_rawEventWeight)

   # truth pt
   np.save(fileDir + "train_truepts_j0_EM.npy",train_truth_Pt) 
   np.save(fileDir + "CV_truepts_j0_EM.npy",   CV_truth_Pt) 
   np.save(fileDir + "test_truepts_j0_EM.npy", test_truth_Pt) 

   # reco pt
   np.save(fileDir + "train_recopts_j0_EM.npy",train_reco_Pt) 
   np.save(fileDir + "CV_recopts_j0_EM.npy",   CV_reco_Pt) 
   np.save(fileDir + "test_recopts_j0_EM.npy", test_reco_Pt) 

   # reco pt
   np.save(fileDir + "train_recopts_jnoarea0_EM.npy",train_reco_noarea_Pt) 
   np.save(fileDir + "CV_recopts_jnoarea0_EM.npy",   CV_reco_noarea_Pt) 
   np.save(fileDir + "test_recopts_jnoarea0_EM.npy", test_reco_noarea_Pt) 

   # npv 
   np.save(fileDir + "train_npvs_j0_EM.npy",train_npvs) 
   np.save(fileDir + "CV_npvs_j0_EM.npy",   CV_npvs) 
   np.save(fileDir + "test_npvs_j0_EM.npy", test_npvs) 

   # mu
   np.save(fileDir + "train_mus_j0_EM.npy",train_mus) 
   np.save(fileDir + "CV_mus_j0_EM.npy",   CV_mus) 
   np.save(fileDir + "test_mus_j0_EM.npy", test_mus) 

   # A_T
   np.save(fileDir + "train_At_j0_EM.npy",train_Ats) 
   np.save(fileDir + "CV_At_j0_EM.npy",   CV_Ats) 
   np.save(fileDir + "test_At_j0_EM.npy", test_Ats) 

   # disk info about the jet
   np.save(fileDir + "train_sumpt_j0_EM.npy",train_sum_pt)
   np.save(fileDir + "CV_sumpt_j0_EM.npy",   CV_sum_pt)
   np.save(fileDir + "test_sumpt_j0_EM.npy", test_sum_pt)

   # isPU
   np.save(fileDir + "train_isPU_j0_EM.npy",train_isPU) 
   np.save(fileDir + "CV_isPU_j0_EM.npy",   CV_isPU) 
   np.save(fileDir + "test_isPU_j0_EM.npy", test_isPU)  

   #isHS
   np.save(fileDir + "train_isHS_j0_EM.npy",train_isHS) 
   np.save(fileDir + "CV_isHS_j0_EM.npy",   CV_isHS) 
   np.save(fileDir + "test_isHS_j0_EM.npy", test_isHS)

   #jet width
   np.save(fileDir + "train_jet_width_j0_EM.npy",train_jet_width) 
   np.save(fileDir + "CV_jet_width_j0_EM.npy",   CV_jet_width) 
   np.save(fileDir + "test_jet_width_j0_EM.npy", test_jet_width)

   #jet Rpt JVF corr JVF
   np.save(fileDir + "train_jet_Rpt_j0_EM.npy",train_jet_Rpt) 
   np.save(fileDir + "CV_jet_Rpt_j0_EM.npy",   CV_jet_Rpt) 
   np.save(fileDir + "test_jet_Rpt_j0_EM.npy", test_jet_Rpt)
   np.save(fileDir + "train_jet_JVF_j0_EM.npy",train_jet_JVF) 
   np.save(fileDir + "CV_jet_JVF_j0_EM.npy",   CV_jet_JVF) 
   np.save(fileDir + "test_jet_JVF_j0_EM.npy", test_jet_JVF)
   np.save(fileDir + "train_jet_corrJVF_j0_EM.npy",train_jet_corrJVF) 
   np.save(fileDir + "CV_jet_corrJVF_EM.npy",   CV_jet_corrJVF) 
   np.save(fileDir + "test_jet_corrJVF_j0_EM.npy", test_jet_corrJVF)

   # pixel image
   np.save(fileDir + "train_pixel_image_j0_EM.npy",train_pixel_image)
   np.save(fileDir + "CV_pixel_image_j0_EM.npy",   CV_pixel_image)
   np.save(fileDir + "test_pixel_image_j0_EM.npy", test_pixel_image)
   np.save(fileDir + "train_pixel_image_onlyClus_j0_EM.npy",train_pixel_image_onlyPt)
   np.save(fileDir + "CV_pixel_image_onlyClus_j0_EM.npy",   CV_pixel_image_onlyPt)
   np.save(fileDir + "test_pixel_image_onlyClus_j0_EM.npy", test_pixel_image_onlyPt)
   np.save(fileDir + "train_pixel_image_onlyTrks_j0_EM.npy",train_pixel_image_onlyTrks)
   np.save(fileDir + "CV_pixel_image_onlyTrks_j0_EM.npy",   CV_pixel_image_onlyTrks)
   np.save(fileDir + "test_pixel_image_onlyTrks_j0_EM.npy", test_pixel_image_onlyTrks)
   np.save(fileDir + "train_pixel_image_clus_trks_j0_EM.npy",train_pixel_image_pT_trks)
   np.save(fileDir + "CV_pixel_image_clus_trks_j0_EM.npy",   CV_pixel_image_pT_trks)
   np.save(fileDir + "test_pixel_image_clus_trks_j0_EM.npy", test_pixel_image_pT_trks)

   # pixel image with the area subtraction
   # np.save(fileDir + "train_pixel_image_areaSub_j0_EM.npy",train_pixel_image_areaSub)
   # np.save(fileDir + "CV_pixel_image_areaSub_j0_EM.npy",   CV_pixel_image_areaSub)
   # np.save(fileDir + "test_pixel_image_areaSub_j0_EM.npy", test_pixel_image_areaSub)

   print("numJets = {0}, badEta  = {1}, badPhi = {2}".format(numJets,badEta,badPhi))
   print("numJets = {0}, badEtaT  = {1}, badPhiT = {2}".format(numJets,badEta,badPhi))


weight_Pt()
