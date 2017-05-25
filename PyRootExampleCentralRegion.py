import ROOT as r
r.gROOT.LoadMacro("giordon.h+")
import numpy
from numpy import save
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--inputDir", help="Directory containing input files",type=str, default="JZ1_EM_ClusterInfo")
parser.add_option("--submitDir", help="Directory containing output files",type=str, default=".")
parser.add_option("--numEvents", help="How many events to include (set to -1 for all events)",type=int, default=100000)
parser.add_option("-i","--identifier", help="sample identifier",type=str, default="jz1_EM")

(options, args) = parser.parse_args()

def readRoot():

  import glob
  filenamebase = '../'
  filenames = glob.glob(options.inputDir + '/*.root')
  tree = r.TChain('oTree')
  for filename in filenames:
    #statinfo = os.stat(filename)
    #if statinfo.st_size < 10000: continue #sometimes batch jobs fail
    print '== Reading in '+filename+' =='
    tree.Add(filename) 

  j0pt_arr       = []
  j0eta_arr      = []
  j0phi_arr      = []
  jnoarea0pt_arr = []
  tj0pt_arr      = []
  tj0eta_arr     = []
  mu_arr         = []
  npv_arr        = []
  At_arr         = []
  evt_wt_arr     = []

  j0_clpt_arr    = []
  j0_cleta_arr   = []
  j0_clphi_arr   = []
  j0_clwidth_lon_arr = []
  j0_clwidth_lat_arr = []
  j0_sumpt_arr   = []

  j0isPU_arr     = []
  j0isHS_arr     = []
  
  j0_Rpt_arr     = []
  j0_JVF_arr     = []
  j0_corrJVF_arr = []
  j0_JVT_arr     = []

  j0_trkpt_arr    = []
  j0_trketa_arr   = []
  j0_trkphi_arr   = []
  j0_trkm_arr     = []
  j0_trkisHS_arr   = []
  j0_trkisPU_arr   = []

  nentries = tree.GetEntries()
  events_of_interest = 0
  print 'Number of events: ' + str(nentries)
  for jentry in xrange(nentries):
      if jentry==options.numEvents: break
      tree.GetEntry(jentry)

      # jet branches
      j0pt_branch  = getattr(tree,'j0pt')
      jnoarea0pt_branch  = getattr(tree,'jnoarea0pt')
      j0eta_branch = getattr(tree,'j0eta')
      j0phi_branch = getattr(tree,'j0phi')
      j0area_branch = getattr(tree,'j0area')

      j0sumpt5_branch  = getattr(tree,'j0sumpt5') 
      j0sumpt10_branch = getattr(tree,'j0sumpt10') 
      j0sumpt15_branch = getattr(tree,'j0sumpt15') 
      j0sumpt20_branch = getattr(tree,'j0sumpt20') 
      j0sumpt25_branch = getattr(tree,'j0sumpt25') 
      j0sumpt30_branch = getattr(tree,'j0sumpt30') 
      j0sumpt35_branch = getattr(tree,'j0sumpt35') 
      j0sumpt40_branch = getattr(tree,'j0sumpt40') # Note, j0sumpt40 = jnoare0pt 

      j0isPU_branch    = getattr(tree,'j0isPU')

      j0_Rpt_branch     = getattr(tree,'j0Rpt') 
      j0_JVF_branch     = getattr(tree,'j0JVF') 
      j0_corrJVF_branch = getattr(tree,'j0corrJVF') 
      j0_JVT_branch     = getattr(tree,'j0JVT') 

      # cluster branches
      j0_clpt_branch  = getattr(tree,'j0_clpt')
      j0_cleta_branch = getattr(tree,'j0_cleta')
      j0_clphi_branch = getattr(tree,'j0_clphi')
      j0_clwidth_lat_branch = getattr(tree,'j0_cllatwidth')
      j0_clwidth_lon_branch = getattr(tree,'j0_cllongwidth')
   
      j0_trkpt_branch  = getattr(tree,'j0_trkpt')
      j0_trketa_branch = getattr(tree,'j0_trketa')
      j0_trkphi_branch = getattr(tree,'j0_trkphi')
      j0_trkm_branch = getattr(tree,'j0_trkm')
      j0_trkisHS_branch = getattr(tree,'j0_trkisHS')
      j0_trkisPU_branch = getattr(tree,'j0_trkisPU')


      # truth jet info
      tj0pt_branch  = getattr(tree,'tj0pt')
      tj0eta_branch = getattr(tree,'tj0eta')
      flag = False
      if jentry % 5000 == 0:
        print 'jentry = {0}'.format(jentry)
      for cl_pts,cl_etas,cl_phis,j0pt,jnoarea0pt,j0eta,j0phi,tj0pt,tj0eta,d5,d10,d15,d20,d25,d30,d35,d40,At,isPU,cl_lns,cl_lts,j0Rpt,j0JVF,j0cJVF,j0JVT,trkpts,trketas,trkphis,trkms,trkisPUs,trkisHSs in zip(j0_clpt_branch,
                                                     j0_cleta_branch,j0_clphi_branch,j0pt_branch,
                                                     jnoarea0pt_branch,j0eta_branch,j0phi_branch,tj0pt_branch,tj0eta_branch,
                                                     j0sumpt5_branch,j0sumpt10_branch,j0sumpt15_branch,j0sumpt20_branch,
                                                     j0sumpt25_branch,j0sumpt30_branch,j0sumpt35_branch,j0sumpt40_branch,
                                                     j0area_branch,j0isPU_branch,j0_clwidth_lon_branch,j0_clwidth_lat_branch,
                                                     j0_Rpt_branch,j0_JVF_branch,j0_corrJVF_branch,j0_JVT_branch,
                                                     j0_trkpt_branch,j0_trketa_branch,j0_trkphi_branch,j0_trkm_branch,
                                                     j0_trkisPU_branch,j0_trkisHS_branch):

          # Eta cut
          if np.absolute(j0eta) > 0.8 or j0pt < 8.18 or j0pt > 36.03:
          #print "Ooops! We're only looking at central region jets with reco Pt > 5 GeV!"
            continue
          flag = True
          npv_arr.append(tree.NPV)
          mu_arr.append(tree.mu)
          At_arr.append(At)
          evt_wt_arr.append(tree.event_weight)
          j0pt_arr.append(j0pt)
          jnoarea0pt_arr.append(jnoarea0pt)
          j0eta_arr.append(j0eta)
          j0phi_arr.append(j0phi)
          tj0pt_arr.append(tj0pt)
          tj0eta_arr.append(tj0eta)
          j0_sumpt_arr.append(np.array([d5,d10,d15,d20,d25,d30,d35,d40]))
          j0isPU_arr.append(isPU)
          if tj0pt > 5 :
            j0isHS_arr.append(True)
          else:
            j0isHS_arr.append(False)
          j0_Rpt_arr.append(j0Rpt)
          j0_JVF_arr.append(j0JVF)
          j0_corrJVF_arr.append(j0cJVF)
          j0_JVT_arr.append(j0JVT)

          # Get the cluster information
          clpt_arr  = []
          cleta_arr = []
          clphi_arr = []
          clln_arr  = []
          cllt_arr  = []

          #print "true jet eta = {0}, pT = {1}".format(tj0eta, tj0pt)
          #print "reco jet (eta,phi) = {0},{1}, pT = {2}".format(j0eta,j0phi,d40)
          cluster_sum = 0
          for cl_pt,cl_eta,cl_phi,cl_ln,cl_lt in zip(cl_pts,cl_etas,cl_phis,cl_lns,cl_lts):
            clpt_arr.append(cl_pt)        
            cleta_arr.append(cl_eta)        
            clphi_arr.append(cl_phi)
            cllt_arr.append(cl_lt)
            clln_arr.append(cl_ln)   
            cluster_sum = cluster_sum + cl_pt     
          #print "cluster (eta,phi) = {0},{1}".format(cl_eta,cl_phi)

          trkpt_arr  = []
          trketa_arr = []
          trkphi_arr = []
          trkm_arr  = []
          trkisPU_arr  = []
          trkisHS_arr  = []
          for trkpt,trketa,trkphi,trkm,trkisPU,trkisHS in zip(trkpts,trketas,trkphis,trkms,trkisPUs,trkisHSs):
            trkpt_arr.append(trkpt)        
            trketa_arr.append(trketa)        
            trkphi_arr.append(trkphi)
            trkm_arr.append(trkm)
            trkisPU_arr.append(trkisPU)
            trkisHS_arr.append(trkisHS)   

          #print "sum clusters = {0}".format(cluster_sum)

          j0_clpt_arr.append(clpt_arr)
          j0_cleta_arr.append(cleta_arr)
          j0_clphi_arr.append(clphi_arr)
          j0_clwidth_lon_arr.append(clln_arr)
          j0_clwidth_lat_arr.append(cllt_arr)
          j0_trkpt_arr.append(trkpt_arr)
          j0_trketa_arr.append(trketa_arr)
          j0_trkphi_arr.append(trkphi_arr)
          j0_trkm_arr.append(trkm_arr)
          j0_trkisHS_arr.append(trkisHS_arr)
          j0_trkisPU_arr.append(trkisPU_arr)
      if flag == True:
        events_of_interest = events_of_interest + 1

  save(options.submitDir+'/recopts_j0_'      + options.identifier,j0pt_arr)
  save(options.submitDir+'/recopts_jnoarea0_'+ options.identifier,jnoarea0pt_arr)
  save(options.submitDir+'/recoeta_j0_'      + options.identifier,j0eta_arr)
  save(options.submitDir+'/recophi_j0_'      + options.identifier,j0phi_arr)
  save(options.submitDir+'/truepts_j0_'      + options.identifier,tj0pt_arr)
  save(options.submitDir+'/weights_j0_'      + options.identifier,evt_wt_arr)
  save(options.submitDir+'/npvs_j0_'         + options.identifier,npv_arr)
  save(options.submitDir+'/mus_j0_'          + options.identifier,mu_arr)
  save(options.submitDir+'/At_j0_'           + options.identifier,At_arr)
  save(options.submitDir+'/j0_sumpt_'        + options.identifier,j0_sumpt_arr)
  save(options.submitDir+'/isPU_j0_'         + options.identifier,j0isPU_arr)
  save(options.submitDir+'/isHS_j0_'         + options.identifier,j0isHS_arr)
  save(options.submitDir+'/clstr_pts_j0_'    + options.identifier,np.array(j0_clpt_arr, dtype=object))
  save(options.submitDir+'/clstr_etas_j0_'   + options.identifier,np.array(j0_cleta_arr, dtype=object))
  save(options.submitDir+'/clstr_phis_j0_'   + options.identifier,np.array(j0_clphi_arr, dtype=object))
  save(options.submitDir+'/clstr_widths_lon_j0_'   + options.identifier,np.array(j0_clwidth_lon_arr, dtype=object))
  save(options.submitDir+'/clstr_widths_lat_j0_'   + options.identifier,np.array(j0_clwidth_lat_arr, dtype=object))

  save(options.submitDir+'/Rpt_j0_'         + options.identifier,j0_Rpt_arr)
  save(options.submitDir+'/JVF_j0_'         + options.identifier,j0_JVF_arr)
  save(options.submitDir+'/corrJVF_j0_'         + options.identifier,j0_corrJVF_arr)
  save(options.submitDir+'/JVT_j0_'         + options.identifier,j0_JVT_arr)

  save(options.submitDir+'/trk_pts_j0_'    + options.identifier,np.array(j0_trkpt_arr, dtype=object))
  save(options.submitDir+'/trk_etas_j0_'   + options.identifier,np.array(j0_trketa_arr, dtype=object))
  save(options.submitDir+'/trk_phis_j0_'   + options.identifier,np.array(j0_trkphi_arr, dtype=object))
  save(options.submitDir+'/trk_ms_j0_'    + options.identifier,np.array(j0_trkm_arr, dtype=object))
  save(options.submitDir+'/trk_isPU_j0_'   + options.identifier,np.array(j0_trkisPU_arr, dtype=object))
  save(options.submitDir+'/trk_isHS_j0_'   + options.identifier,np.array(j0_trkisHS_arr, dtype=object))

  print('Ev o I = {0}'.format(events_of_interest))
readRoot()
