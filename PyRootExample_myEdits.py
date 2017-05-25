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

  nentries = tree.GetEntries()
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

      # cluster branches
      j0_clpt_branch  = getattr(tree,'j0_clpt')
      j0_cleta_branch = getattr(tree,'j0_cleta')
      j0_clphi_branch = getattr(tree,'j0_clphi')
      j0_clwidth_lat_branch = getattr(tree,'j0_cllatwidth')
      j0_clwidth_lon_branch = getattr(tree,'j0_cllongwidth')
   
      # truth jet info
      tj0pt_branch  = getattr(tree,'tj0pt')
      tj0eta_branch = getattr(tree,'tj0eta')

      if jentry % 5000 == 0:
        print 'jentry = {0}'.format(jentry)
      for cl_pts,cl_etas,cl_phis,j0pt,jnoarea0pt,j0eta,j0phi,tj0pt,tj0eta,d5,d10,d15,d20,d25,d30,d35,d40,At,isPU,cl_lns,cl_lts in zip(j0_clpt_branch,
                                                     j0_cleta_branch,j0_clphi_branch,j0pt_branch,
                                                     jnoarea0pt_branch,j0eta_branch,j0phi_branch,tj0pt_branch,tj0eta_branch,
                                                     j0sumpt5_branch,j0sumpt10_branch,j0sumpt15_branch,j0sumpt20_branch,
                                                     j0sumpt25_branch,j0sumpt30_branch,j0sumpt35_branch,j0sumpt40_branch,
                                                     j0area_branch,j0isPU_branch,j0_clwidth_lon_branch,j0_clwidth_lat_branch):

          # Eta cut
          if np.absolute(tj0eta) < 2.5 or j0pt < 5:
          #print "Ooops! We're only looking at forward jets with reco Pt > 5 GeV!"
            continue

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

          #print "sum clusters = {0}".format(cluster_sum)

          j0_clpt_arr.append(clpt_arr)
          j0_cleta_arr.append(cleta_arr)
          j0_clphi_arr.append(clphi_arr)
          j0_clwidth_lon_arr.append(clln_arr)
          j0_clwidth_lat_arr.append(cllt_arr)

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

readRoot()
