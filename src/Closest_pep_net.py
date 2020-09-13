import numpy as np
import scipy.stats as ss 
import scipy.sparse.linalg
import scipy.sparse as sps
from MyEncode import blo_encode_920

def closest_pep_net(peptide,pep_net_dict,pep_length = 9):
    if peptide in pep_net_dict.keys():
        tmp = pep_net_dict[peptide].tolist()
        tmp.append(0)
        return(tmp)
    else:
        distances_to_pep = {}
        peptide_squeeze = np.array(blo_encode_920(peptide)).reshape(-1,21*pep_length,1).squeeze()
        for pep in pep_net_dict.keys():
            tmp = np.array(blo_encode_920(pep)).reshape(-1,21*pep_length,1).squeeze()
            distance = scipy.spatial.distance.euclidean(peptide_squeeze,tmp)
            #distance = scipy.spatial.distance.correlation(peptide_squeeze,tmp)
            distances_to_pep[pep] = distance
        ##找到距离最近的那个peptide       
        pep_list = sorted(distances_to_pep.items(),key = lambda item:item[1],reverse = False)
        closest_pep = pep_list[0][0]
        tmp = pep_net_dict[closest_pep].tolist()
        tmp.append(pep_list[0][1])
        
        
        return(tmp)
        #return(dist)
      

#mhc_network = [mhci.hla_between.median(),mhci.hla_close.median(),mhci.hla_degree.median(), mhci.hla_evcent.median()]
def mhc_net(mhc,mhci,mhc_net_dict, pep_length = 9):
    mhc_network = [mhci.hla_between.median(),mhci.hla_close.median(),mhci.hla_degree.median(), mhci.hla_evcent.median(),'median']
    if mhc in mhc_net_dict.keys():
        tmp = mhc_net_dict[mhc].tolist()
        tmp.append('self')
        return tmp
    else:
        return mhc_network