import numpy as np

from . import features

class SamplerTriplet(object):
    
    def __init__(self, trajectories, border=10):
        
        self.trajectories = trajectories
        self.border = border
        
        # dump trajectories that are to short
        self.trajectories_ = [t for t in trajectories if len(t)>2*self.border]
        self.trajectories_indexes_ = [np.array([d.index for d in t]) for t in self.trajectories_]      

    def sample_triplet(self, index_window=50, border=None, return_indexes=False):
        
        if border is None:
            border = self.border 
            
        n = len(self.trajectories_) 

        count_not_found = 0
        count_too_short = 0
        while True:

            # sample one trajectory with unifrom distribution
            itp = np.random.randint(0,n)
            tp = self.trajectories_[itp]

            m = len(tp)
            
            if (m-2*border)<=0:
                count_too_short += 1
                if count_too_short>100:
                    raise RuntimeError("Could not found a long enough trajectory in more than 100 consecutive tries.") 
            count_too_short = 0

            # sample one detection with unifrom distribution
            ida = np.random.randint(border, m-border)
            da = tp[ida]

            # get the possible positive samples
            idps = []
            for idp in range(ida+1, m):
                if (tp[idp].index-da.index)>index_window:
                    break
                idps.append(idp)   

            # sample the second detection on the same trajectory with unifrom distribution
            idp = np.random.choice(idps)
            dp = tp[idp]

            idxs_remaining_traj = np.arange(n)
            idxs_remaining_traj = np.delete(idxs_remaining_traj, itp)
            np.random.shuffle(idxs_remaining_traj)

            for itn in idxs_remaining_traj:

                tn = self.trajectories_[itn]
                tn_indexes = self.trajectories_indexes_[itn]

                # checking if at least on of the remaining trajectories is close in time w.r.t the detectiosn we just sampled
                if da.index>=tn[0].index and da.index<=tn[-1].index:

                    idns = np.where(np.logical_and(tn_indexes>=da.index, tn_indexes<=(da.index+index_window)))[0]
                    idn = np.random.choice(idns)
                    dn = tn[idn]

                    if return_indexes:
                        return (itp,itn),(ida,idp,idn)
                    else:
                        return (da, dp, dn)

            # if we arrive here is because we have not found a negative sample.
            # The best option for is to sample a new set of positives
            count_not_found += 1
            if count_not_found>100:
                raise RuntimeError("Could not found a valid triplet in more than 100 consecutive tries.")

    def sample_triplet_tracklet(self, index_window=50, length_range=(5,25)):
        
        (itp,itn),(ida,idp,idn) = self.sample_triplet(index_window, self.border+length_range[1], return_indexes=True)
        
        length_a = np.random.randint(*length_range)
        length_p = np.random.randint(*length_range)
        length_n = np.random.randint(*length_range)
        
        tp = self.trajectories_[itp]
        tracklet_a = [tp[i] for i in range(ida-length_a+1, ida+1)]
        tracklet_p = [tp[i] for i in range(idp, idp+length_p)]
        
        tn = self.trajectories_[itn]
        tracklet_n = [tp[i] for i in range(idn, idn+length_n)]        
        '''
        tp = self.trajectories_[itp]
        tracklet_a = []
        for i in reversed(range(1, ida+1)):
            if (tp[ida].index-tp[i].index)<index_length:
                tracklet_a.append(tp[i])
        tracklet_a = list(reversed(tracklet_a))
        
        tracklet_p = []
        for i in range(idp,len(tp)):
            if (tp[i].index-tp[idp].index)<index_length:
                tracklet_p.append(tp[i])
      
        tn = self.trajectories_[itn]
        tracklet_n = []
        for i in range(idn,len(tn)):
            if (tn[i].index-tn[idn].index)<index_length:
                tracklet_n.append(tn[i])
        '''        
        return tracklet_a, tracklet_p, tracklet_n
    
def similarity_function_example(d1, d2):
    return features.color_histogram_similarity(d1.features["color_histogram"], d2.features["color_histogram"])
    
def statistics_detections(trajectories, border=10, index_window=50, n=1000,
                          similarity_function=similarity_function_example):
    
    sampler = SamplerTriplet(trajectories, border=border)

    pos = []
    neg = []
    samples = []
    for i in range(n):
        da, dp, dn = sampler.sample_triplet(index_window=index_window)
        samples.append((da, dp, dn))

        pos.append( similarity_function(da, dp) )
        neg.append( similarity_function(da, dn) )
        
    return pos, neg, samples

'''
def similarity_function_tracklets_example(t1, t2):
    
    
    
    return color_histogram_similarity(d1.features["color_histogram"], d2.features["color_histogram"])
    
def statistics_detections(trajectories, border=10, index_window=50, n=1000,
                          similarity_function=similarity_function_example):
    
    sampler = SamplerTriplet(trajectories, border=border)

    pos = []
    neg = []
    for i in range(n):
        da, dp, dn = sampler.sample_triplet(index_window=index_window)
    
        pos.append( similarity_function(da, dp) )
        neg.append( similarity_function(da, dn) )
        
    return pos, neg
'''    