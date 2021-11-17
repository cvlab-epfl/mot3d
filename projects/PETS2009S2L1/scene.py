#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
from mot3d.utils import utils
from mot3d.types import TrajectoryView

class Scene(object):
    def __init__(self, completed_after=30):
        
        self.completed_after = completed_after
        
        self.id_last = 0
        self.active = {}
        self.completed = {}
        
        self.id_last_tracklet = 0
        self.tracklets = {}    
        
    def store_tracklet(self, tracklet):
        self.tracklets[self.id_last_tracklet] = tracklet
        self.id_last_tracklet += 1
        
    def update(self, id, trajectory):
        
        # update an existing trajectory or create a new one
        if id in self.active:
            self.active[id] = trajectory            
        else:
            self.active[self.id_last] = trajectory
            self.id_last += 1  
            
    def update_completed(self, time_index):
        
        # label as completed the trajectories that have not been updated in a while
        for id, trajectory in list(self.active.items()):
            if (trajectory[-1].index+self.completed_after)<time_index:
                self.completed[id] = self.active[id]
                del self.active[id] 
                
    def save(self, filename):
        utils.pickle_write(filename, {"active":self.active, 
                                      "completed":self.completed,
                                      "trackelts":self.tracklets})    
        
    def load(self, filename):
        data = utils.pickle_read(filename)
        self.active = data['active']
        self.completed = data['completed']
        self.trackelts = data['trackelts']  
        self.id_last = len(self.active)+len(self.completed)
        self.id_last_tracklet = len(self.trackelts)

class Scene2d3d(Scene):
    
    def __init__(self, scene3d, completed_after=30):
        super(Scene2d3d, self).__init__(completed_after)
        self.scene3d = scene3d
        
    def update_completed(self, time_index):
        
        # label as completed the trajectories that have not been updated in a while
        for id, trajectory in list(self.active.items()):
            if isinstance(trajectory[-1], TrajectoryView):
                if trajectory[-1].id_view in self.scene3d.active:
                    pass
                else:
                    self.completed[id] = self.active[id]
                    del self.active[id]
                
            elif (trajectory[-1].index+self.completed_after)<time_index:
                self.completed[id] = self.active[id]
                del self.active[id]    