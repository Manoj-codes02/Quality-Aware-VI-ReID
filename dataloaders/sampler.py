import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy

class CrossModalSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances # e.g., 4 (2 RGB, 2 IR)
        assert self.num_instances % 2 == 0, "num_instances must be even to split equally across RGB/IR"
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.rgb_index_dic = defaultdict(list)
        self.ir_index_dic = defaultdict(list)
        
        for index, (_, pid, cam, _) in enumerate(self.data_source):
            # RGB cameras: 1,2,4,5; IR cameras: 3,6 (Using our simple logic, RGB=1, IR=3)
            if cam == 3 or cam == 6:
                self.ir_index_dic[pid].append(index)
            else:
                self.rgb_index_dic[pid].append(index)
                
        # Only keep pids that have both modalities
        self.pids = list(set(self.rgb_index_dic.keys()) & set(self.ir_index_dic.keys()))

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            rgb_idxs = copy.deepcopy(self.rgb_index_dic[pid])
            ir_idxs = copy.deepcopy(self.ir_index_dic[pid])
            
            num_req = self.num_instances // 2
            
            if len(rgb_idxs) < num_req:
                rgb_idxs = np.random.choice(rgb_idxs, size=num_req, replace=True).tolist()
            if len(ir_idxs) < num_req:
                ir_idxs = np.random.choice(ir_idxs, size=num_req, replace=True).tolist()
                
            np.random.shuffle(rgb_idxs)
            np.random.shuffle(ir_idxs)
            
            batch_idxs = []
            for i in range(num_req):
                batch_idxs.append(rgb_idxs[i])
                batch_idxs.append(ir_idxs[i])
            batch_idxs_dict[pid].append(batch_idxs)
            
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return (len(self.pids) // self.num_pids_per_batch) * self.batch_size
