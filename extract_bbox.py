import os
import os.path as opath
import pickle
from multiprocessing import Pool
import numpy as np

def get_bbox(param):
    dir_ = param["dir"]
    pkl_file = param["pkl"]
    with open(opath.join("JAAD_result", dir_, pkl_file), "rb") as f:
        data = pickle.load(f)
        bboxes = data["cls_boxes"][1]
        print(bboxes)
        select_bboxes = bboxes[bboxes[:, 4]>=0.9, :]
        select_bboxes = (select_bboxes[:, 0:4] + 0.5).astype(np.int32)
    return select_bboxes

params = []
dir_list = os.listdir("./JAAD_result")
for dir_ in dir_list:
    os.system("mkdir -p /data/JAAD_detection/"+ dir_)
    pkl_list = os.listdir(opath.join("JAAD_result", dir_))
    for pkl_file in pkl_list:
        params.append({"dir":dir_, "pkl":pkl_file})
print(params[0])
print(get_bbox(params[0]))
#pool = Pool(8)
#results = pool.map(get_bbox, params)
#for i in range(len(params)):
#    np.save(opath.join("/data/JAAD_detection/", params[i]["dir"], params[i]["pkl"].split(".")[0]+".npy"), np.array(results[i]))
