import networkx as nx
import pickle
import sys, os




#merge scenegraphs - 비디오에 중복되는 그래프가 많아서, 각 파일(비디오)에서 하나씩만 가져와서 파일을 만든 후, 검색을 해볼 것

import multiprocessing
import os
import pickle

# def process_file(file_name):
#     vID = file_name.split('_')[-2] + '-' + file_name.split('_')[-1] + '-'
#     with open("data/GEDPair/train/walk4_step3_ged10/" + file_name, "rb") as fr:
#         tmp = pickle.load(fr)
#         return tmp[0][0], [vID + '_' + str(i) for i in range(len(tmp[0]))]

# # 병렬 처리를 위한 작업자 프로세스 수
# num_processes = multiprocessing.cpu_count()
# gevpair_dbScenegraphs = os.listdir('data/GEDPair/train/walk4_step3_ged10/') #scenegraph file이 있는 폴더명
# db = []

# with multiprocessing.Pool(processes=num_processes) as pool:
#     results = pool.map(process_file, gevpair_dbScenegraphs[:100])

# # 결과를 db와 db_idx에 추가합니다.
# for data, idx in results:
#     db.append(data)

# print("len(dataset):", len(db))






# # DB로 사용할 파일 확인
# with open("data/query_graphs.pkl", "rb") as fr: 
#   tmp = pickle.load(fr)
#   print("tmp[0]: ", tmp)
#   # print("tmp[1]: ", tmp[1])
#   # print("tmp[2]: ", tmp[2])
#   print("len(tmp): ",len(tmp)) 
  
# sys.exit()
  


# mk query graph ------
db = []
gevpair_dbScenegraphs = os.listdir('data/GEDPair/train/walk4_step3_ged10/') #scenegraph file이 있는 폴더명   
for file_name in gevpair_dbScenegraphs:
    vID = file_name.split('_')[-2] +'-' + file_name.split('_')[-1]+'-' 
    # vName = file_name.split('_')[1] # video Name
    with open("data/GEDPair/train/walk4_step3_ged10/"+file_name, "rb") as fr:
        tmp = pickle.load(fr)
        db.append(tmp[0][0])
        # print("tmp[0][0]: ", tmp[0][0])
        # print("db : ",db)
with open("data/1graph_per_video.pkl", "wb") as fr:
    pickle.dump(db, fr)

sys.exit()






#-- mk Query Graphs --

queryGraph = []
folder_path = "data/GEDPair/walk4_step3_ged5/"
filenames = os.listdir(folder_path)
for filename in filenames[:20]:
  with open(folder_path+filename, 'rb') as f:  
        data  = pickle.load(f)        
        if(len(data[0])>2):
          queryGraph.append(data[0][0])
          print(data[0][0])
with open("data/query_graphs.pkl", "wb") as fr:
    pickle.dump(queryGraph, fr)

sys.exit()


# folder_path = "data/scenegraph_1/"
# filenames = os.listdir(folder_path)
# for filename in filenames:
#   with open(folder_path+filename, 'rb') as f:  
#         data  = pickle.load(f)        
#         if(len(data[0][0])>2):
#            queryGraph.add()

#            print(data[0][0][0].edges())
#            print(data[0][0][0].edges(data='predicate'))


folder_path = "data/GEDPair/train/walk4_step3_ged10/"
filenames = os.listdir(folder_path)
for filename in filenames:
  with open(folder_path+filename, 'rb') as f:  
        data  = pickle.load(f)        
        if(len(data[0])>2):
           print(data[0][0].nodes())
           print(data[0][0].edges())
           print("==="*10)
           print(data[0][0].nodes(data='rpe'))
           print(data[0][0].edges(data='predicate'))


with open("data/GEDPair/train/walk4_step3_ged10/walk4_step3_ged10_6794_1172.pkl", 'rb') as f:  
    data  = pickle.load(f)         # predicate가 none 인 경우가 있음

    print("data[0] : ",len(data[0])) #3264
    sys.exit()
    print("data[1] : ",data[1])
    print("data[2] : ",data[2])

    sys.exit()




    for idx in range(len(data[1])):
      # print(data[0][idx].edges(data='predicate'))
      # print(data[0][idx].edges())
      # print(len(data[0][idx].edges(data='predicate')))
      # print(len(data[0][idx].edges()))
      [print (i[2]) for i in data[0][idx].edges(data="predicate")] 
      print("idx: ",idx)

      if [i[2] == None for i in data[1][idx].edges(data="predicate")] :
        print(data[0][idx].edges(data="Predicate"))
        sys.exit()
        
      
sys.exit()


# folder_path = "data/scenegraph_1/"
folder_path = "data/GEDPair/train/walk4_step3_ged10_20-40/"
filenames = os.listdir(folder_path)
for filename in filenames:
  with open(folder_path+filename, 'rb') as f:  
        data  = pickle.load(f)        
  for idx in range(len(data[0][0])):
      # print(data[0][idx].edges(data='predicate')) # GEDPair
      if len(data[1][idx].edges(data="predicate")) != len(data[1][idx].edges()):
        print(data[1][idx].edges(data="Predicate"))
        sys.exit()
      print(data[1][idx].edges(data='predicate'))

      # print(data[0][0][idx].edges(data='predicate')) # Scenegraph
      # if len(data[0][0][idx].edges(data="predicate")) != len(data[0][0][idx].edges()):
      #   print(data[0][0][idx].edges(data="Predicate"))
