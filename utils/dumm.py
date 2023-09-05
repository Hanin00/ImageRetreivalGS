import networkx as nx
import pickle
import sys, os




#merge scenegraphs - 비디오에 중복되는 그래프가 많아서, 각 파일(비디오)에서 하나씩만 가져와서 파일을 만든 후, 검색을 해볼 것

import multiprocessing
import os
import pickle


'''
  데이터셋 확인을 위한 코드
  기존에 gev Dataset 생성 시 용량을 core 개수별로 fileNameList_ordered를 만들었었음
  몇 번 비디오를 사용했는지 확인 필요.
  데이터 생성시 사용한 scenegraph_1 내 번호를 다음과 같이 나타냄 
  ged10_1229_50.pkl 이와 같은 파일명이 있을 때, scenegraph_1 내 번호는 1229임

'''


# 파일 idx로 해야하는데 뭔 말도안되는 걸로함... 파일명 뒤 네자리..
# namelist = os.listdir("data/scenegraph_1/")
# # print(namelist)

# for name in namelist:
#   if(name[-8:-4]) == "1229":
#      print("name:", name)



'''
  gedPair 파일의 키([-2]) 네자리로 
  scenegraph(video name)찾음 
  scenegraph의 이름 Video list로 담고
  - scenegraph내의 프레임 수 확인
  - GEVPair의 그래프 개수 확인
'''


videoNameList = []
frameLen = 0
SgraphCnt = []
PairGraphCnt = []

namelist = os.listdir("data/GEDPair/train/walk4_step3_ged10/")
scenegraphList = os.listdir("data/scenegraph_1/")
print("len(namelist): " ,len(namelist)) # GEVPair 파일 개수
for name in namelist:
  key = name[:-4].split('_')[-2]
  for scenegraph in scenegraphList:
     if scenegraph[-8:-4] == key:
        print("name: ", name) # GEVPair에 사용된 scenegraph의 이름
        print("scenegraph: ", scenegraph)
        # videoNameList.append(scenegraph)

        with open('data/GEDPair/train/walk4_step3_ged10/'+name, 'rb') as f:          
          gevPair = pickle.load(f)
          PairGraphCnt.append(len(gevPair[0]))
          print("len(gevPair[0]): ",len(gevPair[0]))
          print("PairGraphCnt: ", PairGraphCnt)

        with open('data/scenegraph_1/'+scenegraph, 'rb') as f:
          scenegraph = pickle.load(f) 
          SgraphCnt.append(len(scenegraph[0][0]))
          # print("SgraphCnt: ",SgraphCnt)  
          print("scenegraph: ",scenegraph)  
          print("scenegraph: ",scenegraph[0][0])
          print("scenegraph: ",len(scenegraph[0][0]))

          sys.exit()    


  sys.exit()



sys.exit()









# Scenegraph 
with open('data/scenegraph_1/3069_6692501229_6692501229.pkl', 'rb') as f:
   scenegraph = pickle.load(f)
print("len(scenegraph[1][0]): " , scenegraph[1][0])
print("len(scenegraph[0]): " ,len(scenegraph[0][0]))
print("scenegraph[0][0][0]: " ,scenegraph[0][0][0])
print("nodes - scenegraph: " ,scenegraph[0][0][51].nodes(data=True))








with open('data/scenegraph_1/3069_6692501229_6692501229.pkl', 'rb') as f:
   scenegraph = pickle.load(f)

print("len(scenegraph[1][0]): " , scenegraph[1][0])
print("len(scenegraph[0]): " ,len(scenegraph[0][0]))
print("scenegraph[0][0][0]: " ,scenegraph[0][0][0])
sys.exit()



with open('data/GEDPair/train/walk4_step3_ged10/walk4_step3_ged10_1229_50.pkl', 'rb') as f:
   GEDPair = pickle.load(f)

originG = GEDPair[0][0]

print(GEDPair[0][0].nodes(data=True))
print(GEDPair[1][0].nodes(data=True))

print(GEDPair[0][0])
print(GEDPair[1][0])
print(GEDPair[2][0])


sys.exit()



with open('data/fileNameList_ordered.pkl', 'rb') as f:
  fileNameList  = pickle.load(f)

print("len(fileNameList): ", len(fileNameList))
print("len(fileNameList): ", len(fileNameList[0])) # 980


sys.exit()



# fileNameList = fileNameList[:][:20] # walk4_step3_ged10





















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
