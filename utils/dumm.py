import networkx as nx
import pickle
import sys, os


#merge scenegraphs - 비디오에 중복되는 그래프가 많아서, 각 파일(비디오)에서 하나씩만 가져와서 파일을 만든 후, 검색을 해볼 것
import multiprocessing
import os
import pickle


'''
    동일한 파일 명, 범위의 GEDPair의 파일명 확인
'''

# ged02 = []
# ged05 = []
# ged10 = []

# # 주어진 문자열
# for foldername in os.listdir('data/train_origin-13-10/'):
#     file_names = os.listdir('data/train_origin-13-10/'+foldername)
#     for fName in file_names:
#     # 문자열을 "_"를 기준으로 분할하고, 뒤에서 두 번째 부분을 추출합니다.
#       split_result = fName.split("_")
#       if len(split_result) >= 2:
#         if foldername == "walk4_step3_ged2":
#           ged02.append('_'.join(split_result[-2:]))
#         elif foldername == 'walk4_step3_ged5':
#           ged05.append('_'.join(split_result[-2:]))
#         else:
#           ged10.append('_'.join(split_result[-2:]))
#       else:
#           extracted_value = None  # 분할된 결과가 충분하지 않을 때 None을 반환하거나 다른 처리를 수행할 수 있습니다.
#     # 추출된 값을 리스트로 저장합니다.


# common_items = []
# for item in ged02:
#     if item in ged05 and item in ged10:
#         common_items.append(item)
        
# print(common_items)
# print(len(common_items))


# sys.exit()

'''
  scenegraph 개수 확인

'''

fileNameList = ['2670202446', '10111475873', '5112223863', '8715080639', '3736942700', '7478064282', '5770462342', '5162090823', '4692611470', '5826071152', '5619125415', '3536560404', '8035629220', '6410117677', '9410134787', '10574181535', '5249474618', '8777792406', '7002082753', '6027450794', '6514888703', '3396669412']

nfileNameList = [item+'.json.pkl' for item in fileNameList]

scnegraphCnt = 0
resultList = []
for item in nfileNameList:
# for foldername in os.listdir('data/scenegraph/'):
  with open("data/scenegraph"+"/"+item,  "rb") as fr:
    tmp = pickle.load(fr)
    scnegraphCnt += len(tmp[0])
print("scenegraphCnt:" ,scnegraphCnt)
sys.exit()
      
#     # for j in range(num if len(file_names)>=num else len(file_names) ):
#     for j in range(len(file_names) ):
#             filename = file_names[j]
#     #  for filename in os.listdir('data/GEDPair/train/'+foldername):
#             with open("data/train"+"/"+foldername+'/'+filename, "rb") as fr:
#                 tmp = pickle.load(fr)                    
#                 for i in range(len(tmp[0])):    
#                     dataset[0].append(tmp[0][i])

sys.exit()


'''
  파일을 옮기는 코드
'''
import os
import shutil

# file_names = []
# file_list_path = "intersection_dataset.txt"
# with open(file_list_path, "r") as file:
#         for line in file:
#             file_names.append(line.strip(','))

file_names= ['9731127827_506.pkl', '6236608754_100.pkl', '4983571737_185.pkl', '10531891206_302.pkl', '7230741446_547.pkl', '2942623423_403.pkl', '4366402470_302.pkl', '7714458016_100.pkl', '4177404442_403.pkl', '11176411044_391.pkl', '6430774273_100.pkl', '12527995483_100.pkl', '2663659410_504.pkl', '2408164669_100.pkl', '12648273934_201.pkl', '5152794851_100.pkl', '6903781188_201.pkl', '7714458016_255.pkl', '10636949046_302.pkl', '10084164905_403.pkl', '3052284826_302.pkl', '9670369477_201.pkl', '3478131928_201.pkl', '5340686202_100.pkl', '4857066914_302.pkl', '12301427405_169.pkl', '6972688351_302.pkl', '2417388607_201.pkl', '4857066914_201.pkl', '6555498113_302.pkl', '4242713789_201.pkl', '12648273934_302.pkl', '6430774273_201.pkl', '5139493061_302.pkl', '3951427609_100.pkl', '6991115222_403.pkl', '5165840822_100.pkl', '6243300202_183.pkl', '4177404442_201.pkl', '3066398259_100.pkl', '6838195930_100.pkl', '3845177702_526.pkl', '3557607884_201.pkl', '10084164905_302.pkl', '2801868426_100.pkl', '4366402470_403.pkl', '5340686202_605.pkl', '7983797184_403.pkl', '5195700916_201.pkl', '9731127827_100.pkl', '6430774273_353.pkl', '3402816015_302.pkl', '4177404442_100.pkl', '2934818100_201.pkl', '6555498113_329.pkl', '2668803970_523.pkl', '6991115222_504.pkl', '4752448635_302.pkl', '2668803970_100.pkl', '6121786803_504.pkl', '6385308823_409.pkl', '2417388607_302.pkl', '7927942358_302.pkl', '4697224919_201.pkl', '10735513724_201.pkl', '4688033528_100.pkl', '4697224919_100.pkl', '6322430094_100.pkl', '8533666429_100.pkl', '4609284606_100.pkl', '6121786803_605.pkl', '3052284826_201.pkl', '6322430094_201.pkl', '4697224919_326.pkl', '11503848155_201.pkl', '4114876025_302.pkl', '10679855585_302.pkl', '4857066914_100.pkl', '3951427609_201.pkl', '5195700916_100.pkl', '5513879635_100.pkl', '6903781188_100.pkl', '6236608754_386.pkl', '6991115222_100.pkl', '7230741446_504.pkl', '12527995483_407.pkl', '4609284606_201.pkl', '5192910012_302.pkl', '5671008554_100.pkl', '4204053300_100.pkl', '5454851316_516.pkl', '2583337639_395.pkl', '2514339576_403.pkl', '2663659410_302.pkl', '10246204023_306.pkl', '7983797184_485.pkl', '4280307962_201.pkl', '4280307962_383.pkl', '4997487936_100.pkl', '2525646520_201.pkl', '5454851316_201.pkl', '8533666429_201.pkl', '4752448635_100.pkl', '7604314754_100.pkl', '5211277413_100.pkl', '12527995483_403.pkl', '3174267702_201.pkl', '4609284606_302.pkl', '3438747281_315.pkl', '6972688351_403.pkl', '4177404442_302.pkl', '5139493061_403.pkl', '2408164669_302.pkl', '8968804598_100.pkl', '6827458013_401.pkl', '2583337639_100.pkl', '5513879635_210.pkl', '4373878857_201.pkl', '12527995483_201.pkl', '3246559605_302.pkl', '6972688351_201.pkl', '10531891206_403.pkl', '5454851316_403.pkl', '9531658428_302.pkl', '2934818100_379.pkl', '2525646520_235.pkl', '10636949046_375.pkl', '2942623423_461.pkl', '10679855585_441.pkl', '12681261204_368.pkl', '5211277413_251.pkl', '3478131928_302.pkl', '9186559718_201.pkl', '6502487733_100.pkl', '4780185309_302.pkl', '4227107409_100.pkl', '2668803970_201.pkl', '2771309743_566.pkl', '2942623423_100.pkl', '5340686202_504.pkl', '6903781188_395.pkl', '10531891206_419.pkl', '10679855585_100.pkl', '4780185309_201.pkl', '5139493061_201.pkl', '3438747281_201.pkl', '3033055753_411.pkl', '5340686202_201.pkl', '3758307290_201.pkl', '6385308823_403.pkl', '9189371905_201.pkl', '5130149072_302.pkl', '4227107409_201.pkl', '7604314754_201.pkl', '5152794851_302.pkl', '2583337639_302.pkl', '3557607884_100.pkl', '12648273934_100.pkl', '5533831551_201.pkl', '2738551693_100.pkl', '5504106604_201.pkl', '3066398259_302.pkl', '6555498113_201.pkl', '2663659410_100.pkl', '10246204023_201.pkl', '4373878857_302.pkl', '3066398259_201.pkl', '5533831551_100.pkl', '6385308823_201.pkl', '11176411044_201.pkl', '4752448635_504.pkl', '10531891206_100.pkl', '2738551693_201.pkl', '2408164669_377.pkl', '6322430094_403.pkl', '7281947184_100.pkl', '3951427609_403.pkl', '6385308823_302.pkl', '3845177702_100.pkl', '12622416203_100.pkl', '6322430094_302.pkl', '5152794851_405.pkl', '7983797184_302.pkl', '6838195930_201.pkl', '11176411044_100.pkl', '7230741446_201.pkl', '11503848155_302.pkl', '3052284826_100.pkl', '3402816015_359.pkl', '2514339576_201.pkl', '10084164905_446.pkl', '7230741446_403.pkl', '4697224919_302.pkl', '7714345064_401.pkl', '2801868426_292.pkl', '2514339576_100.pkl', '4366402470_201.pkl', '2934818100_100.pkl', '4780185309_403.pkl', '5139493061_100.pkl', '5130149072_201.pkl', '7604314754_312.pkl', '12681261204_302.pkl', '8260718865_201.pkl', '4752448635_584.pkl', '14026402694_302.pkl', '4227107409_302.pkl', '5533831551_302.pkl', '8306224570_302.pkl', '10246204023_100.pkl', '3438747281_302.pkl', '9731127827_201.pkl', '3845177702_403.pkl', '4114876025_357.pkl', '5533831551_308.pkl', '3033055753_201.pkl', '11176411044_302.pkl', '7927942358_201.pkl', '3052284826_329.pkl', '3246559605_201.pkl', '9670369477_302.pkl', '5340686202_403.pkl', '4772153724_100.pkl', '2801868426_201.pkl', '6121786803_403.pkl', '6385308823_100.pkl', '5192910012_403.pkl', '6322430094_456.pkl', '5152794851_201.pkl', '3174267702_424.pkl', '4766590433_100.pkl', '2771309743_504.pkl', '2514339576_302.pkl', '4227107409_403.pkl', '5139493061_486.pkl', '8665067833_100.pkl', '2663659410_517.pkl', '3895899954_302.pkl', '6991115222_201.pkl', '5288357456_201.pkl', '10735513724_100.pkl', '8260718865_100.pkl', '6121786803_100.pkl', '2771309743_201.pkl', '12681261204_100.pkl', '2698153481_373.pkl', '5165840822_201.pkl', '4204053300_201.pkl', '9731127827_403.pkl', '6121786803_302.pkl', '11503848155_100.pkl', '7604314754_302.pkl', '3758307290_100.pkl', '7983797184_100.pkl', '9531658428_404.pkl', '3246559605_100.pkl', '9186559718_302.pkl', '4280307962_100.pkl', '5513879635_201.pkl', '3478131928_335.pkl', '3845177702_504.pkl', '5192910012_100.pkl', '9531658428_100.pkl', '5288357456_100.pkl', '6555498113_100.pkl', '5814217146_201.pkl', '5814217146_302.pkl', '4983571737_100.pkl', '2771309743_403.pkl', '2514339576_409.pkl', '6972688351_100.pkl', '4366402470_100.pkl', '8307352463_100.pkl', '5454851316_302.pkl', '7714345064_201.pkl', '2668803970_504.pkl', '3438747281_100.pkl', '9731127827_302.pkl', '6121786803_617.pkl', '6991115222_302.pkl', '10246204023_302.pkl', '2934818100_302.pkl', '4609284606_303.pkl', '8890945814_201.pkl', '6021779287_201.pkl', '10679855585_403.pkl', '4242713789_403.pkl', '8306224570_100.pkl', '14026402694_201.pkl', '7983797184_201.pkl', '10679855585_201.pkl', '9670369477_424.pkl', '4752448635_403.pkl', '4114876025_201.pkl', '10531891206_201.pkl', '3402816015_100.pkl', '3174267702_302.pkl', '5504106604_100.pkl', '4242713789_302.pkl', '8260718865_453.pkl', '6827458013_302.pkl', '4177404442_570.pkl', '2698153481_302.pkl', '8890945814_100.pkl', '3899884605_100.pkl', '2408164669_201.pkl', '10084164905_100.pkl', '4177404442_504.pkl', '5814217146_403.pkl', '3174267702_100.pkl', '12681261204_201.pkl', '6236608754_201.pkl', '5814217146_431.pkl', '10636949046_100.pkl', '4280307962_302.pkl', '5192910012_498.pkl', '10735513724_541.pkl', '7714458016_201.pkl', '9531658428_403.pkl', '7714345064_302.pkl', '4204053300_302.pkl', '3478131928_100.pkl', '9531658428_201.pkl', '4242713789_100.pkl', '7230741446_302.pkl', '2417388607_100.pkl', '3845177702_302.pkl', '5130149072_100.pkl', '5288357456_302.pkl', '5814217146_100.pkl', '6827458013_100.pkl', '8306224570_483.pkl', '8306224570_403.pkl', '3033055753_100.pkl', '9670369477_100.pkl', '3951427609_302.pkl', '2668803970_302.pkl', '3033055753_302.pkl', '8189252857_100.pkl', '3402816015_201.pkl', '5165840822_302.pkl', '2583337639_201.pkl', '6121786803_201.pkl', '8306224570_201.pkl', '6236608754_302.pkl', '7714345064_100.pkl', '3845177702_201.pkl', '4227107409_423.pkl', '3895899954_100.pkl', '7927942358_403.pkl', '6490763929_201.pkl', '3914555606_100.pkl', '10735513724_302.pkl', '4373878857_100.pkl', '14026402694_100.pkl', '2698153481_201.pkl', '3033055753_403.pkl', '3246559605_316.pkl', '6903781188_302.pkl', '9670369477_403.pkl', '3895899954_201.pkl', '2771309743_302.pkl', '4242713789_408.pkl', '12648273934_348.pkl', '6430774273_302.pkl', '2771309743_100.pkl', '6827458013_201.pkl', '4780185309_454.pkl', '2698153481_100.pkl', '6991115222_588.pkl', '8307352463_201.pkl', '9731127827_504.pkl', '6972688351_438.pkl', '12527995483_302.pkl', '7230741446_100.pkl', '4557315741_100.pkl', '5192910012_201.pkl', '7927942358_100.pkl', '4997487936_174.pkl', '10636949046_201.pkl', '10735513724_504.pkl', '5504106604_278.pkl', '5211277413_201.pkl', '4114876025_100.pkl', '5340686202_302.pkl', '2417388607_349.pkl', '6021779287_100.pkl', '2663659410_403.pkl', '5152794851_403.pkl', '9186559718_378.pkl', '5165840822_372.pkl', '8260718865_403.pkl', '10735513724_403.pkl', '4780185309_100.pkl', '4752448635_201.pkl', '2525646520_100.pkl', '3951427609_410.pkl', '3174267702_403.pkl', '5454851316_504.pkl', '7927942358_412.pkl', '4204053300_403.pkl', '5454851316_100.pkl', '10084164905_201.pkl', '9186559718_100.pkl', '6243300202_100.pkl', '4204053300_450.pkl', '3895899954_403.pkl', '4373878857_364.pkl', '2942623423_201.pkl', '6490763929_100.pkl', '2663659410_201.pkl', '8260718865_302.pkl', '6490763929_247.pkl', '12301427405_100.pkl', '2668803970_403.pkl', '9189371905_100.pkl', '6021779287_283.pkl', '2942623423_302.pkl', '3895899954_426.pkl']

# 특정 리스트에 파일명을 담습니다.
# file_names_to_move  = ["file1.txt", "file2.txt", "file3.txt"]  # 파일명을 적절히 업데이트하세요.

# 원본 폴더와 대상 폴더 경로를 정의합니다.
source_folder = "data/train_origin-13-10/walk4_step3_ged2"
target_folder = "data/train/walk4_step3_ged2"

# 각 파일명을 순회하면서 이동 작업을 수행합니다.
for file_name in file_names:
    file_name = 'walk4_step3_ged2_'+file_name
    # 원본 파일 경로
    source_file_path = os.path.join(source_folder, file_name)
    # 대상 파일 경로
    target_file_path = os.path.join(target_folder, file_name)
    
    try:
        # 파일 이동 작업 수행
        shutil.copy(source_file_path, target_file_path)
        print(f"파일 '{file_name}'을 복사했습니다.")
    except FileNotFoundError:
        print(f"파일 '{file_name}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일 이동 중 오류 발생: {e}")

# 모든 파일 이동이 완료되면 대상 폴더에 목록이 비어있는지 확인할 수 있습니다.
if not os.listdir(source_folder):
    print(f"{source_folder} 폴더가 비어있습니다.")



sys.exit()













dataset = [[], [], []]
for foldername in os.listdir('data/train/'):
    file_names = os.listdir('data/train/'+foldername)
    # for j in range(num if len(file_names)>=num else len(file_names) ):
    for j in range(len(file_names) ):
            filename = file_names[j]
    #  for filename in os.listdir('data/GEDPair/train/'+foldername):
            with open("data/train"+"/"+foldername+'/'+filename, "rb") as fr:
                tmp = pickle.load(fr)                    
                for i in range(len(tmp[0])):    
                    dataset[0].append(tmp[0][i])
                    dataset[1].append(tmp[1][i])
                    dataset[2].append(sum(tmp[2][i])) #GEV -> GED
                    print("dataset[0]: " ,dataset[0])
                    print("dataset[1]: " ,dataset[1])
                    print("dataset[2]: " ,dataset[2])
                        
sys.exit()














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
