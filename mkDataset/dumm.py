
import sys, os
import pickle
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import networkx as nx
import multiprocessing as mp

from surel_gacc import run_walk
from utils.mkGraphRPE import *
import random
import math
import os

import pickle
import multiprocessing




#R_BFS 로 서브 그래프 생성

with open(file='data/dataset01/walk4_step3_ged10/walk4_step3_ged10_10679855585_442_884.pickle', mode='rb') as f:
    data=pickle.load(f)
    print(data)
# edge class txtemb load <- mkScenegraph에서 만든 것
# with open('data/dataset01/walk4_step3_ged10/walk4_step3_ged10_10679855585_442_2.pkl', 'rb') as f:
# with open('data/meta_dataset01/walk4_step3_ged10/10679855585_2.pickle', 'rb') as f:




sys.exit()


import os, sys

#폴더 내 파일들의 생성시간 min, max
from datetime import datetime
def get_oldest_and_newest_file_times(directory_path):
    try:
        # 폴더 내의 파일 리스트를 가져옴
        files = os.listdir(directory_path)

        # 파일들의 생성 시간을 기록할 리스트 초기화
        file_times = []

        # 파일들의 생성 시간을 가져와서 리스트에 추가
        for file in files:
            file_path = os.path.join(directory_path, file)
            creation_time = os.path.getctime(file_path)
            file_times.append((file, creation_time))

        # 파일들의 생성 시간을 기준으로 정렬
        sorted_file_times = sorted(file_times, key=lambda x: x[1])

        # 가장 과거와 가장 최근 파일의 정보를 가져옴
        oldest_file, oldest_time = sorted_file_times[0]
        newest_file, newest_time = sorted_file_times[-1]

        # datetime 객체로 변환
        oldest_datetime = datetime.fromtimestamp(oldest_time)
        newest_datetime = datetime.fromtimestamp(newest_time)

        return oldest_file, oldest_datetime, newest_file, newest_datetime

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# 사용 예시
directory_path = 'data/dataset01/walk4_step3_ged10'  # 폴더 경로를 적절히 수정

result = get_oldest_and_newest_file_times(directory_path)

if result:
    oldest_file, oldest_time, newest_file, newest_time = result

    print(f"Oldest File: {oldest_file}, Created at: {oldest_time}")
    print(f"Newest File: {newest_file}, Created at: {newest_time}")






sys.exit()

#이름 파일에서 scenegraph와 subgraph 개수 저장
def calculate_total_graph_counts(file_names):
    total_scenegraphs = 0
    total_subgraphs = 0

    for file_name in file_names:
        # 파일명에서 필요한 부분을 추출하여 처리
        processed_name = file_name.replace("walk4_step3_ged10_", "").replace(".pkl", "")
        parts = processed_name.split("_")

        if len(parts) >= 2:
            scenegraph_count = int(parts[-2])
            subgraph_count = int(parts[-1])

            total_scenegraphs += scenegraph_count
            total_subgraphs += subgraph_count

    return total_scenegraphs, total_subgraphs

def main():
    # file_names.txt 파일을 읽어들임
    with open('file_names.txt', 'r') as file:
        file_names = file.read().splitlines()

    total_scenegraphs, total_subgraphs = calculate_total_graph_counts(file_names)

    print(f"Total Scenegraphs: {total_scenegraphs}")
    print(f"Total Subgraphs: {total_subgraphs}")

if __name__ == "__main__":
    main()


sys.exit()




#폴더 내 이름 모두 읽어서 파일 저장
def save_file_names_to_text(directory_path, output_file_path):
    try:
        # 폴더 내의 파일명을 읽어옴
        file_names = os.listdir(directory_path)

        # 텍스트 파일에 파일명을 저장
        with open(output_file_path, 'w') as output_file:
            for file_name in file_names:
                output_file.write(file_name + '\n')

        print(f"File names saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# 사용 예시
directory_path = 'data/dataset01/walk4_step3_ged10'  # 폴더 경로를 적절히 수정
output_file_path = 'file_names.txt'  # 저장할 텍스트 파일의 경로 및 파일명을 지정

save_file_names_to_text(directory_path, output_file_path)


sys.exit()














