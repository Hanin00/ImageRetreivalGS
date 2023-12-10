import os
import shutil
# import tqdm



source_directory = "data/train_sub-23-22/walk4_step3_ged10"
# source_directory = "data/train_sub-22-21/walk4_step3_ged10"
# source_directory = "data/train_sub-21-20/walk4_step3_ged10"
# source_directory = "data/train_sub-20-19/walk4_step3_ged10"
# source_directory = "data/train_sub-19-18/walk4_step3_ged10"
# source_directory = "data/train_sub-18-17/walk4_step3_ged10"
# source_directory = "data/train_sub-17-16/walk4_step3_ged2"
# source_directory = "data/train_sub-16-13/walk4_step3_ged10"
# source_directory = "data/train_sub-13-10/walk4_step3_ged10"
# source_directory = "data/train_sub-10-7/walk4_step3_ged10"
# source_directory = "data/train_sub-7-1/walk4_step3_ged10"
destination_directory = "data/train/walk4_step3_ged10"

try:
    # 디렉토리 내의 모든 파일을 복사
    for item in os.listdir(source_directory):
        source_item = os.path.join(source_directory, item)
        destination_item = os.path.join(destination_directory, item)
        if os.path.isfile(source_item):
            shutil.copy(source_item, destination_item)
    print("source_directory: ",source_directory)
    print("파일 복사가 완료되었습니다.")
except Exception as e:
    print(f"파일 복사 중 오류 발생: {str(e)}")