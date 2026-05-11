import os

#读取文件夹
file_path='../datasets/train/A'
file_path_2='../datasets/val/A'
file_path_3='../datasets/test/A'
#读取文件
file_names=os.listdir(file_path) # 可以重新排序
file_names_2=os.listdir(file_path_2)
file_names_3=os.listdir(file_path_3)
print(file_names_2)
#拼接路径
file = open("../datasets/train/list.txt", 'a')
for _ in file_names:
    str =  _  +'\n'
#写入txt文件
    file.write(str)

file_2 = open("../datasets/val/list.txt", 'a')
for _ in file_names_2:
    str =  _ + '\n'
#写入txt文件
    file_2.write(str)

file_3 = open("../datasets/test/list.txt", 'a')
for _ in file_names_3:
    str =  _ + '\n'
#写入txt文件
    file_3.write(str)
file.close()

# "train_dataset/image1/train_16.png train_dataset/image2/train_16.png train_dataset/gt/train_16.png"