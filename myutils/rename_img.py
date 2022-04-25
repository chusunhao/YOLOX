import os

# path为批量文件的文件夹的路径
root_path = './datasets/SPARK/val'
new_path = './datasets/SPARK/val'

for sat_name in os.listdir(root_path):

    path = os.path.join(root_path, sat_name)
    # 文件夹中所有文件的文件名
    file_names = os.listdir(path)

    print(len(file_names))
    # 外循环遍历所有文件名，内循环遍历每个文件名的每个字符
    for name in file_names:
        index = name.split('_')[1]
        new_name = sat_name + '_' + index + '_img.png'
        os.renames(os.path.join(path, name), os.path.join(new_path, new_name))
        print(f"image {index} Done!")
