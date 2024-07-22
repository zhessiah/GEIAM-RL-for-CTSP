import os
import shutil
import datetime
# 指定你的目录路径
directory = 'PyTorch/tensorboard_log_dir/0616_22_53/100'
print(datetime.now().strftime('%m%d_%H_%M'))
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 获取文件的完整路径
    file_path = os.path.join(directory, filename)
    
    # 检查是否为文件
    if os.path.isfile(file_path):
        # 创建一个新的目录，其名称与文件名相同
        new_directory = os.path.join(directory, os.path.splitext(filename)[0])
        os.makedirs(new_directory, exist_ok=True)
        
        # 将文件移动到新的目录中
        shutil.move(file_path, new_directory)