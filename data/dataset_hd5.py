import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py


class IpaintDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 读取hd5文件数据并返回
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]

        return data


# if __name__ == '__main__':

#     import os
#     import numpy as np

#     folder_path = 'data'  
#     file_paths = []

#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.h5'):  
#             file_path = os.path.join(folder_path, file_name)
#             file_paths.append(file_path)

#     pre_dataset = PredDataset(file_paths)

#     BATCH_SIZE = 64

#     pre_dataloader = DataLoader(
#         pre_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         drop_last=True
#     )
