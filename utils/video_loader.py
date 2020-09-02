from __future__ import print_function, absolute_import

import json
import os

from torch.utils.data import Dataset
import random
random.seed(1)
from PIL import Image
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
np.random.seed(1)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, data_name='mars', seq_len=15, sample='evenly', transform=None, transform2=None, type=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.transform2 = transform2
        self.type = type
        self.path = ""
        self.data_name = data_name

        if data_name == 'mars':
            if self.type == "train":
                self.path = "./data/datasets/keypoint_train.json"
            else:
                self.path = "./data/datasets/keypoint_test.json"
        elif data_name == 'ilidsvid':
            self.path = "./data/datasets/keypoint_ilids.json"
        elif data_name == 'prid':
            self.path = "./data/datasets/keypoint_prid.json"

        with open(self.path, 'r', encoding='utf8') as load_f:
            json_str = json.load(load_f)
            self.keypoint_dict = json.loads(json_str)
            print("key point load success!")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            imgs = []
            head_map_list = []
            body_map_list = []
            leg_map_list = []
            for i, index in enumerate(indices):
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                img, flag, x1, y1 = self.transform2(img)
                # if i == 0:
                if self.data_name == 'mars':
                    image_id = os.path.basename(img_path)
                    image_id = image_id[0:-4]
                    person_id = image_id[0:4]
                    tracklet_id = image_id[7:11]
                    keypoint = self.keypoint_dict[person_id][tracklet_id][image_id]
                    keypoint = list(keypoint)

                elif self.data_name == 'ilidsvid':
                    image = os.path.basename(img_path)
                    image = image[0:-4]
                    person_id = image[11:14]
                    cam_id = image[0:4]
                    image_id = image[15:]
                    keypoint = self.keypoint_dict[cam_id][person_id][image_id]
                    keypoint = list(keypoint)
                    keypoint = [i * 2 for i in keypoint]
                    # print(img_paths)

                elif self.data_name == 'prid':
                    image_split = os.path.split(img_path)
                    image_id = image_split[1][0:4]
                    person_split = os.path.split(image_split[0])
                    person_id = person_split[1][7:]
                    cam_split = os.path.split(person_split[0])
                    cam_id = cam_split[1][:]
                    keypoint = self.keypoint_dict[cam_id][person_id][image_id]
                    keypoint = list(keypoint)
                    keypoint = [i * 2 for i in keypoint]

                if flag == True:
                    keypoint = [i * 1.125 for i in keypoint]
                    keypoint[2] = min(255, max(0, keypoint[2] - y1))
                    keypoint[6] = min(255, max(0, keypoint[6] - y1))
                    keypoint[10] = min(255, max(0, keypoint[10] - y1))
                    keypoint[3] = min(255, max(0, keypoint[3] - y1))
                    keypoint[7] = min(255, max(0, keypoint[7] - y1))
                    keypoint[11] = min(255, max(0, keypoint[11] - y1))
                    keypoint[0] = min(127, max(0, keypoint[0] - x1))
                    keypoint[4] = min(127, max(0, keypoint[4] - x1))
                    keypoint[8] = min(127, max(0, keypoint[8] - x1))
                    keypoint[1] = min(127, max(0, keypoint[1] - x1))
                    keypoint[5] = min(127, max(0, keypoint[5] - x1))
                    keypoint[9] = min(127, max(0, keypoint[9] - x1))
                for index, point in enumerate(keypoint):
                    keypoint[index] = int(point / 32)
                head_map = np.zeros(shape=(1, 8, 4))
                body_map = np.zeros(shape=(1, 8, 4))
                leg_map = np.zeros(shape=(1, 8, 4))
                head_map[:, keypoint[2]:(keypoint[3]+1), keypoint[0]:(keypoint[1]+1)] = 1
                body_map[:, keypoint[6]:(keypoint[7]+1), keypoint[4]:(keypoint[5]+1)] = 1
                leg_map[:, keypoint[10]:(keypoint[11]+1), keypoint[8]:(keypoint[9]+1)] = 1
                head_map_list.append(torch.Tensor(head_map))
                body_map_list.append(torch.Tensor(body_map))
                leg_map_list.append(torch.Tensor(leg_map))

                if self.transform is not None:
                    img = self.transform(img)

                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            head_map_list = torch.cat(head_map_list, dim=0)
            body_map_list = torch.cat(body_map_list, dim=0)
            leg_map_list = torch.cat(leg_map_list, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid, head_map_list, body_map_list, leg_map_list

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index=0
            frame_indices = list(range(num))
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_stack = []
            head_map_stack = []
            body_map_stack = []
            leg_map_stack = []
            for indices in indices_list:
                imgs = []
                head_map_list = []
                body_map_list = []
                leg_map_list = []
                for i, index in enumerate(indices):
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)

                    # if i == 0:
                    if self.data_name == 'mars':
                        image_id = os.path.basename(img_path)
                        image_id = image_id[0:-4]
                        person_id = image_id[0:4]
                        tracklet_id = image_id[7:11]
                        keypoint = self.keypoint_dict[person_id][tracklet_id][image_id]
                        keypoint = list(keypoint)

                    elif self.data_name == 'ilidsvid':
                        image = os.path.basename(img_path)
                        image = image[0:-4]
                        person_id = image[11:14]
                        cam_id = image[0:4]
                        image_id = image[15:]
                        keypoint = self.keypoint_dict[cam_id][person_id][image_id]
                        keypoint = list(keypoint)
                        keypoint = [i * 2 for i in keypoint]
                        # print(img_paths)

                    elif self.data_name == 'prid':
                        image_split = os.path.split(img_path)
                        image_id = image_split[1][0:4]
                        person_split = os.path.split(image_split[0])
                        person_id = person_split[1][7:]
                        cam_split = os.path.split(person_split[0])
                        cam_id = cam_split[1][:]
                        keypoint = self.keypoint_dict[cam_id][person_id][image_id]
                        keypoint = list(keypoint)
                        keypoint = [i * 2 for i in keypoint]

                    for index, point in enumerate(keypoint):
                        keypoint[index] = int(point / 32)
                    head_map = np.zeros(shape=(1, 8, 4))
                    body_map = np.zeros(shape=(1, 8, 4))
                    leg_map = np.zeros(shape=(1, 8, 4))
                    head_map[:, keypoint[2]:(keypoint[3] + 1), keypoint[0]:(keypoint[1] + 1)] = 1
                    body_map[:, keypoint[6]:(keypoint[7] + 1), keypoint[4]:(keypoint[5] + 1)] = 1
                    leg_map[:, keypoint[10]:(keypoint[11] + 1), keypoint[8]:(keypoint[9] + 1)] = 1

                    head_map_list.append(torch.Tensor(head_map))
                    body_map_list.append(torch.Tensor(body_map))
                    leg_map_list.append(torch.Tensor(leg_map))

                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                head_map_list = torch.cat(head_map_list, dim=0)
                body_map_list = torch.cat(body_map_list, dim=0)
                leg_map_list = torch.cat(leg_map_list, dim=0)
                imgs_stack.append(imgs)
                head_map_stack.append(head_map_list)
                body_map_stack.append(body_map_list)
                leg_map_stack.append(leg_map_list)

            imgs_array = torch.stack(imgs_stack)
            head_map_array = torch.stack(head_map_stack)
            body_map_array = torch.stack(body_map_stack)
            leg_map_array = torch.stack(leg_map_stack)
            return imgs_array, pid, camid, head_map_array, body_map_array, leg_map_array

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))









