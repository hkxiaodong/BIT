

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import re

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomVerticalFlip
from PIL import Image

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16


def dp_txt(txt):
    # nonEnglish_regex = re.compile('[^a-zA-Z0-9\\?\\!\\,\\.@#\\+\\-=\\*\'\"><&\\$%\\(\\)\\[\\]:;]+')
    hashtag_pattern = re.compile('#[a-zA-Z0-9]+')
    at_pattern = re.compile('@[a-zA-Z0-9]+')
    http_pattern = re.compile(
        "((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")
    txt = txt.strip()
    txt_hashtag = re.sub(hashtag_pattern, '', txt)
    txt_nonat = re.sub(at_pattern, '', txt_hashtag)
    txt_nonhttp = re.sub(http_pattern, '', txt_nonat)
    txt = txt_nonhttp
    return txt


class MVSA_Single(Dataset):
    def __init__(self, txt_path, dp = False):
        fh = open(txt_path, 'r', encoding='utf-8')
        self.imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()  # 以空格进行split
            # print(words)
            name = words[0]
            label = int(words[1])
            text = ''
            for p in words[2:]:
                text += p
                text += ' '  # 先加上空格还原最初的文本，放到bert中会被处理掉
            text = text.rstrip()  # 把末尾的空格去掉
            self.imgs.append((name, label, text))
        self.dp = dp

    def __getitem__(self, index):
        name_path, emo_label, text = self.imgs[index]
        if self.dp:
            text = dp_txt(text)
        image = _transform(n_px=224)(Image.open(name_path))
        return image, text, emo_label

    def __len__(self):
        return len(self.imgs)


train_path = './MVSA_Single/train_0.9.txt'  #
#test_path = './MVSA_Single/test_0.1.txt'  #
valid_path = './MVSA_Single/valid_0.1.txt'
train_d = MVSA_Single(train_path)
test_d = MVSA_Single(valid_path)

def get_dataset(batch_size = 16, drop_last=False):
    train_data = data.DataLoader(train_d, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_data = data.DataLoader(test_d, batch_size=batch_size, shuffle=False)
    return train_data, test_data

if __name__ == '__main__':
    # training
    train_data, test_data = get_dataset(64)
    for t in range(1):
        for i, data in enumerate(train_data):
            image, text, label = data
            print(image.shape)


