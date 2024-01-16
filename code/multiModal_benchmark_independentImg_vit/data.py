import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from torchvision import transforms
from torchvision.io import read_image
# from PIL import Image
from collections import defaultdict
from transformers import BertTokenizer
import pandas as pd
# from util import OCR
import torch.nn.functional as F 
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import StratifiedKFold
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

class myDataset(Dataset):
    """ dataset reader
    """

    def __init__(self, args, data, dataType):
        super(myDataset, self).__init__()
        self.args = args
        self.dataType = dataType

        text, images, labels = data

        self.images = self._imageHandler(images)
        self.text = self._textHandler(text)
        self.labels = self._labelHandler(labels)

    def _labelHandler(self, labels):
        return torch.tensor(labels)

    def _textHandler(self, text):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer._add_tokens(["[desc]", "[ocr]"])
        text = torch.tensor(self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"])
        return text

    def _imageHandler(self, images):
        # starting here are the special handling of image data
        pad_length = self.args["image_pad_length"]
        images_post_process = []
        # hardcoded, need to address later
        # ------------image transform---------------------------------
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_transform_size = 224

        if self.dataType == "train":
            transform = self._imagenet_train(mean, std, size=img_transform_size)
        else:
            transform = self._imagenet_val(mean, std, size=img_transform_size)

        for items in images:
            # fix minor bug, in case len(items) > pad_length
            source_size = len(items)
            
            items = items + [torch.zeros((3, img_transform_size,img_transform_size))] * max(pad_length - len(items), 0)
           
            # some images have 4th channel, i.e., an alpha channel, throwing it away for now
            items = [item[:3][:][:] for item in items]  

            # items = [transform(item)[None, :, :, :] for item in items] # for downsampling
            # items = [torch.squeeze(F.interpolate(item, size=img_transform_size//2)) for item in items]

            items = torch.cat([transform(item).unsqueeze(0) for item in items],dim=0)
     
            # items_block = []
            # for block in range(0, pad_length, 2):
            #     items_block.append(torch.cat(items[block: block+2], dim=1))
            # items = torch.cat(items_block, dim=2)
            
            # this will actually duplicate items 
            # items = torch.cat(items, dim=1).reshape(3,224,224) 
            images_post_process.append(items.unsqueeze(0))

        images_post_process = torch.cat(images_post_process, dim=0)

        return torch.tensor(images_post_process)

    def _imagenet_train(self, mean=None, std=None, size=224):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return trans

    def _imagenet_val(self, mean=None, std=None, size=224):
        trans = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return trans

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.text[index], self.images[index], self.labels[index]


class DataHandler(object):
    """docstring for DataHandler"""
    def __init__(self, args):
        super(DataHandler, self).__init__()
        self.args = args
        # self.original_data = self._readData(original=True)
        if self.args["use_synthetic"]:
            self.synthetic_data = self._readData(original=False)
            

    def fullTrainLoadData(self):
        # use this method for final retrain
        dataType = 'train'
        train_data = [item1+item2 for item1,item2 in zip(self.original_data, self.synthetic_data)]
        # full train set
        train_data = myDataset(self.args, train_data, dataType)
        # self.args["logger"].info("{} size: {}".format(dataType, len(train_data.labels)))
        train_loader = DataLoader(train_data, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"])

        dataType = 'test'
        test_data = self.original_data
        # full train set
        test_data = myDataset(self.args, test_data, dataType)
        # self.args["logger"].info("{} size: {}".format(dataType, len(test_data.labels)))
        test_loader = DataLoader(test_data, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"])
        return {'train':train_loader, 'test':test_loader}

    def inferLoadData(self):
        # use this method for inference
        # test set, please fill in label column with placeholder 1s or 0s
        dataType = 'test_true'

        test_data = myDataset(self.args, self._readData(original=True,datatype='test_true'), dataType)
        # self.args["logger"].info("{} size: {}".format(dataType, len(test_data.labels)))
        test_loader = DataLoader(test_data, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
        return {'test':test_loader}
        
    def validLoadData(self):
        # use this method for inference
        # test set, please fill in label column with placeholder 1s or 0s
        dataType = 'test'
        test_data = myDataset(self.args, self._readData(original=True,datatype='test'), dataType)
        # self.args["logger"].info("{} size: {}".format(dataType, len(test_data.labels)))
        test_loader = DataLoader(test_data, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])
        return {'valid':test_loader}

    def cvLoadData(self, fold):
        # use this method when grid searching
        self.data_fold = self._cv_split()
        loaders = {}
        for dataType in ['train', 'test']:
            shuffle = (dataType=='train')

            data = myDataset(self.args, self.data_fold[dataType][fold], dataType)
            # self.args["logger"].info("{} size: {}".format(dataType, len(data.labels)))
            target = DataLoader(data, batch_size=self.args["batch_size"], shuffle=shuffle, num_workers=self.args["num_workers"])
            loaders[dataType] = target

        return loaders

    def _readData(self, original=True,datatype='train'):
        if original:
            dataPath = self.args["dataPath"]
            textPath = self.args["textPath"]
            imagePath = self.args["imagePath"]
        else:
            dataPath = self.args["syntheticDataPath"]
            textPath = self.args["syntheticTextPath"]
            imagePath = self.args["syntheticImagePath"]

        # read text
        pad_length = self.args["image_pad_length"]
        if datatype == 'test':
            tweets = pd.read_csv(dataPath+'valid'+textPath)
        elif datatype =='test_true':
            tweets = pd.read_csv(dataPath+'test'+textPath)
        else:
            tweets = pd.read_csv(dataPath+datatype+textPath)
        tweets.fillna('', inplace=True)

        text = []
        if self.args["use_desc"]:
            tweets["return_text"] = tweets.tweet_text_cleaned + ' [desc] ' + tweets.user_description_cleaned
        else:
            tweets["return_text"] = tweets.tweet_text_cleaned

        if self.args["use_ocr"] and original:
            tweets["return_text"] = tweets.return_text + ' [ocr] ' + tweets.ocr

        image_files = glob.glob(dataPath+imagePath+'*')
        images = defaultdict(list)
        for filename in image_files:
            index = ('_').join(filename.split('/')[-1].split('_')[3:5])     
            try:
                images[index].append(read_image(filename))
            except:
                pass

        # align and pad data
        images_in_order = [images[str(index)][:pad_length] for index in tweets.original_tid]
        
        return list(tweets.return_text), images_in_order, list(tweets.label.astype(int))

    # def _cv_split(self):
    #     text, image, label = self.original_data
    #     splitter = StratifiedKFold(n_splits=self.args["cv"], random_state=1, shuffle=True)
    #     train_fold, test_fold = [], []
    #     # x can be anything of same length here, just a place holder
    #     for train_index, test_index in splitter.split(self.original_data[0], self.original_data[2]):
    #         train = [[item[index] for index in train_index] for item in self.original_data]
    #         test = [[item[index] for index in test_index] for item in self.original_data]

    #         if self.args["use_synthetic"]:
    #             train_fold.append([item1+item2 for item1, item2 in zip(train, self.synthetic_data)])
    #         else:
    #             train_fold.append(train)
    #         test_fold.append(test)

    #     return {'train':train_fold, 'test':test_fold}

    def _cv_split(self):
        train_fold = [self._readData(original=True,datatype='train')]
        test_fold = [self._readData(original=True, datatype='test')]
        return {'train':train_fold, 'test':test_fold}