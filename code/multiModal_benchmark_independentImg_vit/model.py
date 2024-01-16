import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.loss import _WeightedLoss
import torchvision
from transformers import BertModel
import logging
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

def save_model(model, args):
    """Save model."""
    torch.save(model.state_dict(), args["outputPath"] + args["exp_name"] + 'retrain_phase{}_'.format(args["phase"]) + args["modelPath"])
    # args["logger"].info("Saved better model selected by validation.")
    return True


class TextEncoder(nn.Module):
    """docstring for TextEncoder"""
    def __init__(self, args, **kwargs):
        super(TextEncoder, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder.resize_token_embeddings(kwargs["encoder_vocab_len"])
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, text):
        return self.encoder(text)[1] # pooler output

class ImageEncoder(nn.Module):
    """docstring for ImageEncoder"""
    def __init__(self, args, **kwargs):
        super(ImageEncoder, self).__init__()
        self.args = args

        self.encoder = torchvision.models.vit_b_16(pretrained=True)
        self.out_dim = self.encoder.hidden_dim

    def encode(self, x):
        # Reshape and permute the input tensor
        x = self.encoder._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.encoder.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder.encoder(x)[:, 0]
        return x

    def forward(self, image):
        # fix minor bug, if batch is 1 then simple squeeze will remove batch dim
        images = torch.split(image, 1, dim=-4)
        return torch.cat([self.encode(img.squeeze(dim=1)).squeeze(dim=-1).squeeze(dim=-1) for img in images],dim=-1)


class MultiModalWLT(nn.Module):
    def __init__(self, args, **kwargs):
        super(MultiModalWLT, self).__init__()
        self.args = args
        # hardcoded, in the future make it if-else from argparse
        self.text_encoder = TextEncoder(self.args, encoder_vocab_len=kwargs["tokenizer_len"])
        self.image_encoder = ImageEncoder(self.args)

        self.dropout1 = nn.Dropout(p=0.05)

        if self.args["use_image"]:
            self.fc1 = nn.Linear(self.text_encoder.out_dim + self.image_encoder.out_dim*self.args["image_pad_length"], self.args["hidden_dim"])
        else:
            self.fc1 = nn.Linear(self.text_encoder.out_dim, self.args["hidden_dim"])

        self.dropout2 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(self.args["hidden_dim"], self.args["num_classes"])

        self.act = nn.ReLU()
        # self.act = nn.Softmax(dim=-1) # do not use softmax if crossentropy loss, already implemented in the loss

        if self.args["weighted_loss"]:
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor([self.args["weighted_0"], self.args["weighted_1"]]))
        else:
            self.loss = nn.CrossEntropyLoss()


    def forward(self, text, image):
        text_features = self.text_encoder(text)
        if self.args["use_image"]:
            image_features = self.image_encoder(image)
            # direct concatenation, maybe softmax first or projection or attention?
            features = torch.cat([text_features, image_features], dim=-1)
        else:
            fetures = text_features

        output = self.act(self.fc1(self.dropout1(features)))
        output = self.fc2(self.dropout2(output))
        return output


