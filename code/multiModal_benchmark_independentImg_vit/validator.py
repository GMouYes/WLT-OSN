import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from pprint import pprint
import logging

from model import MultiModalWLT
from model import save_model
from util import *
from data import DataHandler
from warnings import simplefilter
from torch.nn import utils
simplefilter(action='ignore', category=UserWarning)

class Predictor(object):
    """An interface for predicting outputs based on trained models.
    """

    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args["use_cuda"] else 'cpu'

    def predict(self, network, test_data):
        network = network.to(self.device)
        network.eval()

        pred_list, truthList = [], []
        test_acc = 0.0

        # we need tqdm for infernce because this can be very large
        for data in tqdm(test_data, desc="inference"):
            text, image, label = [item.to(self.device) for item in data]

            with torch.no_grad():
                pred = network(text, image)

            pred_list.append(pred.detach().cpu())
            truthList.append(label.detach().cpu())

        pred_list = torch.cat(pred_list, axis=0)
        truthList = torch.cat(truthList, axis=0)

        # Compute the average acc and loss over all test instances
        # test_acc = metric(truthList, pred_list, self.args)

        # self.args["logger"].info("[Final Tester] Accuracy: {:.4f}".format(test_acc))
        return pred_list


class Simulation(object):
    """docstring for Simulation"""
    def __init__(self, cfg):
        super(Simulation, self).__init__()
        self._init_config(cfg)

        self._init_path()
        self.dataloader = DataHandler(self.args)
        self.data = self.dataloader.validLoadData()

    def _init_config(self, cfg):
        args = readConfig(cfg.configReadPath)
        args["logger"] = setupLogger(args["loggerConfigPath"])
        self.args = {**args, **vars(cfg)} # argparse namespace will overwrite yml config if they overlap
        return True

    def _init_path(self):
        if not os.path.isdir(self.args["outputPath"]):
            os.mkdir(self.args["outputPath"])

        expPath = self.args["outputPath"] + self.args["exp_name"]
        if not os.path.isdir(expPath):
            os.mkdir(expPath)

        return True

    def run(self):
        seed_all(self.args["seed"])
        for phase in range(self.args["rerun"]):
            modelPath = self.args["outputPath"] + self.args["exp_name"] + 'retrain_phase{}_'.format(phase) + self.args["modelPath"]
            predictPath = self.args["newOutputPath"] + 'retrain_phase{}'.format(phase) + '.npy'
            self.args["logger"].info("Phase {}: [modelPath]: {}; [predictPath]: {}".format(phase, modelPath, predictPath))
            # only test on the best model
            model = MultiModalWLT(self.args, tokenizer_len=len(self.data["valid"].dataset.tokenizer))
            model.load_state_dict(torch.load(modelPath))
            predictor = Predictor(self.args)
            pred = predictor.predict(model, self.data["valid"])
            np.save(predictPath, pred)

        self.args["logger"].info("Experiment Done!")
        return True

    def checkInput(self, args):
        # placeholder for sanity check
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug/')
    parser.add_argument('--configReadPath', type=str, default="../../config/debug.yml")
    parser.add_argument('--newOutputPath', type=str, default="../../output/")
    parser.add_argument('--dataPath', type=str, default="../../data/")
    parser.add_argument('--textPath', type=str, default="")
    parser.add_argument('--imagePath', type=str, default="../../data/images/")
    args = parser.parse_args()

    simulator = Simulation(args)
    simulator.run()
    return

if __name__ == '__main__':
    main()
