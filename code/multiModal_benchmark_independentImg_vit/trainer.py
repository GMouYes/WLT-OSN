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


class Trainer(object):
    """Trainer."""

    def __init__(self, args, bestResult=np.inf):
        super(Trainer, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args["use_cuda"] else 'cpu'
        self.bestResult = bestResult
        self.best_report = None

    def train(self, network, train_data, dev_data=None, plot_loss=False, save_better_model=False):
        network = network.to(self.device)
        train_loss, valid_loss = [], []
        validator = Tester(self.args)
        self.optimizer = torch.optim.RAdam(network.parameters(), lr=self.args["lr"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args["gamma"])

        for epoch in range(self.args["epoch"]):
            logString = "Train epoch {}: ".format(epoch)
            network.train()
            
            train_epoch_loss = self._train_step(train_data, network, epoch=epoch)
            train_loss.append(train_epoch_loss)
            logString += "[Train] loss: {:.6f} ".format(train_epoch_loss)

            # validation
            test_epoch_loss, test_report = validator.test(network, dev_data, epoch=epoch)
            valid_loss.append(test_epoch_loss)
            logString += "[Validate] Loss: {:.6f} ".format(test_epoch_loss)

            # check if result is better
            better_result = self.best_eval_result(test_epoch_loss)
            self.scheduler.step()

            if better_result:
                self.best_report = test_report
                logString += "Found better eval result."
                # only save models during retrain
                if save_better_model:
                    save_model(network, self.args)

            self.args["logger"].info(logString)

        self.args["logger"].info("Grid best cls result\n" + self.best_report)

        # only plot loss during retrain
        if plot_loss:
            self.plot_loss(train_loss, valid_loss)

        return self.bestResult

    def plot_loss(self, train_loss, valid_loss):
        plotPath = self.args["outputPath"] + self.args["exp_name"] + 'retrain_phase{}_'.format(self.args["phase"]) + self.args["lossPath"]
        plt.figure()
        ax = plt.subplot(121)
        ax.set_title('train loss')
        ax.plot(train_loss, 'r-')

        ax = plt.subplot(122)
        ax.set_title('validation loss')
        ax.plot(valid_loss, 'b-')

        plt.savefig(plotPath)
        plt.close()

        return True

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        # train_acc = 0.
        loss_record = 0.
        
        for data in data_iterator:
            text, image, label = [item.to(self.device) for item in data]

            self.optimizer.zero_grad()
            pred = network(text, image)
            loss = network.loss(pred, label)
            loss_record += loss.item()
            loss.backward()
            utils.clip_grad_norm_(network.parameters(), self.args["clip_grad"])
            self.optimizer.step()

        return loss_record / len(data_iterator.dataset)

    def best_eval_result(self, test_loss):
        """ Check if the current epoch yields better validation results.
            We may switch to other metrics, right now it is loss, maybe acc?

        :param test_loss, a floating number
        :return: bool, True means current results on dev set is the best.
        """

        if test_loss < self.bestResult:
            self.bestResult = test_loss
            
            # self.args["bestResult"] = self.bestResult
            return True
        return False


class Tester(object):
    """Tester."""

    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() and args["use_cuda"] else 'cpu'

    def test(self, network, dev_data, **kwargs):
        network = network.to(self.device)
        network.eval()

        # test_acc = 0.0
        valid_loss = 0.0
        # self.args["logger"].info("Eval epoch {}".format(kwargs["epoch"]))
        pred_list, truthList = [], []
        for data in dev_data:
            text, image, label = [item.to(self.device) for item in data]

            with torch.no_grad():
                pred = network(text, image)
                loss = network.loss(pred, label)
                valid_loss += loss.item()

            pred_list.append(pred.detach().cpu())
            truthList.append(label.detach().cpu())

        pred_list = torch.cat(pred_list, axis=0)
        truthList = torch.cat(truthList, axis=0)

        test_report = metric(truthList, pred_list)
        valid_loss /= len(dev_data.dataset)

        return valid_loss, test_report


class Simulation(object):
    """docstring for Simulation"""
    def __init__(self, cfg):
        super(Simulation, self).__init__()
        self._init_config(cfg)

        self.fixed_args = {k:v for k,v in self.args.items() if k != "grid_search"}
        self.bestResult = np.inf
        self.bestConfigPath = self.args["outputPath"]+self.args["exp_name"]+self.args["configWritePath"]

        self._init_path()
        self.data = DataHandler(self.args)

    def _init_config(self, cfg):
        args = readConfig(cfg.configReadPath)
        args["logger"] = setupLogger(args["loggerConfigPath"])
        self.args = {**args, **vars(cfg)} # argparse namespace will overwrite yml config if they overlap
        return True

    def _init_path(self):
        if not os.path.isdir(self.args["outputPath"]):
            os.mkdir(self.args["outputPath"])

        expPath = self.args["outputPath"]+self.args["exp_name"]
        if not os.path.isdir(expPath):
            os.mkdir(expPath)
        return True

    # def _readData(self):
    #     data_handler = DataHandler(self.args)
    #     self.trainData, self.validData = DataHandler.cvLoadData()
    #     return True

    def _gridSearch(self):
        gridSearch = self.args["grid_search"]
        for grid in [dict(zip(gridSearch.keys(),v)) for v in itertools.product(*gridSearch.values())]:
            new_args = {**self.fixed_args, **grid}
            self.args["logger"].info("Grid Search: {}".format(grid))
            fold_result = []
            for fold in range(self.args["cv"]):
                self.args["logger"].info("Fold {}:".format(fold))
                data = self.data.cvLoadData(fold=fold)
                # self.args["logger"].info(new_args) # debug purpose
                avg_rerun_result = self._rerun(new_args, data["train"], data["test"])
                fold_result.append(avg_rerun_result)
            
            avg_fold_result = sum(fold_result) / self.args["cv"]
            if avg_fold_result < self.bestResult:
                self.bestResult = avg_fold_result
                saveConfig(self.bestConfigPath, new_args)
                self.args["logger"].info("Better config with avg valid loss {}\n".format(avg_fold_result))

        return True

    # def _retrain(self, args):
    #     
    #     data = self.data.fullTrainLoadData()
    #     avg_result = self._rerun(args, data["train"], data["test"], plot_loss=True, save_better_model=True)
    #     if avg_result < self.bestResult:
    #         self.bestResult = avg_result
    #         # during retrain, no need to repeatedly store same config
    #         # saveConfig(self.bestConfigPath, args)
    #         args["logger"].info("Better config with avg valid loss {}".format(avg_result))
    #         # args["logger"].info(args) # debug purpose
    #     return avg_result

    def _rerun(self, args, trainData, validData, plot_loss=False, save_better_model=True):
        rerun_results = [np.inf] * args["rerun"]
        seed_all(self.args["seed"])
        for phase in range(args["rerun"]):
            self.args["logger"].info("Phase {}:".format(phase))
            args["phase"] = phase
            model = MultiModalWLT(args, tokenizer_len=len(trainData.dataset.tokenizer))
            trainer = Trainer(args)
            rerun_results[phase] = trainer.train(model, trainData, validData, plot_loss=plot_loss, save_better_model=save_better_model)

        avg_result = sum(rerun_results) / args["rerun"]
        
        return avg_result

    def run(self):
        if self.args["do_grid_search"]:
            self._gridSearch()
        else:
            self._retrain(self.args)
        self.args["logger"].info("Experiment Done!")
        return True

    def _checkInput(self):
        # placeholder for sanity check
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug/')
    parser.add_argument('--configReadPath', type=str, default="../../config/debug.yml")
    parser.add_argument('--do_grid_search', action='store_true')
    args = parser.parse_args()

    simulator = Simulation(args)
    simulator.run()
    return

if __name__ == '__main__':
    main()
