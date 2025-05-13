from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from tqdm import tqdm, trange

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.test_loss_list = []
        self.test_acc_list = []
        self.test_recall_list = []
        self.test_f1_list = []

    def _build_model(self):
        # model input depends on data
        self.train_data, self.train_loader = self._get_data(flag="train")
        self.vali_data, self.vali_loader = self._get_data(flag="val")
        self.test_data, self.test_loader = self._get_data(flag="test")

        # self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        # self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(self.train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def confusion(self, trues, predictions, folder_path):
        cm = confusion_matrix(trues, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.test_data.class_names,
            yticklabels=self.test_data.class_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        confusion_path = os.path.join(
            folder_path, self.args.model_id + "confusion_matrix.png"
        )
        plt.savefig(confusion_path, bbox_inches="tight")
        plt.close()

    def acc_loss(self, folder_path):
        epochs = list(range(1, len(self.test_acc_list)+1))
        plt.figure(figsize=(10, 8))
        # 创建主坐标轴（左侧Y轴）
        ax1 = plt.gca()
        ax1.plot(epochs, self.test_loss_list, "r-", label="Test Loss", linewidth=2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="r")
        ax1.tick_params(axis="y", labelcolor="r")
        ax1.grid(True, linestyle="--", alpha=0.3)

        # 创建次坐标轴（右侧Y轴）
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.test_acc_list, "b-", label="Test Accuracy", linewidth=2)
        ax2.set_ylabel("Accuracy ", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

        # 设置标题和图例
        # plt.title('VNAT 10 Epochs Test Metrics (standard)')
        fig = plt.gcf()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")

        # 保存并显示
        plt.tight_layout()
        training_path = os.path.join(folder_path, self.args.model_id + "training.png")
        plt.savefig(training_path, bbox_inches="tight")
        plt.close()

    def result(self, folder_path, setting, accuracy, precision, recall, f1):
        print(
            "accuracy:{}, precision:{}, recall:{}, f1:{}".format(
                accuracy, precision, recall, f1
            )
        )
        file_name = "result_classification.txt"
        with open(os.path.join(folder_path, file_name), "a") as f:
            f.write(setting + "\n")
            f.write(
                "accuracy:{}, precision:{}, recall:{}, f1:{}".format(
                    accuracy, precision, recall, f1
                )
                + "\n\n"
            )

    def calculate_metrics(self, trues, predictions):
        """
        计算准确率、精确率、召回率、F1分数
        :param trues: 真实标签
        :param predictions: 预测标签
        :return: 准确率、精确率、召回率、F1分数
        """
        accuracy = cal_accuracy(predictions, trues)
        precision = precision_score(trues, predictions, average="macro")
        recall = recall_score(trues, predictions, average="macro")
        f1 = f1_score(trues, predictions, average="macro")
        return accuracy, precision, recall, f1

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(
                epoch_bar := tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.args.train_epochs}",
                    total=len(self.train_loader),
                    position=1,
                    leave=False,
                )
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                # print(label)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                if i % 20 == 0:
                    epoch_bar.set_postfix(
                        loss=f"{loss.item():.6f}",
                        learning=f"{self.args.learning_rate:.4f}",
                    )

                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()
                # if i % 10 == 0:
                #     tqdm.write(f"Loss: {loss.item():.4f}")

            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(
                self.vali_data, self.vali_loader, criterion
            )
            test_loss, test_accuracy = self.vali(
                self.test_data, self.test_loader, criterion
            )

            self.test_loss_list.append(test_loss)
            self.test_acc_list.append(test_accuracy)

            tqdm.write(
                " Epoch: {0}, cost time: {1:.3f} | Train Loss: {2:.3f}\tVali Loss: {3:.3f}\tVali Acc: {4:.3f}\tTest Loss: {5:.3f}\tTest Acc: {6:.3f}".format(
                    epoch + 1,
                    time.time() - epoch_time,
                    train_loss,
                    vali_loss,
                    val_accuracy,
                    test_loss,
                    test_accuracy,
                )
            )
            early_stopping(-test_accuracy, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(
                    model_optim, scheduler, epoch + 1, self.args, printout=False
                )

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if test:
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach())
                trues.append(label)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print("test shape:", preds.shape, trues.shape)
        probs = torch.nn.functional.softmax(preds)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy, precision, recall, f1 = self.calculate_metrics(trues, predictions)
        folder_path = "./results/" + setting + "/"
        os.makedirs(folder_path, exist_ok=True)
        trues = trues.flatten()
        predictions = predictions.flatten()
        self.confusion(trues, predictions, folder_path)
        self.acc_loss(folder_path)
        self.result(folder_path, setting, accuracy, precision, recall, f1)
        return
