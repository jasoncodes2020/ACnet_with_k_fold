import pandas as pd
import torch
from torch import nn
import time
import numpy as np
import torch.utils.data as Data
from ACNet import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, \
    confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
import warnings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
warnings.filterwarnings('ignore')
writer = SummaryWriter()
def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid
net = EARCNN(num_classes=3)
net = net.to(device)
learningRate = 0.0001  # 学习率（可以改）
optimizer = torch.optim.AdamW(net.parameters(), lr=learningRate, weight_decay=0.02)  # 优化器（这个可以换Adam等）
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
def train(epoch, Model, dataloader):
    Model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(dataloader, 0):
        X_train, Y_train = data
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        # X_train, Y_train = X_train.to(device), Y_train.to(device)
        optimizer.zero_grad()

        outputs = Model(X_train)
        loss = loss_fn(outputs, Y_train.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch , batch_idx + 1, running_loss / 300))
            # running_loss = 0.0
    return Model,np.mean(running_loss)


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # 获取标签的间隔数
    plt.xticks(num_class, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.imsave('./result/cm.png',cm)
    plt.show()
def test(Model, dataloader):
    correct = 0
    total = 0
    y_true = 0
    y_pred = 0
    acc_mean=[]
    precision_mean = []
    recall_mean = []
    f1_mean = []
    kappa_mean=[]
    cm_mean = []
    Model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            X, Y = data[0], data[1]
            X = X.to(device)
            Y = Y.to(device)
            # X_test, Y_test = X_test.to(device), Y_test.to(device)
            outputs = Model(X)
            _, predicted = torch.max(outputs.data, dim=1)

            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            # print('accuracy on test set: %.2f %% ' % (100 * correct / total))
            acc_mean.append(accuracy_score(Y, predicted))
            precision_mean.append(precision_score(Y, predicted,average='weighted'))
            recall_mean.append(recall_score(Y, predicted,average='weighted'))
            f1_mean.append(f1_score(Y, predicted,average='weighted'))
            kappa_mean.append(cohen_kappa_score(Y, predicted))
            cm = confusion_matrix(y_true=Y, y_pred=predicted)
            if np.mean(acc_mean) > 0.98:
                plot_confusion_matrix(cm, ['0', '1', '2'], 'confusion_matrix')
            # plt.show()
            # print(cm)
            # y_true += Y
            # y_pred += predicted
    # print('accuracy on test set: %.2f %% ' % (100 * correct / total))
    # print('准确率:',np.mean(acc_mean))
    # print('精准率:',np.mean(precision_mean))
    # print('召回率:', np.mean(recall_mean))
    # print('F1:', np.mean(f1_mean))
    # print('Kappa:', np.mean(kappa_mean))
    # print('accuracy on test set: %.2f %% ' % (100 * correct / total))
    # cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # #
    # # # 打印混淆矩阵
    # print("Confusion Matrix: ")
    # print(cm)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    return np.mean(acc_mean),np.mean(precision_mean),np.mean(recall_mean),np.mean(f1_mean),np.mean(kappa_mean)
def k_fold(k, X_train, y_train):
    acc_final = []
    precision_final = []
    recall_final = []
    f1_final = []
    kappa_final = []
    for i in range(k):
        print('k-flod',i)
        X_train, y_train, X_valid,y_valid = get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
         ### 实例化模型
        batch_size = 64  # batch_size（可以改）
        trainData = Data.TensorDataset(X_train, y_train)
        testData = Data.TensorDataset(X_valid, y_valid)  # <class 'torch.utils.data.dataset.TensorDataset'>
        train_dataloader = DataLoader(trainData, batch_size=batch_size, drop_last=True,shuffle=True)
        test_dataloader = DataLoader(testData, batch_size=batch_size, drop_last=True,shuffle=True)
        ### 每份数据进行训练,体现步骤三####
        k_loss=[]
        k_acc=[]
        k_precision=[]
        k_recall = []
        k_f1 = []
        k_kappa=[]
        for epoch in range(20):
            myModel,loss = train(epoch, net, train_dataloader)
            acc_mean, precision_mean, recall_mean, f1_mean, kappa_mean = test(myModel, test_dataloader)

            writer.add_scalar('loss',loss,epoch)
            writer.add_scalar('acc',acc_mean,epoch)
            writer.add_scalar('precision', precision_mean, epoch)
            writer.add_scalar('recall', recall_mean, epoch)
            writer.add_scalar('f1', f1_mean, epoch)
            writer.add_scalar('kappa', kappa_mean, epoch)
            # if acc_mean > 0.8 and i==5 and epoch==20:
            #     torch.save(myModel, "model.pt")
            k_loss.append(loss)
            k_acc.append(acc_mean)
            k_precision.append(precision_mean)
            k_recall.append(recall_mean)
            k_f1.append(f1_mean)
            k_kappa.append(kappa_mean)
        # k_loss.reverse()
        k_acc.reverse()
        k_precision.reverse()
        k_recall.reverse()
        k_f1.reverse()
        k_kappa.reverse()
        pd.DataFrame(k_loss).to_csv('result/loss%d折.csv' % i, index_label=False, index=False)
        pd.DataFrame(k_acc).to_csv('result/acc%d折.csv' % i, index_label=False, index=False)
        pd.DataFrame(k_precision).to_csv('result/precision%d折.csv' % i, index_label=False, index=False)
        pd.DataFrame(k_recall).to_csv('result/recall%d折.csv' % i, index_label=False, index=False)
        pd.DataFrame(k_f1).to_csv('result/f1%d折.csv' % i, index_label=False, index=False)
        pd.DataFrame(k_kappa).to_csv('result/kappa%d折.csv' % i, index_label=False, index=False)
        index = np.argmax(k_acc) #返回最大acc索引
        acc = k_acc[index]
        precision = k_precision[index]
        recall = k_recall[index]
        f1 = k_f1[index]
        kappa = k_kappa[index]
        acc_final.append(acc)
        precision_final.append(precision)
        recall_final.append(recall)
        f1_final.append(f1)
        kappa_final.append(kappa)
        print('#' * 10, '第%d折交叉验证的结果为:\n'%i)
        print('准确率:', acc)
        print('精准率:', precision)
        print('召回率:', recall)
        print('F1:', f1)
        print('Kappa:', kappa)
        print('#' * 10,'第%d折结束'%i)
    index_best = np.argmax(acc_final)
    acc_best = acc_final[index_best]
    precision_best = precision_final[index_best]
    recall_best = recall_final[index_best]
    f1_best = f1_final[index_best]
    kappa_best = kappa_final[index_best]

    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('第%d折效果最好'%index_best)
    print('最优的准确率为:',acc_best)
    print('最优精准率:', precision_best)
    print('最优召回率:', recall_best)
    print('最优F1:', f1_best)
    print('最优Kappa:', kappa_best)
    # ####体现步骤四#####
    # print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k), \
    #       'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))


if __name__ == '__main__':

    # Configuration options
    loss_function = nn.CrossEntropyLoss()
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)
    falx = np.load('data/t6x_89.npy')  # shape(new_x)=(45,1126,6,8,9,5)  class 'numpy.ndarray'
    faly = np.load('data/t6y_89.npy')
    falx = falx.reshape((50670, 6, 8, 9, 5))

    data2 = torch.from_numpy(falx)  # numpy->Tensor 这里看你输入需不需要改，会不会保存，我这里需要转一下
    label2 = torch.from_numpy(faly)
    data1 = data2.type(torch.FloatTensor)
    label1 = label2.type(torch.LongTensor)
    data = torch.FloatTensor(data1)
    label = torch.LongTensor(label1)
    # X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, shuffle=True, random_state=2022)

    # Start print
    print('--------------------------------')
    k_fold(3,data,label)

