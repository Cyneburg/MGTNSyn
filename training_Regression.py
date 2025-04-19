import random
import torch.nn.functional as F
import torch.nn as nn
from models.TransformerDDS import MGTNSyn
from utils_test import *
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error, r2_score
from scipy import stats
from torch_geometric.loader import DataLoader
from tqdm import trange


def median_absolute_percentage_error(y_true,y_pred):
    return np.median(np.abs((y_pred-y_true)/y_true))

def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('\n Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} === {:.0f}% === Loss: {:.6f}'.format(epoch,
                                                                        100. * batch_idx / len(drug1_loader_train),
                                                                        loss.item()))

def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

modeling = MGTNSyn

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

datafile = 'drug_synergy'
drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)
print('model', modeling)
print('data', datafile)
#5:1划分数据集(训练集:测试集)
random.seed(2)
random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    drug1_data_train = [data for data in drug1_data[train_num]]
    drug1_data_test = [data for data in drug1_data[test_num]]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE)

    drug2_data_test = [data for data in drug2_data[test_num]]
    drug2_data_train = [data for data in drug2_data[train_num]]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE)
   

    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5)

    model_file_name = 'data/result/model/regression/' + str(i) + '--model_' + datafile +  '.model'
    file_AUCs = 'data/result/regression/' + str(i) + datafile + '.txt'
    AUCs = ('Epoch\tMSE\tR2\tPearsonr\tP_value\tSpearman\tMAE\tMedianAPE\tMAPE')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_loss = float("inf")
    for epoch in trange(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        # T is correct score
        # S is predict score
        T, S = predicting(model, device, drug1_loader_test, drug2_loader_test)

        # compute preformence
        MedianAPE = median_absolute_percentage_error(T, S)
        MSE = mean_squared_error(T, S)
        MAE = mean_absolute_error(T, S)
        MAPE = mean_absolute_percentage_error(T, S)
        R2 = r2_score(T, S)
        Pearsonr = pearson(T, S)
        P_value = p_value(T, S)
        Spearman = spearman(T, S) #

        # save data
        if best_loss > MSE:
            best_loss = MSE
            AUCs = [epoch, MSE, R2, Pearsonr, P_value, Spearman, MAE, MedianAPE, MAPE]
            save_AUCs(AUCs, file_AUCs)
            torch.save(model.state_dict(), model_file_name)

        print('MSE:{:.3f},R2:{:.3f},Pearsonr:{:.3f}'.format(MSE, R2, Pearsonr))
