import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.SDDSynergy import SDDSynergyNet
from utils_test import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
# training function at each epoch

def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        x1, edge_index1, batch1, cell = data1.x.to(device), data1.edge_index.to(device), data1.batch.to(device), data1.cell.to(device)
        x2, edge_index2, batch2 = data2.x.to(device), data2.edge_index.to(device), data2.batch.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(x1,x2, edge_index1,edge_index2, batch1,batch2, cell)
        # output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx,
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))

def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            # data1 = data1.to(device)
            # data2 = data2.to(device)
            x1, edge_index1, batch1, cell = data1.x.to(device), data1.edge_index.to(device), data1.batch.to(
                device), data1.cell.to(device)
            x2, edge_index2, batch2 = data2.x.to(device), data2.edge_index.to(device), data2.batch.to(device)
            output = model(x1,x2, edge_index1,edge_index2, batch1,batch2, cell)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


modeling = SDDSynergyNet #model_name

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'synergy_data_examples'
# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)

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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) #, betas=(0.9,0.999)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9, nesterov=True)#momentum=0.9, nesterov=True
    # optimizer = swats.SWATS(model.parameters(), lr=LR)

    model_file_name = 'data/SDDSynergy/' + 'SDDSynergyNet_'+ str(i) + '.model'
    file_AUCs = 'data/SDDSynergy/SDDSynergyNet' + '_' + str(LR) + '_' + str(i) + '_' + str(NUM_EPOCHS) + '_Adam'+'.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        # T is correct label
        # S is predict score
        # Y is predict label
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)


        # compute preformence
        print(T,Y)
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)


        # save data
        # if best_auc < AUC:
        #     best_auc = AUC
        #     AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        #     save_AUCs(AUCs, file_AUCs)
        #     torch.save(model.state_dict(), model_file_name)
            # independent_num = []
            # independent_num.append(test_num)
            # independent_num.append(T)
            # independent_num.append(Y)
            # independent_num.append(S)
            # txtDF = pd.DataFrame(data=independent_num)
            # txtDF.to_csv(result_file_name, index=False, header=False)

        print('ROC:{:.3f},PR:{:.3f},ACC:{:.3f}'.format(AUC,PR_AUC,ACC) )
