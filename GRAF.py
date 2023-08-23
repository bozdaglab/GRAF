print("GRAF is running..")
import os
import torch
import argparse
import errno
import warnings

base_path = os.getcwd() + '/'
dataset_name = 'sampledata'
folder_ext = 'GRAF_results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='''An integrative node classification framework, called GRAF (Graph Attention-aware Fusion Networks)''')
parser.add_argument('-lr', "--learning_rate", nargs = 1, default = ['0.01'])
parser.add_argument('-hs', "--hidden_size", nargs = 1, default = ['32'])
parser.add_argument('-pat', "--patience", nargs = 1, default = ['10'])
parser.add_argument('-data', "--data", nargs = 1, default = ['DrugADR'])
args = parser.parse_args()
fixed_lr = float(args.learning_rate[0])
fixed_hs = int(args.hidden_size[0])
start_patience = int(args.patience[0])
this_data = args.data[0]
    
# should be defined specific to data [this_sample: node of interest in HeteroData, meta_size: number of association in HeteroData]
if this_data == 'DrugADR':
    this_sample = 'sample' ; meta_size = 4

this_data_folder = this_data + '_'  + str(fixed_lr) + '_' + str(fixed_hs)+ '_' + str(start_patience)

if not os.path.exists(base_path  + folder_ext + this_data_folder):
    os.makedirs(base_path  + folder_ext + this_data_folder + '/')

xtimes = 10 # times repeated to get performance metrics (median and std dev) - also number of runs for attention repeats
fixed_dropout=0.6 
fixed_heads = 8 
fixed_wdecay = 0.001
max_epochs =  200

# Import libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.metrics import f1_score, accuracy_score
import statistics
import pickle, time

excel_file2 = base_path + folder_ext+ this_data_folder +"/Attentions.xlsx"
df1 = pd.read_excel(excel_file2, engine='openpyxl',)
df1
for i in range(xtimes):
    if i == 0:
        c = [float(x) for x in df1['att'][i].replace('[', '').replace(']', '').replace(' ', '').split(",")]
    else:
        c2 = [float(x) for x in df1['att'][i].replace('[', '').replace(']', '').replace(' ', '').split(",")]
        c = [a + b for (a,b) in zip(c,c2)]

init_att = torch.tensor([round(x/xtimes,2) for x in c])
print("Association-based attentions in order: " + str([round(x,2) for x in init_att.tolist()]))

# Network Fusion
edge_ind_dict = {}
edge_att_dict = {}
flag2 = 0
for netw in range(meta_size):
    flag = 0
    file = base_path + dataset_name +'/'+this_data+'.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    edge_index = pd.DataFrame(data[this_sample, 'metapath_' + str(netw), this_sample].edge_index).astype("int").transpose()
    
    for trials in range(xtimes):
        out_file = base_path + folder_ext+ this_data_folder +"/Attention_" + str(trials) + ".pkl"
        with open(out_file, 'rb') as f:
            attention = pickle.load(f)
        
        temp_list = attention[this_sample +'__' + 'metapath_' + str(netw) + '__'+this_sample].tolist()
        edge_attention = [round(statistics.mean(record), 3) for record in temp_list]

        if flag == 0:
            sum_attention = edge_attention
            flag = 1
        else:
            sum_attention = [x + y for x, y in zip(sum_attention, edge_attention)]
        
    sum_attention =  [round(record/xtimes, 3) for record in sum_attention]
    sum_attention = [x * init_att.tolist()[netw] for x in sum_attention]
    edge_att_dict[netw] = sum_attention
    edge_index['value'] = sum_attention
    edge_ind_dict[netw] = edge_index
    
    if flag2 == 0:
        weights = edge_ind_dict[netw]
        flag2 = 1
    else:
        weights = pd.concat([weights,edge_ind_dict[netw]])
    
weights.columns = ['Var1', 'Var2', 'value']
agg_weights = weights.groupby(['Var1', 'Var2']).agg({'value': ['sum']})
agg_weights.columns = ['value']
agg_weights = agg_weights.reset_index()
edge_index = agg_weights
edge_index.drop(edge_index[edge_index['value'] == 0].index, inplace = True)
edge_index['prob'] = edge_index['value']
edge_index = edge_index[edge_index.prob != 0]
in_file = base_path + folder_ext + this_data_folder + '/' + 'Fused_network.pkl'
with open(in_file, 'wb') as f:
    pickle.dump(edge_index, f)
    
# Downstream task with GCN

class Net(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb

    
def train():
    model.train()
    optimizer.zero_grad()
    out, emb1 = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return emb1


def validate():
    model.eval()
    with torch.no_grad():
        out, emb2 = model(data)
        pred = out.argmax(dim=1)
    return pred, emb2

criterion = torch.nn.CrossEntropyLoss()
i = 0

for fusion_perc in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:

    patience = start_patience
    run_name = folder_ext +this_data_folder+'/GRAF_prob_' + str(fusion_perc)
    save_path = base_path + run_name + '/'
    excel_file = save_path + "_results.xlsx"

    if not os.path.exists(base_path  + run_name):
        os.makedirs(base_path + run_name + '/')

    start = time.time()
    with open(base_path + folder_ext + this_data_folder + '/Fused_network.pkl', 'rb') as f:
        all_index = pickle.load(f)

    current_best_val = 0
    file = base_path + dataset_name +'/'+this_data+'.pkl'
    with open(file, 'rb') as f:
        data2 = pickle.load(f)

    if fusion_perc < 1:
        prev_edges = all_index
        p = np.array(prev_edges['prob'])
        p /= p.sum()
        a = np.random.choice(prev_edges.shape[0], round(prev_edges.shape[0] * fusion_perc), p=p, replace=False)
        edge_index = prev_edges.iloc[a,:]
    else:
        edge_index = all_index

    df2 = pd.DataFrame(columns=['Repeat No', 'LearningRate', 'HiddenSize', 'Test Acc', 'Test wF1', 'Test mF1'])
    av_this_epoch = list()
    av_result_acc = list()
    av_result_wf1 = list()
    av_result_mf1 = list()
    vl_result_mf1 = list()

    for ii in range(xtimes):
        data = Data(x=data2[this_sample].x, edge_index=torch.tensor(edge_index[edge_index.columns[0:2]].transpose().values, device=device).long(), edge_attr=torch.tensor(edge_index[edge_index.columns[2]].transpose().values, device=device).float(), y=data2[this_sample].y)

        data.train_mask = data2[this_sample].train_mask
        data.valid_mask = data2[this_sample].val_mask
        data.test_mask = data2[this_sample].test_mask
        in_size = data.x.shape[1]
        out_size = torch.unique(data.y).shape[0]
        model = Net(in_size=in_size, hid_size=fixed_hs, out_size=out_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=fixed_lr, weight_decay=fixed_wdecay)
        min_valid_F1 = 0
        patience_count = 0

        for epoch in range(max_epochs):
            emb = train()
            pred, emb = validate()
            this_valid_F1 = f1_score(pred[data.valid_mask], data.y[data.valid_mask], average='macro')

            if this_valid_F1 > min_valid_F1:
                min_valid_F1 = this_valid_F1
                this_pred = pred
                patience_count = 0
                selected_emb = emb
                this_epoch = epoch
            else:
                patience_count += 1

            if patience_count >= start_patience:
                break

        av_this_epoch.append(this_epoch)
        av_result_acc.append(round(accuracy_score(data.y[data.test_mask], this_pred[data.test_mask]), 3))
        av_result_wf1.append(round(f1_score(this_pred[data.test_mask], data.y[data.test_mask], average='weighted'), 3))
        av_result_mf1.append(round(f1_score(this_pred[data.test_mask], data.y[data.test_mask], average='macro'), 3))
        vl_result_mf1.append(round(f1_score(this_pred[data.valid_mask], data.y[data.valid_mask], average='macro'), 3))

        emb_file = save_path + 'Emb_' + str(ii) + '.pkl'
        with open(emb_file, 'wb') as f:
            pickle.dump(selected_emb, f)

        x = [ii, fixed_lr, fixed_hs, av_result_acc[ii], av_result_wf1[ii], av_result_mf1[ii]]
        df2 = df2.append(pd.Series(x, index=df2.columns), ignore_index=True)

    if xtimes == 1:
        av_this_epoch.append(this_epoch)
        av_result_acc.append(round(accuracy_score(data.y[data.test_mask], this_pred[data.test_mask]), 3))
        av_result_wf1.append(round(f1_score(this_pred[data.test_mask], data.y[data.test_mask], average='weighted'), 3))
        av_result_mf1.append(round(f1_score(this_pred[data.test_mask], data.y[data.test_mask], average='macro'), 3))
        vl_result_mf1.append(round(f1_score(this_pred[data.valid_mask], data.y[data.valid_mask], average='macro'), 3))

    sel_epoch = str(round(statistics.median(av_this_epoch), 3)) + '+-' + str(round(statistics.stdev(av_this_epoch), 3))
    result_acc = str(round(statistics.median(av_result_acc), 3)) + '+-' + str(round(statistics.stdev(av_result_acc), 3))
    result_wf1 = str(round(statistics.median(av_result_wf1), 3)) + '+-' + str(round(statistics.stdev(av_result_wf1), 3))
    result_mf1 = str(round(statistics.median(av_result_mf1), 3)) + '+-' + str(round(statistics.stdev(av_result_mf1), 3))
    result_mf1_vl = str(round(statistics.median(vl_result_mf1), 3)) + '+-' + str(round(statistics.stdev(vl_result_mf1), 3))

    df = pd.DataFrame(columns=['lr', 'hs', 'Test Acc', 'Test wF1','Test mF1'])
    x = [fixed_lr, fixed_hs, result_acc, result_wf1, result_mf1]
    df = df.append(pd.Series(x, index=df.columns), ignore_index=True)
    
    end = time.time()
    print("GRAF with " +  str(int(fusion_perc*100)) + "% of edges: Macro F1: " + str(df.iloc[0]['Test mF1']) + ", Accuracy: " + str(df.iloc[0]['Test Acc']) + ", Weighted F1 " + str(df.iloc[0]['Test wF1']) + ". Time: " + str(round(end - start, 1)) + ' seconds/10 runs.')
    
print('GRAF is done.')
    