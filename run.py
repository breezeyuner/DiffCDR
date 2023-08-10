
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
from models import MFBasedModel
import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR

import pickle

class Run():
    def __init__(self,
                 config
                 ):
        self.use_cuda = config['use_cuda'] 
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_ss = config['src_tgt_pairs'][self.task]['batchsize_ss']
        self.batchsize_la = config['src_tgt_pairs'][self.task]['batchsize_la']
        self.batchsize_diff = config['src_tgt_pairs'][self.task]['batchsize_diff']

        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_diff_test = config['src_tgt_pairs'][self.task]['batchsize_diff_test']

        self.batchsize_aug = self.batchsize_src


        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.lr = config['lr']
        self.la_lr = config['la_lr']
        

        self.wd = config['wd']

        self.ratio = [float(self.ratio.split(',')[0][1:]),float(self.ratio.split(',')[1][:-1])]

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        

        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'

        self.warm_tgt_train_path = self.input_root + '/warm_start_tgt_train.csv'
        self.warm_train_path = self.input_root + '/warm_start_train.csv'
        self.warm_test_path = self.input_root + '/warm_start_test.csv'

        self.results = {'tgt_mae': 10, 'tgt_rmse': 10, 
                        'aug_mae': 10, 'aug_rmse': 10, 
                        'emcdr_mae': 10, 'emcdr_rmse': 10, 
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10, 
                        'diff_mae': 10, 'diff_rmse': 10, 
                        'sscdr_mae': 10, 'sscdr_rmse': 10 , 
                        'lacdr_mae': 10, 'lacdr_rmse': 10 , 
                        }
        
        self.device = "cuda" if config['use_cuda'] else "cpu"

        self.diff_lr = config['diff_lr']
        self.diff_steps = config['diff_steps']
        self.diff_sample_steps = config['diff_sample_steps']
        self.diff_scale = config['diff_scale']
        self.diff_dim   = config['diff_dim'] 
        self.diff_task_lambda = config['diff_task_lambda'] 
        self.diff_mask_rate = config["diff_mask_rate"]

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False, shuffle = True ):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle= shuffle )
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle= shuffle)
            return data_iter

    def read_map_data(self,data_path):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(data_path , header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter


    def read_diff_data(self,data_path,batch_size, shuffle=True):
        
        meta_uid_seq = pd.read_csv(data_path , header=None)
        meta_uid_seq.columns = ['meta_uid', 'iid', 'y', 'pos_seq']
        meta_uid   = torch.tensor(meta_uid_seq['meta_uid'].values, dtype=torch.long)

        iid_input=  torch.tensor(meta_uid_seq[['iid']].values, dtype=torch.long)
        y_input= torch.tensor(meta_uid_seq[['y']].values, dtype=torch.long) 

        if self.use_cuda:
            meta_uid = meta_uid.cuda()
            iid_input = iid_input.cuda()
            y_input = y_input.cuda()

        dataset = TensorDataset(meta_uid,iid_input,y_input)
        data_iter = DataLoader(dataset,batch_size , shuffle= shuffle) 
        return data_iter
    

    def read_ss_data(self,data_path):
        '''
        '''
        cols = ['uid', 'iid', 'y', 'pos_seq']
        meta_data = pd.read_csv(data_path , header=None)
        meta_data.columns = cols
        meta_data.drop(['y'],axis=1,inplace=True)

        #neg sample 
        meta_data['pos_seq'] = meta_data['pos_seq'].str[1:-1]
        meta_data['pos_seq'] = meta_data['pos_seq'].str.split(',')
        meta_data['pos_split_len'] = [  len(x) for x in   meta_data['pos_seq']]
        meta_data['positive_s_i'] = [ np.random.choice(x,1)[0] for x in meta_data['pos_split_len']  ]
        meta_data['positive_s_i'] = [ int(x[y]) if x !=[''] else 0 for x,y in  zip(meta_data['pos_seq'],meta_data['positive_s_i']) ]
        
        #hist item
        all_his_item = set()
        for x_seq in meta_data['pos_seq']:
            for x in x_seq:
                if x != '':
                    all_his_item.add( int(x))
        
        all_his_item = list(all_his_item)
        neg_s_i = np.random.choice( len( all_his_item), meta_data.shape[0])
        
        meta_data['negetive_s_i'] = [ all_his_item[x] for x in neg_s_i]

        x_u = torch.tensor(meta_data['uid'], dtype=torch.long)
        x_p_i = torch.tensor(meta_data['positive_s_i'], dtype=torch.long)
        x_n_i = torch.tensor(meta_data['negetive_s_i'], dtype=torch.long)
        x_t_u = torch.tensor(meta_data['uid'], dtype=torch.long)
        
        del meta_data,all_his_item,neg_s_i

        if self.use_cuda:
            x_u = x_u.cuda()
            x_p_i = x_p_i.cuda()
            x_n_i = x_n_i.cuda()
            x_t_u = x_t_u.cuda()
        dataset = TensorDataset(x_u,x_p_i,x_n_i,x_t_u)
        data_iter = DataLoader(dataset, self.batchsize_ss, shuffle=True)
        
        return data_iter

    def read_la_data(self):
        
        
        #overlap 
        cols = ['uid', 'iid', 'y', 'pos_seq']
        meta_data = pd.read_csv(self.meta_path , header=None)
        meta_data.columns = cols
        meta_data.drop(['y'],axis=1,inplace=True)

        #full_uid = meta_data[['uid']].drop_duplicates()
        full_uid = meta_data[['uid']]

        full_uid['mask_src'] = 1
        full_uid['mask_tgt'] = 1

        x_uid = torch.tensor(full_uid['uid'], dtype=torch.long)
        x_mask_src = torch.tensor(full_uid['mask_src'], dtype=torch.long)
        x_mask_tgt = torch.tensor(full_uid['mask_tgt'], dtype=torch.long)

        del meta_data,full_uid
        
        if self.use_cuda:
            x_uid = x_uid.cuda()
            x_mask_src = x_mask_src.cuda()
            x_mask_tgt = x_mask_tgt.cuda()
        dataset = TensorDataset(x_uid,x_mask_src,x_mask_tgt)
        data_iter = DataLoader(dataset, self.batchsize_la, shuffle=True)

        return data_iter

    def read_aug_data(self, tgt_path ):
        #merge source train , target train 

        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

        
    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data(self.meta_path)
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_diff = self.read_diff_data(self.meta_path,batch_size=self.batchsize_diff)
        print('diff {} iter / batchsize = {} '.format(len(data_diff), self.batchsize_diff))

        data_aug = self.read_aug_data(self.tgt_path)
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_ss = self.read_ss_data(self.meta_path)
        print('ss {} iter / batchsize = {} '.format(len(data_ss), self.batchsize_ss))

        data_la = self.read_la_data()
        print('la {} iter / batchsize = {} '.format(len(data_la), self.batchsize_la))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True,shuffle=False)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        data_diff_test = self.read_diff_data(self.test_path,batch_size=self.batchsize_diff_test,shuffle=False)
        print('diff {} iter / batchsize = {} '.format(len(data_diff_test), self.batchsize_diff_test))

        return data_src, data_tgt, data_meta, data_map, data_diff, data_aug,data_ss, data_la, data_test,data_diff_test


    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.uid_all, self.iid_all, self.emb_dim, self.meta_dim )
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model,diff_model=None,ss_model=None  ,la_model=None):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        
        if diff_model is None and ss_model is None and la_model is None :
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map
        
        elif diff_model is None and ss_model is not None and la_model is None :
            optimizer_ss = torch.optim.Adam(params=ss_model.parameters(), lr=self.lr, weight_decay=self.wd)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,optimizer_ss, optimizer_map
        
        elif diff_model is None and la_model is not None :
            optimizer_la = torch.optim.Adam(params=la_model.parameters(), lr= self.la_lr , weight_decay=self.wd)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,optimizer_la, optimizer_map

        elif diff_model is not None:
            optimizer_diff = torch.optim.Adam(params = diff_model.parameters(),lr= self.diff_lr)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,optimizer_diff, optimizer_map

    def eval_mae(self, model, data_loader, stage ):
        print('Evaluating MAE:')
        
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()

        with torch.no_grad():
            if stage in ('test_diff'):
                for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval() 
                    model[1].eval() 
                    pred = model[0](X, stage,self.device, diff_model = model[1])

                    y_input = X[-1]
                    targets.extend(y_input.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

            elif stage in ('test_ss'):
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval() 
                    model[1].eval() 
                    pred = model[0](X, stage,self.device, diff_model = None, ss_model= model[1])
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

            elif stage in ('test_la'):
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval() 
                    model[1].eval() 
                    pred = model[0](X, stage,self.device, diff_model = None, la_model= model[1])
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

            else:
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model.eval() 
                    pred = model(X, stage,self.device)
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())
        

        targets = torch.tensor(targets).float()
        predicts  = torch.tensor(predicts)

        return loss(targets , predicts ).item(), torch.sqrt(mse_loss(targets , predicts )).item()
            

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False,diff=False ,ss=False , la=False  ):
        print('Training Epoch {}:'.format(epoch + 1))

        loss_ls = []
        if diff == False and ss==False and la==False:
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                if mapping:
                    model.train()

                    src_emb, tgt_emb = model(X, stage,self.device)
                    loss = criterion(src_emb, tgt_emb) 

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    model.train()

                    pred = model(X, stage,self.device)
                    loss = criterion(pred, y.squeeze().float())

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_ls.append(loss.item())
            return torch.tensor(loss_ls).mean()

        elif diff == False and ss==True and la==False:
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()
                loss = model[0]( X ,stage,self.device,diff_model = None, ss_model= model[1])

                model[1].zero_grad()
                loss.backward()
                optimizer.step()

                loss_ls.append(loss.item())
            return torch.tensor(loss_ls).mean()

        elif diff == False and la==True:
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()
                loss = model[0]( X ,stage,self.device,diff_model = None, la_model= model[1])

                model[1].zero_grad()
                loss.backward()
                optimizer.step()

                loss_ls.append(loss.item())
            return torch.tensor(loss_ls).mean()

        elif diff == True:
            task_loss_ls = []
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()

                #diff first, then task 
                #diff
                loss = model[0]( X ,stage,self.device,diff_model = model[1],is_task=False)
                model[1].zero_grad()
                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(),1.)
                optimizer.step()
                #task
                task_loss = model[0]( X ,stage,self.device,diff_model = model[1],is_task=True)
                model[1].zero_grad()
                task_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(),1.)
                optimizer.step()
                
                loss_ls.append(loss.item())
                task_loss_ls.append(task_loss.item())
            #return torch.tensor(loss_ls).mean()
            return torch.tensor(loss_ls).mean() ,torch.tensor(task_loss_ls).mean() 


    def update_results(self, mae, rmse,  phase):

        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse
        
    def reset_results(self):
        self.results = {'tgt_mae': 10, 'tgt_rmse': 10, 
                        'aug_mae': 10, 'aug_rmse': 10, 
                        'emcdr_mae': 10, 'emcdr_rmse': 10, 
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10, 
                        'diff_mae': 10, 'diff_rmse': 10, 
                        'sscdr_mae': 10, 'sscdr_rmse': 10 , 
                        'lacdr_mae': 10, 'lacdr_rmse': 10 , 
                        }

    def TgtOnly(self, model, data_tgt, data_test , criterion, optimizer):
        print('=========TgtOnly========')
        n_epoch = self.epoch

        for i in range(n_epoch):
            loss = self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt' )
            mae, rmse = self.eval_mae(model, data_test , stage='test_tgt')
            self.update_results(mae, rmse ,'tgt')
            print('MAE: {} RMSE: {} '.format(mae, rmse))

    def SrcOnly(self, model, data_src, criterion, optimizer_src): 
        print('=====SrcOnly=====')
        for i in range(self.epoch):
            loss = self.train(data_src, model, criterion, optimizer_src, i, stage='train_src' )

    def DataAug(self, model, data_aug, data_test , criterion, optimizer):
        print('=========DataAug========')
        n_epoch = self.epoch

        for i in range( n_epoch ):
            loss = self.train(data_aug, model, criterion, optimizer, i, stage='train_aug')
            mae, rmse = self.eval_mae(model, data_test , stage='test_aug')
            self.update_results(mae, rmse,  'aug')
            print('MAE: {} RMSE: {} '.format(mae, rmse))

    def Diff_CDR(self, model,diff_model,  data_diff, data_test ,optimizer):
        print('=========Diff_CDR========')
        for i in range(self.epoch):
            loss,task_loss = self.train(data_diff,[model,diff_model],None,optimizer,i,stage='train_diff' , mapping=False,diff=True )
 
            mae, rmse = self.eval_mae( [model,diff_model] , data_test , stage='test_diff')
            self.update_results(mae, rmse  , 'diff')
            print('DIFF LOSS',loss.item(),'TASK LOSS',task_loss.item() ,
                  'MAE: {} RMSE: {}'.format(mae, rmse))


    def SS_CDR(self, model,ss_model,  data_ss, data_test  ,optimizer_ss):
        print('==========SS_CDR==========')
        for i in range(self.epoch):
            loss = self.train(data_ss, [model,ss_model],None,  optimizer_ss, i, stage='train_ss' , mapping=False,diff=False,ss=True)
            mae, rmse = self.eval_mae([model,ss_model], data_test , stage='test_ss')
            self.update_results(mae, rmse, 'sscdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def LA_CDR(self, model,la_model,  data_la, data_test , test_uid ,optimizer_la):
        print('==========LA_CDR==========')
        for i in range(self.epoch):
            loss = self.train(data_la, [model,la_model],None,  optimizer_la, i, stage='train_la' , mapping=False,diff=False,ss=False,la=True)
            mae, rmse = self.eval_mae([model,la_model], data_test , stage='test_la')
            self.update_results(mae, rmse,  'lacdr')
            print('LA LOSS',loss.item(),'MAE: {} RMSE: {}  '.format(mae, rmse ))


    def CDR(self, model,  data_map, data_meta, data_test ,
            criterion,  optimizer_map, optimizer_meta):

        print('==========EMCDR==========')
        for i in range(self.epoch):
            loss = self.train(data_map, model, criterion, optimizer_map, i, stage='train_map' , mapping=True)
            mae, rmse = self.eval_mae(model, data_test , stage='test_map')
            self.update_results(mae, rmse , 'emcdr')
            print('MAE: {} RMSE: {}  '.format(mae, rmse ))
        print('==========PTUPCDR==========')
        for i in range(self.epoch):
            loss = self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta' )
            mae, rmse = self.eval_mae(model, data_test , stage='test_meta')
            self.update_results(mae, rmse , 'ptupcdr')
            print('MAE: {} RMSE: {} '.format(mae, rmse ))



    def model_save(self,model,path):
        torch.save(model.state_dict(),path)

    def model_load(self,model,path):
        if self.device == 'cuda':
            model.load_state_dict(torch.load(path))  
        else:
            model.load_state_dict(torch.load(path,map_location='cpu'))  

    def result_print(self, phase):
        print_str=''
        for p in phase:
            for m in ['_mae', '_rmse']:
                metric_name = p + m
                print_str += metric_name + ': {:.6f} '.format(self.results[metric_name])
        print(print_str)

    def main(self, exp_part = 'None_CDR',save_path=None):
        if exp_part == 'diff_CDR':
            diff_model = Diff.DiffCDR(self.diff_steps, self.diff_dim, self.emb_dim,self.diff_scale,self.diff_sample_steps,self.diff_task_lambda,self.diff_mask_rate )
            diff_model = diff_model.cuda() if self.use_cuda else diff_model

            model = self.get_model()

            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,optimizer_diff, optimizer_map = self.get_optimizer(model,diff_model)
        
        elif exp_part == 'ss_CDR':
            ss_model = SSCDR.SSCDR(self.emb_dim )
            ss_model = ss_model.cuda() if self.use_cuda else ss_model

            model = self.get_model()
            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,optimizer_ss, optimizer_map = self.get_optimizer(model,None,ss_model)

        elif exp_part == 'la_CDR':
            la_model = LACDR.LACDR(self.emb_dim )
            la_model = la_model.cuda() if self.use_cuda else la_model

            model = self.get_model()
            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,optimizer_la, optimizer_map = self.get_optimizer(model,None,None,la_model)
        
        else:
            model = self.get_model()
            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)

        data_src, data_tgt, data_meta, data_map, data_diff, data_aug, data_ss, data_la, data_test,data_diff_test = self.get_data()

        criterion = torch.nn.MSELoss()
        
        if exp_part == 'None_CDR':
            self.TgtOnly(model, data_tgt, data_test , criterion, optimizer_tgt)
            self.SrcOnly(model, data_src, criterion, optimizer_src)
            #CMF
            self.DataAug(model, data_aug, data_test , criterion, optimizer_aug) 
            self.result_print(['tgt','aug'])
            self.model_save(model,path =  save_path )
            
        elif exp_part == 'CDR':
            self.model_load(model,path =  save_path )
            print('None_CDR model loaded')
            self.CDR(model,  data_map, data_meta, data_test ,
                     criterion,  optimizer_map, optimizer_meta)
            self.result_print(['emcdr','ptupcdr'])

        elif exp_part == 'ss_CDR':
            self.model_load(model,path =  save_path )
            print('None_CDR model loaded')
            self.SS_CDR(model,ss_model,  data_ss, data_test ,optimizer_ss)
            self.result_print(['sscdr'])

        elif exp_part == 'la_CDR':
            self.model_load(model,path =  save_path )
            print('None_CDR model loaded')
            self.LA_CDR(model,la_model,  data_la, data_test,optimizer_la)
            self.result_print(['lacdr'])

        elif exp_part == 'diff_CDR':    
            self.model_load(model,path =  save_path )
            print('None_CDR model loaded')
            self.Diff_CDR(model,diff_model,  data_diff, data_diff_test  ,optimizer_diff)
            self.result_print(['diff'])
            




