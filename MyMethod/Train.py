from Bert_Transformer import BertTransformer
from DataReader import Reader
from Mapper import Mapper
from Model_with_weights import ModelWithWeights
import torch
import math
import random
from create_batch import get_pair_batch_test, toarray, get_pair_batch_train_common
from transformers import  BertModel, BertTokenizer
from model import BiLSTM_Attention

from NodeClassifier import NodeClassifier
import numpy as np
# import args
#from dataset import Reader
# import utils
from create_batch import get_pair_batch_test, toarray, get_pair_batch_train_common,get_pair_all_test_common
import torch
#from model import BiLSTM_Attention
import torch.nn as nn
import os
import logging
import math
# import time
import argparse
import random
from transformers import  BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def main():
    parser = argparse.ArgumentParser(add_help=False)
    # args, _ = parser.parse_known_args()
    parser.add_argument('--model', default='WTT', help='model name')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='run training or evaluation')
    parser.add_argument('-ds', '--dataset', default='WN18RR', help='dataset')
    args, _ = parser.parse_known_args()
    parser.add_argument('--save_dir', default=f'./checkpoints/{args.dataset}/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=f'./checkpoints/{args.dataset}')
    parser.add_argument('--log_folder', default=f'./checkpoints/{args.dataset}/', help='model output directory')


    # data
    parser.add_argument('--data_path', default=f'./data/{args.dataset}/', help='path to the dataset')
    parser.add_argument('--dir_emb_ent', default="entity2vec.txt", help='pretrain entity embeddings')
    parser.add_argument('--dir_emb_rel', default="relation2vec.txt", help='pretrain entity embeddings')
    parser.add_argument('--num_batch', default=2740, type=int, help='number of batch')
    parser.add_argument('--num_train', default=0, type=int, help='number of triples')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--total_ent', default=0, type=int, help='number of entities')
    parser.add_argument('--total_rel', default=0, type=int, help='number of relations')

    # model architecture
    parser.add_argument('--BiLSTM_input_size', default=100, type=int, help='BiLSTM input size')
    parser.add_argument('--BiLSTM_hidden_size', default=100, type=int, help='BiLSTM hidden size')
    parser.add_argument('--BiLSTM_num_layers', default=2, type=int, help='BiLSTM layers')
    parser.add_argument('--BiLSTM_num_classes', default=1, type=int, help='BiLSTM class')
    parser.add_argument('--num_neighbor', default=39, type=int, help='number of neighbors')
    parser.add_argument('--embedding_dim', default=100, type=int, help='embedding dim')

    # regularization
    parser.add_argument('--alpha', type=float, default=0.2, help='hyperparameter alpha')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for EaGNN')

    # optimization
    parser.add_argument('--max_epoch', default=10, help='max epochs')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
    parser.add_argument('--gama', default=0.5, type=float, help="margin parameter")
    parser.add_argument('--lam', default=0.1, type=float, help="trade-off parameter")
    parser.add_argument('--mu', default=0.001, type=float, help="gated attention parameter")
    parser.add_argument('--anomaly_ratio', default=0.05, type=float, help="anomaly ratio")
    parser.add_argument('--num_anomaly_num', default=300, type=int, help="number of anomalies")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    #dataset = Reader(args, args.data_path)
    dataset = Reader("../data_process/wiki/single_data_label_description_1975_1980.pkl",if_wiki=True)
    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')
    if args.mode == 'train':
        train(args, dataset, device)
    elif args.mode == 'test':
        # raise NotImplementedError
        test(args, dataset, device)
    else:
        raise ValueError('Invalid mode')

def train(args,dataset,device):
    num_classes=2
    import pickle
    # with open("../data_process/wiki/single_data_label_description_1975_1980.pkl", "rb") as file:
    #     data = pickle.load(file)
    # for d in data:
    #     print(d[9])
    #
    transformer_small = BertTransformer(
        hidden_size=768,  # BERT的隐藏层维度
        nhead=8,          # 多头注意力机制的头数
        dim_feedforward=2048,  # 前馈网络的维度
        dropout=0.1,# Dropout概率
    )
    transformer_combine = BertTransformer(
        hidden_size=2304,  # BERT的隐藏层维度
        nhead=8,  # 多头注意力机制的头数
        dim_feedforward=2048,  # 前馈网络的维度
        dropout=0.1,  # Dropout概率
    )
    model = BiLSTM_Attention(args, args.BiLSTM_input_size, args.BiLSTM_hidden_size, args.BiLSTM_num_layers,
                             args.dropout,
                             args.alpha, args.mu, device)
    mapper=Mapper(input_dim=2304,output_dim=600)
    classifier = NodeClassifier(hidden_size=600, num_classes=num_classes)
    model_with_weights=ModelWithWeights(transformer_small=transformer_small,transformer_combine=transformer_combine,classifier=classifier,CAGED_model=model,mapper=mapper,args=args)

    # w_1 = torch.nn.Parameter(torch.tensor(1.0))
    # w_2 = torch.nn.Parameter(torch.tensor(1.0))
    #optimizer = torch.optim.Adam(list(transformer_small.parameters())+list(transformer_combine.parameters())+list(classifier.parameters())+list(model.parameters())+list(mapper.parameters()), lr=0.0001)
    optimizer = torch.optim.Adam(model_with_weights.parameters(), lr=0.0001)
    criterion2 = nn.MarginRankingLoss(args.gama)#共享头实体和共享尾实体模型用的损失
    criterion1 = nn.CrossEntropyLoss()#交叉熵损失

    all_triples = dataset.train_data
    train_idx = list(range(len(all_triples) // 2))#前一半是正样本，后一半是负样本
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / args.batch_size)#正样本数目/每个batch的长度=一共的epoch轮数
    total_num_anomalies = dataset.num_anomalies#没啥用
    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation
    model_name = args.model

    model_saved_path = model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + ".ckpt"
    model_saved_path = os.path.join(args.save_dir, model_saved_path)
    print("hh")
    best_loss=100
    for k in range(args.max_epoch):
        print(f"epoch:{k}")

        for it in range(num_iterations):

            print(f"iteration:{it}")
            # start_read_time = time.time()
            # 对于wikilaishuo ,train_idx就是一个长度为4141的从0-4140的list，
            # it 就是epoch轮数
            # batch_size num_neighbor(邻居是自己)
            batch_h, batch_r, batch_t, batch_size,bert_embedding,label= get_pair_batch_train_common(args, dataset, it, train_idx,
                                                                                args.batch_size,
                                                                                args.num_neighbor)
            #break
            # end_read_time = time.time()
            # print("Time used in loading data", it)
            #model.train()
            transformer_small.train()
            transformer_combine.train()
            classifier.train()
            #optimizer.zero_grad()
            model.train()

            label=torch.tensor(label)
            batch_h=torch.LongTensor(batch_h)
            batch_r=torch.LongTensor(batch_r)
            batch_t=torch.LongTensor(batch_t)
            transformer_cls=[]

            outputs,out,out_att = model_with_weights(batch_h, batch_r, batch_t, bert_embedding,batch_size)

            # #--------------------------------bert code-----------------------------------------------
            # for embedding in bert_embedding:
            #     head_embedding=embedding[0][0]
            #     head_mask=embedding[0][1]
            #     relation_embedding=embedding[1][0]
            #     relation_mask=embedding[1][1]
            #     tail_embedding=embedding[2][0]
            #     tail_mask=embedding[2][1]
            #     #combined_embeddings=torch.cat([head_embedding,relation_embedding,tail_embedding],dim=-1)
            #     #首先分别根据掩码计算每一个bert的embedding
            #     head_output = transformer_small(head_embedding, src_key_padding_mask=~head_mask.bool())
            #     relation_output=transformer_small(relation_embedding,src_key_padding_mask=~relation_mask.bool())
            #     tail_output=transformer_small(tail_embedding,src_key_padding_mask=~tail_mask.bool())
            #     combined_embeddings=torch.cat([head_output,relation_output,tail_output],dim=-1)
            #     combined_mask = torch.logical_or(torch.logical_or(head_mask, relation_mask), tail_mask)
            #     #讲所有的embedding通过拼接的方式汇总，再过一遍transformer
            #     transformer_x=transformer_combine(combined_embeddings,src_key_padding_mask=~combined_mask.bool())
            #     transformer_single_cls=transformer_x[:,0,:][0]
            #     transformer_cls.append(transformer_single_cls)
            #     #optimizer.zero_grad()
            #
            # transformer_cls=torch.stack(transformer_cls)
            # #-------------------------------------------------------bert code--------------------------
            # #label#01代码，0代表过期了，1代表没过期
            #
            #
            # #w_3 = torch.nn.Parameter(torch.tensor(1.0))
            #
            # #----------------------------kg code------------------------------------------
            # out, out_att = model(batch_h, batch_r, batch_t)
            # # out.shape 512.600
            # # out_att.shape 1024,600
            out = out.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
            # 256,4,600
            #out_att = out_att.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
            pos_h = out[:, 0, :]
            # 256,2,600
            pos_z0 = out_att[:, 0, :]
            pos_z1 = out_att[:, 1, :]
            neg_h = out[:, 1, :]
            neg_z0 = out_att[:, 2, :]
            neg_z1 = out_att[:, 3, :]
            #
            # # loss function
            # # positive
            # # 上面的是对比损失
            # # 下面的是嵌入损失
            # #pos_h 前200 是头，中间200是关系 后面200是尾巴
            # #pos_z 是正样本增强后生成的两个视图
            pos_loss = args.lam * torch.norm(pos_z0 - pos_z1, p=2, dim=1) + \
                       torch.norm(pos_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)
            # negative
            neg_loss = args.lam * torch.norm(neg_z0 - neg_z1, p=2, dim=1) + \
                       torch.norm(neg_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)

            y = -torch.ones(batch_size)



            #----------------------------kg code------------------------------------------




            #mapper_trans=mapper(transformer_cls)
            #outputs =classifier(mapper_trans+w_1*pos_z0+w_2*pos_z1)


            loss = criterion1(outputs, label)+criterion2(pos_loss, neg_loss, y)
            #loss=criterion1(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Loss:", loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(transformer_small, './model/transformer_small.pth')
                torch.save(transformer_combine, './model/transformer_combine.pth')
                torch.save(classifier, './model/classifier.pth')
                torch.save(model,'./model/CAGED_model.pth')
                torch.save(mapper,'./model/mapper.pth')
                torch.save(model_with_weights,'./model/model_with_weights.pth')

        test(args=args,dataset=dataset,train_idx=train_idx)
                #torch.save(classifier, 'model/classifier.pth')
            # #--------------------------------bert code-----------------------------------------------------
            #
            #
            #
            # batch_h = torch.LongTensor(batch_h).to(device)
            # batch_t = torch.LongTensor(batch_t).to(device)
            # batch_r = torch.LongTensor(batch_r).to(device)
            #
            # #out, out_att = model(batch_h, batch_r, batch_t)
            #
            # # running_time = time.time()
            # # print("Time used in running model", math.fabs(end_read_time - running_time))
            #
            # out = out.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
            # out_att = out_att.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
            #
            # pos_h = out[:, 0, :]
            # pos_z0 = out_att[:, 0, :]
            # pos_z1 = out_att[:, 1, :]
            # neg_h = out[:, 1, :]
            # neg_z0 = out_att[:, 2, :]
            # neg_z1 = out_att[:, 3, :]
            #
            # # loss function
            # # positive
            # pos_loss = args.lam * torch.norm(pos_z0 - pos_z1, p=2, dim=1) + \
            #            torch.norm(pos_h[:, 0:2 * args.BiLSTM_hidden_size] +
            #                       pos_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
            #                       pos_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
            #                       dim=1)
            # # negative
            # neg_loss = args.lam * torch.norm(neg_z0 - neg_z1, p=2, dim=1) + \
            #            torch.norm(neg_h[:, 0:2 * args.BiLSTM_hidden_size] +
            #                       neg_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
            #                       neg_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
            #                       dim=1)
            #
            # y = -torch.ones(batch_size).to(device)
            # loss = criterion(pos_loss, neg_loss, y)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # pos_loss_value = torch.sum(pos_loss) / (batch_size * 2.0)
            # neg_loss_value = torch.sum(neg_loss) / (batch_size * 2.0)
            # logging.info('There are %d Triples in this batch.' % batch_size)
            # logging.info('Epoch: %d-%d, pos_loss: %f, neg_loss: %f, Loss: %f' % (
            #     k, it + 1, pos_loss_value.item(), neg_loss_value.item(), loss.item()))
            #
            # # final_time = time.time()
            # # print("BP time:", math.fabs(final_time - running_time))
            #
            # torch.save(model.state_dict(), model_saved_path)

    print("The training ends!")









def test(args,dataset,train_idx):
    print("test")
    with torch.no_grad():  # 关闭梯度计算
        # print("Loss:",loss.item())

        # inputs = weighted_pooling_embeddings
        # labels = batch['labels']
        batch_h, batch_r, batch_t, bert_embedding, label,batch_size = get_pair_all_test_common(args, dataset, train_idx,
                                                                                    args.batch_size,
                                                                                    args.num_neighbor)#其实batch_sizejiushi
        label = torch.tensor(label)
        batch_h=torch.tensor(batch_h)
        batch_r=torch.tensor(batch_r)
        batch_t=torch.tensor(batch_t)


        transformer_small = torch.load('./model/transformer_small.pth',weights_only=False)
        transformer_combine = torch.load('./model/transformer_combine.pth',weights_only=False)
        CAGED_model=torch.load("./model/CAGED_model.pth",weights_only=False)
        classifier = torch.load('./model/classifier.pth',weights_only=False)
        mapper=torch.load('./model/mapper.pth',weights_only=False)
        model_with_weights=torch.load('./model/model_with_weights',weights_only=False)

        # transformer_cls = []
        # for embedding in bert_embedding:
        #     head_embedding = embedding[0][0]
        #     head_mask = embedding[0][1]
        #     relation_embedding = embedding[1][0]
        #     relation_mask = embedding[1][1]
        #     tail_embedding = embedding[2][0]
        #     tail_mask = embedding[2][1]
        #     # combined_embeddings=torch.cat([head_embedding,relation_embedding,tail_embedding],dim=-1)
        #     # 首先分别根据掩码计算每一个bert的embedding
        #     head_output = transformer_small(head_embedding, src_key_padding_mask=~head_mask.bool())
        #     relation_output = transformer_small(relation_embedding, src_key_padding_mask=~relation_mask.bool())
        #     tail_output = transformer_small(tail_embedding, src_key_padding_mask=~tail_mask.bool())
        #     combined_embeddings = torch.cat([head_output, relation_output, tail_output], dim=-1)
        #     combined_mask = torch.logical_or(torch.logical_or(head_mask, relation_mask), tail_mask)
        #     # 讲所有的embedding通过拼接的方式汇总，再过一遍transformer
        #     transformer_x = transformer_combine(combined_embeddings, src_key_padding_mask=~combined_mask.bool())
        #     transformer_single_cls = transformer_x[:, 0, :][0]
        #     transformer_cls.append(transformer_single_cls)
        #     # optimizer.zero_grad()
        #
        # transformer_cls = torch.stack(transformer_cls)
        # out,out_att=CAGED_model(batch_h, batch_r, batch_t)
        # out_att=out_att.reshape(args.batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
        #
        #
        # pos_z0=out_att[:, 0, :]
        # pos_z1=out_att[:, 1, :]
        # mapper_trans = Mapper(transformer_cls)
        # outputs = classifier(mapper_trans + w_1 * pos_z0 + w_2 * pos_z1)
        outputs,_,_=model_with_weights(batch_h, batch_r, batch_t, bert_embedding,batch_size)




        _, preds = torch.max(outputs, 1)  # 获取预测结果
        accuracy = accuracy_score(label, preds)
        precision = precision_score(label, preds, average='weighted')
        recall = recall_score(label, preds, average='weighted')
        f1 = f1_score(label, preds, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()