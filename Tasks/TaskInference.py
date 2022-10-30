from ast import arg
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging
import os
from utils import logger_init
from model import BertConfig
from model import BertForMaskedLM
from utils import Load1KGPDataset
from transformers import BertTokenizer
import torch
import time
import argparse


class ModelConfig:
    def __init__(self, masked_rate=0.4, train_set="hap_train.csv", test_set="hap_test.csv", first_train_start=512, last_train_start=512, first_test_start=512, last_test_start=512, do_logging=True):
        self.project_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'dnabert')
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "bert_dna")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_set = train_set
        self.test_set = test_set
        self.train_file_path = os.path.join(
            self.dataset_dir, train_set)
        self.val_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.test_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        # self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.logs_save_dir = os.path.join(self.project_dir, '../records/bertmlm/raw')
        self.data_name = 'dnabert'
        self.model_save_path = os.path.join(
            self.model_save_dir, f'model_{self.data_name}.pt')
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 24
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2021
        self.learning_rate = 5e-5
        self.masked_rate = masked_rate
        self.masked_token_rate = 1
        self.masked_token_unchanged_rate = 1
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 100
        self.model_val_per_epoch = 1
        self.first_train_start = first_train_start
        self.last_train_start = last_train_start
        self.first_test_start = first_test_start
        self.last_test_start = last_test_start
        self.do_logging = do_logging
        self.attention = ''

        if do_logging:
            logger_init(log_file_name='dnabert', log_level=self.log_level,
                        log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(
            self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        if do_logging:
            logging.info(" ### 将当前配置打印到日志文件中 ")
            for key, value in self.__dict__.items():
                logging.info(f"### {key} = {value}")


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def fakeTokenizer(s):
    return list(s)


def roc_auc(mlm_logits, mlm_labels, vocab_size, PAD_IDX):
    with torch.no_grad():
        mlm_pred = mlm_logits.transpose(0, 1).reshape(-1, mlm_logits.shape[2])
        mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
        mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 获取mask位置的行索引
        mlm_pred = mlm_pred[mask, 5:]  # 去除预测为特殊标记可能性
        mlm_pred_sm = torch.softmax(mlm_pred, dim=1).cpu()
        mlm_true = mlm_true[mask]
        mlm_true = mlm_true.reshape(-1, 1).cpu()
        mlm_true = torch.zeros(mlm_true.shape[0], vocab_size).scatter_(
            dim=1, index=mlm_true, value=1)
        mlm_true = mlm_true[:, 5:]
        return roc_auc_score(y_true=mlm_true, y_score=mlm_pred_sm, average='macro', multi_class='ovr')


def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
    # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    return (mlm_acc, mlm_correct, mlm_total)


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, auc, cnt = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            mlm_logits = model(input_ids=b_token_ids,
                                          attention_mask=b_mask,
                                          token_type_ids=None)
            result = accuracy(mlm_logits, b_mlm_label, PAD_IDX)
            _, mlm_cor, mlm_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            auc += roc_auc(mlm_logits, b_mlm_label, config.vocab_size, PAD_IDX)
            cnt += 1
    model.train()
    return (float(mlm_corrects) / mlm_totals, auc / cnt)


def inference(config):
    model = BertForMaskedLM(config)
    if os.path.exists(config.model_save_path):
        # loaded_paras = torch.load(config.model_save_path)
        # model.load_state_dict(loaded_paras)
        model = torch.load(config.model_save_path, map_location={'cuda:1':'cuda:0'})
        logging.info("## 成功载入已有模型，进行推断......")
    else:
        logging.info("## 已训练模型不存在，退出......")
        return

    model = model.to(config.device)
    bert_tokenize = fakeTokenizer
    #bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = Load1KGPDataset(vocab_path=config.vocab_path,
                                  tokenizer=bert_tokenize,
                                  batch_size=config.batch_size,
                                  max_sen_len=config.max_sen_len,
                                  max_position_embeddings=config.max_position_embeddings,
                                  pad_index=config.pad_index,
                                  is_sample_shuffle=config.is_sample_shuffle,
                                  random_state=config.random_state,
                                  data_name=config.data_name,
                                  masked_rate=config.masked_rate,
                                  masked_token_rate=config.masked_token_rate,
                                  masked_token_unchanged_rate=config.masked_token_unchanged_rate,
                                  first_train_start=config.first_train_start,
                                  last_train_start=config.last_train_start,
                                  first_test_start=config.first_test_start,
                                  last_test_start=config.last_test_start)
    _, _, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)

    mlm_acc, auc = evaluate(config, val_iter, model, data_loader.PAD_IDX)
    logging.info(
        f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, roc_auc: {auc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="To do MLM task on 1KGP dataset.")
    parser.add_argument('-mr', '--masked_rate', type=float,
                        help='Masking rate of SNPs.', default=0.4)
    parser.add_argument('-ftrs', '--first_train_start', type=int,
                        help='Position where first 512 SNPs in training set start.', default=0)
    parser.add_argument('-ltrs', '--last_train_start', type=int,
                        help='Position where last 512 SNPs in training set start.', default=1000)
    parser.add_argument('-ftes', '--first_test_start', type=int,
                        help='Position where first 512 SNPs in testing set start.', default=0)
    parser.add_argument('-ltes', '--last_test_start', type=int,
                        help='Position where last 512 SNPs in testing set start.', default=1000)
    parser.add_argument('-trs', '--train_set', type=str,
                        help='train_set', default='hap_train.csv')
    parser.add_argument('-tes', '--test_set', type=str,
                        help='test_set', default='hap_test.csv')

    args = parser.parse_args()
    if args.last_train_start < args.first_train_start:
        args.last_train_start = args.first_train_start
    if args.last_test_start < args.first_test_start:
        args.last_test_start = args.first_test_start

    config = ModelConfig(masked_rate=args.masked_rate,
                         first_train_start=args.first_train_start,
                         last_train_start=args.last_train_start,
                         first_test_start=args.first_test_start,
                         last_test_start=args.last_test_start,
                         train_set=args.train_set,
                         test_set=args.test_set)

    files = os.listdir('../data/dnabert/')
    for file in files:
        if file.startswith('hap_test_'):
            os.remove('../data/dnabert/' + file)
    inference(config)

