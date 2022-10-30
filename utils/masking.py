import logging
import random
from re import T
from tqdm import tqdm
from .data_helpers import build_vocab
from .data_helpers import pad_sequence
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np


def vec2str(vec, start, max_len=512):
    v = vec[start: start + max_len]
    s = ''.join([str(i) for i in v])
    # logging.info(s)
    return s


def read_dnaseq(filepath=None, first_start=0, last_start=0, max_len=512):
    df = pd.read_csv(filepath, header=None)
    df = df.to_numpy()
    paragraphs = []
    start = first_start
    while start <= last_start:
        for vec in df:
            paragraphs.append(vec2str(vec, start, max_len=max_len))
        start = start + max_len
    return paragraphs


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class Load1KGPDataset(object):
    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2021,
                 data_name='dnabert',
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 first_train_start=0,
                 last_train_start=0,
                 first_test_start=0,
                 last_test_start=0):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        self.first_train_start = first_train_start
        self.last_train_start = last_train_start
        self.first_test_start = first_test_start
        self.last_test_start = last_test_start
        random.seed(random_state)

    def span_mask(self,span_pby,span_max):
        return None
        


    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = token_ids[mlm_pred_position]  # 10%的时间：保持词不变
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            rand_t=random.random()
            if rand_t < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            elif random.random() > self.masked_token_unchanged_rate:  # 0.5
                # 10%的时间：用随机词替换该词
                    masked_token_id = random.randint(0, len(self.vocab.stoi) - 1)
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        # print(mlm_input_tokens_id.shape)
        #print([mlm_input_tokens_id,mlm_label])
        return mlm_input_tokens_id, mlm_label

    @cache
    def data_process(self, filepath, istraining=True, postfix='cache'):
        """
        本函数的作用是是根据格式化后的数据制作NSP和MLM两个任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        if istraining:
            first_start = self.first_train_start
            last_start = self.last_train_start
        else:
            first_start = self.first_test_start
            last_start = self.last_test_start
        paragraphs = read_dnaseq(filepath, first_start=first_start, last_start=last_start, max_len=self.max_position_embeddings)
        # 返回的是一个二维列表，每个列表可以看做是一个段落（其中每个元素为一句话）
        data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        desc = f" ## 正在构造MLM样本({filepath.split('.')[1]})"
        for paragraph in tqdm(paragraphs, ncols=80, desc=desc):  # 遍历每个
            token_ids = [self.vocab[token] for token in self.tokenizer(paragraph)]
            if len(token_ids) > self.max_position_embeddings:
                # BERT预训练模型只取前512个字符
                token_ids = token_ids[:self.max_position_embeddings]
            #logging.debug(f" ## Mask之前token ids:{token_ids}")
            mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
            token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
            mlm_label = torch.tensor(mlm_label, dtype=torch.long)
            max_len = max(max_len, token_ids.size(0))
            #logging.debug(f" ## Mask之后token ids:{token_ids.tolist()}")
            #logging.debug(f" ## Mask之后label ids:{mlm_label.tolist()}")
            #logging.debug(f" ## 当前样本构造结束================== \n\n")
            data.append([token_ids, mlm_label])

        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_mlm_label = [], []
        for (token_ids, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        return b_token_ids, b_mask, b_mlm_label

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"
        test_data = self.data_process(filepath=test_file_path, istraining=False,
                                      postfix='test' + postfix)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            logging.info(f"## 成功返回测试集，一共包含样本{len(test_iter.dataset)}个")
            return test_iter
        data = self.data_process(filepath=train_file_path, istraining=True, postfix='train' + postfix)
        train_data, max_len = data['data'], data['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_data = self.data_process(
            filepath=val_file_path, istraining=False, postfix='val' + postfix)['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch)
        logging.info(f"## 成功返回训练集样本（{len(train_iter.dataset)}）个、开发集样本（{len(val_iter.dataset)}）个"
                     f"测试集样本（{len(test_iter.dataset)}）个.")
        return train_iter, test_iter, val_iter

def span_masking(sentence, spans, tokens, MASK_IDS, mask_id, pad_len, mask, replacement='word_piece', endpoints='external'):
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.full(sent_length, pad)
    pair_targets = []
    spans = merge_intervals(spans)
    assert len(mask) == sum([e - s + 1 for s,e in spans])
    # print(list(enumerate(sentence)))
    for start, end in spans:
        lower_limit = 0 if endpoints == 'external' else -1
        upper_limit = sent_length - 1 if endpoints == 'external' else sent_length
        if start > lower_limit and end < upper_limit:
            if endpoints == 'external':
                pair_targets += [[start - 1, end + 1]]
            else:
                pair_targets += [[start, end]]
            pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
        rand = np.random.random()
        for i in range(start, end + 1):
            assert i in mask
            target[i] = sentence[i]
            if replacement == 'word_piece':
                rand = np.random.random()
            if rand < 0.8:
                masked_token_id = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                token= np.random.choice(tokens)
                sentence[i] = np.random.choice(tokens)
    pair_targets = pad_to_len(pair_targets, MASK_IDS, pad_len + 2)
    # if pair_targets is None:
    return sentence, target, pair_targets

def mask(self, sentence, entity_map=None):
        """mask tokens for masked language model training
        Args:
            sentence: 1d tensor, token list to be masked
            mask_ratio: ratio of tokens to be masked in the sentence
        Return:
            masked_sent: masked sentence
        """
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
        # get entity spans
        entity_spans, spans = [], []
        new_entity = True
        for i in range(entity_map.length()):
            if entity_map[i] and new_entity:
                entity_spans.append([i, i])
                new_entity = False
            elif entity_map[i] and not new_entity:
                entity_spans[-1][-1] = i
            else:
                new_entity = True
        while len(mask) < mask_num:
            if np.random.random() <= self.args.ner_masking_prob:
                self.mask_entity(sentence, mask_num, word_piece_map, spans, mask, entity_spans)
            else:
                span_len = np.random.choice(self.lens, p=self.len_distrib)
                anchor  = np.random.choice(sent_length)
                if anchor in mask:
                    continue
                self.mask_random_span(sentence, mask_num, word_piece_map, spans, mask, span_len, anchor)
        sentence, target, pair_targets = span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.args.replacement_method, endpoints=self.args.endpoints)
        if self.args.return_only_spans:
            pair_targets = None
        return sentence, target, pair_targets

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x : x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


if __name__ == '__main__':
    read_dnaseq(filepath='/home/linwenhao/Berthap/data/dnabert/hap_test.csv')
