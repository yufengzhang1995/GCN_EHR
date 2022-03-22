from numpy import genfromtxt
import csv
import numpy as np
import pandas as pd
import os
import glob 
import sklearn
import itertools
import random
import tensorflow as tf
import scipy.special
from scipy.sparse import csr_matrix
import scipy.sparse as sp


data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/Yufeng/'
raw_data_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/NSF HF Dataset/'
cohort_1_root = os.path.join(data_root,'cohort_1')
cohort_23_root = os.path.join(data_root,'cohort_23')

file_code_name = ['VAclass','PCT','ICD','LOINC']


root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/'
save_root = os.path.join(root,'Data/Processed/Yufeng/')



class sentence_generator():
    def __init__(self):
        self.years = np.arange(2013,2021) 
        super(sentence_generator, self).__init__()

    def str2lst(self,lst):
        if isinstance(lst, str):
            new_lst =  [x.strip(' ').strip(" ' ").strip(' " ').strip("/").strip(" ' ").strip(' " ') for x in lst.split("[")[-1].split("]")[0].split(',')]
            if '' in new_lst:
                new_lst.remove('')
            return new_lst
        else :
            return []
        
    def code_column_process(self,cohort):
        cohort['VAclass'] = cohort['VAclass'].apply(lambda x: self.str2lst(x))
        
        cohort['CPT4_code'] = cohort['CPT4_code'].apply(lambda x: self.str2lst(x))
        cohort['ICD_code'] = cohort['ICD_code'].apply(lambda x: self.str2lst(x))
        cohort['LOINC'] = cohort['LOINC'].apply(lambda x: self.str2lst(x))
        cohort['full_code'] = cohort['VAclass'] + cohort['CPT4_code'] + cohort['ICD_code'] + cohort['LOINC']
        return cohort

    def process_cohort_1(self):
        files = [os.path.join(cohort_1_root,'cohort1_{}_update.csv').format(c) for c in file_code_name]
        for i,f in enumerate(files):
            if i == 0:
                cohort_1 = pd.read_csv(f)
            else:
                tbl = pd.read_csv(f)
                print(tbl.shape)
                cohort_1 = pd.merge(cohort_1,tbl,on = ['HFID', 'EncID'],how = 'outer')
        cohort_1 = self.code_column_process(cohort_1)
        print(cohort_1.columns)
        return cohort_1

    def process_cohort_23(self):
        cohort_23 = []
        
        for y in self.years:
            files = [os.path.join(cohort_23_root,'cohort23_{}_{}_update.csv').format(c,y) for c in file_code_name]
            for i,f in enumerate(files):
                if i == 0:
                    cohort = pd.read_csv(f)
                else:
                    tbl = pd.read_csv(f)
                    cohort = pd.merge(cohort,tbl,on = ['HFID', 'EncID'],how = 'outer')
                    
            cohort_23.append(cohort)
        
        cohort_23 = pd.concat(cohort_23) 
        cohort_23 = self.code_column_process(cohort_23)
        print(cohort_23.columns)
        return cohort_23

    def process_two_cohorts(self):
        cohort_1 = self.process_cohort_1()
        print('cohort 1 processed',cohort_1.shape)
        cohort_23 = self.process_cohort_23()
        print('cohort 23 processed',cohort_23.shape)
        cohort = pd.concat([cohort_1,cohort_23])
        print('cohort 1 and 23 merged',cohort.shape)
        
        VA = np.unique([item for sublist in cohort['VAclass'].tolist() for item in sublist])
        CPT = np.unique([item for sublist in cohort['CPT4_code'].tolist() for item in sublist])
        ICD = np.unique([item for sublist in cohort['ICD_code'].tolist() for item in sublist])
        LOINC = np.unique([item for sublist in cohort['LOINC'].tolist() for item in sublist])
        
    
        return cohort,VA,CPT,ICD,LOINC


def construct_conditioanl_mat(sentence):
    num_allpairs = int(scipy.special.comb(len(sentence), 2))
    num_allcodes = len(sentence)
    
    uniq_code = np.unique(sentence)
    value_record = np.zeros((len(uniq_code),len(uniq_code)))
    index_record = np.zeros((len(uniq_code),len(uniq_code)),dtype='i,i')
    tmp_idx_word = dict(zip(uniq_code, range(len(uniq_code))))

    for i in range(len(sentence)-1):
        for j in range(i+1,len(sentence)):
            value_record[tmp_idx_word[sentence[i]],tmp_idx_word[sentence[j]]] +=1
            index_record[tmp_idx_word[sentence[i]],tmp_idx_word[sentence[j]]] = (word_index[sentence[i]],word_index[sentence[j]])
            
    value_record = value_record * num_allcodes / num_allpairs
    
    B_cnt = np.array([sentence.count(i) for i in uniq_code])
    con_mat = value_record / B_cnt[np.newaxis,:]
    
    index_record = index_record.flatten()
    row_idx, col_idx = zip(*index_record)
    con_mat = con_mat.flatten()
    
    spa_con_mat = csr_matrix((con_mat, (row_idx,col_idx)), shape = (len(embedding_names), len(embedding_names)))
    return spa_con_mat

def sp_matrix_to_sp_tensor(M):
    """ Convert a sparse matrix to a SparseTensor
    Parameters
    ----------
    M: a scipy.sparse matrix
    Returns
    -------
    X: a tf.SparseTensor
    Notes
    -----
    Also see tf.SparseTensor, scipy.sparse.csr_matrix
    H. J. @ 2019-02-12
    """
    if not isinstance(M, sp.csr.csr_matrix):
        M = M.tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    row, col = M.nonzero()
    X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
    X = tf.cast(X, tf.float32)
    return X






# generate tables with labels
sentence_gen = sentence_generator()
cohort,VA,CPT,ICD,LOINC = sentence_gen.process_two_cohorts()

Cohort_assign = pd.read_csv('/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Processed/NSF HF Dataset/Cohort Assignment.csv')
Cohort_assign = Cohort_assign[['HFID','EncID','Cohort']]
Cohort = pd.merge(cohort,Cohort_assign,on = ['HFID', 'EncID'],how = 'inner')
d_x = Cohort['Date_x'].isna()
d_y = Cohort['Date_y'].isna()
Cohort['Date_combo'] = np.select([d_x, d_y, d_x & d_y], [Cohort['Date_y'], Cohort['Date_x'], np.nan])
labels = Cohort['Cohort'].to_numpy()
sentence_ls = Cohort.full_code.tolist()
print(len(sentence_ls) == len(labels))


# READ EMBEDDINGS AND EMBEDDING ROW NAMES
embedding = genfromtxt('test_sgns_embedding.csv', delimiter=',')
embedding_names = []
with open("test_sgns_row_names.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        embedding_names.extend(line)

word_index =  dict(zip(embedding_names, range(len(embedding_names))))


tmp_root = os.path.join(save_root,'Graph_indep')
if not os.path.isdir(tmp_root):
    os.mkdir(tmp_root)




for i in range(len(sentence_ls)):
    try:
        spa_con_mat = tf.io.serialize_sparse(sp_matrix_to_sp_tensor(construct_conditioanl_mat(sentence_ls[i]))).numpy()  
        label = labels[i]
        f_name = "_".join(list(map(str,Cohort[["HFID",'EncID']].iloc[i])))
        f_name = f_name + "_" + str(i)
        p_name = os.path.join(tmp_root,'{}.tfrecord'.format(f_name))

        if not os.path.isfile(p_name): 
            subject_id =  bytes(f_name, encoding="ascii")
            feature = {
                    'graph/adj': tf.train.Feature(bytes_list=tf.train.BytesList(value=spa_con_mat)),
                    'graph/subject_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_id])),
                    'graph/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            with tf.io.TFRecordWriter(p_name) as writer:
                writer.write(tf_example.SerializeToString())
                writer.close()
            print("write out {}".format(subject_id))
        else:
            print('{} exists!'.format(p_name))
    
    
    except ValueError:
        print('Enter valueerror')
        pass

































