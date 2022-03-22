import os
import glob
import numpy as np
from shutil import copyfile


root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/'
file_root = os.path.join(root,'Data/Processed/Yufeng/Graph_indep')

filenames = []
filenames += glob.glob(os.path.join(file_root,'*tfrecord'))
files = np.array(filenames)
np.random.shuffle(files)

tfrecord_root = os.path.join(root,'Data/Processed/Yufeng/Graph_tfrecords')


fold_names = ['cv_test']
fold_names.extend("fold_"+ str(i) for i in range(3))



split = False

# if split:

# for k in fold_names:
#     fold_dir = os.path.join(tfrecord_root,k)
#     if not os.path.isdir(fold_dir):
#         os.mkdir(fold_dir)



# def chunkIt(seq, num):
#     avg = len(seq) / float(num)
#     out = []
#     last = 0.0
#     while last < len(seq):
#         out.append(seq[int(last):int(last + avg)])
#         last += avg
#     return out

# whole_fold_lens = chunkIt(range(len(files)), len(fold_names))

# for i,s in enumerate(fold_names):
#     save_root = os.path.join(tfrecord_root,s)
#     f = files[whole_fold_lens[i]]
#     for record in f:
#         src = record
#         name = record.split('/')[-1]
#         dst = os.path.join(save_root,name)
#         copyfile(src, dst)
#     print("Save {} files in {}".format(len(f),save_root))

check_overlap = True
if check_overlap:
    files = [glob.glob(os.path.join(tfrecord_root,fold_name,'*tfrecord')) for fold_name in fold_names]
    files[0] = [f.split('/')[-1] for f in files[0]]

    for i in range(1,4):
        f_ls = files[i]
        f_ls = set([f.split('/')[-1] for f in f_ls])
        print(len(f_ls.intersection(set(files[0]))))
        for i in  f_ls.intersection(set(files[0])):
            os.remove(os.path.join(tfrecord_root,'cv_test',i))

        print(len(f_ls.intersection(set(files[0]))))



























# files = [glob.glob(os.path.join(tfrecord_root,fold_name,'*tfrecord')) for fold_name in fold_names]
# files[0] = [f.split('/')[-1] for f in files[0]]

# for i in range(1,4):
#     f_ls = files[i]
#     f_ls = set([f.split('/')[-1] for f in f_ls])
    # print(len(f_ls.intersection(set(files[0]))))
    # for i in  f_ls.intersection(set(files[0])):
    #     os.remove(os.path.join(tfrecord_root,'cv_test',i))

    # print(len(f_ls.intersection(set(files[0]))))












# cv_test_files = np.array(glob.glob(os.path.join(tfrecord_root,'cv_test','*tfrecord')))

# np.random.shuffle(cv_test_files)
# s = ["fold_"+ str(i) for i in range(3)]

# for i in range(3):
#     save_root = os.path.join(tfrecord_root,s[i])
#     f = cv_test_files[50*i:50*(i+1)]
#     for record in f:
#         src = record
#         name = record.split('/')[-1]
#         dst = os.path.join(save_root,name)
#         copyfile(src, dst)
#     print(f)
#     print("Save {} files in {}".format(len(f),save_root))





#     f = files[whole_fold_lens[i]]
#     for record in f:
#         src = record
#         name = record.split('/')[-1]
#         dst = os.path.join(save_root,name)
#         copyfile(src, dst)
#     print("Save {} files in {}".format(len(f),save_root))
 















# train_dir = os.path.join(tfrecord_root,'train')
# val_dir = os.path.join(tfrecord_root,'val')
# test_dir = os.path.join(tfrecord_root,'test')


# train_files = []
# train_files += glob.glob(os.path.join(train_dir,'*tfrecord'))
# val_files = []
# val_files += glob.glob(os.path.join(val_dir,'*tfrecord'))

# train_files.extend(val_files)

# train_files = [os.path.basename(f) for f in train_files]
# filenames = [os.path.basename(f) for f in filenames]




# test_files = list(set(filenames).difference(set(train_files)))
# print(test_files[0])


# for f in test_files:
#     src = os.path.join(file_root,f)
#     dst = os.path.join(test_dir,f)
#     copyfile(src, dst)









# if not os.path.isdir(train_dir):
#     os.mkdir(train_dir)
# if not os.path.isdir(val_dir):
#     os.mkdir(val_dir)
# if not os.path.isdir(test_dir):
#     os.mkdir(test_dir)

# filenames = np.array(filenames)
# np.random.shuffle(filenames)
# train_lens = int(len(filenames) * 0.8)
# val_lens = int(len(filenames) * 0.2)
# # test_lens = int(len(filenames) * 0.2)

# train_files =  filenames[range(train_lens+1)]
# print(range(train_lens+1))
# val_files = filenames[range(train_lens+1, train_lens+val_lens+1)]
# print(range(train_lens+1, train_lens+val_lens+1))
# print(len(filenames))

# for f in train_files:
#     src = f
#     new_f_name = f.split('/')[-1]
#     dst = os.path.join(train_dir,new_f_name)
#     copyfile(src, dst)

# for f in val_files:
#     src = f
#     new_f_name = f.split('/')[-1]
#     dst = os.path.join(val_dir,new_f_name)
#     copyfile(src, dst)







