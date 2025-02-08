import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D,Dropout,Input, Flatten,BatchNormalization
from keras.layers import concatenate, Dropout, Activation,Embedding
import pandas as pd, numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys, os
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

memory_limit = 4500
if memory_limit:
    tf.config.set_logical_device_configuration(tf.config.list_physical_devices('GPU')[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)] )

def enc_list_bl_max_len(aa_seqs, blosum, max_seq_len):
    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        e_seq=np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                print(aa)
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)

    # pad sequences:
    #max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq


blosum50_20aa = {
        'A': np.array((5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0)),
        'R': np.array((-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3)),
        'N': np.array((-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3)),
        'D': np.array((-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4)),
        'C': np.array((-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1)),
        'Q': np.array((-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3)),
        'E': np.array((-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3)),
        'G': np.array((0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4)),
        'H': np.array((-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4)),
        'I': np.array((-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4)),
        'L': np.array((-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1)),
        'K': np.array((-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3)),
        'M': np.array((-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1)),
        'F': np.array((-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1)),
        'P': np.array((-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3)),
        'S': np.array((1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2)),
        'T': np.array((0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0)),
        'W': np.array((-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3)),
        'Y': np.array((-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1)),
        'V': np.array((0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5))
    }

opt = keras.optimizers.legacy.Adam(learning_rate=0.001)

def net_conv(L):
    input_in = Input(shape=(L,20))
    cdr_conv1 = Conv1D(16, 1, padding='same', activation='relu')(input_in)
    batch1 = BatchNormalization()(cdr_conv1)
    cdr_conv3 = Conv1D(16, 3, padding='same', activation='relu')(input_in)
    batch3 = BatchNormalization()(cdr_conv3)
    cdr_conv5 = Conv1D(16, 5, padding='same', activation='relu')(input_in)
    batch5 = BatchNormalization()(cdr_conv5)
    cdr_conv7 = Conv1D(16, 7, padding='same', activation='relu')(input_in)
    batch7 = BatchNormalization()(cdr_conv7)
    cdr_conv9 = Conv1D(16, 9, padding='same', activation='relu')(input_in)
    batch9 = BatchNormalization()(cdr_conv9)
    
    #cdr_cat = concatenate([cdr_conv1, cdr_conv3, cdr_conv5, cdr_conv7, cdr_conv9])
    cdr_cat = concatenate([batch1,batch3,batch5,batch7,batch9])

    #convs1 = Conv1D(filters=16, kernel_size=2, padding='same', 
    #batch1 = BatchNormalization()(convs1)
    #convs2 = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu')(batch1)
    #batch2 = BatchNormalization()(convs2)
    flat = Flatten()(cdr_cat)
    dense1 = Dense(16, activation='relu')(flat)
    batch3 = BatchNormalization()(dense1)
    
    out = Dense(4, activation='softmax')(batch3)
    
    model = (Model(inputs=[input_in],outputs=[out]))
    
    return model

P_samples = int(sys.argv[1])
L =30; N_samples = int(P_samples)
EPOCHS, batch_size = 25, 128
repeats = int(25)

pm,beta = 0.2,1.00

peptide1, peptide2, peptide3 = 'AMFWSVPTV','YLQPRTFLL','ATDALMTGY'
check1, check2, check3 = 1500,2100,170

df1_1 = pd.read_csv(f'./Binders_{peptide1}.csv').drop_duplicates()
df2 = pd.read_csv(f'./Generated_binders_{peptide1}_RBM_beta{beta:.2f}.csv').drop_duplicates()
df2 = df2[df2['CDR3b'].str.len()>7]
df2 = df2[~df2['CDR3b'].isin(df1_1['CDR3b'])].dropna()

data_pos_1 = pd.concat((df1_1.sample(50),df2.sample(P_samples - 50 ) )).drop_duplicates()
data_pos_1['labels'] = 0
df1_1['labels'] = 0

data_pos_2 = pd.read_csv(f'./tchard_{peptide2}.csv').drop_duplicates()
data_pos_2['labels'] = 1

df1_3 = pd.read_csv(f'./Binders_{peptide3}.csv').drop_duplicates()
df2 = pd.read_csv(f'./Generated_binders_{peptide3}_RBM_beta{beta:.2f}.csv').drop_duplicates()
df2 = df2[df2['CDR3b'].str.len()>7]
df2 = df2[~df2['CDR3b'].isin(df1_3['CDR3b'])].dropna()

data_pos_3 = pd.concat((df1_3.sample(135),df2.sample(P_samples - 135) )).drop_duplicates()
data_pos_3['labels'] = 2
df1_3['labels'] = 2

T = int(24)

results = []
data_neg = pd.read_csv('./Background_notaligned.csv')
data_neg['labels'] = 3

Test_insample = []; Test_ext = []
print('Training... and testing over %d repeats with entry-state: ' %(repeats))

df_tchar = pd.read_csv(f'./tchard_{peptide1}.csv').drop_duplicates();
ext_eval_1 = df_tchar[~df_tchar['CDR3b'].isin(data_pos_1['CDR3b'])]
ext_eval_1['labels']=0

df_tchar = pd.read_csv(f'./Binders_{peptide2}.csv').drop_duplicates();
ext_eval_2 = df_tchar[~df_tchar['CDR3b'].isin(data_pos_2['CDR3b'])]
ext_eval_2['labels']=1

df_tchar = pd.read_csv(f'./tchard_{peptide3}.csv').drop_duplicates();
ext_eval_3 = df_tchar[~df_tchar['CDR3b'].isin(data_pos_3['CDR3b'])]
ext_eval_3['labels']=2


for K in tqdm(range(repeats)):
    
    if (K == 0 or K == repeats-1):
        print('\t \t repeat n: %d' %K)
    
    data_pos_in_1 = data_pos_1.sample(P_samples)
    data_pos_in_2 = data_pos_2.sample(P_samples)
    data_pos_in_3 = data_pos_3.sample(P_samples)
    data_neg_in = data_neg.sample(N_samples)
    
    
    cdr_in = pd.concat((data_pos_in_1,data_pos_in_2,data_pos_in_3,data_neg_in ))
    cdr_in.reset_index(inplace=True,drop=True)
    y_train = cdr_in['labels']; y_train = to_categorical(y_train)
    cdr_in = enc_list_bl_max_len(cdr_in['CDR3b'], blosum50_20aa, L)
    
    mdl = net_conv(L); mdl.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['AUC','accuracy'])
    hist = mdl.fit(cdr_in, y_train, epochs=EPOCHS, batch_size=batch_size,verbose=0)
    
    pos_test_1 = df1_1[~df1_1['CDR3b'].isin(data_pos_in_1['CDR3b'])].dropna().sample(T)
    pos_test_2 = data_pos_2[~data_pos_2['CDR3b'].isin(data_pos_in_2['CDR3b'])].dropna().sample(T)
    pos_test_3 = df1_3[~df1_3['CDR3b'].isin(data_pos_in_3['CDR3b'])].dropna().sample(T)
    neg_test = data_neg[~data_neg.isin(data_neg_in)].dropna()
    cdr_test = pd.concat((pos_test_1,pos_test_2,pos_test_3,neg_test.sample(T) ))
    cdr_test.reset_index(inplace=True,drop=True)
    y_test = cdr_test['labels']; y_test = to_categorical(y_test)
    cdr_test = enc_list_bl_max_len(cdr_test['CDR3b'], blosum50_20aa, L)
    Test_insample.append(mdl.evaluate(cdr_test,y_test,verbose=0)[1:])
    
    neg_test = data_neg[~data_neg.isin(data_neg_in)].dropna().sample(2*T)
    cdr_test_ext = pd.concat((ext_eval_1.sample(2*T),ext_eval_2.sample(2*T),
                          ext_eval_3.sample(2*T),neg_test ))
    cdr_test_ext.reset_index(inplace=True,drop=True)
    y_test_ext = cdr_test_ext['labels']; y_test_ext = to_categorical(y_test_ext)
    cdr_test_ext = enc_list_bl_max_len(cdr_test_ext['CDR3b'], blosum50_20aa, L)
    
    Test_ext.append(mdl.evaluate(cdr_test_ext,y_test_ext,verbose=0)[1:])
    
    #print(entry_state, k,mdl.evaluate(cdr_unseen,y_unseen,verbose=0))
    del mdl
    keras.backend.clear_session()

Tback_n=np.mean(np.array(Test_insample),axis=0); ETback_n= np.std(np.array(Test_insample),axis=0)
Text_n=np.mean(np.array(Test_ext),axis=0); EText_n= np.std(np.array(Test_ext),axis=0)

    
results.append(np.concatenate(([P_samples],[N_samples],Tback_n,ETback_n,Text_n,EText_n )) )

columns_name = ['Psamples','Nsamples','AUC-back','ACC-back','std-back','stdACC-back','AUC-ext','ACC-ext','std-ext','stdACC-ext']

df = pd.DataFrame(results,columns=columns_name)
f ='./AUCACC_performance_multiclass_epochs_{}_batchsize_{}_three_min_RBM.csv'.format(EPOCHS,batch_size)
if os.path.isfile(f):
    df.to_csv(f, mode = 'a', header = None,index=False)
else:
    df.to_csv(f,index=False)
