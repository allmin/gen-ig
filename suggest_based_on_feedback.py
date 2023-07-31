from sklearn.neighbors import KNeighborsClassifier
from config import scope_name, system_name
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D,MaxPooling2D, SimpleRNN, Bidirectional, Flatten, Dense, Concatenate, Subtract, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import ThresholdedReLU, LeakyReLU
from tensorflow.keras.layers import Input as Inp
from tensorflow.keras import optimizers
from sklearn.cluster import KMeans
from numpy.random import seed
from sklearn.utils import shuffle


import regex as re
seed(1)
import tensorflow as tf
tf.random.set_seed(2) 
import numpy as np
import os
import pandas as pd
import pdb
from time import time

def get_inp(df, label_name):
    x1 = np.array([df['meanA'].tolist()]).T
    x2 = np.array([df['meanB_'].tolist()]).T
    feats = np.array([list(i) for i in df['feature_vector'].values])
    
    labels = df[label_name].values
    if label_name == 'feedback':
        labels = np.array([fb_dict[i] for i in labels])
    return x1, x2, feats, labels

def train_model(aug_model,all_insight_df,gt_df):
    x1, x2, feats, labels = get_inp(all_insight_df, 'labels')
    x1_val, x2_val, feats_val, labels_val = get_inp(gt_df, 'feedback')
    
    if len(feats[0]) != len(feats_val[0]):
        len_diff = (len(feats[0]) - len(feats_val[0]))
        feats_val = np.pad(feats_val,((0,0),(0,len_diff)))

    all_insight_data = [[x1,x2 ],feats]
    feat_val = [[x1_val, x2_val],feats_val]
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    aug_model.fit(all_insight_data,labels,epochs = 100,validation_data = (feat_val, labels_val), callbacks=[checkpoint])

def get_predictions(aug_model,all_insight_df):
    x1 = np.array([all_insight_df['meanA'].tolist()]).T
    x2 = np.array([all_insight_df['meanB_'].tolist()]).T
    feats = np.array([list(i) for i in all_insight_df['feature_vector'].values])
    all_insight_data = [[x1,x2 ],feats]
    return [float(float('{:0.02f}'.format(i))>0.5) for i in aug_model.predict(all_insight_data)[:,0]]

def get_base_model_icml(NUMBER_OF_BINS):
    input_layer1 = Inp(shape=(NUMBER_OF_BINS, 1),name='input_1')
    input_layer2 = Inp(shape=(NUMBER_OF_BINS, 1),name='input_2')
    act1 = LeakyReLU(alpha = 0.3)
    act2 = ThresholdedReLU(theta=0.05)
    y1 = Dense(50)(input_layer1)
    y1 = act1(y1)
    y1 = Dense(20)(y1)
    y1 = act1(y1)
    y2 = Dense(50)(input_layer2)
    y2 = act1(y2)
    y2 = Dense(20)(y2)
    y2 = act1(y2)
    y = Concatenate(axis=-1)([y1, y2])
    y = Bidirectional(SimpleRNN(100, return_sequences=True))(y)
    # y = Dense(100)(y)
    y = act1(y)
    y = Flatten()(y)
    y = Dense(100)(y)
    y = act1(y)
    # Output layer
    output_layer = Dense(1)(y)
    output_layer = act2(output_layer)
    model = Model([input_layer1, input_layer2], output_layer)
#    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    return model

def load_base_model(base_path, NUMBER_OF_BINS):
    base_model = get_base_model_icml(NUMBER_OF_BINS)
    if os.path.exists(base_path):
        base_model.load_weights(base_path)
    return base_model

def get_augmented_model(user, num_bins, f_size=100):
    base_path="systems/{}/outputs/feedback_models/base_model.hdf5".format(system_name)
    LR = 10e-3
    sgd = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    base_model = load_base_model(base_path, num_bins)
    FREEZE_BASE = True
    if FREEZE_BASE:
        for i in range(11):
            base_model.layers[i].trainable = False
    layer_name = base_model.layers[-3].name
    base_model_cropped_layer = base_model.get_layer(layer_name).output
    input_layer3 = Inp(shape = (f_size , 1), name = 'fvector')
    dense_new = Dense(50)(input_layer3)
    delta_wing = Flatten()(dense_new)
    y = Concatenate(axis=-1, name = 'final_concatenate')([base_model_cropped_layer, delta_wing])
    y = Dense(100, activation='relu')(y)
    out = Dense(1, activation='sigmoid', name = 'final_output')(y)
    augmented_model = Model(base_model.inputs + [input_layer3], out)
    sgd1 = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.0, nesterov=False, name="SGD")
    augmented_model.compile(optimizer=sgd1, loss='mse', metrics=['accuracy'])
    if os.path.exists('systems/{}/outputs/feedback_models/model_{}.hdf5'.format(system_name, user)):
        os.remove('systems/{}/outputs/feedback_models/model_{}.hdf5'.format(system_name, user))
    #   augmented_model.load_weights('systems/{}/outputs/feedback_models/model_{}.hdf5'.format(system_name, user))
    return augmented_model

def generate_pseudo_labeled_data(labelled_data, unlabelled_data):
    knn = KNeighborsClassifier(n_neighbors=1)
    # len_recent_feature = len(unlabelled_data['feature_vector'].values[0])
    feat = np.array([list(i) for i in labelled_data['feature_vector'].values])
    feat2 = np.array([list(i) for i in unlabelled_data['feature_vector'].values])
    if len(feat2[0]) != len(feat[0]):
        len_diff = (len(feat2[0]) - len(feat[0]))
        feat = np.pad(feat,((0,0),(0,len_diff)))
    labels = [fb_dict[i] for i in labelled_data['feedback']]
    knn.fit(feat,labels)
    pred = knn.predict(feat2)
    pseudo_labelled_data = unlabelled_data.copy()
    pseudo_labelled_data['labels'] = pred
    return pseudo_labelled_data

def perform_kmeans(K,X):
    X = np.array([i for i in X.values])
    K = min(K,len(X))
    if K == 0:
        return []
    kms = KMeans(n_clusters=K, random_state=123).fit(X)
    return kms.labels_

if __name__ == '__main__':
    fb_dict = {'veryrelevant':1, 'relevant':1, 'neutral':1, 'limitedutility':0, 'notrelevantatall':0}
    fb_dict = {'Very relevant':1, 'Relevant':1, 'Neutral':1, 'Limited utility':0, 'Useless':0}
    num_insights_needed = 23

    user_list = [scope_name]
    user_algorithm_dict = {k:2 for k in user_list}
    for (user, algorithm) in user_algorithm_dict.items():
        if algorithm == 2:
            fb_path = 'systems/{}/outputs/feedbacks/{}.xlsx'.format(system_name,user)
            fb_path2 = 'systems/{}/outputs/feedbacks/{}.pickle'.format(system_name,user)
            if os.path.exists(fb_path):
                fb_df = pd.read_excel(fb_path, engine='openpyxl')
                fb_df2 = pd.read_pickle(fb_path2)
                
                fb_df['feature_vector'] = fb_df2['feature_vector']
                fb_df = fb_df.dropna(subset=['feature_vector'])
                fb_df = fb_df.dropna(subset=['feedback'])
                cluster_labels = perform_kmeans(K=num_insights_needed,X=fb_df['feature_vector'])
                fb_df['cluster_labels'] = cluster_labels#fb_df2['cluster_labels']
                past_time = time()-(100*24*3600)
                fb_df = fb_df[fb_df['timestamp_feedback'] > past_time]
            else:
                continue
            insight_path = 'systems/{}/outputs/scored_insights_final_with_clusters_{}_{}.pickle'.format(system_name,user,system_name)
            if os.path.exists(insight_path):
                insight_df = pd.read_pickle(insight_path)
            feat_len = len(insight_df.feature_vector.iloc[0])
            model = get_augmented_model(user, num_bins=1, f_size=feat_len)
            pseudo_labelled_df = generate_pseudo_labeled_data(fb_df, insight_df)
            model_path = 'systems/{}/outputs/feedback_models/model_{}.hdf5'.format(system_name,user)
            
            print('training neural model')

            pseudo_labelled_df['meanB_'] = pseudo_labelled_df['meanB'].astype('str')

            pseudo_labelled_df['meanB_'] = [float(re.findall('([0-9]+[.]*[0-9]*)', i)[0]) for i in pseudo_labelled_df['meanB_']]
            
            train_model(model, pseudo_labelled_df, fb_df)
            pseudo_labelled_df.to_excel('systems/{}/outputs/pseudo_labelled_{}.xlsx'.format(system_name,user), index=False)
            model.load_weights(model_path)
            # feats = np.array([list(i) for i in pseudo_labelled_df['feature_vector'].values])
            neural_predicted_labels = get_predictions(model,pseudo_labelled_df)
            # neural_predicted_labels = pseudo_labelled_df['labels']
            pseudo_labelled_df['neural_predicted_labels'] = neural_predicted_labels
            pseudo_labelled_df.to_excel('systems/{}/outputs/neuro_labelled_{}.xlsx'.format(system_name,user), index=False)
            pseudo_labelled_df = pseudo_labelled_df[(pseudo_labelled_df['neural_predicted_labels'] == 1) & (pseudo_labelled_df['labels']==1)]
            cluster_labels = perform_kmeans(K=num_insights_needed,X=pseudo_labelled_df['feature_vector'])
            pseudo_labelled_df['cluster_labels'] = cluster_labels#fb_df2['cluster_labels']
            for iid in fb_df['insight_identifier']:
                pseudo_labelled_df = pseudo_labelled_df[pseudo_labelled_df['insight_identifier']!=iid]
            print('picking from each cluster')
            recommended_insights = pd.DataFrame()
            for i in range(num_insights_needed):
                sorting_criteria = 'pscore_final'
                insight_candidates = pseudo_labelled_df[pseudo_labelled_df['cluster_labels'] == i]
                insight_candidate = insight_candidates[insight_candidates[sorting_criteria] == (insight_candidates[sorting_criteria].max())]
                if len(insight_candidate) > 1:
                        insight_candidate = shuffle(insight_candidate).iloc[:1]
                if len(recommended_insights) == 0:
                    recommended_insights = insight_candidate
                else:
                    recommended_insights = pd.concat([recommended_insights,insight_candidate])

            recommended_insights.to_excel('systems/{}/outputs/neural_recommended_{}.xlsx'.format(system_name,user), index=False)


        


