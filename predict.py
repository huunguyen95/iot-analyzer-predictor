import os
import sys
import logging
from glob import glob
from shutil import rmtree
from hashlib import md5

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


logger = logging.getLogger(__name__)

AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = (78, 4)
BATCH_SIZE = 1024
NUM_CLASSES = 6

MODEL_PATH = glob('model/multiclass_*.h5')[0]
TEMP_PATH = 'temp'
FEATURES = ['dst_port', 'protocol', 'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len', 'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s',
            'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']
LABELS = ['Benign', 'Botnet', 'Bruteforce', 'DoS', 'Tấn công xâm nhập']


class NetworkClassifier:
    def __init__(self, batch_size=512):
        self.model = self.load_model()
        self.batch_size = batch_size

    def load_model(self):
        model = keras.models.load_model(MODEL_PATH)
        return model

    def create_dataset(self, paths):
        def decode_image(path):
            bits = tf.io.read_file(path)
            image = tf.image.decode_png(bits, channels=4)
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.reshape(image, IMAGE_SIZE)
            return image

        dataset = (
            tf.data.TFRecordDataset
            .from_tensor_slices(paths)
            .map(decode_image, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
        )
        return dataset

    def predict(self, csv_path):
        test_df = pd.read_csv(csv_path)
        feat_df = test_df.loc[:, FEATURES]

        # Convert network flow to RGBA image
        if os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
            os.mkdir(TEMP_PATH)
        else:
            os.mkdir(TEMP_PATH)
        for index, row in feat_df.iterrows():
            try:
                data = row.values.astype('float')
                image = Image.fromarray(data, mode='RGBA')
                image.save(f'{TEMP_PATH}/{index}.png')
            except Exception as e:
                print(e)

        # Create test dataset
        test_paths = glob(f'{TEMP_PATH}/*.png')
        test_dataset = self.create_dataset(test_paths)

        y_probs = self.model.predict(test_dataset)
        y_pred = np.argmax(y_probs, axis=1)

        ret_df = test_df.loc[:, ['src_ip', 'dst_ip']]
        ret_df.loc[:, 'pred'] = y_pred
        ret_df.drop(ret_df[ret_df['pred'] == 0].index, inplace=True)
        ret_df.loc[:, 'pred'] = ret_df.pred.apply(lambda x: LABELS[x])

        ret_df = ret_df.groupby(ret_df.columns.tolist(), as_index=False).size()
        ret_df.drop(ret_df[ret_df['size'] <= 5].index, inplace=True)
        ret_df.rename(columns={'size': 'N of flows'}, inplace=True)

        return ret_df


if __name__ == '__main__':
    csv_path = sys.argv[1]
    model = NetworkClassifier()
    ret = model.predict(csv_path)
    print(ret)
