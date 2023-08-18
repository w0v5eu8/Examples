
import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point #교육 또는 추론을 위한 컨테이너 이미지를 생성하는 데 사용됩니다.
from sagemaker_xgboost_container.data_utils import get_dmatrix #dmatrix 형태 
from sagemaker_xgboost_container import distributed

import xgboost as xgb


def _xgb_train(params, dtrain, evals, num_boost_round, model_dir):

    model = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round = num_boost_round)
    
    model_location = model_dir + '/xgboost-model'
    pkl.dump(model, open(model_location, 'wb'))
        
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # XGB의 파라미터
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--seed', type=int)


    #train,validation 파일이 있는 위치
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")) #경로
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    

    args, _ = parser.parse_known_args()

    dtrain = get_dmatrix(args.train, 'libsvm')
    dval = get_dmatrix(args.validation, 'libsvm')
    watchlist = [(dtrain, 'train'), (dval, 'validation')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
    }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir)

    #training
    _xgb_train(**xgb_train_args)

