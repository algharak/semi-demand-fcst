# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

'''
Note: A substantial amount of the following code is sourced from Amazon.com's gluon-ts package and the author here
using this code for experimental and non-commercial purpuses
'''


import pprint
from functools import partial

import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

datasets_di = {
                "m4_hourly":    False,
                "m4_daily":     True,
                "m4_weekly":    False,
                "m4_monthly":   False,
                "m4_quarterly": False,
                "m4_yearly":    False
                }
epochs = 10
num_batches_per_epoch = 5

sff_est = partial(
        SimpleFeedForwardEstimator,
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    )

dar_est = partial(
        DeepAREstimator,
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    )

dar_pl_est = partial(
        DeepAREstimator,
        distr_output=PiecewiseLinearOutput(8),
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    )
mqcnn_est = partial(
        MQCNNEstimator,
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    )

estimators = [sff_est,mqcnn_est,dar_pl_est,dar_est]
estimators_mask = [True,True,True,True]


def evaluate(dataset_name, estimator):
    dataset = get_dataset(dataset_name)
    estimator = estimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
        use_feat_static_cat=True,
        cardinality=[
            feat_static_cat.cardinality
            for feat_static_cat in dataset.metadata.feat_static_cat
        ],
    )
    print(f"evaluating {estimator} on {dataset}")

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_samples=100
    )
    # note: set num_workers to 0 since otherwise it assumes multiple workers and that does not work
    agg_metrics, item_metrics = Evaluator(num_workers=0)(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint.pprint(agg_metrics)

    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    return eval_dict


if __name__ == "__main__":
    results = []

    '''
    for dataset_name in datasets:
    for estimator in estimators:
        # catch exceptions that are happening during training to avoid failing the whole evaluation
        try:
            results.append(evaluate(dataset_name, estimator))
        except Exception as e:
            print(str(e))
    '''
    for dataset_name,dataset_mask in datasets_di.items():
        if dataset_mask:
            for estimator,est_mask in zip(estimators,estimators_mask):
                if est_mask:
                    try:
                        results.append(evaluate(dataset_name, estimator))
                    except Exception as e:
                        print(str(e))

    df = pd.DataFrame(results)
    sub_df = df[
        [
            "dataset",
            "estimator",
            "RMSE",
            "mean_wQuantileLoss",
            "MASE",
            "sMAPE",
            "OWA",
            "MSIS",
        ]
    ]

    print(sub_df.to_string())
