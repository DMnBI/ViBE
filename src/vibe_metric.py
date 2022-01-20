# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import datasets
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# TODO: Add BibTeX citation
_CITATION = """\
@article={scikit-learn,
title = {Scikit-learn: Machine Learning in {P}ython},
authors={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
journal={Journal of Machine Learning Research},
volume={12},
pages={2825--2830},
year={2011}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
A set of 5 metrics for evaluating ViBE model.
(Accuracy, Area Under the ROC curve (AUC), F1-score, precision, and recall)
All metrics is calculated using scikit-learn package.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a predicted labels.
    references: list of reference for each prediction. Each
        reference should be a ground truth label.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> vibe_metric = datasets.load_metric("vibe_metric")
    >>> results = vibe_metric.compute(references=[0, 1, 2], predictions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> print(results)
    {'Accuracy': 1.0, 'AUC': 1.0, 'F1-score': 1.0, 'precision': 1.0, 'recall': 1.0}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ViBEMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('float32')),
                'references': datasets.Value('int64'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # TODO: Download external resources if needed

    def _compute(self, predictions, references):
        def simple_accuracy(y_true, y_pred):
            return (y_true == y_pred).mean()
        """Returns the scores"""
        # TODO: Compute the different scores of the metric
        probabilities = np.array(predictions)
        predictions = np.argmax(probabilities, axis = 1)
        accuracy = simple_accuracy(y_true = references, y_pred = predictions)
        precision = precision_score(y_true = references, y_pred = predictions, average = 'macro')
        recall = recall_score(y_true = references, y_pred = predictions, average = 'macro')
        f1 = f1_score(y_true = references, y_pred = predictions, average = 'macro')
        if probabilities.shape[1] != 2:
            auc = roc_auc_score(y_true = references, y_score = probabilities, average = 'macro', multi_class = 'ovo')
        else:
            auc = roc_auc_score(y_true = references, y_score = probabilities[:, 1])

        return {
            "Accuracy": accuracy,
            "AUC": auc,
            "F1-score": f1,
            "precision": precision,
            "recall": recall,
        }
