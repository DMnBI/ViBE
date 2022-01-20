# coding=utf-8
# Copyright hjgwak and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ViBE."""
from transformers.utils import logging
from transformers.models.bert.tokenization_bert import BertTokenizer
import math

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "ViBE-base-cased": "https://huggingface.co/ViBE-base-cased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ViBE-base-cased": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "ViBE-base-cased": {"do_lower_case": False},
}


class ViBETokenizer(BertTokenizer):
    r"""
    Construct a ViBE tokenizer.

    :class:`~transformers.ViBETokenizer` is derived from :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting only.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        do_basic_tokenize=True,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            **kwargs,
        )

        self.k = math.log(len(self.vocab) - len(self.all_special_tokens), 4)
        if self.k.is_integer() == False:
            raise ValueError(
                "vocab.txt does not involve enough number of k-mers. "
                "It should contain as many words as powers of 4. "
            )
        else:
            self.k = int(self.k)

    def _tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text, never_split = self.all_special_tokens):
            split_tokens.append(token)
        return split_tokens
