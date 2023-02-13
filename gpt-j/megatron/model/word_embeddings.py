# Copyright (c) 2021, EleutherAI
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

import torch
import torch.nn as nn
import math

class SoftEmbedding(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_size,
        seq_length,
        wte,
        n_tokens: int = 10,
        init_range: float = 0.5,
        init_string: str = "",
    ):
        super(SoftEmbedding, self).__init__()
        self.random_range = 0.5
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.n_tokens = n_tokens
        self.init_range = init_range
        self.init_string = init_string
        self.soft_embedding_weight = torch.nn.parameter.Parameter(
            self.initialize_embedding()
        )

    def initialize_embedding(self):
        if self.init_string:
            embeds = torch.LongTensor(
                self.tokenizer.tokenize(self.init_string)
            ).to(self.embedding_module.weight.device)
            embeds = self.embedding_module(embeds)
            if embeds.shape[0] >= self.n_tokens:
                embeds = embeds[: self.n_tokens, :]  # slice
            else:
                embeds = embeds.repeat(math.ceil(self.n_tokens / embeds.shape[0]), 1)[
                    : self.n_tokens, :
                ]  # pad up to n_tokens
            return embeds
        return torch.Tensor(self.n_tokens, self.hidden_size).uniform_(
            -self.random_range, self.random_range
        )

    def forward(self, args: tuple):
        in_inference = len(args) == 3  # embeddings, layer_past, attention_mask
        in_train = len(args) == 2  # embeddings, attention_mask
        if in_train:
            embedding, attention_mask = args
        else:
            embedding, layer_past, attention_mask = args
        soft_embedding = self.soft_embedding_weight.repeat(
            embedding.shape[0], 1, 1
        )  # repeat batch_size times
        if in_train:
            # append soft embedding at the beginning in training
            embedding = torch.cat((soft_embedding, embedding), dim=1)
            embedding = embedding[:, : self.seq_length, ...]
            return embedding, attention_mask
        else:
            if not (layer_past is not None and layer_past.numel() > 0):
                # if in inference, on the first forward pass, we want to do the same as in training (append soft embedding)
                embedding = torch.cat((soft_embedding, embedding), dim=1)
                embedding = embedding[:, : self.seq_length, ...]
            # otherwise, we're in incremental mode, and just want to forward the single embedding (since the soft prompt has already been cached)
            return embedding, layer_past, attention_mask

class SoftEmbedding2(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding2, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.soft_embedding_weight = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        soft_embedding_weight = self.soft_embedding_weight.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([soft_embedding_weight, input_embedding], 1)