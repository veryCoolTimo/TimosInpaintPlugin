"""
TokenizerWrapper and add_tokens for PowerPaint task prompts.
Adapted from open-mmlab/PowerPaint powerpaint/utils/utils.py
"""
import copy
import logging
import random
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import transformers

logger = logging.getLogger(__name__)


class EmbeddingLayerWithFixes(nn.Module):
    """Embedding layer that supports external embeddings for task tokens."""

    def __init__(self, wrapped: nn.Embedding, external_embeddings=None):
        super().__init__()
        self.wrapped = wrapped
        self.num_embeddings = wrapped.weight.shape[0]
        self.external_embeddings = []
        if external_embeddings:
            self.add_embeddings(external_embeddings)
        self.trainable_embeddings = nn.ParameterDict()

    @property
    def weight(self):
        return self.wrapped.weight

    def check_duplicate_names(self, embeddings):
        names = [emb["name"] for emb in embeddings]
        assert len(names) == len(set(names)), f"Duplicated names: {names}"

    def check_ids_overlap(self, embeddings):
        ids_range = [[emb["start"], emb["end"], emb["name"]] for emb in embeddings]
        ids_range.sort()
        for idx in range(len(ids_range) - 1):
            assert ids_range[idx][1] <= ids_range[idx + 1][0], (
                f"IDs overlap between '{ids_range[idx][-1]}' and '{ids_range[idx+1][-1]}'"
            )

    def add_embeddings(self, embeddings):
        if isinstance(embeddings, dict):
            embeddings = [embeddings]
        self.external_embeddings += embeddings
        self.check_duplicate_names(self.external_embeddings)
        self.check_ids_overlap(self.external_embeddings)
        for embedding in embeddings:
            if embedding.get("trainable", False):
                name = embedding["name"]
                embedding["embedding"] = torch.nn.Parameter(embedding["embedding"])
                self.trainable_embeddings[name] = embedding["embedding"]
        names = ", ".join(emb["name"] for emb in embeddings)
        logger.info(f"Added external embeddings: {names}")

    def replace_input_ids(self, input_ids):
        input_ids_fwd = input_ids.clone()
        input_ids_fwd[input_ids_fwd >= self.num_embeddings] = 0
        return input_ids_fwd

    def replace_embeddings(self, input_ids, embedding, external_embedding):
        name = external_embedding["name"]
        start = external_embedding["start"]
        end = external_embedding["end"]
        target_ids = list(range(start, end))
        ext_emb = external_embedding["embedding"]

        if not (input_ids == start).any():
            return embedding

        new_embedding = []
        s_idx, e_idx = 0, 0
        while e_idx < len(input_ids):
            if input_ids[e_idx] == start:
                if e_idx != 0:
                    new_embedding.append(embedding[s_idx:e_idx])
                actual_ids = [int(i) for i in input_ids[e_idx:e_idx + end - start]]
                assert actual_ids == target_ids, (
                    f"Invalid input_ids at {s_idx}-{e_idx}: "
                    f"expected {target_ids} for '{name}', got {actual_ids}"
                )
                new_embedding.append(ext_emb)
                s_idx = e_idx + end - start
                e_idx = s_idx + 1
            else:
                e_idx += 1
        if e_idx == len(input_ids):
            new_embedding.append(embedding[s_idx:e_idx])
        return torch.cat(new_embedding, dim=0)

    def forward(self, input_ids, external_embeddings=None):
        assert input_ids.ndim in [1, 2]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if external_embeddings is None and not self.external_embeddings:
            return self.wrapped(input_ids)

        input_ids_fwd = self.replace_input_ids(input_ids)
        inputs_embeds = self.wrapped(input_ids_fwd)
        vecs = []
        if external_embeddings is None:
            external_embeddings = []
        elif isinstance(external_embeddings, dict):
            external_embeddings = [external_embeddings]
        embeddings = self.external_embeddings + external_embeddings

        for input_id, embedding in zip(input_ids, inputs_embeds):
            new_embedding = embedding
            for ext_emb in embeddings:
                new_embedding = self.replace_embeddings(input_id, new_embedding, ext_emb)
            vecs.append(new_embedding)
        return torch.stack(vecs)


class TokenizerWrapper:
    """Wrapper for CLIPTokenizer that supports placeholder task tokens."""

    def __init__(self, from_pretrained=None, from_config=None, *args, **kwargs):
        module_cls = transformers.CLIPTokenizer

        if from_config:
            from_pretrained = from_config

        if from_pretrained:
            self.wrapped = module_cls.from_pretrained(from_pretrained, *args, **kwargs)
        else:
            self.wrapped = module_cls(*args, **kwargs)

        self._from_pretrained = from_pretrained
        self.token_map = {}

    def __getattr__(self, name):
        if name == "wrapped":
            return super().__getattribute__("wrapped")
        try:
            return getattr(self.wrapped, name)
        except AttributeError:
            raise AttributeError(
                f"'{name}' not found in {self.__class__.__name__} or its wrapped tokenizer"
            )

    def try_adding_tokens(self, tokens, *args, **kwargs):
        num_added = self.wrapped.add_tokens(tokens, *args, **kwargs)
        assert num_added != 0, f"Token {tokens} already exists in tokenizer"

    def get_token_info(self, token):
        token_ids = self.__call__(token).input_ids
        start, end = token_ids[1], token_ids[-2] + 1
        return {"name": token, "start": start, "end": end}

    def add_placeholder_token(self, placeholder_token, *args, num_vec_per_token=1, **kwargs):
        output = []
        if num_vec_per_token == 1:
            self.try_adding_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                self.try_adding_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)

        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f"Token {token} conflicts with {placeholder_token}"
                )
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(self, text, vector_shuffle=False, prop_tokens_to_load=1.0):
        if isinstance(text, list):
            return [self.replace_placeholder_tokens_in_text(t, vector_shuffle=vector_shuffle) for t in text]
        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[:1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return text

    def replace_text_with_placeholder_tokens(self, text):
        if isinstance(text, list):
            return [self.replace_text_with_placeholder_tokens(t) for t in text]
        for placeholder_token, tokens in self.token_map.items():
            merged = " ".join(tokens)
            if merged in text:
                text = text.replace(merged, placeholder_token)
        return text

    def __call__(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        replaced = self.replace_placeholder_tokens_in_text(
            text, vector_shuffle=vector_shuffle, prop_tokens_to_load=prop_tokens_to_load
        )
        return self.wrapped.__call__(replaced, *args, **kwargs)

    def encode(self, text, *args, **kwargs):
        replaced = self.replace_placeholder_tokens_in_text(text)
        return self.wrapped(replaced, *args, **kwargs)

    def decode(self, token_ids, return_raw=False, *args, **kwargs):
        text = self.wrapped.decode(token_ids, *args, **kwargs)
        if return_raw:
            return text
        return self.replace_text_with_placeholder_tokens(text)


def add_tokens(tokenizer, text_encoder, placeholder_tokens, initialize_tokens=None, num_vectors_per_token=1):
    """Add task prompt tokens (P_ctxt, P_shape, P_obj) to tokenizer and text encoder."""
    if initialize_tokens is not None:
        assert len(initialize_tokens) == len(placeholder_tokens)

    for token in placeholder_tokens:
        tokenizer.add_placeholder_token(token, num_vec_per_token=num_vectors_per_token)

    embedding_layer = text_encoder.text_model.embeddings.token_embedding
    text_encoder.text_model.embeddings.token_embedding = EmbeddingLayerWithFixes(embedding_layer)
    embedding_layer = text_encoder.text_model.embeddings.token_embedding

    initialize_embedding = []
    for ii in range(len(placeholder_tokens)):
        if initialize_tokens is not None:
            init_id = tokenizer(initialize_tokens[ii]).input_ids[1]
        else:
            init_id = tokenizer("a").input_ids[1]
        temp_embedding = embedding_layer.weight[init_id]
        if initialize_tokens is not None:
            initialize_embedding.append(temp_embedding[None, ...].repeat(num_vectors_per_token, 1))
        else:
            len_emb = temp_embedding.shape[0]
            init_weight = (torch.rand(num_vectors_per_token, len_emb) - 0.5) / 2.0
            initialize_embedding.append(init_weight)

    token_info_all = []
    for ii in range(len(placeholder_tokens)):
        token_info = tokenizer.get_token_info(placeholder_tokens[ii])
        token_info["embedding"] = initialize_embedding[ii]
        token_info["trainable"] = True
        token_info_all.append(token_info)
    embedding_layer.add_embeddings(token_info_all)
