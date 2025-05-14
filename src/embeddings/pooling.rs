use crate::embeddings::Features;
use rust_bert::Config;
use serde::{Deserialize, Serialize};
use std::f32;
use tch::{Kind, Tensor, nn};

#[derive(Debug, Serialize, Deserialize)]
pub struct PoolingConfig {
    pub word_embedding_dimension: i16,
    pub pooling_mode_cls_token: Option<bool>,
    pub pooling_mode_max_tokens: Option<bool>,
    pub pooling_mode_mean_tokens: Option<bool>,
    pub pooling_mode_mean_sqrt_len_tokens: Option<bool>,
}

impl Config<PoolingConfig> for PoolingConfig {}

#[derive(Debug)]
pub struct Pooling {
    pooling_mode_cls_token: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
    pooling_output_dimension: i16,
}

impl Pooling {
    pub fn new(_p: &nn::Path, config: &PoolingConfig) -> Pooling {
        let pooling_mode_cls_token = if let Some(value) = config.pooling_mode_cls_token {
            value
        } else {
            false
        };
        let pooling_mode_max_tokens = if let Some(value) = config.pooling_mode_max_tokens {
            value
        } else {
            false
        };
        let pooling_mode_mean_tokens = if let Some(value) = config.pooling_mode_mean_tokens {
            value
        } else {
            // Forcing MEAN to be the default
            true
        };
        let pooling_mode_mean_sqrt_len_tokens =
            if let Some(value) = config.pooling_mode_mean_sqrt_len_tokens {
                value
            } else {
                false
            };

        let pooling_mode_multiplier: i16 = (pooling_mode_cls_token
            || pooling_mode_max_tokens
            || pooling_mode_mean_tokens
            || pooling_mode_mean_sqrt_len_tokens) as i16;
        let pooling_output_dimension = pooling_mode_multiplier * config.word_embedding_dimension;

        Pooling {
            pooling_mode_cls_token,
            pooling_mode_max_tokens,
            pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens,
            pooling_output_dimension,
        }
    }

    pub fn forward_t(&self, features: Features) -> Features {
        let mut output_vectors: Vec<Tensor> = Vec::new();

        if self.pooling_mode_cls_token {
            let cls = features
                .cls_token_embeddings
                .as_ref()
                .unwrap()
                .shallow_clone();
            output_vectors.push(cls);
        }

        if self.pooling_mode_max_tokens {
            let mask = features
                .input_mask
                .as_ref()
                .unwrap()
                .unsqueeze(-1)
                .expand(&features.token_embeddings.as_ref().unwrap().size(), false)
                .to_kind(Kind::Float);

            let min_value = Tensor::of_slice(&[f32::MIN]);
            let token_embeddings = features
                .token_embeddings
                .as_ref()
                .unwrap()
                .where1(&mask.ne(0), &min_value);

            let max_over_time = token_embeddings.max_dim(1, false).0;
            output_vectors.push(max_over_time);
        }

        // 3) MEAN / MEAN_SQRT_LEN pooling
        if self.pooling_mode_mean_tokens || self.pooling_mode_mean_sqrt_len_tokens {
            let mask = features
                .input_mask
                .as_ref()
                .unwrap()
                .unsqueeze(-1)
                .expand(&features.token_embeddings.as_ref().unwrap().size(), false)
                .to_kind(Kind::Float);

            let token_emb = features.token_embeddings.as_ref().unwrap();
            let sum_embeddings = (token_emb * &mask).sum_dim_intlist(&[1], false, Kind::Float);

            let sum_mask = match &features.token_weights_sum {
                Some(w) => w
                    .shallow_clone()
                    .unsqueeze(-1)
                    .expand(&sum_embeddings.size(), false),
                None => mask.sum_dim_intlist(&[1], false, Kind::Float),
            }
            .clamp_min(1e-9);

            if self.pooling_mode_mean_tokens {
                output_vectors.push(&sum_embeddings / &sum_mask);
            }
            if self.pooling_mode_mean_sqrt_len_tokens {
                output_vectors.push(&sum_embeddings / &sum_mask.sqrt());
            }
        }

        let output_vector = Tensor::cat(&output_vectors, 1);

        Features {
            sentence_embedding: Some(output_vector),
            ..features
        }
    }
}
