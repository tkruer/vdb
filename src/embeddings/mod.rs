use rust_bert::Config;
use std::path::Path;
use tch::{Device, Kind, Tensor, no_grad};

pub mod bert;
pub mod pooling;

use bert::{Bert, Features};
use pooling::{Pooling, PoolingConfig};

pub struct SentenceTransformer {
    pub bert: Bert,
    pub pooling: Pooling,
}

impl SentenceTransformer {
    pub fn new(model_path: &Path, device: Device) -> Result<SentenceTransformer, tch::TchError> {
        let bert_model_path = model_path.join("0_BERT");
        let pooling_config_path = model_path.join("1_Pooling/config.json");

        let bert = Bert::new(&bert_model_path, None, None, device);
        let pooling = Pooling::new(
            &(&bert.vs.root() / "pooling"),
            &PoolingConfig::from_file(&pooling_config_path),
        );

        Ok(SentenceTransformer { bert, pooling })
    }

    pub fn encode(&self, text: &str) -> Vec<f64> {
        // 1) tokenize & build batched feature tensors
        let tokens = self.bert.tokenize(text);
        let max_len = tokens.len();
        let (input_ids, token_type_ids, input_mask, _len) =
            self.bert.get_sentence_features(&tokens, max_len);

        let mut ids = Vec::with_capacity(1);
        let mut types = Vec::with_capacity(1);
        let mut masks = Vec::with_capacity(1);

        ids.push(Tensor::from_slice(&input_ids));
        types.push(Tensor::from_slice(&token_type_ids));
        masks.push(Tensor::from_slice(&input_mask));

        let device = self.bert.vs.device();
        let mut features = Features::default();
        features.input_ids = Some(Tensor::stack(&ids, 0).to(device));
        features.token_type_ids = Some(Tensor::stack(&types, 0).to(device));
        features.input_mask = Some(Tensor::stack(&masks, 0).to(device));

        // 2) forward passes (no_grad)
        let features = no_grad(|| self.bert.forward_t(features));
        let features = no_grad(|| self.pooling.forward_t(features));

        // 3) extract the final sentence_embedding, turn it into Vec<f64>
        let sent = features
            .sentence_embedding
            .unwrap() // take ownership of the Tensor
            .to_kind(Kind::Double); // convert from f32→f64
        Vec::<f64>::from(sent) // now From<Tensor> for Vec<f64> is available
    }
}

pub fn generate_emdedding(text: &str) -> Vec<f64> {
    let device = Device::cuda_if_available();
    // unwrap here (or propagate error if you prefer a Result return type)
    let svc =
        SentenceTransformer::new(Path::new("models/bert-base-nli-mean-tokens"), device).unwrap();
    svc.encode(text)
}
