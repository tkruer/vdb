use crate::database::db::{DatabaseOperations, Document};
use crate::embeddings::generate_emdedding;

pub struct CosineDatabase {
    pub documents: Vec<Document>,
}

impl DatabaseOperations for CosineDatabase {
    fn load(&mut self, texts: &Vec<String>) {
        for text in texts {
            let embedding = generate_emdedding(text);
            let document = Document {
                id: text.clone(),
                text: text.clone(),
                embedding,
                score: 0.0,
                metadata: vec![],
            };
            self.documents.push(document);
        }
    }

    fn query(&self, query: String, n: u32) -> Vec<Document> {
        let mut result = vec![];
        let query_embedding = generate_emdedding(&query);
        for document in &self.documents {
            let score = cosine_similarity(&query_embedding, &document.embedding);
            let mut doc = document.clone();
            doc.score = score;
            result.push(doc);
        }
        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        result.drain(..n as usize).collect()
    }

    fn insert(&self, _document: Document) -> Result<(), String> {
        // Implementation here
        Ok(())
    }

    fn update(&self, _document: Document) -> Result<(), String> {
        // Implementation here
        Ok(())
    }
    fn delete(&self, _id: &str) -> Result<(), String> {
        // Implementation here
        Ok(())
    }

    fn search(&self, _query: &str) -> Result<Vec<Document>, String> {
        // Implementation here
        Ok(vec![])
    }
    fn get(&self, id: &str) -> Result<Document, String> {
        // Implementation here
        Ok(Document {
            id: id.to_string(),
            text: id.to_string(),
            embedding: vec![],
            score: 0.0,
            metadata: vec![],
        })
    }

    fn list(&self) -> Result<Vec<Document>, String> {
        // Implementation here
        Ok(vec![])
    }

    fn count(&self) -> Result<usize, String> {
        // Implementation here
        Ok(0)
    }

    fn clear(&self) -> Result<(), String> {
        // Implementation here
        Ok(())
    }

    fn close(&self) -> Result<(), String> {
        // Implementation here
        Ok(())
    }

    fn get_metadata(&self, _id: &str) -> Result<Vec<String>, String> {
        // Implementation here
        Ok(vec![])
    }
}

fn cosine_similarity(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let norms = norm(a) * norm(b);
    if norms == 0.0 {
        return 0.0;
    }
    let result = dot_product(a, b) / norms;
    if result.is_nan() {
        return 0.0;
    }
    result
}

fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |sum, (&a, &b)| sum + (a * b))
}

fn norm(a: &Vec<f64>) -> f64 {
    dot_product(a, a).sqrt()
}
