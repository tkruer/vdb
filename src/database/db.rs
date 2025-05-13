use crate::database::cosine::CosineDatabase;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub embedding: Vec<f64>,
    pub score: f64,
    pub metadata: Vec<String>,
}

pub enum Database {
    CosineDatabase(CosineDatabase),
}

pub fn new(database_method: &str) -> Database {
    match database_method {
        "cosine" => Database::CosineDatabase(CosineDatabase { documents: vec![] }),
        _ => panic!("Unsupported database method"),
    }
}

pub trait DatabaseOperations {
    fn insert(&self, document: Document) -> Result<(), String>;
    fn update(&self, document: Document) -> Result<(), String>;
    fn delete(&self, id: &str) -> Result<(), String>;
    fn search(&self, query: &str) -> Result<Vec<Document>, String>;
    fn get(&self, id: &str) -> Result<Document, String>;
    fn list(&self) -> Result<Vec<Document>, String>;
    fn count(&self) -> Result<usize, String>;
    fn clear(&self) -> Result<(), String>;
    fn close(&self) -> Result<(), String>;
    fn get_metadata(&self, id: &str) -> Result<Vec<String>, String>;
    fn load(&mut self, texts: &Vec<String>);
    fn query(&self, query: String, n: u32) -> Vec<Document>;
}

impl DatabaseOperations for Database {
    fn insert(&self, document: Document) -> Result<(), String> {
        // Implementation here
        Ok(())
    }
    fn update(&self, document: Document) -> Result<(), String> {
        // Implementation here
        Ok(())
    }
    fn delete(&self, id: &str) -> Result<(), String> {
        // Implementation here
        Ok(())
    }
    fn search(&self, query: &str) -> Result<Vec<Document>, String> {
        // Implementation here
        Ok(vec![])
    }
    fn get(&self, id: &str) -> Result<Document, String> {
        // Implementation here
        Ok(Document {
            id: id.to_string(),
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

    fn load(&mut self, texts: &Vec<String>) {
        match self {
            Database::CosineDatabase(db) => db.load(texts),
            _ => (),
        }
    }

    fn query(&self, query: String, n: u32) -> Vec<Document> {
        match self {
            Database::CosineDatabase(db) => db.query(query, n),
            _ => vec![],
        }
    }
}
