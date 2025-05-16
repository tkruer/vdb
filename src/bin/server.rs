use vdb::database::db;

#[tokio::main]
pub async fn main() {
    db::new().await.unwrap();
}
