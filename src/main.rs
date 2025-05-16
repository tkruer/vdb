pub use vdb;

#[tokio::main]
async fn main() {
    vdb::database::db::new().await.unwrap();
}
