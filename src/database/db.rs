use serde_json::to_string;
use std::{collections::HashMap, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};
use tracing::{error, info};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub embedding: Vec<f64>,
    pub text: String,
    pub score: f64,
    pub metadata: Vec<String>,
}

pub type Database = Arc<Mutex<HashMap<String, String>>>;

pub async fn new() -> std::io::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = FmtSubscriber::builder().with_env_filter(filter).finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");

    info!("starting vdb on 127.0.0.1:6379");

    let db = Arc::new(Mutex::new(HashMap::new()));
    let listener = TcpListener::bind("127.0.0.1:6379").await?;
    loop {
        let (socket, addr) = listener.accept().await?;
        info!(%addr, "accepted connection");
        let db = db.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_connection(socket, db).await {
                error!(error = %err, "connection error");
            }
        });
    }
}

async fn handle_connection(stream: TcpStream, db: Database) -> std::io::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    writer.write_all(b"+OK vdb 0.1\r\n").await?;

    while let Some(line) = lines.next_line().await? {
        info!("received command: {}", line);
        let mut parts = line.splitn(3, ' ');
        match parts.next() {
            Some(cmd) if cmd.eq_ignore_ascii_case("GET") => {
                if let Some(key) = parts.next() {
                    let map = db.lock().await;
                    if let Some(val) = map.get(key) {
                        writer.write_all(format!("+{}\r\n", val).as_bytes()).await?;
                    } else {
                        writer.write_all(b"$-1\r\n").await?;
                    }
                } else {
                    writer.write_all(b"-ERR missing key\r\n").await?;
                }
            }

            Some(cmd) if cmd.eq_ignore_ascii_case("SET") => {
                if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                    let mut map = db.lock().await;
                    map.insert(key.to_string(), value.to_string());
                    writer.write_all(b"+OK\r\n").await?;
                } else {
                    writer.write_all(b"-ERR missing key or value\r\n").await?;
                }
            }

            Some(cmd) if cmd.eq_ignore_ascii_case("SEARCH") => {
                if let Some(query) = parts.next() {
                    let map = db.lock().await;
                    let results: Vec<String> = map
                        .iter()
                        .filter(|(k, v)| k.contains(query) || v.contains(query))
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect();

                    if results.is_empty() {
                        writer.write_all(b"$-1\r\n").await?;
                    } else {
                        let json = to_string(&results).unwrap();
                        writer
                            .write_all(format!("+{}\r\n", json).as_bytes())
                            .await?;
                    }
                } else {
                    writer.write_all(b"-ERR missing query\r\n").await?;
                }
            }
            Some(cmd) if cmd.eq_ignore_ascii_case("QUIT") => {
                writer.write_all(b"+BYE\r\n").await?;
                break;
            }

            _ => {
                writer.write_all(b"-ERR unknown command\r\n").await?;
            }
        }
    }

    Ok(())
}
