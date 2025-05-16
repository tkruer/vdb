use serde_json::from_str;
use std::error::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::TcpStream;
use tracing::info;

/// A simple async client for vdb (GET/SET/QUIT text protocol).
pub struct Client {
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
    writer: BufWriter<tokio::net::tcp::OwnedWriteHalf>,
}

impl Client {
    /// Connects to the vdb server at the given address (e.g. "127.0.0.1:6379").
    /// Reads the initial greeting and returns a usable Client.
    pub async fn connect(addr: &str) -> Result<Self, Box<dyn Error>> {
        let stream = TcpStream::connect(addr).await?;
        let (read_half, write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Read and ignore the server greeting
        let mut greeting = String::new();
        reader.read_line(&mut greeting).await?;

        let writer = BufWriter::new(write_half);
        Ok(Client { reader, writer })
    }

    /// Sends `SET key value` to the server. Returns Ok if server replies +OK.
    pub async fn set(&mut self, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
        // Format and send
        let cmd = format!("SET {} {}\r\n", key, value);
        self.writer.write_all(cmd.as_bytes()).await?;
        self.writer.flush().await?;

        // Read response
        let mut resp = String::new();
        self.reader.read_line(&mut resp).await?;
        if resp.starts_with("+OK") {
            Ok(())
        } else {
            Err(format!("SET error: {}", resp.trim_end()).into())
        }
    }

    /// Sends `GET key` to the server. Returns Ok(Some(value)) or Ok(None) if nil.
    pub async fn get(&mut self, key: &str) -> Result<Option<String>, Box<dyn Error>> {
        let cmd = format!("GET {}\r\n", key);
        self.writer.write_all(cmd.as_bytes()).await?;
        self.writer.flush().await?;

        let mut resp = String::new();
        self.reader.read_line(&mut resp).await?;
        // $-1 => nil
        if resp.starts_with("$-1") {
            Ok(None)
        } else if resp.starts_with('+') {
            // remove the leading '+' and trailing CRLF
            let val = resp.trim_start_matches('+').trim_end().to_string();
            Ok(Some(val))
        } else {
            Err(format!("GET error: {}", resp.trim_end()).into())
        }
    }

    pub async fn search(&mut self, query: &str) -> Result<Option<Vec<String>>, Box<dyn Error>> {
        let cmd = format!("SEARCH {}\r\n", query);
        self.writer.write_all(cmd.as_bytes()).await?;
        self.writer.flush().await?;

        let mut resp = String::new();
        self.reader.read_line(&mut resp).await?;

        if resp.starts_with("$-1") {
            info!("SEARCH error: {}", resp.trim_end());
            Ok(None)
        } else if resp.starts_with('+') {
            let json = resp.trim_start_matches('+').trim_end();
            let results: Vec<String> = from_str(json)?;
            info!("SEARCH results: {}", json);
            Ok(Some(results))
        } else {
            Err(format!("SEARCH error: {}", resp.trim_end()).into())
        }
    }

    /// Sends `QUIT` and closes the connection.
    pub async fn quit(&mut self) -> Result<(), Box<dyn Error>> {
        self.writer.write_all(b"QUIT\r\n").await?;
        self.writer.flush().await?;
        Ok(())
    }
}
