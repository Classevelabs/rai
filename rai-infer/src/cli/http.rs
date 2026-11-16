//! The loopback HTTP/1.1 server under `rai serve`.
//!
//! Deliberately small: one request per connection, a body cap enforced before
//! a body byte is read, a deadline on every socket, and a bounded number of
//! connections being read at once. Requests are read and parsed on short-lived
//! reader threads and handed, complete, to the caller's single-threaded loop,
//! so a client that dribbles its request out cannot stall anyone else.
//!
//! It replaced `tiny_http` 0.12, the crate's last release (2022). That crate
//! read nothing it was not asked to and then, when a request was dropped with
//! its body unread, discarded the remainder by allocating the whole declared
//! `Content-Length` in one buffer: a request declaring 10^18 bytes aborted the
//! process before sending one, and since it set no socket timeouts a client
//! that declared a body and never sent it held the server forever.
//!
//! Parsing is `httparse`, the parser under `hyper`.

use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Largest request head (request line and headers) accepted, in bytes.
const MAX_HEAD_BYTES: usize = 16 * 1024;
/// Most header fields one request may carry.
const MAX_HEADERS: usize = 64;
/// Connections read concurrently; past this a new one is answered `503`.
const MAX_READING: usize = 32;
/// Requests parsed and waiting for the handler.
const QUEUE_DEPTH: usize = 64;
/// How long a client has to deliver its whole request.
const READ_TIMEOUT: Duration = Duration::from_secs(15);
/// How long one write may block before the client is taken to be gone.
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);
/// After a response to a request whose body was refused unread, how long and
/// how much of it is drained so closing does not reset the connection before
/// the client has read the refusal.
const LINGER: Duration = Duration::from_millis(500);
const LINGER_BYTES: usize = 256 * 1024;

/// The request methods the server distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Head,
    Post,
    Options,
    Other,
}

/// What became of the request body.
#[derive(Debug)]
pub enum Body {
    /// Read in full; at most the server's body limit.
    Complete(Vec<u8>),
    /// Declared larger than the limit, so never read.
    TooLarge,
}

/// One parsed request and the connection it arrived on.
pub struct Request {
    method: Method,
    url: String,
    minor_version: u8,
    headers: Vec<(String, String)>,
    body: Body,
    stream: TcpStream,
}

impl Request {
    pub fn method(&self) -> Method {
        self.method
    }

    /// The request target exactly as sent: path and query.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// True for HTTP/1.1, false for HTTP/1.0.
    pub fn is_http11(&self) -> bool {
        self.minor_version >= 1
    }

    /// The first header with this name, compared case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(field, _)| field.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub fn body(&self) -> &Body {
        &self.body
    }

    /// Send a complete response and close the connection.
    pub fn respond(mut self, response: Response) -> io::Result<()> {
        let head_only = self.method == Method::Head;
        let mut head = status_line(response.status);
        for (name, value) in &response.headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str(&format!(
            "Content-Length: {}\r\nConnection: close\r\n\r\n",
            response.body.len()
        ));
        let written = self
            .stream
            .write_all(head.as_bytes())
            .and_then(|()| {
                if head_only {
                    Ok(())
                } else {
                    self.stream.write_all(&response.body)
                }
            })
            .and_then(|()| self.stream.flush());
        self.close();
        ignore_client_gone(written)
    }

    /// Start a chunked response and hand back the writer for its body.
    pub fn into_chunked(mut self, headers: &[(&str, &str)]) -> io::Result<ChunkedWriter> {
        let mut head = status_line(200);
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
        self.stream.write_all(head.as_bytes())?;
        self.stream.flush()?;
        Ok(ChunkedWriter {
            stream: self.stream,
        })
    }

    /// Close after responding. If the body was refused unread, part of it may
    /// still be in flight; closing on unread data makes the operating system
    /// reset the connection, which can destroy the response before the client
    /// reads it, so a bounded amount is drained first.
    fn close(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Write);
        if matches!(self.body, Body::TooLarge) {
            let _ = self.stream.set_read_timeout(Some(LINGER));
            let mut sink = [0u8; 8192];
            let mut drained = 0usize;
            while drained < LINGER_BYTES {
                match self.stream.read(&mut sink) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => drained += read,
                }
            }
        }
    }
}

/// A response with a complete body.
pub struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body,
        }
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

/// The body of a chunked response, one chunk per [`Self::send`].
pub struct ChunkedWriter {
    stream: TcpStream,
}

impl ChunkedWriter {
    pub fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            // An empty chunk would terminate the body.
            return Ok(());
        }
        write!(self.stream, "{:x}\r\n", bytes.len())?;
        self.stream.write_all(bytes)?;
        self.stream.write_all(b"\r\n")?;
        self.stream.flush()
    }

    pub fn finish(mut self) -> io::Result<()> {
        self.stream.write_all(b"0\r\n\r\n")?;
        self.stream.flush()?;
        let _ = self.stream.shutdown(Shutdown::Write);
        Ok(())
    }
}

/// A listening loopback socket and the requests parsed off it.
pub struct HttpServer {
    local: SocketAddr,
    requests: Mutex<Receiver<Request>>,
    stopping: Arc<AtomicBool>,
}

impl HttpServer {
    /// Listen on `127.0.0.1:port`, or on `[::1]:port` when this machine has no
    /// IPv4 loopback address.
    ///
    /// Falling back on *any* IPv4 error would let a second server bind `[::1]`
    /// while the first holds `127.0.0.1` on the same port, and `localhost`
    /// would then reach whichever one the client happens to try first. A port
    /// in use is an error; only a missing address family moves to IPv6.
    pub fn bind_loopback(port: u16, max_body: usize) -> io::Result<Self> {
        let listener = match TcpListener::bind((Ipv4Addr::LOCALHOST, port)) {
            Ok(listener) => listener,
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => return Err(error),
            Err(v4_error) => {
                TcpListener::bind((Ipv6Addr::LOCALHOST, port)).map_err(|v6_error| {
                    io::Error::new(
                        v6_error.kind(),
                        format!(
                            "cannot listen on 127.0.0.1:{port} ({v4_error}) or [::1]:{port} \
                         ({v6_error})"
                        ),
                    )
                })?
            }
        };
        let local = listener.local_addr()?;
        let (sender, receiver) = sync_channel(QUEUE_DEPTH);
        let stopping = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stopping);
        thread::Builder::new()
            .name("rai-http-accept".into())
            .spawn(move || accept_loop(listener, sender, flag, max_body))?;
        Ok(Self {
            local,
            requests: Mutex::new(receiver),
            stopping,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// The parsed requests, in arrival order; blocks until the next one, and
    /// ends after [`Self::unblock`].
    pub fn incoming_requests(&self) -> impl Iterator<Item = Request> + '_ {
        std::iter::from_fn(move || {
            let receiver = self
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            receiver.recv().ok()
        })
    }

    /// Stop accepting; [`Self::incoming_requests`] ends once the requests
    /// already being read have been delivered or have timed out.
    pub fn unblock(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        // `accept` has no timeout; a connection of our own wakes it.
        let _ = TcpStream::connect_timeout(&self.local, Duration::from_secs(1));
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        self.unblock();
    }
}

fn accept_loop(
    listener: TcpListener,
    sender: SyncSender<Request>,
    stopping: Arc<AtomicBool>,
    max_body: usize,
) {
    let reading = Arc::new(AtomicUsize::new(0));
    for connection in listener.incoming() {
        if stopping.load(Ordering::SeqCst) {
            break;
        }
        let mut stream = match connection {
            Ok(stream) => stream,
            Err(_) => {
                // Out of descriptors and the like: back off instead of spinning
                // on an error that will repeat until something closes.
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        if reading.fetch_add(1, Ordering::SeqCst) >= MAX_READING {
            reading.fetch_sub(1, Ordering::SeqCst);
            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                  Connection: close\r\n\r\n",
            );
            continue;
        }
        let sender = sender.clone();
        let slot = Arc::clone(&reading);
        let spawned = thread::Builder::new()
            .name("rai-http-read".into())
            .spawn(move || {
                if let Some(request) = read_request(stream, max_body) {
                    let _ = sender.send(request);
                }
                slot.fetch_sub(1, Ordering::SeqCst);
            });
        if spawned.is_err() {
            // The thread never ran, so it never gave its slot back.
            reading.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Read and parse one request, answering protocol errors itself.
fn read_request(mut stream: TcpStream, max_body: usize) -> Option<Request> {
    stream.set_read_timeout(Some(READ_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT)).ok()?;
    let _ = stream.set_nodelay(true);

    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let (head_len, method, url, minor_version, headers) = loop {
        let mut fields = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut parsed = httparse::Request::new(&mut fields);
        match parsed.parse(&buffer) {
            Ok(httparse::Status::Complete(head_len)) => {
                let method = match parsed.method.unwrap_or_default() {
                    "GET" => Method::Get,
                    "HEAD" => Method::Head,
                    "POST" => Method::Post,
                    "OPTIONS" => Method::Options,
                    _ => Method::Other,
                };
                let url = parsed.path.unwrap_or("/").to_string();
                let minor_version = parsed.version.unwrap_or(1);
                let mut headers = Vec::with_capacity(parsed.headers.len());
                for field in parsed.headers.iter() {
                    let Ok(value) = std::str::from_utf8(field.value) else {
                        reject(&mut stream, 400, "header value is not UTF-8");
                        return None;
                    };
                    headers.push((field.name.to_string(), value.trim().to_string()));
                }
                break (head_len, method, url, minor_version, headers);
            }
            Ok(httparse::Status::Partial) => {}
            Err(httparse::Error::TooManyHeaders) => {
                reject(&mut stream, 431, "too many header fields");
                return None;
            }
            Err(_) => {
                reject(&mut stream, 400, "malformed request");
                return None;
            }
        }
        if buffer.len() >= MAX_HEAD_BYTES {
            reject(&mut stream, 431, "request head is too large");
            return None;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(read) => buffer.extend_from_slice(&chunk[..read]),
        }
    };

    let find = |name: &str| {
        headers
            .iter()
            .filter(|(field, _)| field.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>()
    };
    if !find("Transfer-Encoding").is_empty() {
        // Every client this server has sends a Content-Length; a chunked body
        // is refused rather than decoded, which keeps the limit a single
        // number checked before reading.
        reject(
            &mut stream,
            411,
            "send the request body with a Content-Length",
        );
        return None;
    }
    let declared = match find("Content-Length").as_slice() {
        [] => 0,
        lengths => {
            let mut values = lengths.iter().map(|value| value.parse::<u64>());
            let first = values.next();
            match first {
                Some(Ok(length)) if values.all(|other| other == Ok(length)) => length,
                _ => {
                    reject(&mut stream, 400, "invalid Content-Length");
                    return None;
                }
            }
        }
    };

    let body = if declared > max_body as u64 {
        Body::TooLarge
    } else {
        if find("Expect")
            .iter()
            .any(|value| value.eq_ignore_ascii_case("100-continue"))
            && minor_version >= 1
        {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").ok()?;
        }
        let declared = declared as usize;
        let mut body = buffer.split_off(head_len);
        body.truncate(declared);
        while body.len() < declared {
            let want = (declared - body.len()).min(chunk.len());
            match stream.read(&mut chunk[..want]) {
                Ok(0) | Err(_) => return None,
                Ok(read) => body.extend_from_slice(&chunk[..read]),
            }
        }
        Body::Complete(body)
    };

    Some(Request {
        method,
        url,
        minor_version,
        headers,
        body,
        stream,
    })
}

/// Answer a request the server could not parse, and close.
fn reject(stream: &mut TcpStream, status: u16, message: &str) {
    let body = format!("{message}\n");
    let response = format!(
        "{}Content-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        status_line(status),
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Write);
}

fn status_line(status: u16) -> String {
    let reason = match status {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "",
    };
    format!("HTTP/1.1 {status} {reason}\r\n")
}

/// A client that hung up before reading its response is not the server's
/// failure.
fn ignore_client_gone(result: io::Result<()>) -> io::Result<()> {
    result.or_else(|error| match error.kind() {
        io::ErrorKind::BrokenPipe
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::ConnectionReset => Ok(()),
        _ => Err(error),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange(server: &HttpServer, raw: &[u8]) -> String {
        let mut socket = TcpStream::connect(server.local_addr()).expect("connect");
        socket.write_all(raw).expect("write");
        let mut response = String::new();
        let _ = socket.read_to_string(&mut response);
        response
    }

    /// Serve every request on a thread with a fixed answer, recording what
    /// arrived.
    fn echo_server(max_body: usize) -> (Arc<HttpServer>, thread::JoinHandle<Vec<(String, usize)>>) {
        let server = Arc::new(HttpServer::bind_loopback(0, max_body).expect("bind"));
        let handle = {
            let server = Arc::clone(&server);
            thread::spawn(move || {
                let mut seen = Vec::new();
                for request in server.incoming_requests() {
                    let length = match request.body() {
                        Body::Complete(bytes) => bytes.len(),
                        Body::TooLarge => usize::MAX,
                    };
                    seen.push((request.url().to_string(), length));
                    let status = if length == usize::MAX { 413 } else { 200 };
                    let _ = request.respond(Response::new(status, b"ok".to_vec()));
                }
                seen
            })
        };
        (server, handle)
    }

    #[test]
    fn a_huge_declared_body_is_refused_without_reading_or_allocating_it() {
        let (server, handle) = echo_server(1024);
        // The declaration that used to abort the process: nothing follows it.
        let response = exchange(
            &server,
            b"POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: 1000000000000000000\r\n\r\n",
        );
        assert!(response.starts_with("HTTP/1.1 413"), "{response}");
        server.unblock();
        let seen = handle.join().unwrap();
        assert_eq!(seen, vec![("/x".to_string(), usize::MAX)]);
    }

    #[test]
    fn a_body_within_the_limit_arrives_whole_even_when_split() {
        let (server, handle) = echo_server(1024);
        let mut socket = TcpStream::connect(server.local_addr()).expect("connect");
        socket
            .write_all(b"POST /y HTTP/1.1\r\nHost: a\r\nContent-Length: 10\r\n\r\n01234")
            .unwrap();
        socket.flush().unwrap();
        thread::sleep(Duration::from_millis(50));
        socket.write_all(b"56789").unwrap();
        let mut response = String::new();
        let _ = socket.read_to_string(&mut response);
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("Connection: close"), "{response}");
        server.unblock();
        assert_eq!(handle.join().unwrap(), vec![("/y".to_string(), 10)]);
    }

    #[test]
    fn chunked_request_bodies_and_conflicting_lengths_are_refused() {
        let (server, handle) = echo_server(1024);
        let chunked = exchange(
            &server,
            b"POST /z HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
        );
        assert!(chunked.starts_with("HTTP/1.1 411"), "{chunked}");
        let conflicting = exchange(
            &server,
            b"POST /z HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nabc",
        );
        assert!(conflicting.starts_with("HTTP/1.1 400"), "{conflicting}");
        let garbage = exchange(&server, b"\x00\x01 not http\r\n\r\n");
        assert!(garbage.starts_with("HTTP/1.1 400"), "{garbage}");
        server.unblock();
        assert!(handle.join().unwrap().is_empty());
    }

    #[test]
    fn an_oversized_head_is_refused() {
        let (server, handle) = echo_server(1024);
        let mut raw = b"GET / HTTP/1.1\r\nHost: a\r\nX-Pad: ".to_vec();
        raw.extend(std::iter::repeat_n(b'a', MAX_HEAD_BYTES + 10));
        raw.extend_from_slice(b"\r\n\r\n");
        let response = exchange(&server, &raw);
        assert!(response.starts_with("HTTP/1.1 431"), "{response}");
        server.unblock();
        assert!(handle.join().unwrap().is_empty());
    }

    #[test]
    fn a_silent_client_does_not_hold_up_the_next_one() {
        let (server, handle) = echo_server(1024);
        // Declares a body and never sends it.
        let mut stalled = TcpStream::connect(server.local_addr()).expect("connect");
        stalled
            .write_all(b"POST /stalled HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n")
            .unwrap();
        let response = exchange(&server, b"GET /next HTTP/1.1\r\nHost: a\r\n\r\n");
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        drop(stalled);
        server.unblock();
        let seen = handle.join().unwrap();
        assert_eq!(seen, vec![("/next".to_string(), 0)]);
    }

    #[test]
    fn a_port_in_use_is_an_error_not_an_ipv6_fallback() {
        let first = HttpServer::bind_loopback(0, 16).expect("bind");
        let port = first.local_addr().port();
        let error = HttpServer::bind_loopback(port, 16)
            .err()
            .expect("the second bind must fail");
        assert_eq!(error.kind(), io::ErrorKind::AddrInUse, "{error}");
    }

    #[test]
    fn chunked_responses_are_framed() {
        let server = Arc::new(HttpServer::bind_loopback(0, 16).expect("bind"));
        let handle = {
            let server = Arc::clone(&server);
            thread::spawn(move || {
                if let Some(request) = server.incoming_requests().next() {
                    let mut writer = request
                        .into_chunked(&[("Content-Type", "application/x-ndjson")])
                        .unwrap();
                    writer.send(b"{\"a\":1}\n").unwrap();
                    writer.send(b"").unwrap();
                    writer.finish().unwrap();
                }
            })
        };
        let response = exchange(
            &server,
            b"POST /s HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n\r\n",
        );
        assert!(
            response.contains("Transfer-Encoding: chunked"),
            "{response}"
        );
        assert!(
            response.ends_with("8\r\n{\"a\":1}\n\r\n0\r\n\r\n"),
            "{response:?}"
        );
        handle.join().unwrap();
    }
}
