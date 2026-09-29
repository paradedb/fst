//! Random-access FST routing without materializing the complete encoded graph.

use super::{CompiledAddr, Node, Output, EMPTY_ADDRESS, VERSION};
use std::io;
use std::ops::{Deref, Range};

const HEADER_LEN: usize = 16;
const FOOTER_LEN: usize = 16;
const NODE_HEADER_LEN: usize = 3;

/// Supplies byte windows; implementations should cache storage pages across reads.
pub trait ReadBytes {
    /// Owns or pins the returned window.
    type Bytes: Deref<Target = [u8]>;
    /// Total length of the encoded FST.
    fn num_bytes(&self) -> usize;
    /// Reads exactly the requested range.
    fn read_bytes(&self, range: Range<usize>) -> io::Result<Self::Bytes>;
}

/// Reads FST routing paths without retaining the whole graph.
#[derive(Debug)]
pub struct PagedFst<R> {
    reader: R,
    version: u64,
    root: CompiledAddr,
    len: u64,
}

impl<R: ReadBytes> PagedFst<R> {
    /// Reads only the fixed header and footer.
    pub fn new(reader: R) -> io::Result<Self> {
        let size = reader.num_bytes();
        if size < HEADER_LEN + FOOTER_LEN {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated FST"));
        }
        let header = reader.read_bytes(0..HEADER_LEN)?;
        let footer = reader.read_bytes(size - FOOTER_LEN..size)?;
        if header.len() != HEADER_LEN || footer.len() != FOOTER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete FST header or footer",
            ));
        }
        let version = u64::from_le_bytes(header[..8].try_into().unwrap());
        let len = u64::from_le_bytes(footer[..8].try_into().unwrap());
        let root = usize::try_from(u64::from_le_bytes(footer[8..].try_into().unwrap()))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid FST root"))?;
        if version == 0
            || version > VERSION
            || (root == EMPTY_ADDRESS && size != HEADER_LEN + FOOTER_LEN)
            || (root != EMPTY_ADDRESS && root.checked_add(FOOTER_LEN + 1) != Some(size))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid FST header",
            ));
        }
        Ok(Self {
            reader,
            version,
            root,
            len,
        })
    }

    fn with_node<T>(&self, addr: CompiledAddr, visit: impl FnOnce(Node<'_>) -> T) -> io::Result<T> {
        if addr == EMPTY_ADDRESS {
            return Ok(visit(Node::from_bytes(self.version, addr, &[])));
        }
        if addr < HEADER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid FST node address",
            ));
        }
        let end = addr
            .checked_add(1)
            .filter(|end| *end <= self.reader.num_bytes())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid FST node address")
            })?;
        let tail = self
            .reader
            .read_bytes(end.saturating_sub(NODE_HEADER_LEN)..end)?;
        let len = Node::encoded_len(self.version, &tail)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid FST node header"))?;
        let start = end
            .checked_sub(len)
            .filter(|&start| start >= HEADER_LEN)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid FST node size"))?;
        if len <= tail.len() {
            return Ok(visit(Node::from_bytes(
                self.version,
                addr,
                &tail[tail.len() - len..],
            )));
        }
        let bytes = self.reader.read_bytes(start..end)?;
        Ok(visit(Node::from_bytes(self.version, addr, &bytes)))
    }

    /// Returns the output of the lexicographically first key greater than or equal to `key`.
    pub fn lower_bound(&self, key: &[u8]) -> io::Result<Option<Output>> {
        if self.len == 0 {
            return Ok(None);
        }
        let mut addr = self.root;
        let mut output = Output::zero();
        let mut successor = None;
        for &byte in key {
            let (equal, greater) = self.with_node(addr, |node| {
                let equal = node.find_input(byte);
                let greater = match equal {
                    Some(i) if i + 1 < node.len() => Some(i + 1),
                    Some(_) => None,
                    None => node.transitions().position(|t| t.inp > byte),
                };
                (
                    equal.map(|i| node.transition(i)),
                    greater.map(|i| node.transition(i)),
                )
            })?;
            if let Some(next) = greater {
                successor = Some((next.addr, output.cat(next.out)));
            }
            if let Some(next) = equal {
                addr = next.addr;
                output = output.cat(next.out);
            } else {
                return match successor {
                    Some((addr, output)) => self.first_output(addr, output).map(Some),
                    None => Ok(None),
                };
            }
        }
        self.first_output(addr, output).map(Some)
    }

    fn first_output(&self, mut addr: CompiledAddr, mut output: Output) -> io::Result<Output> {
        loop {
            let next = self.with_node(addr, |node| {
                if node.is_final() {
                    Ok(node.final_output())
                } else {
                    Err(node.transition(0))
                }
            })?;
            match next {
                Ok(final_output) => return Ok(output.cat(final_output)),
                Err(next) => {
                    addr = next.addr;
                    output = output.cat(next.out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntoStreamer, Map, MapBuilder, Streamer};
    use proptest::prelude::*;
    use std::cell::RefCell;
    use std::sync::Arc;

    #[derive(Clone)]
    struct Source {
        data: Arc<Vec<u8>>,
        reads: Arc<RefCell<Vec<Range<usize>>>>,
    }
    impl ReadBytes for Source {
        type Bytes = Vec<u8>;
        fn num_bytes(&self) -> usize {
            self.data.len()
        }
        fn read_bytes(&self, range: Range<usize>) -> io::Result<Vec<u8>> {
            self.reads.borrow_mut().push(range.clone());
            Ok(self.data[range].to_vec())
        }
    }
    fn check(mut keys: Vec<Vec<u8>>, probes: Vec<Vec<u8>>) {
        keys.sort();
        keys.dedup();
        let mut builder = MapBuilder::memory();
        for (i, key) in keys.iter().enumerate() {
            builder.insert(key, i as u64 * 1234567).unwrap();
        }
        let data = Arc::new(builder.into_inner().unwrap());
        let source = Source {
            data: data.clone(),
            reads: Arc::new(RefCell::new(Vec::new())),
        };
        let paged = PagedFst::new(source.clone()).unwrap();
        let eager = Map::from_bytes(data.as_ref().clone()).unwrap();
        for key in probes.iter().chain(&keys) {
            let expected = eager
                .range()
                .ge(key)
                .into_stream()
                .next()
                .map(|(_, value)| value);
            assert_eq!(
                paged.lower_bound(key).unwrap().map(|v| v.value()),
                expected,
                "key {key:?}"
            );
        }
        if data.len() > 64 {
            assert!(source.reads.borrow().iter().all(|r| r.len() < data.len()));
        }
    }
    #[test]
    fn empty_and_dense_nodes() {
        check(vec![], vec![vec![], vec![0], vec![255]]);
        check(vec![vec![]], vec![vec![], vec![0]]);
        check(
            (0..=255).map(|b| vec![b]).collect(),
            (0..=255).map(|b| vec![b, 0]).collect(),
        );
        check(
            vec![
                b"a".to_vec(),
                b"ab".to_vec(),
                b"abd".to_vec(),
                b"b".to_vec(),
            ],
            vec![b"abc".to_vec(), b"az".to_vec(), b"z".to_vec()],
        );
    }
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn agrees_with_eager(keys in prop::collection::vec(prop::collection::vec(any::<u8>(),0..40),0..300), probes in prop::collection::vec(prop::collection::vec(any::<u8>(),0..40),0..100)) {
            check(keys,probes);
        }
    }
    #[test]
    fn large_fst_only_reads_lookup_path() {
        let mut builder = MapBuilder::memory();
        for i in 0u64..50000 {
            builder
                .insert(
                    format!("term-{i:05}-{:016x}", i.wrapping_mul(0x9e3779b97f4a7c15)),
                    i,
                )
                .unwrap();
        }
        let source = Source {
            data: Arc::new(builder.into_inner().unwrap()),
            reads: Arc::new(RefCell::new(Vec::new())),
        };
        let paged = PagedFst::new(source.clone()).unwrap();
        assert_eq!(
            source.reads.borrow().iter().map(|r| r.len()).sum::<usize>(),
            32
        );
        assert_eq!(
            paged.lower_bound(b"term-25000").unwrap().unwrap().value(),
            25000
        );
        let bytes: usize = source.reads.borrow().iter().map(|r| r.len()).sum();
        assert!(bytes < source.num_bytes() / 100);
    }
}
