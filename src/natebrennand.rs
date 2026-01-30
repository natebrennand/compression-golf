//! Arrow IPC codec with TSV mapping table for repo info.
//!
//! Strategy:
//! - Sort by event_id for optimal delta encoding
//! - Dictionary encode event_type (14 unique values)
//! - Store unique (repo_id, repo_name) pairs in a TSV mapping table (compressed once)
//! - Per event, store only an index into the mapping table
//! - Store first event_id in header, deltas as UInt8 (max delta is 251)
//! - Store first timestamp in header, deltas as Int16 (range -2540 to +2540)
//! - Reconstruct repo.url from repo.name during decode
//! - Use zstd level 22 compression
//!
//! Set NATE_DEBUG=1 to see column size statistics.

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Int16Array, RecordBatch, StringArray, UInt32Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use bytes::Bytes;
use chrono::DateTime;
use std::collections::HashMap;
use std::error::Error;
use std::io::Cursor;
use std::sync::Arc;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;

pub struct NatebrennandCodec;

impl NatebrennandCodec {
    pub fn new() -> Self {
        Self
    }
}

fn debug_enabled() -> bool {
    std::env::var("NATE_DEBUG").is_ok()
}

fn print_column_stats(batch: &RecordBatch, mapping_compressed_size: usize) {
    let num_rows = batch.num_rows();
    eprintln!("\n=== Per-Column Compressed Size Estimates ===");
    eprintln!("Total rows: {}", num_rows);
    eprintln!("{:<20} {:>10} {:>10} {:>8} {:>10}", "Column", "Raw", "Zstd", "Ratio", "B/Row");
    eprintln!("{}", "-".repeat(64));

    let mut total_raw = 0usize;
    let mut total_compressed = 0usize;

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let col = batch.column(i);
        let raw_size = col.get_array_memory_size();

        let single_schema = Arc::new(Schema::new(vec![field.clone()]));
        let single_batch = RecordBatch::try_new(
            single_schema.clone(),
            vec![col.clone()],
        ).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &single_schema).unwrap();
            writer.write(&single_batch).unwrap();
            writer.finish().unwrap();
        }
        let compressed = zstd::encode_all(buf.as_slice(), ZSTD_LEVEL).unwrap();
        let compressed_size = compressed.len();

        total_raw += raw_size;
        total_compressed += compressed_size;

        eprintln!(
            "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
            field.name(),
            raw_size,
            compressed_size,
            100.0 * compressed_size as f64 / raw_size as f64,
            compressed_size as f64 / num_rows as f64
        );
    }

    eprintln!(
        "{:<20} {:>10} {:>10} {:>7}  {:>10.2}",
        "repo mapping (TSV)",
        "-",
        mapping_compressed_size,
        "-",
        mapping_compressed_size as f64 / num_rows as f64
    );
    total_compressed += mapping_compressed_size;

    eprintln!("{}", "-".repeat(64));
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "TOTAL",
        total_raw,
        total_compressed,
        100.0 * total_compressed as f64 / total_raw as f64,
        total_compressed as f64 / num_rows as f64
    );
    eprintln!();
}

fn parse_timestamp(ts: &str) -> i64 {
    DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.timestamp())
        .unwrap_or(0)
}

fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

fn build_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            "event_type",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("event_id_delta", DataType::UInt8, false),
        Field::new("repo_pair_idx", DataType::UInt32, false),
        Field::new("created_at_delta", DataType::Int16, false),
    ])
}

/// Header: first_event_id (u64) + first_timestamp (i64) + mapping_size (u32)
struct Header {
    first_event_id: u64,
    first_timestamp: i64,
    mapping_size: u32,
}

const HEADER_SIZE: usize = 20; // 8 + 8 + 4

impl Header {
    fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.first_event_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.first_timestamp.to_le_bytes());
        buf[16..20].copy_from_slice(&self.mapping_size.to_le_bytes());
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let first_event_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let first_timestamp = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let mapping_size = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        Self { first_event_id, first_timestamp, mapping_size }
    }
}

/// Build TSV mapping table from events, returns (tsv_bytes, pair_to_idx mapping)
fn build_mapping_table(events: &[(EventKey, EventValue)]) -> (Vec<u8>, HashMap<(u32, String), u32>) {
    // Collect unique (repo_id, repo_name) pairs
    let mut unique_pairs: Vec<(u32, String)> = events
        .iter()
        .map(|(_, v)| (v.repo.id as u32, v.repo.name.clone()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Sort for deterministic ordering
    unique_pairs.sort();

    // Build TSV and index mapping
    let mut tsv = String::new();
    let mut pair_to_idx: HashMap<(u32, String), u32> = HashMap::new();

    for (idx, (repo_id, repo_name)) in unique_pairs.iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\n", repo_id, repo_name));
        pair_to_idx.insert((*repo_id, repo_name.clone()), idx as u32);
    }

    (tsv.into_bytes(), pair_to_idx)
}

/// Parse TSV mapping table, returns Vec of (repo_id, repo_name) pairs
fn parse_mapping_table(tsv_bytes: &[u8]) -> Vec<(u32, String)> {
    let tsv = std::str::from_utf8(tsv_bytes).unwrap();
    tsv.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut parts = line.split('\t');
            let repo_id: u32 = parts.next().unwrap().parse().unwrap();
            let repo_name = parts.next().unwrap().to_string();
            (repo_id, repo_name)
        })
        .collect()
}

fn compute_u8_deltas(values: &[u64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    deltas.push(0u8);
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        deltas.push(delta as u8);
    }
    deltas
}

fn compute_i16_deltas(values: &[i64]) -> Vec<i16> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    deltas.push(0i16);
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        deltas.push(delta as i16);
    }
    deltas
}

fn restore_u64_from_deltas(first: u64, deltas: &[u8]) -> Vec<u64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(deltas.len());
    values.push(first);
    for i in 1..deltas.len() {
        values.push(values[i - 1] + deltas[i] as u64);
    }
    values
}

fn restore_i64_from_deltas(first: i64, deltas: &[i16]) -> Vec<i64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(deltas.len());
    values.push(first);
    for i in 1..deltas.len() {
        values.push(values[i - 1] + deltas[i] as i64);
    }
    values
}

fn encode_events(events: &[(EventKey, EventValue)]) -> Result<(Header, Vec<u8>, RecordBatch), Box<dyn Error>> {
    let schema = Arc::new(build_schema());

    // Build mapping table first
    let (mapping_tsv, pair_to_idx) = build_mapping_table(events);

    // Sort by event_id for optimal delta encoding
    let mut sorted_indices: Vec<usize> = (0..events.len()).collect();
    sorted_indices.sort_by_key(|&i| events[i].0.id.parse::<u64>().unwrap_or(0));

    // Collect values in sorted order
    let event_types: Vec<&str> = sorted_indices
        .iter()
        .map(|&i| events[i].0.event_type.as_str())
        .collect();
    let event_ids: Vec<u64> = sorted_indices
        .iter()
        .map(|&i| events[i].0.id.parse::<u64>().unwrap_or(0))
        .collect();
    let repo_pair_indices: Vec<u32> = sorted_indices
        .iter()
        .map(|&i| {
            let repo_id = events[i].1.repo.id as u32;
            let repo_name = events[i].1.repo.name.clone();
            pair_to_idx[&(repo_id, repo_name)]
        })
        .collect();
    let timestamps: Vec<i64> = sorted_indices
        .iter()
        .map(|&i| parse_timestamp(&events[i].1.created_at))
        .collect();

    // Create header
    let header = Header {
        first_event_id: event_ids[0],
        first_timestamp: timestamps[0],
        mapping_size: mapping_tsv.len() as u32,
    };

    // Delta encode
    let event_id_deltas = compute_u8_deltas(&event_ids);
    let timestamp_deltas = compute_i16_deltas(&timestamps);

    // Build arrays
    let event_type_array: DictionaryArray<Int8Type> = event_types.into_iter().collect();
    let event_id_array = UInt8Array::from(event_id_deltas);
    let repo_pair_idx_array = UInt32Array::from(repo_pair_indices);
    let timestamp_array = Int16Array::from(timestamp_deltas);

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(event_type_array) as ArrayRef,
            Arc::new(event_id_array) as ArrayRef,
            Arc::new(repo_pair_idx_array) as ArrayRef,
            Arc::new(timestamp_array) as ArrayRef,
        ],
    )?;

    Ok((header, mapping_tsv, batch))
}

fn decode_batch(header: &Header, mapping: &[(u32, String)], batch: &RecordBatch) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
    let event_type_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .ok_or("Failed to downcast event_type")?;

    let event_id_delta_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or("Failed to downcast event_id_delta")?;

    let repo_pair_idx_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or("Failed to downcast repo_pair_idx")?;

    let timestamp_delta_array = batch
        .column(3)
        .as_any()
        .downcast_ref::<Int16Array>()
        .ok_or("Failed to downcast created_at_delta")?;

    // Restore values from deltas
    let event_id_deltas: Vec<u8> = event_id_delta_array.values().iter().copied().collect();
    let timestamp_deltas: Vec<i16> = timestamp_delta_array.values().iter().copied().collect();
    let event_ids = restore_u64_from_deltas(header.first_event_id, &event_id_deltas);
    let timestamps = restore_i64_from_deltas(header.first_timestamp, &timestamp_deltas);

    // Get string values from dictionary
    let event_type_values = event_type_array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("Failed to get event_type values")?;

    let events: Vec<(EventKey, EventValue)> = (0..batch.num_rows())
        .map(|i| {
            let event_type_idx = event_type_array.keys().value(i) as usize;
            let event_type = event_type_values.value(event_type_idx).to_string();

            let event_id = event_ids[i];

            // Look up repo info from mapping table
            let pair_idx = repo_pair_idx_array.value(i) as usize;
            let (repo_id, repo_name) = &mapping[pair_idx];

            let timestamp = timestamps[i];
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            (
                EventKey {
                    id: event_id.to_string(),
                    event_type,
                },
                EventValue {
                    repo: Repo {
                        id: *repo_id as u64,
                        name: repo_name.clone(),
                        url: repo_url,
                    },
                    created_at: format_timestamp(timestamp),
                },
            )
        })
        .collect();

    Ok(events)
}

impl EventCodec for NatebrennandCodec {
    fn name(&self) -> &str {
        "natebrennand"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let (header, mapping_tsv, batch) = encode_events(events)?;

        // Compute mapping compressed size for stats
        let mapping_compressed_size = zstd::encode_all(mapping_tsv.as_slice(), ZSTD_LEVEL)?.len();

        if debug_enabled() {
            print_column_stats(&batch, mapping_compressed_size);
        }

        // Write Arrow IPC stream
        let mut ipc_buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc_buf, &batch.schema())?;
            writer.write(&batch)?;
            writer.finish()?;
        }

        if debug_enabled() {
            let ipc_compressed = zstd::encode_all(ipc_buf.as_slice(), ZSTD_LEVEL)?;
            eprintln!("Mapping TSV: {} -> {} bytes ({:.1}%)",
                mapping_tsv.len(), mapping_compressed_size,
                100.0 * mapping_compressed_size as f64 / mapping_tsv.len() as f64);
            eprintln!("IPC: {} -> {} bytes ({:.1}%)",
                ipc_buf.len(), ipc_compressed.len(),
                100.0 * ipc_compressed.len() as f64 / ipc_buf.len() as f64);
        }

        // Compress mapping + IPC together
        let mut uncompressed = Vec::with_capacity(mapping_tsv.len() + ipc_buf.len());
        uncompressed.extend_from_slice(&mapping_tsv);
        uncompressed.extend_from_slice(&ipc_buf);
        let compressed = zstd::encode_all(uncompressed.as_slice(), ZSTD_LEVEL)?;

        // Final output: header + compressed(mapping + IPC)
        let mut buf = Vec::with_capacity(HEADER_SIZE + compressed.len());
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(&compressed);

        if debug_enabled() {
            eprintln!("Compressed size (zstd {}): {} bytes", ZSTD_LEVEL, buf.len());
            eprintln!("Compressed bytes/row: {:.2}", buf.len() as f64 / events.len() as f64);
        }

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let header = Header::decode(&bytes[0..HEADER_SIZE]);

        // Decompress
        let decompressed = zstd::decode_all(&bytes[HEADER_SIZE..])?;

        // Split into mapping and IPC
        let mapping_bytes = &decompressed[0..header.mapping_size as usize];
        let ipc_buf = &decompressed[header.mapping_size as usize..];

        // Parse mapping table
        let mapping = parse_mapping_table(mapping_bytes);

        // Read Arrow IPC stream
        let cursor = Cursor::new(ipc_buf);
        let reader = StreamReader::try_new(cursor, None)?;

        let mut all_events = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            let events = decode_batch(&header, &mapping, &batch)?;
            all_events.extend(events);
        }

        // Sort by EventKey to match expected output
        all_events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(all_events)
    }
}
