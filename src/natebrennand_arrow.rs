//! Arrow IPC codec with zstd-22 compression.
//!
//! Strategy:
//! - Dictionary encode event_type and repo_name
//! - Store repo_id as UInt64, event_id as UInt64, timestamp as Int64
//! - Reconstruct repo.url from repo.name during decode
//! - Use Arrow IPC wrapped with zstd level 22 compression

use arrow::array::{ArrayRef, DictionaryArray, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Int32Type, Int8Type, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use bytes::Bytes;
use chrono::DateTime;
use std::error::Error;
use std::io::Cursor;
use std::sync::Arc;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;

pub struct NateBrennandArrowCodec;

impl NateBrennandArrowCodec {
    pub fn new() -> Self {
        Self
    }
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
        Field::new("event_id", DataType::UInt64, false),
        Field::new("repo_id", DataType::UInt64, false),
        Field::new(
            "repo_name",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("created_at", DataType::Int64, false),
    ])
}


impl EventCodec for NateBrennandArrowCodec {
    fn name(&self) -> &str {
        "natebrennand-arrow"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let schema = Arc::new(build_schema());

        // Collect column data
        let event_types: Vec<&str> = events.iter().map(|(k, _)| k.event_type.as_str()).collect();
        let event_ids: Vec<u64> = events
            .iter()
            .map(|(k, _)| k.id.parse::<u64>().unwrap_or(0))
            .collect();
        let repo_ids: Vec<u64> = events.iter().map(|(_, v)| v.repo.id).collect();
        let repo_names: Vec<&str> = events.iter().map(|(_, v)| v.repo.name.as_str()).collect();
        let timestamps: Vec<i64> = events
            .iter()
            .map(|(_, v)| parse_timestamp(&v.created_at))
            .collect();

        // Build arrays
        let event_type_array: DictionaryArray<Int8Type> = event_types.into_iter().collect();
        let event_id_array = UInt64Array::from(event_ids);
        let repo_id_array = UInt64Array::from(repo_ids);
        let repo_name_array: DictionaryArray<Int32Type> = repo_names.into_iter().collect();
        let timestamp_array = Int64Array::from(timestamps);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(event_type_array) as ArrayRef,
                Arc::new(event_id_array) as ArrayRef,
                Arc::new(repo_id_array) as ArrayRef,
                Arc::new(repo_name_array) as ArrayRef,
                Arc::new(timestamp_array) as ArrayRef,
            ],
        )?;

        // Write Arrow IPC stream
        let mut ipc_buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc_buf, &schema)?;
            writer.write(&batch)?;
            writer.finish()?;
        }

        // Compress with zstd level 22
        let compressed = zstd::encode_all(ipc_buf.as_slice(), ZSTD_LEVEL)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        // Decompress with zstd
        let decompressed = zstd::decode_all(bytes)?;
        let cursor = Cursor::new(decompressed);
        let reader = StreamReader::try_new(cursor, None)?;

        let mut all_events = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;

            let event_type_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<DictionaryArray<Int8Type>>()
                .ok_or("Failed to downcast event_type")?;
            let event_type_values = event_type_array
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("Failed to get event_type values")?;

            let event_id_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or("Failed to downcast event_id")?;

            let repo_id_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or("Failed to downcast repo_id")?;

            let repo_name_array = batch
                .column(3)
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or("Failed to downcast repo_name")?;
            let repo_name_values = repo_name_array
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("Failed to get repo_name values")?;

            let timestamp_array = batch
                .column(4)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("Failed to downcast created_at")?;

            for i in 0..batch.num_rows() {
                let event_type_idx = event_type_array.keys().value(i) as usize;
                let event_type = event_type_values.value(event_type_idx).to_string();

                let event_id = event_id_array.value(i);

                let repo_id = repo_id_array.value(i);
                let repo_name_idx = repo_name_array.keys().value(i) as usize;
                let repo_name = repo_name_values.value(repo_name_idx).to_string();
                let repo_url = format!("https://api.github.com/repos/{repo_name}");

                let timestamp = timestamp_array.value(i);

                all_events.push((
                    EventKey {
                        id: event_id.to_string(),
                        event_type,
                    },
                    EventValue {
                        repo: Repo {
                            id: repo_id,
                            name: repo_name,
                            url: repo_url,
                        },
                        created_at: format_timestamp(timestamp),
                    },
                ));
            }
        }

        // Sort by EventKey to match expected output
        all_events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(all_events)
    }
}
