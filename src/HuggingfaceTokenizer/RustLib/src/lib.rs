extern crate libc;
extern crate lazy_static;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;
use std::sync::Mutex;
use tokenizers::Tokenizer;

lazy_static::lazy_static! {
    static ref LOADED_TOKENIZER: Mutex<Option<Tokenizer>> = Mutex::new(None);
}


use std::fs::OpenOptions;
use std::io::Write;
use std::time::SystemTime;

fn log_to_file(message: &str) {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open("rust_logs.txt")
        .unwrap();

    writeln!(file, "[{}]: {}", now, message).unwrap();
}


#[no_mangle]
pub extern "C" fn load_tokenizer(tokenizer_path: *const c_char, seq_len: usize) -> bool {
    let path = unsafe { CStr::from_ptr(tokenizer_path) }.to_str().unwrap();
    let tokenizer = Tokenizer::from_file(path);

    match tokenizer {
        Ok(mut tok) => {
            // Set truncation parameters with max_length = 512
            tok.with_truncation(Some(tokenizers::TruncationParams {
                max_length: seq_len,
                ..Default::default() // You can customize other truncation behaviors here if needed
            })).unwrap();

            tok.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(seq_len),
                pad_id: 0,
                ..Default::default()
            }));

            let mut global_tokenizer = LOADED_TOKENIZER.lock().unwrap();
            *global_tokenizer = Some(tok);
            true
        },
        Err(_) => false
    }
}

#[no_mangle]
pub extern "C" fn tokenize_and_get_ids(
    input: *const c_char,
    output_ids: *mut u32,
    ids_len: usize,
    output_attention_mask: *mut u32,
    attention_mask_len: usize,
) -> bool {
    let input_string = unsafe { CStr::from_ptr(input as *const i8) }.to_str().map_err(|e| {
        log_to_file(&format!("Failed to convert CStr to Rust string: {}", e))
    }).unwrap();

    let tokenizer = LOADED_TOKENIZER.lock().unwrap();

    if let Some(tokenizer) = &*tokenizer {
        let encoding = match tokenizer.encode(&*input_string, true) {
            Ok(enc) => enc,
            Err(e) => {
                log_to_file(&format!("Failed to tokenize: {}", e));
                return false;
            }
        };
        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Write to output_ids
        let ids_output_slice = unsafe { slice::from_raw_parts_mut(output_ids, ids_len) };
        for (i, &id) in ids.iter().enumerate() {
            ids_output_slice[i] = id;
        }

        // Write to output_attention_mask
        let attention_mask_output_slice = unsafe { slice::from_raw_parts_mut(output_attention_mask, attention_mask_len) };
        for (i, &mask) in attention_mask.iter().enumerate() {
            attention_mask_output_slice[i] = mask;
        }
        true
    } else {
        false // Tokenizer not loaded
    }
}
