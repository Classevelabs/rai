//! What a loaded model's prompts look like, and which tokens end its replies.
//!
//! A conversion records the checkpoint's own chat template, its special
//! tokens and its end-of-sequence ids beside the model, as
//! `<name>.chat.json` ([`ChatRecord`](crate::chat_format::ChatRecord)).
//! [`ChatFormat`](crate::chat_format::ChatFormat) renders that template the
//! way transformers' `apply_chat_template` does, or a built-in
//! [`ChatTemplate`](crate::chat_template::ChatTemplate) when one is asked
//! for by name or no record exists.
//!
//! Probing the vocabulary for sentinel tokens, the only option before a
//! record existed, cannot tell most families apart: TinyLlama-Chat, Zephyr
//! and Mistral v0.1/v0.2 spell their turn markers as plain text, so all of
//! them fell through to the few-shot prompt. The record is the checkpoint's
//! own statement, so it needs no guessing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::chat_template::{ChatTemplate, Role};

/// End-of-sequence and end-of-turn markers across the verified families:
/// Llama 2 and Mistral, Llama 3, Qwen and SmolLM, Phi-3, Gemma, OLMo.
///
/// Matched against the tokenizer's added tokens only. A byte-level vocabulary
/// can hold the same spelling as an ordinary merged token — Qwen2.5 has `</s>`
/// at id 128247 — and treating that as an end would cut off any reply that
/// writes it.
const END_MARKERS: [&str; 9] = [
    "</s>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<|end|>",
    "<eos>",
    "<end_of_turn>",
];

/// What a conversion records about how its model is prompted.
///
/// Every field is the checkpoint's own, read from the files transformers
/// reads it from: `chat_template` from `chat_template.jinja` or
/// `tokenizer_config.json`, the two token strings from `tokenizer_config.json`
/// (or `special_tokens_map.json` for a config saved before transformers wrote
/// `added_tokens_decoder`), and the ids from `generation_config.json` (or
/// `config.json` when there is none). A field the checkpoint does not provide
/// is `None` or empty.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRecord {
    /// The Jinja chat template, as transformers renders it.
    #[serde(default)]
    pub chat_template: Option<String>,
    /// The beginning-of-sequence token the template spells as `bos_token`.
    #[serde(default)]
    pub bos_token: Option<String>,
    /// The end-of-sequence token the template spells as `eos_token`.
    #[serde(default)]
    pub eos_token: Option<String>,
    /// Every id that ends generation, from `generation_config.json`.
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
}

impl ChatRecord {
    /// Where the record for the model at `model` lives.
    pub fn path_beside(model: &Path) -> PathBuf {
        model.with_extension("chat.json")
    }

    /// Read what a Hugging Face checkpoint directory says about prompting it.
    ///
    /// Each file is optional: a base model ships no template, and older
    /// checkpoints no `generation_config.json`. A file that is present but
    /// malformed is an error rather than a silently empty record.
    pub fn from_checkpoint(dir: &Path) -> Result<Self> {
        let tokenizer_config = read_json_if_present(&dir.join("tokenizer_config.json"))?;
        let generation_config = read_json_if_present(&dir.join("generation_config.json"))?;

        // transformers 4.47+ saves the template as its own file and prefers it
        // over the copy in tokenizer_config.json; processors keep theirs in
        // chat_template.json.
        let jinja = dir.join("chat_template.jinja");
        let chat_template = match std::fs::read_to_string(&jinja) {
            Ok(source) => Some(source),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match tokenizer_config
                    .as_ref()
                    .map(|config| config.get("chat_template"))
                {
                    Some(Some(field)) => template_field(field, "tokenizer_config.json")?,
                    _ => match read_json_if_present(&dir.join("chat_template.json"))? {
                        Some(config) => match config.get("chat_template") {
                            Some(field) => template_field(field, "chat_template.json")?,
                            None => None,
                        },
                        None => None,
                    },
                }
            }
            Err(error) => {
                return Err(error).with_context(|| format!("reading {}", jinja.display()));
            }
        };

        // transformers reads special_tokens_map.json only for a tokenizer
        // config saved before it wrote `added_tokens_decoder`, and then lets
        // the map's entries override the config's.
        let legacy = tokenizer_config
            .as_ref()
            .is_none_or(|config| config.get("added_tokens_decoder").is_none());
        let special_tokens_map = if legacy {
            read_json_if_present(&dir.join("special_tokens_map.json"))?
        } else {
            None
        };
        let token = |name: &str| -> Option<String> {
            // `None` when the file has no such entry; `Some(None)` when it
            // has one that spells no token.
            let entry = |config: &serde_json::Value| -> Option<Option<String>> {
                let field = config.get(name)?;
                // A plain string, or the serialized AddedToken transformers
                // writes.
                Some(
                    field
                        .as_str()
                        .or_else(|| field.get("content").and_then(|content| content.as_str()))
                        .map(str::to_owned),
                )
            };
            special_tokens_map
                .as_ref()
                .and_then(entry)
                .or_else(|| tokenizer_config.as_ref().and_then(entry))
                .flatten()
        };

        // Without a generation_config.json, transformers builds the generation
        // settings from config.json, end ids included.
        let (end_source, end_file) = match generation_config {
            Some(config) => (Some(config), "generation_config.json"),
            None => (
                read_json_if_present(&dir.join("config.json"))?,
                "config.json",
            ),
        };
        let token_id = |value: &serde_json::Value| -> Result<u32> {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .with_context(|| format!("{end_file} has the eos_token_id {value}, not a token id"))
        };
        let eos_token_ids = match end_source
            .as_ref()
            .and_then(|config| config.get("eos_token_id"))
        {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::Array(ids)) => {
                ids.iter().map(token_id).collect::<Result<Vec<u32>>>()?
            }
            Some(id) => vec![token_id(id)?],
        };

        Ok(Self {
            chat_template,
            bos_token: token("bos_token"),
            eos_token: token("eos_token"),
            eos_token_ids,
        })
    }

    /// The record beside `model`, or `None` when the model has none: it was
    /// converted before records existed, or renamed without its record.
    pub fn read_beside(model: &Path) -> Result<Option<Self>> {
        let path = Self::path_beside(model);
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .with_context(|| format!("parsing {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// The record as written beside a model: pretty JSON, one trailing newline.
    pub fn to_json(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(self).expect("a ChatRecord always serializes");
        bytes.push(b'\n');
        bytes
    }
}

fn read_json_if_present(path: &Path) -> Result<Option<serde_json::Value>> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .with_context(|| format!("parsing {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

/// A `chat_template` field: one template, or transformers' list of named
/// ones, of which the one named `default` is what `apply_chat_template` uses.
fn template_field(field: &serde_json::Value, file: &str) -> Result<Option<String>> {
    match field {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(source) => Ok(Some(source.clone())),
        serde_json::Value::Array(named) => Ok(named
            .iter()
            .find(|entry| entry.get("name").and_then(|name| name.as_str()) == Some("default"))
            .and_then(|entry| entry.get("template"))
            .and_then(|template| template.as_str())
            .map(str::to_owned)),
        other => bail!("{file} has a chat_template that is neither text nor a list: {other}"),
    }
}

/// How a loaded model is prompted, and where its replies end.
pub struct ChatFormat {
    style: Style,
    end_ids: Vec<usize>,
    stops: Vec<String>,
}

enum Style {
    Builtin(ChatTemplate),
    Recorded(Box<Recorded>),
}

struct Recorded {
    env: minijinja::Environment<'static>,
    bos_token: String,
    eos_token: String,
    /// The built-in family whose turn markers the template uses, if any: its
    /// text stop sequences still apply, and it names the format to a reader.
    family: Option<ChatTemplate>,
}

const TEMPLATE_NAME: &str = "chat_template";

impl ChatFormat {
    /// The format `choice` asks for, for the model at `model`.
    ///
    /// `choice` is `auto` (the model's recorded template; the few-shot prompt
    /// for a model recorded without one, which is a base model; sentinel
    /// probing for a model with no record), `model` (the recorded template,
    /// or an error), or a built-in template's name. Anything else is an error.
    pub fn load(choice: &str, model: &Path, tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let record = ChatRecord::read_beside(model)?;
        Self::new(choice, record.as_ref(), tokenizer)
    }

    /// [`ChatFormat::load`] with the record already in hand.
    pub fn new(
        choice: &str,
        record: Option<&ChatRecord>,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Self> {
        let recorded = record.and_then(|record| record.chat_template.as_deref());
        let style = match choice {
            "model" => match (record, recorded) {
                (Some(record), Some(source)) => Style::recorded(source, record)?,
                (Some(_), None) => bail!(
                    "this model was converted from a checkpoint without a chat template; pass a \
                     template by name ({})",
                    ChatTemplate::NAMES.join(", ")
                ),
                (None, _) => bail!(
                    "this model has no <name>.chat.json beside it, so it carries no chat \
                     template of its own; convert it again, or pass a template by name ({})",
                    ChatTemplate::NAMES.join(", ")
                ),
            },
            "auto" => match (record, recorded) {
                (Some(record), Some(source)) => match Style::recorded(source, record) {
                    Ok(style) => style,
                    // A template this engine cannot parse must not make the
                    // model unusable: probe as before records existed, and
                    // say so. `model` asks for the template and gets the error.
                    Err(error) => {
                        eprintln!("warning: {error:#}; using a built-in template instead");
                        Style::Builtin(ChatTemplate::auto_detect(tokenizer))
                    }
                },
                (Some(_), None) => Style::Builtin(ChatTemplate::FewShot),
                (None, _) => Style::Builtin(ChatTemplate::auto_detect(tokenizer)),
            },
            name => Style::Builtin(ChatTemplate::from_name(name).with_context(|| {
                format!(
                    "unknown chat template '{name}'; expected auto, model, or one of {}",
                    ChatTemplate::NAMES.join(", ")
                )
            })?),
        };

        let mut stops: Vec<String> = match &style {
            Style::Builtin(template) => template
                .stop_sequences()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Style::Recorded(recorded) => recorded
                .family
                .map(|family| {
                    family
                        .stop_sequences()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default(),
        };
        if let Some(eos) = record.and_then(|record| record.eos_token.as_deref()) {
            if !eos.is_empty() && !stops.iter().any(|stop| stop == eos) {
                stops.push(eos.to_owned());
            }
        }
        let added: HashMap<String, u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| (token.content, id))
            .collect();
        let end_ids = end_ids(tokenizer, &added, record, &stops);
        // A stop that is a control token ends the reply by its id. Matched as
        // text too, it could only fire on ordinary tokens that happen to spell
        // it, cutting a reply that quotes `</s>`, where transformers stops on
        // ids alone. Text stops stay for markers the tokenizer spells as text.
        stops.retain(|stop| !added.contains_key(stop));
        Ok(Self {
            style,
            end_ids,
            stops,
        })
    }

    /// The prompt for `turns`, oldest first, ending with the opener of the
    /// assistant's reply.
    ///
    /// A recorded template can refuse a conversation, as transformers would
    /// (Gemma's insists that roles alternate); that is an error here.
    pub fn render(&self, turns: &[(Role, &str)]) -> Result<String> {
        match &self.style {
            Style::Builtin(template) => Ok(template.format_conversation(turns)),
            Style::Recorded(recorded) => recorded.render(turns),
        }
    }

    /// `turns` rendered and tokenized, with the special-token policy this
    /// format needs: a recorded template spells `bos_token` itself, exactly as
    /// `apply_chat_template` tokenizes, and a built-in one leaves it to the
    /// tokenizer, as `tokenizer(text)` does.
    pub fn encode(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        turns: &[(Role, &str)],
    ) -> Result<Vec<usize>> {
        let prompt = self.render(turns)?;
        let add_special_tokens = matches!(self.style, Style::Builtin(_));
        let encoding = tokenizer
            .encode(prompt.as_str(), add_special_tokens)
            .map_err(|error| anyhow::anyhow!("could not tokenize the prompt: {error}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Token ids that end a reply.
    pub fn end_ids(&self) -> &[usize] {
        &self.end_ids
    }

    /// Text that ends a reply when it appears in the decoded output, for the
    /// markers a model spells across several tokens.
    pub fn stop_sequences(&self) -> &[String] {
        &self.stops
    }

    /// The format's name, as `--chat-template` spells it, for a reader.
    pub fn name(&self) -> String {
        match &self.style {
            Style::Builtin(template) => template.display_name().to_string(),
            Style::Recorded(recorded) => match recorded.family {
                Some(family) => format!("the model's own ({})", family.display_name()),
                None => "the model's own".to_string(),
            },
        }
    }

    /// The built-in template in use, or `None` for the model's own.
    pub fn builtin(&self) -> Option<ChatTemplate> {
        match &self.style {
            Style::Builtin(template) => Some(*template),
            Style::Recorded(_) => None,
        }
    }
}

impl Style {
    fn recorded(source: &str, record: &ChatRecord) -> Result<Self> {
        let mut env = minijinja::Environment::new();
        // transformers renders chat templates with exactly these two settings.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(TEMPLATE_NAME, source.to_owned())
            .context("the model's chat template does not parse")?;
        Ok(Style::Recorded(Box::new(Recorded {
            env,
            bos_token: record.bos_token.clone().unwrap_or_default(),
            eos_token: record.eos_token.clone().unwrap_or_default(),
            family: family_of(source),
        })))
    }
}

impl Recorded {
    fn render(&self, turns: &[(Role, &str)]) -> Result<String> {
        let messages: Vec<HashMap<&str, &str>> = turns
            .iter()
            .map(|&(role, content)| HashMap::from([("role", role.as_str()), ("content", content)]))
            .collect();
        let template = self.env.get_template(TEMPLATE_NAME)?;
        // The variables transformers passes: no tools or documents, the
        // generation prompt on, and the tokenizer's special tokens.
        template
            .render(minijinja::context! {
                messages => messages,
                add_generation_prompt => true,
                tools => (),
                documents => (),
                bos_token => self.bos_token,
                eos_token => self.eos_token,
            })
            .map_err(|error| {
                anyhow::anyhow!("the model's chat template refused this conversation: {error}")
            })
    }
}

/// The built-in family whose turn markers a template uses.
fn family_of(source: &str) -> Option<ChatTemplate> {
    if source.contains("<start_of_turn>") {
        Some(ChatTemplate::Gemma)
    } else if source.contains("<|start_header_id|>") {
        Some(ChatTemplate::Llama3Instruct)
    } else if source.contains("<|im_start|>") {
        Some(ChatTemplate::ChatML)
    } else if source.contains("[INST]") {
        Some(ChatTemplate::MistralInstruct)
    } else if source.contains("<|user|>") && source.contains("<|end|>") {
        Some(ChatTemplate::Phi3)
    } else if source.contains("<|user|>") {
        Some(ChatTemplate::Zephyr)
    } else {
        None
    }
}

/// Every id that ends a reply: the end markers among the tokenizer's added
/// tokens, the ids and token the checkpoint declares, and the stop markers
/// that are tokens of their own.
fn end_ids(
    tokenizer: &tokenizers::Tokenizer,
    added: &HashMap<String, u32>,
    record: Option<&ChatRecord>,
    stops: &[String],
) -> Vec<usize> {
    let vocab = tokenizer.get_vocab_size(true);
    let mut ids: Vec<u32> = END_MARKERS
        .iter()
        .copied()
        .chain(stops.iter().map(String::as_str))
        .filter_map(|marker| added.get(marker).copied())
        .collect();
    if let Some(record) = record {
        ids.extend(&record.eos_token_ids);
        // The checkpoint's own eos is authoritative wherever the tokenizer
        // keeps it.
        if let Some(eos) = record.eos_token.as_deref() {
            ids.extend(
                added
                    .get(eos)
                    .copied()
                    .or_else(|| tokenizer.token_to_id(eos)),
            );
        }
    }
    ids.retain(|&id| (id as usize) < vocab);
    ids.sort_unstable();
    ids.dedup();
    ids.into_iter().map(|id| id as usize).collect()
}

fn raise_exception(message: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        message,
    ))
}

/// transformers' `strftime_now`, which is `datetime.now().strftime(format)`:
/// local time. Llama 3.1 and 3.2 put today's date in their system header with
/// it; without it they fall back to a fixed date.
fn strftime_now(format: String) -> Result<String, minijinja::Error> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    strftime(
        &format,
        seconds.saturating_add_signed(local_offset_seconds(seconds)),
    )
}

/// How far local time is ahead of UTC at `unix_seconds`, by the system's own
/// zone rules; 0, which is UTC, where the system cannot say.
// `tm_gmtoff` is a C `long`: already 64 bits on 64-bit targets, 32 on others.
#[allow(clippy::useless_conversion)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn local_offset_seconds(unix_seconds: u64) -> i64 {
    let Ok(time) = libc::time_t::try_from(unix_seconds) else {
        return 0;
    };
    // SAFETY: an all-zero `tm` is a valid value, which `localtime_r` overwrites.
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers are to live locals, and `localtime_r` writes only
    // through the second.
    if unsafe { libc::localtime_r(&time, &mut local) }.is_null() {
        return 0;
    }
    i64::from(local.tm_gmtoff)
}

/// How far local time is ahead of UTC now, by the system's own zone rules; 0,
/// which is UTC, where the system cannot say.
#[cfg(windows)]
fn local_offset_seconds(_unix_seconds: u64) -> i64 {
    use windows_sys::Win32::System::SystemServices::{
        TIME_ZONE_ID_DAYLIGHT, TIME_ZONE_ID_STANDARD, TIME_ZONE_ID_UNKNOWN,
    };
    use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};
    // SAFETY: an all-zero TIME_ZONE_INFORMATION is a valid value, which the
    // call overwrites.
    let mut zone: TIME_ZONE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `zone` is a live, writable TIME_ZONE_INFORMATION.
    let bias = match unsafe { GetTimeZoneInformation(&mut zone) } {
        TIME_ZONE_ID_DAYLIGHT => zone.Bias + zone.DaylightBias,
        TIME_ZONE_ID_STANDARD => zone.Bias + zone.StandardBias,
        TIME_ZONE_ID_UNKNOWN => zone.Bias,
        _ => return 0,
    };
    // The bias is the minutes added to local time to reach UTC.
    -i64::from(bias) * 60
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn local_offset_seconds(_unix_seconds: u64) -> i64 {
    0
}

fn strftime(format: &str, unix_seconds: u64) -> Result<String, minijinja::Error> {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const WEEKDAYS: [&str; 7] = [
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
    ];
    let days = unix_seconds / 86_400;
    let (year, month, day) = civil_from_days(days);
    let second_of_day = unix_seconds % 86_400;
    let weekday = WEEKDAYS[(days % 7) as usize];
    let month_name = MONTHS[(month - 1) as usize];

    let mut out = String::with_capacity(format.len() + 16);
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('Y') => out.push_str(&year.to_string()),
            Some('y') => out.push_str(&format!("{:02}", year % 100)),
            Some('b') => out.push_str(&month_name[..3]),
            Some('B') => out.push_str(month_name),
            Some('a') => out.push_str(&weekday[..3]),
            Some('A') => out.push_str(weekday),
            Some('H') => out.push_str(&format!("{:02}", second_of_day / 3600)),
            Some('M') => out.push_str(&format!("{:02}", second_of_day % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", second_of_day % 60)),
            Some('%') => out.push('%'),
            other => {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!(
                        "strftime_now does not support %{}",
                        other.map(String::from).unwrap_or_default()
                    ),
                ))
            }
        }
    }
    Ok(out)
}

/// Days since 1970-01-01 to a proleptic Gregorian (year, month, day), by
/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// A word-level tokenizer whose added tokens are `<s>`, `</s>` and
    /// `<|im_end|>`, plus an ordinary vocabulary entry spelled `<eos>`: the
    /// shape of a byte-level vocabulary that merged an end marker's spelling.
    fn tokenizer() -> tokenizers::Tokenizer {
        let added = |id: u32, content: &str| {
            serde_json::json!({
                "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        };
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(1, "<s>"), added(2, "</s>"), added(3, "<|im_end|>")],
            "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "unk_token": "[UNK]", "vocab": {
                "[UNK]": 0, "<s>": 1, "</s>": 2, "<|im_end|>": 3, "<eos>": 4, "hi": 5
            }}
        });
        tokenizers::Tokenizer::from_str(&json.to_string()).expect("stub tokenizer")
    }

    fn record(template: Option<&str>) -> ChatRecord {
        ChatRecord {
            chat_template: template.map(str::to_owned),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            eos_token_ids: vec![2],
        }
    }

    /// TinyLlama-Chat's template, verbatim.
    const ZEPHYR: &str = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";

    #[test]
    fn a_recorded_template_renders_as_transformers_does() {
        let format = ChatFormat::new("auto", Some(&record(Some(ZEPHYR))), &tokenizer()).unwrap();
        let turns = [
            (Role::User, "Hi"),
            (Role::Assistant, "Hello!"),
            (Role::User, "Bye"),
        ];
        // What transformers' own Jinja environment renders for this thread
        // (trim_blocks and lstrip_blocks on, as it sets them).
        assert_eq!(
            format.render(&turns).unwrap(),
            "<|user|>\nHi</s>\n<|assistant|>\nHello!</s>\n<|user|>\nBye</s>\n<|assistant|>\n"
        );
        assert_eq!(format.name(), "the model's own (Zephyr)");
        assert!(format.builtin().is_none());
        // The family's text stops still apply.
        assert!(format
            .stop_sequences()
            .iter()
            .any(|stop| stop == "<|user|>"));
    }

    #[test]
    fn python_string_methods_and_template_errors_work() {
        let template = "{{ bos_token }}{% for m in messages %}\
            {% if (m['role'] == 'user') != (loop.index0 % 2 == 0) %}\
            {{ raise_exception('Conversation roles must alternate') }}{% endif %}\
            {{ m['content'].strip().upper() }}|{% endfor %}";
        let format = ChatFormat::new("model", Some(&record(Some(template))), &tokenizer()).unwrap();
        assert_eq!(format.render(&[(Role::User, "  hi ")]).unwrap(), "<s>HI|");
        let error = format
            .render(&[(Role::User, "a"), (Role::User, "b")])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Conversation roles must alternate"),
            "{error}"
        );
        // A recorded template spells bos_token itself, so encoding adds none:
        // `<s>` once, then `HI` and `|`, both unknown to the stub vocabulary.
        assert_eq!(
            format.encode(&tokenizer(), &[(Role::User, "hi")]).unwrap(),
            vec![1, 0, 0]
        );
    }

    #[test]
    fn end_ids_come_from_control_tokens_and_the_record() {
        let format = ChatFormat::new("chatml", Some(&record(None)), &tokenizer()).unwrap();
        // `<|im_end|>` and `</s>` are control tokens; `<eos>` here is only an
        // ordinary vocabulary entry with that spelling, so it is not an end.
        assert_eq!(format.end_ids(), &[2, 3]);
        let unrecorded = ChatFormat::new("none", None, &tokenizer()).unwrap();
        assert_eq!(unrecorded.end_ids(), &[2, 3]);
    }

    /// A stop marker that is a control token ends the reply by id, and is not
    /// also a text stop: matched as text it could only cut a reply that
    /// quotes it. A marker spelled as plain text stays a text stop.
    #[test]
    fn control_token_stops_end_by_id_not_by_text() {
        let chatml = ChatFormat::new("chatml", Some(&record(None)), &tokenizer()).unwrap();
        assert!(chatml.end_ids().contains(&3));
        assert_eq!(chatml.stop_sequences(), &["<|im_start|>".to_string()]);
        let zephyr = ChatFormat::new("auto", Some(&record(Some(ZEPHYR))), &tokenizer()).unwrap();
        assert!(zephyr.end_ids().contains(&2));
        assert_eq!(zephyr.stop_sequences(), &["<|user|>".to_string()]);
    }

    #[test]
    fn choices_resolve_or_are_refused_by_name() {
        let tokenizer = tokenizer();
        // Recorded without a template: a base model, prompted few-shot.
        let base = ChatFormat::new("auto", Some(&record(None)), &tokenizer).unwrap();
        assert_eq!(base.builtin(), Some(ChatTemplate::FewShot));
        // No record at all: the tokenizer is probed, as before records.
        let probed = ChatFormat::new("auto", None, &tokenizer).unwrap();
        assert_eq!(probed.builtin(), Some(ChatTemplate::FewShot));
        for (choice, record) in [("model", None), ("model", Some(record(None)))] {
            assert!(ChatFormat::new(choice, record.as_ref(), &tokenizer).is_err());
        }
        let error = ChatFormat::new("chatlm", None, &tokenizer)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("unknown chat template 'chatlm'"), "{error}");
        // A template that does not parse is an error when asked for, and a
        // fallback to probing under `auto`.
        let broken = record(Some("{% for m in messages %}"));
        assert!(ChatFormat::new("model", Some(&broken), &tokenizer).is_err());
        let fallback = ChatFormat::new("auto", Some(&broken), &tokenizer).unwrap();
        assert_eq!(fallback.builtin(), Some(ChatTemplate::FewShot));
    }

    #[test]
    fn a_checkpoint_says_how_it_is_prompted() {
        let dir = std::env::temp_dir().join(format!("rai-chat-record-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Nothing at all is an empty record, not an error.
        assert_eq!(
            ChatRecord::from_checkpoint(&dir).unwrap(),
            ChatRecord::default()
        );

        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": [
                    { "name": "tool_use", "template": "tools" },
                    { "name": "default", "template": "chat" }
                ],
                "bos_token": { "__type": "AddedToken", "content": "<s>" },
                "eos_token": "</s>"
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": [2, 32000]}"#,
        )
        .unwrap();
        let read = ChatRecord::from_checkpoint(&dir).unwrap();
        assert_eq!(read.chat_template.as_deref(), Some("chat"));
        assert_eq!(read.bos_token.as_deref(), Some("<s>"));
        assert_eq!(read.eos_token.as_deref(), Some("</s>"));
        assert_eq!(read.eos_token_ids, vec![2, 32000]);

        // transformers 4.47+ keeps the template in its own file, which wins.
        std::fs::write(dir.join("chat_template.jinja"), "from the file").unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 7}"#).unwrap();
        let read = ChatRecord::from_checkpoint(&dir).unwrap();
        assert_eq!(read.chat_template.as_deref(), Some("from the file"));
        assert_eq!(read.eos_token_ids, vec![7]);

        // The record survives its own round trip.
        let model = dir.join("model.raimodel");
        std::fs::write(ChatRecord::path_beside(&model), read.to_json()).unwrap();
        assert_eq!(ChatRecord::read_beside(&model).unwrap(), Some(read));
        assert_eq!(ChatRecord::path_beside(&model), dir.join("model.chat.json"));

        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": -1}"#,
        )
        .unwrap();
        assert!(ChatRecord::from_checkpoint(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The record reads each field where transformers 5 does: a legacy
    /// tokenizer config's tokens give way to special_tokens_map.json, and a
    /// checkpoint without generation_config.json ends where config.json says.
    #[test]
    fn the_record_reads_each_field_where_transformers_does() {
        let dir = std::env::temp_dir().join(format!("rai-chat-sources-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, value: serde_json::Value| {
            std::fs::write(dir.join(name), value.to_string()).unwrap();
        };
        write(
            "tokenizer_config.json",
            serde_json::json!({ "bos_token": "<s>", "eos_token": "</s>" }),
        );
        write(
            "special_tokens_map.json",
            serde_json::json!({ "bos_token": { "content": "<|begin|>" } }),
        );
        write("config.json", serde_json::json!({ "eos_token_id": 5 }));
        let read = ChatRecord::from_checkpoint(&dir).unwrap();
        assert_eq!(read.bos_token.as_deref(), Some("<|begin|>"));
        assert_eq!(read.eos_token.as_deref(), Some("</s>"));
        assert_eq!(read.eos_token_ids, vec![5]);

        // A config that lists its added tokens is read as it stands, and a
        // generation config, once there, is the only source of end ids.
        write(
            "tokenizer_config.json",
            serde_json::json!({ "bos_token": "<s>", "added_tokens_decoder": {} }),
        );
        write("generation_config.json", serde_json::json!({}));
        let read = ChatRecord::from_checkpoint(&dir).unwrap();
        assert_eq!(read.bos_token.as_deref(), Some("<s>"));
        assert!(read.eos_token_ids.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dates_are_formatted_as_python_formats_them() {
        assert_eq!(strftime("%d %b %Y", 0).unwrap(), "01 Jan 1970");
        // 2023-11-14 22:13:20 UTC, a Tuesday.
        assert_eq!(
            strftime("%A %d %B %y %H:%M:%S %a %m %%", 1_700_000_000).unwrap(),
            "Tuesday 14 November 23 22:13:20 Tue 11 %"
        );
        // 2024-02-29, a leap day.
        assert_eq!(strftime("%Y-%m-%d", 1_709_164_800).unwrap(), "2024-02-29");
        assert!(strftime("%Q", 0).is_err());
    }

    /// The clock the template sees is the system's local one: every zone in
    /// use is within 14 hours of UTC and a whole number of minutes from it.
    #[test]
    fn the_template_clock_is_local_time() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let offset = local_offset_seconds(now);
        assert!(offset.abs() <= 14 * 3600, "{offset} s from UTC");
        assert_eq!(offset % 60, 0, "{offset} s from UTC");
        let today = strftime_now("%Y-%m-%d".to_string()).unwrap();
        let expected = strftime("%Y-%m-%d", now.saturating_add_signed(offset)).unwrap();
        let next_second = strftime("%Y-%m-%d", (now + 1).saturating_add_signed(offset)).unwrap();
        assert!(today == expected || today == next_second, "{today}");
    }
}
