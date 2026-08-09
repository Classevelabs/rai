//! Chat template formatting for instruction-tuned models.
//!
//! Supports:
//! - `MistralInstruct`: `<s>[INST] {msg} [/INST]` (Mistral-7B-Instruct)
//! - `Llama3Instruct`: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`
//! - `FewShot`: Simple few-shot prompt for base models (SmolLM-135M etc.)
//! - `None`: Raw prompt passthrough

/// Chat template variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatTemplate {
    /// No template — raw prompt passthrough.
    None,
    /// Few-shot prompt for base models: `User: ...\nAssistant: ...`
    FewShot,
    /// Mistral-Instruct: `<s>[INST] {msg} [/INST]`
    MistralInstruct,
    /// Llama-3-Instruct: header-based format
    Llama3Instruct,
}

impl ChatTemplate {
    /// Format a user message into a model-ready prompt.
    pub fn format_prompt(&self, user_message: &str) -> String {
        match self {
            ChatTemplate::None => user_message.to_string(),
            ChatTemplate::FewShot => format!(
                "User: Hello!\n\
                 Assistant: Hello! How can I help you today?\n\
                 User: What is the capital of France?\n\
                 Assistant: The capital of France is Paris.\n\
                 User: {}\n\
                 Assistant:",
                user_message
            ),
            ChatTemplate::MistralInstruct => format!("<s>[INST] {} [/INST]", user_message),
            ChatTemplate::Llama3Instruct => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
                 {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user_message
            ),
        }
    }

    /// Stop sequences for this template.
    pub fn stop_sequences(&self) -> &'static [&'static str] {
        match self {
            ChatTemplate::None => &[],
            ChatTemplate::FewShot => &["\nUser:", "\nHuman:", "\nStudent", "\nQ:", "\n\n\n"],
            ChatTemplate::MistralInstruct => &["</s>", "[INST]"],
            ChatTemplate::Llama3Instruct => &["<|eot_id|>", "<|end_of_text|>"],
        }
    }

    /// Display name for the UI.
    pub fn display_name(&self) -> &'static str {
        match self {
            ChatTemplate::None => "None",
            ChatTemplate::FewShot => "Few-Shot",
            ChatTemplate::MistralInstruct => "Mistral Instruct",
            ChatTemplate::Llama3Instruct => "Llama-3 Instruct",
        }
    }

    /// Auto-detect template from tokenizer vocabulary.
    /// Checks for sentinel tokens that identify the model family.
    ///
    /// Only available with the default `cli` feature — template formatting
    /// itself has no tokenizer dependency.
    #[cfg(feature = "cli")]
    pub fn auto_detect(tokenizer: &tokenizers::Tokenizer) -> Self {
        // Check for Mistral-Instruct sentinel: [INST] token
        if tokenizer.token_to_id("[INST]").is_some() {
            return ChatTemplate::MistralInstruct;
        }
        // Check for Llama-3-Instruct sentinel: <|start_header_id|>
        if tokenizer.token_to_id("<|start_header_id|>").is_some() {
            return ChatTemplate::Llama3Instruct;
        }
        // Default: few-shot for base models
        ChatTemplate::FewShot
    }

    /// Parse from CLI string.
    ///
    /// Only available with the default `cli` feature (unknown and `"auto"`
    /// values fall back to tokenizer auto-detection).
    #[cfg(feature = "cli")]
    pub fn from_str_arg(s: &str, tokenizer: &tokenizers::Tokenizer) -> Self {
        match s {
            "auto" => Self::auto_detect(tokenizer),
            "none" => ChatTemplate::None,
            "few-shot" => ChatTemplate::FewShot,
            "mistral" => ChatTemplate::MistralInstruct,
            "llama3" => ChatTemplate::Llama3Instruct,
            _ => {
                eprintln!("Warning: unknown chat template '{s}', using auto-detect");
                Self::auto_detect(tokenizer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_prompt_none_is_passthrough() {
        assert_eq!(ChatTemplate::None.format_prompt("hi there"), "hi there");
    }

    #[test]
    fn format_prompt_few_shot_embeds_message_and_ends_at_assistant_turn() {
        let prompt = ChatTemplate::FewShot.format_prompt("What is 2+2?");
        assert!(prompt.contains("User: What is 2+2?\n"));
        assert!(prompt.ends_with("Assistant:"));
        // The canned examples precede the real message.
        assert!(prompt.starts_with("User: Hello!\n"));
        assert!(prompt.contains("The capital of France is Paris."));
    }

    #[test]
    fn format_prompt_mistral_wraps_message_in_inst_markers() {
        assert_eq!(
            ChatTemplate::MistralInstruct.format_prompt("Say hi"),
            "<s>[INST] Say hi [/INST]"
        );
    }

    #[test]
    fn format_prompt_llama3_uses_header_format() {
        let prompt = ChatTemplate::Llama3Instruct.format_prompt("Say hi");
        assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("Say hi<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn stop_sequences_match_each_template() {
        assert!(ChatTemplate::None.stop_sequences().is_empty());
        assert_eq!(
            ChatTemplate::FewShot.stop_sequences(),
            &["\nUser:", "\nHuman:", "\nStudent", "\nQ:", "\n\n\n"]
        );
        assert_eq!(
            ChatTemplate::MistralInstruct.stop_sequences(),
            &["</s>", "[INST]"]
        );
        assert_eq!(
            ChatTemplate::Llama3Instruct.stop_sequences(),
            &["<|eot_id|>", "<|end_of_text|>"]
        );
    }

    #[test]
    fn display_names_are_stable() {
        assert_eq!(ChatTemplate::None.display_name(), "None");
        assert_eq!(ChatTemplate::FewShot.display_name(), "Few-Shot");
        assert_eq!(
            ChatTemplate::MistralInstruct.display_name(),
            "Mistral Instruct"
        );
        assert_eq!(
            ChatTemplate::Llama3Instruct.display_name(),
            "Llama-3 Instruct"
        );
    }

    /// Tokenizer-dependent tests: build a minimal in-memory WordLevel
    /// tokenizer so no model files are needed.
    #[cfg(feature = "cli")]
    mod with_tokenizer {
        use super::super::*;
        use tokenizers::models::wordlevel::WordLevel;

        fn stub_tokenizer(extra_tokens: &[&str]) -> tokenizers::Tokenizer {
            let mut pairs: Vec<(String, u32)> = vec![("[UNK]".to_string(), 0)];
            for (i, token) in extra_tokens.iter().enumerate() {
                pairs.push(((*token).to_string(), (i + 1) as u32));
            }
            // The builder takes tokenizers' re-exported hash map; collect
            // into it by inference rather than naming the ahash type.
            let model = WordLevel::builder()
                .vocab(pairs.into_iter().collect())
                .unk_token("[UNK]".to_string())
                .build()
                .expect("stub WordLevel model");
            tokenizers::Tokenizer::new(model)
        }

        #[test]
        fn auto_detect_recognizes_sentinel_tokens() {
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["[INST]"])),
                ChatTemplate::MistralInstruct
            );
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["<|start_header_id|>"])),
                ChatTemplate::Llama3Instruct
            );
            // No sentinel tokens → base-model few-shot default.
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["hello"])),
                ChatTemplate::FewShot
            );
            // Mistral sentinel wins when both are present (documented order).
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["[INST]", "<|start_header_id|>"])),
                ChatTemplate::MistralInstruct
            );
        }

        #[test]
        fn from_str_arg_maps_known_names_without_consulting_the_tokenizer() {
            // The tokenizer has Mistral's sentinel, but explicit names win.
            let tokenizer = stub_tokenizer(&["[INST]"]);
            assert_eq!(
                ChatTemplate::from_str_arg("none", &tokenizer),
                ChatTemplate::None
            );
            assert_eq!(
                ChatTemplate::from_str_arg("few-shot", &tokenizer),
                ChatTemplate::FewShot
            );
            assert_eq!(
                ChatTemplate::from_str_arg("mistral", &tokenizer),
                ChatTemplate::MistralInstruct
            );
            assert_eq!(
                ChatTemplate::from_str_arg("llama3", &tokenizer),
                ChatTemplate::Llama3Instruct
            );
        }

        #[test]
        fn from_str_arg_auto_and_unknown_fall_back_to_detection() {
            let mistral_tok = stub_tokenizer(&["[INST]"]);
            assert_eq!(
                ChatTemplate::from_str_arg("auto", &mistral_tok),
                ChatTemplate::MistralInstruct
            );
            // Unknown value warns and auto-detects (few-shot without sentinels).
            let plain_tok = stub_tokenizer(&["hello"]);
            assert_eq!(
                ChatTemplate::from_str_arg("not-a-template", &plain_tok),
                ChatTemplate::FewShot
            );
        }
    }
}
