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
