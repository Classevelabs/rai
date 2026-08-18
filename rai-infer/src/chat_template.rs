//! Chat template formatting for instruction-tuned models.
//!
//! Supports:
//! - `MistralInstruct`: `<s>[INST] {msg} [/INST]` (Mistral-7B-Instruct)
//! - `Llama3Instruct`: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`
//! - `ChatML`: `<|im_start|>user ... <|im_start|>assistant` (Qwen, many others)
//! - `Zephyr`: `<|user|> ... </s> <|assistant|>` (TinyLlama-Chat, Zephyr, StableLM)
//! - `Phi3`: `<|user|> ... <|end|> <|assistant|>` (Phi-3, Phi-3.5)
//! - `Gemma`: `<start_of_turn>user ... <start_of_turn>model` (Gemma 1/2/3)
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
    /// ChatML: `<|im_start|>user ... <|im_end|>` (Qwen and many fine-tunes)
    ChatML,
    /// Zephyr-style: `<|user|> ... </s> <|assistant|>` (TinyLlama-Chat, Zephyr)
    Zephyr,
    /// Phi-3: `<|user|> ... <|end|> <|assistant|>`.
    ///
    /// Shares Zephyr's role markers but ends a turn with `<|end|>` rather than
    /// `</s>`. It needs its own variant because formatting Phi-3 as Zephyr
    /// produces a prompt shape it was never trained on and leaves generation
    /// with a stop sequence the model never emits.
    Phi3,
    /// Gemma 1/2/3: `<start_of_turn>user ... <end_of_turn><start_of_turn>model`.
    Gemma,
}

/// Who said a turn. The template decides how each side is spelled — Gemma
/// writes the assistant as `model`, Llama-3 as `assistant` — so callers hand
/// over the role, never the marker text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl ChatTemplate {
    /// Format a user message into a model-ready prompt.
    ///
    /// Single-turn shorthand for [`format_conversation`](Self::format_conversation);
    /// both produce identical bytes for a one-message thread.
    pub fn format_prompt(&self, user_message: &str) -> String {
        self.format_conversation(&[(Role::User, user_message)])
    }

    /// Format a whole conversation into a model-ready prompt, ending with the
    /// opener for the assistant's next turn.
    ///
    /// `turns` is ordered oldest first and normally ends with the user message
    /// being answered. Each family renders history the way its reference
    /// template does: Llama-3 repeats header blocks, Mistral closes an
    /// assistant turn with `</s>` before the next `[INST]`, ChatML/Zephyr/
    /// Phi-3/Gemma repeat their turn markers. `None` has no turn syntax to
    /// spell, so the texts are joined with newlines.
    pub fn format_conversation(&self, turns: &[(Role, &str)]) -> String {
        match self {
            ChatTemplate::None => turns
                .iter()
                .map(|&(_, text)| text)
                .collect::<Vec<_>>()
                .join("\n"),
            ChatTemplate::FewShot => {
                let mut out = String::from(
                    "User: Hello!\n\
                     Assistant: Hello! How can I help you today?\n\
                     User: What is the capital of France?\n\
                     Assistant: The capital of France is Paris.\n",
                );
                for &(role, text) in turns {
                    match role {
                        Role::User => out.push_str("User: "),
                        Role::Assistant => out.push_str("Assistant: "),
                    }
                    out.push_str(text);
                    out.push('\n');
                }
                out.push_str("Assistant:");
                out
            }
            ChatTemplate::MistralInstruct => {
                let mut out = String::from("<s>");
                for &(role, text) in turns {
                    match role {
                        Role::User => {
                            out.push_str("[INST] ");
                            out.push_str(text);
                            out.push_str(" [/INST]");
                        }
                        Role::Assistant => {
                            out.push_str(text);
                            out.push_str("</s>");
                        }
                    }
                }
                out
            }
            ChatTemplate::Llama3Instruct => {
                let mut out = String::from("<|begin_of_text|>");
                for &(role, text) in turns {
                    let name = match role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    out.push_str("<|start_header_id|>");
                    out.push_str(name);
                    out.push_str("<|end_header_id|>\n\n");
                    out.push_str(text);
                    out.push_str("<|eot_id|>");
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }
            ChatTemplate::ChatML => {
                let mut out = String::new();
                for &(role, text) in turns {
                    let name = match role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    };
                    out.push_str("<|im_start|>");
                    out.push_str(name);
                    out.push('\n');
                    out.push_str(text);
                    out.push_str("<|im_end|>\n");
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
            ChatTemplate::Zephyr => {
                let mut out = String::new();
                for &(role, text) in turns {
                    out.push_str(match role {
                        Role::User => "<|user|>\n",
                        Role::Assistant => "<|assistant|>\n",
                    });
                    out.push_str(text);
                    out.push_str("</s>\n");
                }
                out.push_str("<|assistant|>\n");
                out
            }
            ChatTemplate::Phi3 => {
                let mut out = String::new();
                for &(role, text) in turns {
                    out.push_str(match role {
                        Role::User => "<|user|>\n",
                        Role::Assistant => "<|assistant|>\n",
                    });
                    out.push_str(text);
                    out.push_str("<|end|>\n");
                }
                out.push_str("<|assistant|>\n");
                out
            }
            ChatTemplate::Gemma => {
                let mut out = String::new();
                for &(role, text) in turns {
                    out.push_str(match role {
                        Role::User => "<start_of_turn>user\n",
                        Role::Assistant => "<start_of_turn>model\n",
                    });
                    out.push_str(text);
                    out.push_str("<end_of_turn>\n");
                }
                out.push_str("<start_of_turn>model\n");
                out
            }
        }
    }

    /// Stop sequences for this template.
    pub fn stop_sequences(&self) -> &'static [&'static str] {
        match self {
            ChatTemplate::None => &[],
            ChatTemplate::FewShot => &["\nUser:", "\nHuman:", "\nStudent", "\nQ:", "\n\n\n"],
            ChatTemplate::MistralInstruct => &["</s>", "[INST]"],
            ChatTemplate::Llama3Instruct => &["<|eot_id|>", "<|end_of_text|>"],
            ChatTemplate::ChatML => &["<|im_end|>", "<|im_start|>"],
            ChatTemplate::Zephyr => &["</s>", "<|user|>"],
            ChatTemplate::Phi3 => &["<|end|>", "<|user|>", "<|endoftext|>"],
            ChatTemplate::Gemma => &["<end_of_turn>", "<start_of_turn>"],
        }
    }

    /// Whether the tokenizer should prepend its beginning-of-sequence token.
    ///
    /// Most templates spell their own opening marker into the prompt text
    /// (`<s>` for Mistral, `<|begin_of_text|>` for Llama-3), so asking the
    /// tokenizer for one as well would emit it twice. Gemma's reference
    /// template starts with a `<bos>` that is *not* part of the turn markers,
    /// and Gemma without it degrades badly rather than failing outright.
    pub fn prepends_bos(&self) -> bool {
        matches!(self, ChatTemplate::Gemma)
    }

    /// Display name for the UI.
    pub fn display_name(&self) -> &'static str {
        match self {
            ChatTemplate::None => "None",
            ChatTemplate::FewShot => "Few-Shot",
            ChatTemplate::MistralInstruct => "Mistral Instruct",
            ChatTemplate::Llama3Instruct => "Llama-3 Instruct",
            ChatTemplate::ChatML => "ChatML",
            ChatTemplate::Zephyr => "Zephyr",
            ChatTemplate::Phi3 => "Phi-3",
            ChatTemplate::Gemma => "Gemma",
        }
    }

    /// Auto-detect template from tokenizer vocabulary.
    /// Checks for sentinel tokens that identify the model family.
    ///
    /// Only available with the default `cli` feature — template formatting
    /// itself has no tokenizer dependency.
    #[cfg(feature = "cli")]
    pub fn auto_detect(tokenizer: &tokenizers::Tokenizer) -> Self {
        // Gemma's turn markers are unique to it, so the order does not matter
        // for correctness; it goes first because the check is one lookup and a
        // Gemma tokenizer carries no other family's sentinel.
        if tokenizer.token_to_id("<start_of_turn>").is_some() {
            return ChatTemplate::Gemma;
        }
        // Check for Mistral-Instruct sentinel: [INST] token
        if tokenizer.token_to_id("[INST]").is_some() {
            return ChatTemplate::MistralInstruct;
        }
        // Check for Llama-3-Instruct sentinel: <|start_header_id|>
        if tokenizer.token_to_id("<|start_header_id|>").is_some() {
            return ChatTemplate::Llama3Instruct;
        }
        // ChatML sentinel: <|im_start|> (Qwen and many fine-tunes)
        if tokenizer.token_to_id("<|im_start|>").is_some() {
            return ChatTemplate::ChatML;
        }
        // Phi-3 is tested before Zephyr: it carries <|assistant|> too, so
        // checking Zephyr's sentinel first would silently select a template
        // whose turn terminator (</s>) Phi-3 never emits. <|end|> separates
        // them.
        if tokenizer.token_to_id("<|end|>").is_some()
            && tokenizer.token_to_id("<|assistant|>").is_some()
        {
            return ChatTemplate::Phi3;
        }
        // Zephyr sentinel: <|assistant|> (TinyLlama-Chat, Zephyr, StableLM-Zephyr)
        if tokenizer.token_to_id("<|assistant|>").is_some() {
            return ChatTemplate::Zephyr;
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
            "chatml" => ChatTemplate::ChatML,
            "zephyr" => ChatTemplate::Zephyr,
            "phi3" => ChatTemplate::Phi3,
            "gemma" => ChatTemplate::Gemma,
            _ => {
                eprintln!("warning: unknown chat template '{s}'; using auto-detect");
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

    #[test]
    fn format_prompt_chatml_uses_im_markers() {
        let prompt = ChatTemplate::ChatML.format_prompt("What is 2+2?");
        assert_eq!(
            prompt,
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn format_prompt_zephyr_closes_the_user_turn_and_opens_the_assistant_turn() {
        let prompt = ChatTemplate::Zephyr.format_prompt("What is 2+2?");
        assert_eq!(prompt, "<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n");
    }

    #[test]
    fn gemma_uses_its_own_turn_markers() {
        assert_eq!(
            ChatTemplate::Gemma.format_prompt("What is 2+2?"),
            "<start_of_turn>user
What is 2+2?<end_of_turn>
<start_of_turn>model
"
        );
        assert!(ChatTemplate::Gemma
            .stop_sequences()
            .contains(&"<end_of_turn>"));
    }

    /// A one-message thread through `format_conversation` must be the same
    /// bytes `format_prompt` produced before multi-turn existed — the serve
    /// path switched to the conversation form and single-turn requests must
    /// not shift by a byte under a model that was working.
    #[test]
    fn single_turn_conversation_matches_format_prompt() {
        let all = [
            ChatTemplate::None,
            ChatTemplate::FewShot,
            ChatTemplate::MistralInstruct,
            ChatTemplate::Llama3Instruct,
            ChatTemplate::ChatML,
            ChatTemplate::Zephyr,
            ChatTemplate::Phi3,
            ChatTemplate::Gemma,
        ];
        for template in all {
            assert_eq!(
                template.format_conversation(&[(Role::User, "What is 2+2?")]),
                template.format_prompt("What is 2+2?"),
                "{}",
                template.display_name()
            );
        }
    }

    #[test]
    fn multi_turn_llama3_repeats_header_blocks() {
        let turns = [
            (Role::User, "Hi"),
            (Role::Assistant, "Hello!"),
            (Role::User, "What is 2+2?"),
        ];
        assert_eq!(
            ChatTemplate::Llama3Instruct.format_conversation(&turns),
            "<|begin_of_text|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn multi_turn_mistral_closes_assistant_turns_with_eos() {
        let turns = [
            (Role::User, "Hi"),
            (Role::Assistant, "Hello!"),
            (Role::User, "What is 2+2?"),
        ];
        assert_eq!(
            ChatTemplate::MistralInstruct.format_conversation(&turns),
            "<s>[INST] Hi [/INST]Hello!</s>[INST] What is 2+2? [/INST]"
        );
    }

    #[test]
    fn multi_turn_chatml_repeats_im_blocks() {
        let turns = [
            (Role::User, "Hi"),
            (Role::Assistant, "Hello!"),
            (Role::User, "What is 2+2?"),
        ];
        assert_eq!(
            ChatTemplate::ChatML.format_conversation(&turns),
            "<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nHello!<|im_end|>\n\
             <|im_start|>user\nWhat is 2+2?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn multi_turn_gemma_spells_the_assistant_as_model() {
        let turns = [
            (Role::User, "Hi"),
            (Role::Assistant, "Hello!"),
            (Role::User, "More"),
        ];
        assert_eq!(
            ChatTemplate::Gemma.format_conversation(&turns),
            "<start_of_turn>user\nHi<end_of_turn>\n\
             <start_of_turn>model\nHello!<end_of_turn>\n\
             <start_of_turn>user\nMore<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    /// Phi-3 and Zephyr differ by one token, and picking the wrong one costs
    /// nothing at load time and everything at generation time: the prompt is
    /// malformed and no stop sequence ever fires, so the model runs to
    /// max-tokens emitting turn markers.
    #[test]
    fn phi3_ends_its_user_turn_with_end_not_eos() {
        let prompt = ChatTemplate::Phi3.format_prompt("What is 2+2?");
        assert_eq!(prompt, "<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n");
        assert_ne!(
            ChatTemplate::Phi3.format_prompt("x"),
            ChatTemplate::Zephyr.format_prompt("x"),
            "Phi-3 must not collapse into the Zephyr template"
        );
        assert!(
            ChatTemplate::Phi3.stop_sequences().contains(&"<|end|>"),
            "generation would never stop"
        );
    }

    #[test]
    fn new_templates_stop_on_their_own_turn_markers() {
        assert_eq!(
            ChatTemplate::ChatML.stop_sequences(),
            &["<|im_end|>", "<|im_start|>"]
        );
        assert_eq!(ChatTemplate::Zephyr.stop_sequences(), &["</s>", "<|user|>"]);
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
        fn auto_detect_recognizes_chatml_and_zephyr_sentinels() {
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["<|im_start|>"])),
                ChatTemplate::ChatML
            );
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["<|assistant|>"])),
                ChatTemplate::Zephyr
            );
            // Mistral and Llama-3 sentinels still win over the newer families.
            assert_eq!(
                ChatTemplate::auto_detect(&stub_tokenizer(&["[INST]", "<|im_start|>"])),
                ChatTemplate::MistralInstruct
            );
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
