#[allow(non_camel_case_types)]
pub enum ModelTypes {
    FLAN_T5_XXL,
    FLAN_UL2,
    MT0_XXL,
    GPT_NEOX,
    MPT_7B_INSTRUCT2,
    STARCODER,
    LLAMA_2_70B_CHAT,
    LLAMA_2_13B_CHAT,
    GRANITE_13B_INSTRUCT,
    GRANITE_13B_CHAT,
    FLAN_T5_XL,
    GRANITE_13B_CHAT_V2,
    GRANITE_13B_INSTRUCT_V2,
}

impl ModelTypes {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelTypes::FLAN_T5_XXL => "google/flan-t5-xxl",
            ModelTypes::FLAN_UL2 => "google/flan-ul2",
            ModelTypes::MT0_XXL => "bigscience/mt0-xxl",
            ModelTypes::GPT_NEOX => "eleutherai/gpt-neox-20b",
            ModelTypes::MPT_7B_INSTRUCT2 => "ibm/mpt-7b-instruct2",
            ModelTypes::STARCODER => "bigcode/starcoder",
            ModelTypes::LLAMA_2_70B_CHAT => "meta-llama/llama-2-70b-chat",
            ModelTypes::LLAMA_2_13B_CHAT => "meta-llama/llama-2-13b-chat",
            ModelTypes::GRANITE_13B_INSTRUCT => "ibm/granite-13b-instruct-v1",
            ModelTypes::GRANITE_13B_CHAT => "ibm/granite-13b-chat-v1",
            ModelTypes::FLAN_T5_XL => "google/flan-t5-xl",
            ModelTypes::GRANITE_13B_CHAT_V2 => "ibm/granite-13b-chat-v2",
            ModelTypes::GRANITE_13B_INSTRUCT_V2 => "ibm/granite-13b-instruct-v2",
        }
    }

    pub fn to_string(&self) -> String {
        self.as_str().to_string()
    }
}

pub enum DecodingMethods {
    SAMPLE,
    GREEDY,
}

impl DecodingMethods {
    pub fn as_str(&self) -> &'static str {
        match self {
            DecodingMethods::SAMPLE => "sample",
            DecodingMethods::GREEDY => "greedy",
        }
    }
}

#[allow(non_snake_case)]
pub struct GenTextParamsMetaNames {
    pub DECODING_METHOD: String,
    pub LENGTH_PENALTY: LengthPenalty,
    pub TEMPERATURE: f64,
    pub TOP_P: f64,
    pub TOP_K: u64,
    pub RANDOM_SEED: u64,
    pub REPETITION_PENALTY: f64,
    pub MIN_NEW_TOKENS: u64,
    pub MAX_NEW_TOKENS: u64,
    pub STOP_SEQUENCES: Vec<String>,
    pub TIME_LIMIT: u64,
    pub TRUNCATE_INPUT_TOKENS: u64,
    pub RETURN_OPTIONS: ReturnOptions,
}

impl Default for GenTextParamsMetaNames {
    fn default() -> Self {
        GenTextParamsMetaNames {
            DECODING_METHOD: "sample".to_string(),
            LENGTH_PENALTY: LengthPenalty::default(),
            TEMPERATURE: 0.5,
            TOP_P: 0.2,
            TOP_K: 1,
            RANDOM_SEED: 33,
            REPETITION_PENALTY: 2.0,
            MIN_NEW_TOKENS: 50,
            MAX_NEW_TOKENS: 200,
            STOP_SEQUENCES: vec!["fail".to_string()],
            TIME_LIMIT: 600000,
            TRUNCATE_INPUT_TOKENS: 200,
            RETURN_OPTIONS: ReturnOptions::default(),
        }
    }
}

pub struct LengthPenalty {
    pub decay_factor: f64,
    pub start_index: u64,
}

impl Default for LengthPenalty {
    fn default() -> Self {
        LengthPenalty {
            decay_factor: 2.5,
            start_index: 5,
        }
    }
}

pub struct ReturnOptions {
    pub input_text: bool,
    pub generated_tokens: bool,
    pub input_tokens: bool,
    pub token_logprobs: bool,
    pub token_ranks: bool,
    pub top_n_tokens: bool,
}

impl Default for ReturnOptions {
    fn default() -> Self {
        ReturnOptions {
            input_text: true,
            generated_tokens: true,
            input_tokens: true,
            token_logprobs: true,
            token_ranks: false,
            top_n_tokens: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gentextparams() {
        let genparams = GenTextParamsMetaNames::default();
        assert_eq!(genparams.LENGTH_PENALTY.decay_factor, 2.5);
    }
}
