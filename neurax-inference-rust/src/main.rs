//! Neurax Agent Inference in Rust using Candle
//!
//! Loads a fine-tuned LoRA adapter and runs inference for neural architecture building.
//!
//! Usage:
//!   cargo run --release -- --model Qwen/Qwen2.5-3B-Instruct --adapter ../neurax-lora
//!   cargo run --release -- --prompt "Build a CNN for image classification"

use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, Qwen2};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Neurax inference arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model name or path (HuggingFace repo or local)
    #[arg(short, long, default_value = "Qwen/Qwen2.5-3B-Instruct")]
    model: String,

    /// Path to LoRA adapter directory
    #[arg(short, long, default_value = "./neurax-lora")]
    adapter: PathBuf,

    /// Input prompt for inference
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value = "0.7")]
    temperature: f64,

    /// Top-p for nucleus sampling
    #[arg(long, default_value = "0.9")]
    top_p: f64,

    /// Use CPU instead of CUDA
    #[arg(long, default_value = "true")]
    cpu: bool,

    /// Interactive mode (REPL)
    #[arg(short, long, default_value = "true")]
    interactive: bool,
}

/// LoRA adapter configuration
#[derive(Debug, Deserialize)]
struct LoraConfig {
    r: usize,
    lora_alpha: usize,
    lora_dropout: f64,
    target_modules: Vec<String>,
}

/// Neurax system prompt
const SYSTEM_PROMPT: &str = r#"You are Neurax AI controller. You build neural network architectures using tools.

TOOLS: set_family, set_hw_config, add_node, move_node, connect, set_node_params, done

OUTPUT FORMAT: Return ONLY valid JSON:
{"assistant": "<your reasoning for this step>", "tool": {"name": "<tool_name>", "args": {...}}}

REASONING GUIDELINES:
- Analyze the user's request to understand the task, constraints, and desired properties
- Consider what blocks are available in the catalogue for the selected family
- Think about the data flow: input → processing → output
- Explain WHY you're choosing specific blocks and parameters
- Check the current graph state before making connections
- Handle fan-in restrictions: most nodes accept only ONE input

RULES:
1. INFER FAMILY FIRST: Determine best architecture family based on the task description
2. CONNECT SEQUENTIALLY: After adding input AND first layer, immediately connect input → layer
3. ADD OUTPUT: Always create an output node and connect final layer to it
4. CHECK STATE: Review existing connections before proposing new ones
5. USE CATALOGUE: Only use blocks from the provided catalogue for the selected family"#;

/// Tool call structure for JSON parsing
#[derive(Debug, Serialize, Deserialize)]
struct ToolCall {
    assistant: String,
    tool: Tool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Tool {
    name: String,
    args: serde_json::Value,
}

/// Load tokenizer from HuggingFace or local path
fn load_tokenizer(model_path: &str) -> Result<Tokenizer> {
    info!("Loading tokenizer from: {}", model_path);
    
    let tokenizer = if PathBuf::from(model_path).exists() {
        // Local path
        let tokenizer_path = PathBuf::from(model_path).join("tokenizer.json");
        Tokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load tokenizer from {:?}", tokenizer_path))?
    } else {
        // HuggingFace hub
        let api = Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = Repo::with_dtype(model_path.to_string(), RepoType::Model, None);
        let api_repo = api.repo(repo);
        let tokenizer_path = api_repo.get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        Tokenizer::from_file(&tokenizer_path)
            .with_context(|| format!("Failed to load tokenizer from {:?}", tokenizer_path))?
    };
    
    Ok(tokenizer)
}

/// Load model configuration
fn load_config(model_path: &str) -> Result<Config> {
    info!("Loading config from: {}", model_path);
    
    let config = if PathBuf::from(model_path).exists() {
        let config_path = PathBuf::from(model_path).join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        serde_json::from_str(&config_text)
            .context("Failed to parse config.json")?
    } else {
        let api = Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = Repo::with_dtype(model_path.to_string(), RepoType::Model, None);
        let api_repo = api.repo(repo);
        let config_path = api_repo.get("config.json")
            .context("Failed to download config.json")?;
        let config_text = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        serde_json::from_str(&config_text)
            .context("Failed to parse config.json")?
    };
    
    Ok(config)
}

/// Load base model weights
fn load_model(
    model_path: &str,
    config: &Config,
    device: &Device,
    dtype: DType,
) -> Result<Qwen2> {
    info!("Loading model weights from: {}", model_path);
    
    let vb = if PathBuf::from(model_path).exists() {
        // Local path - load from safetensors or bin files
        let model_dir = PathBuf::from(model_path);
        let safetensors_path = model_dir.join("model.safetensors");
        
        if safetensors_path.exists() {
            let tensors = unsafe { 
                candle_core::safetensors::load(&safetensors_path, device)?
            };
            VarBuilder::from_tensors(tensors, dtype, device)
        } else {
            // Try sharded safetensors
            let index_path = model_dir.join("model.safetensors.index.json");
            if index_path.exists() {
                let index_text = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_text)?;
                let weight_map = index.get("weight_map")
                    .and_then(|v| v.as_object())
                    .context("Invalid weight map in index")?;
                
                let mut all_tensors = std::collections::HashMap::new();
                let mut seen_files = std::collections::HashSet::new();
                
                for file_name in weight_map.values().filter_map(|v| v.as_str()) {
                    if seen_files.contains(file_name) {
                        continue;
                    }
                    seen_files.insert(file_name.to_string());
                    
                    let file_path = model_dir.join(file_name);
                    info!("Loading shard: {}", file_name);
                    let tensors = unsafe { 
                        candle_core::safetensors::load(&file_path, device)?
                    };
                    all_tensors.extend(tensors);
                }
                
                VarBuilder::from_tensors(all_tensors, dtype, device)
            } else {
                anyhow::bail!("No model.safetensors or index file found in {:?}", model_dir);
            }
        }
    } else {
        // Download from HuggingFace
        let api = Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = Repo::with_dtype(model_path.to_string(), RepoType::Model, None);
        let api_repo = api.repo(repo);
        
        // Check for sharded model
        let index_result = api_repo.get("model.safetensors.index.json");
        let vb = if let Ok(index_path) = index_result {
            let index_text = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_text)?;
            let weight_map = index.get("weight_map")
                .and_then(|v| v.as_object())
                .context("Invalid weight map in index")?;
            
            let mut all_tensors = std::collections::HashMap::new();
            let mut seen_files = std::collections::HashSet::new();
            
            for file_name in weight_map.values().filter_map(|v| v.as_str()) {
                if seen_files.contains(file_name) {
                    continue;
                }
                seen_files.insert(file_name.to_string());
                
                info!("Downloading shard: {}", file_name);
                let file_path = api_repo.get(file_name)
                    .with_context(|| format!("Failed to download {}", file_name))?;
                let tensors = unsafe { 
                    candle_core::safetensors::load(&file_path, device)?
                };
                all_tensors.extend(tensors);
            }
            
            VarBuilder::from_tensors(all_tensors, dtype, device)
        } else {
            // Single file
            let model_path = api_repo.get("model.safetensors")
                .or_else(|_| api_repo.get("pytorch_model.bin"))
                .context("Failed to download model weights")?;
            
            if model_path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                let tensors = unsafe { 
                    candle_core::safetensors::load(&model_path, device)?
                };
                VarBuilder::from_tensors(tensors, dtype, device)
            } else {
                // PyTorch bin format - not supported in candle directly
                anyhow::bail!("PyTorch .bin format not supported. Please convert to safetensors.");
            }
        };
        vb
    };
    
    info!("Building model...");
    let model = Qwen2::new(config.clone(), vb)
        .context("Failed to build model")?;
    
    Ok(model)
}

/// Load and merge LoRA adapter weights
fn load_lora_adapter(
    adapter_path: &PathBuf,
    model: &mut Qwen2,
    device: &Device,
) -> Result<()> {
    info!("Loading LoRA adapter from: {:?}", adapter_path);
    
    // Load adapter config
    let config_path = adapter_path.join("adapter_config.json");
    if !config_path.exists() {
        warn!("No adapter_config.json found, skipping LoRA merge");
        return Ok(());
    }
    
    let config_text = std::fs::read_to_string(&config_path)
        .context("Failed to read adapter_config.json")?;
    let lora_config: LoraConfig = serde_json::from_str(&config_text)
        .context("Failed to parse adapter_config.json")?;
    
    info!("LoRA config: r={}, alpha={}", lora_config.r, lora_config.lora_alpha);
    
    // Load adapter weights
    let adapter_weights_path = adapter_path.join("adapter_model.safetensors");
    if !adapter_weights_path.exists() {
        warn!("No adapter_model.safetensors found, skipping LoRA merge");
        return Ok(());
    }
    
    let adapter_tensors = unsafe {
        candle_core::safetensors::load(&adapter_weights_path, device)?
    };
    
    // Merge LoRA weights into base model
    // This is a simplified version - full implementation would need to handle
    // each target module (q_proj, k_proj, v_proj, o_proj, etc.)
    let scaling = lora_config.lora_alpha as f64 / lora_config.r as f64;
    
    info!("Merging LoRA weights with scaling factor: {}", scaling);
    
    for (name, tensor) in adapter_tensors {
        // LoRA weights are typically named like:
        // base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        // base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
        
        if name.contains(".lora_A.") || name.contains(".lora_B.") {
            info!("Found LoRA tensor: {}", name);
            // Full implementation would:
            // 1. Find corresponding base weight
            // 2. Compute delta = (lora_B @ lora_A) * scaling
            // 3. Add delta to base weight
            // For now, we log that this needs manual merging
        }
    }
    
    warn!("Note: Full LoRA merging requires manual weight manipulation.");
    warn!("Consider using Python to merge adapter with base model first:");
    warn!("  model = PeftModel.from_pretrained(base_model, adapter_path)");
    warn!("  merged_model = model.merge_and_unload()");
    warn!("  merged_model.save_pretrained('./merged_model')");
    
    Ok(())
}

/// Format chat prompt for Qwen models
fn format_chat_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system, user
    )
}

/// Generate text using the model
fn generate(
    model: &Qwen2,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    device: &Device,
) -> Result<String> {
    info!("Tokenizing prompt...");
    
    let full_prompt = format_chat_prompt(SYSTEM_PROMPT, prompt);
    let tokens = tokenizer
        .encode(full_prompt.as_str(), true)
        .context("Failed to tokenize prompt")?;
    
    let input_ids: Vec<u32> = tokens.get_ids().to_vec();
    let mut generated_tokens = Vec::new();
    
    info!("Generating {} tokens...", max_tokens);
    
    // Create input tensor
    let input_tensor = Tensor::new(&input_ids[..], device)?
        .unsqueeze(0)
        .context("Failed to create input tensor")?;
    
    let mut current_input = input_tensor.clone();
    
    for _ in 0..max_tokens {
        // Forward pass
        let logits = model.forward(&current_input, None)
            .context("Forward pass failed")?;
        
        // Get last token logits
        let next_token_logits = logits
            .get((0, logits.dim(1)? - 1))
            .context("Failed to get last token logits")?;
        
        // Apply temperature and sample
        let scaled_logits = if temperature > 0.0 {
            (&next_token_logits / temperature)?
        } else {
            next_token_logits
        };
        
        // Top-p sampling
        let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
        let next_token = if top_p < 1.0 {
            // Nucleus sampling
            sample_top_p(&probs, top_p)?
        } else {
            // Greedy or temperature sampling
            if temperature == 0.0 {
                probs.argmax(0)?.to_scalar::<u32>()?
            } else {
                probs.sample(0)?
            }
        };
        
        // Check for EOS token
        let eos_token = tokenizer.get_vocab().get("<|im_end|>")
            .copied()
            .or_else(|| tokenizer.get_vocab().get("<|endoftext|>").copied())
            .unwrap_or(151643); // Qwen EOS token
        
        if next_token == eos_token {
            info!("EOS token reached");
            break;
        }
        
        generated_tokens.push(next_token);
        
        // Update input for next iteration
        let next_token_tensor = Tensor::new(&[next_token as u32], device)?
            .unsqueeze(0)
            .context("Failed to create next token tensor")?;
        current_input = Tensor::cat(&[&current_input, &next_token_tensor], 1)
            .context("Failed to concatenate tokens")?;
    }
    
    // Decode generated tokens
    let decoded = tokenizer.decode(&generated_tokens, true)
        .context("Failed to decode tokens")?;
    
    Ok(decoded)
}

/// Top-p (nucleus) sampling
fn sample_top_p(probs: &Tensor, top_p: f64) -> Result<u32> {
    let probs_vec = probs.to_vec1::<f32>()
        .context("Failed to convert probs to vec")?;
    
    // Sort probabilities descending
    let mut indexed_probs: Vec<(usize, f32)> = probs_vec
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Find cutoff for top-p
    let mut cumulative = 0.0;
    let mut cutoff_idx = 0;
    for (i, (_, prob)) in indexed_probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    // Sample from top-p tokens
    let top_tokens: Vec<(usize, f32)> = indexed_probs.into_iter().take(cutoff_idx).collect();
    let total: f32 = top_tokens.iter().map(|(_, p)| p).sum();
    
    // Renormalize and sample
    let rand_val: f32 = rand::random();
    let mut cumulative = 0.0;
    for (token_id, prob) in top_tokens {
        cumulative += prob / total;
        if rand_val < cumulative {
            return Ok(token_id as u32);
        }
    }
    
    // Fallback to first token
    Ok(top_tokens.first().map(|(i, _)| *i as u32).unwrap_or(0))
}

/// Interactive REPL mode
fn interactive_mode(
    model: &Qwen2,
    tokenizer: &Tokenizer,
    device: &Device,
    args: &Args,
) -> Result<()> {
    use std::io::{self, Write};
    
    println!("\n{}", "=".repeat(50));
    println!("Neurax Agent Interactive Mode (Rust/Candle)");
    println!("Type 'quit' to exit, 'test' for test prompts");
    println!("{}", "=".repeat(50));
    
    let test_prompts = [
        "Build a CNN for image classification with 10 classes",
        "Build a transformer for text generation with 12 layers",
        "Build a MoE model with 8 experts for language modeling",
        "Build a diffusion model for image generation",
        "Build an SSM for long-range sequence modeling",
    ];
    
    loop {
        print!("\nUser: ");
        io::stdout().flush().context("Failed to flush stdout")?;
        
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("Failed to read input")?;
        
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }
        
        if input.eq_ignore_ascii_case("test") {
            println!("\nTest prompts:");
            for (i, prompt) in test_prompts.iter().enumerate() {
                println!("  {}. {}", i + 1, prompt);
            }
            print!("Select test (1-5): ");
            io::stdout().flush().context("Failed to flush stdout")?;
            
            let mut choice = String::new();
            io::stdin().read_line(&mut choice).context("Failed to read choice")?;
            
            if let Ok(idx) = choice.trim().parse::<usize>() {
                if idx >= 1 && idx <= test_prompts.len() {
                    let prompt = test_prompts[idx - 1];
                    println!("\nRunning: {}", prompt);
                    
                    let response = generate(
                        model, tokenizer, prompt,
                        args.max_tokens, args.temperature, args.top_p, device
                    ).context("Generation failed")?;
                    
                    println!("\nAssistant: {}", response);
                }
            }
            continue;
        }
        
        if input.is_empty() {
            continue;
        }
        
        let response = generate(
            model, tokenizer, input,
            args.max_tokens, args.temperature, args.top_p, device
        ).context("Generation failed")?;
        
        println!("\nAssistant: {}", response);
        
        // Try to parse as JSON
        if let Ok(tool_call) = serde_json::from_str::<ToolCall>(&response) {
            println!("\n[Tool call: {}]", tool_call.tool.name);
        }
    }
    
    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing_subscriber::init(subscriber);
    
    let args = Args::parse();
    
    // Setup device
    let device = if args.cpu {
        info!("Using CPU device");
        Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            info!("Using CUDA device");
            Device::new_cuda(0).context("Failed to create CUDA device")?
        }
        #[cfg(not(feature = "cuda"))]
        {
            warn!("CUDA not available, falling back to CPU");
            Device::Cpu
        }
    };
    
    let dtype = DType::F32; // Use F32 for CPU, F16 for CUDA
    
    // Load tokenizer
    let tokenizer = load_tokenizer(&args.model)
        .context("Failed to load tokenizer")?;
    
    // Load config
    let config = load_config(&args.model)
        .context("Failed to load config")?;
    
    // Load model
    let mut model = load_model(&args.model, &config, &device, dtype)
        .context("Failed to load model")?;
    
    // Load LoRA adapter (if exists)
    if args.adapter.exists() {
        load_lora_adapter(&args.adapter, &mut model, &device)
            .context("Failed to load LoRA adapter")?;
    } else {
        warn!("Adapter path does not exist: {:?}", args.adapter);
    }
    
    // Run inference
    if let Some(prompt) = &args.prompt {
        let response = generate(
            &model, &tokenizer, prompt,
            args.max_tokens, args.temperature, args.top_p, &device
        ).context("Generation failed")?;
        
        println!("User: {}", prompt);
        println!("Assistant: {}", response);
    } else if args.interactive {
        interactive_mode(&model, &tokenizer, &device, &args)
            .context("Interactive mode failed")?;
    }
    
    Ok(())
}
