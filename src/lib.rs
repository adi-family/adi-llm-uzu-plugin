//! ADI Uzu LLM Plugin
//!
//! Provides local LLM inference on Apple Silicon using the Uzu engine.
//! Optimized for M1/M2/M3 chips with Metal acceleration.

use lib_client_uzu::{Client, GenerateRequest};
use lib_plugin_abi_v3::{
    async_trait,
    cli::{CliCommand, CliCommands, CliContext, CliResult},
    Plugin, PluginContext, PluginMetadata, PluginType, Result as PluginResult, SERVICE_CLI_COMMANDS,
};
use once_cell::sync::Mutex;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

/// Loaded models (path -> Client)
static MODELS: Mutex<Option<HashMap<String, Client>>> = Mutex::new(None);

/// Uzu LLM Plugin
pub struct UzuLlmPlugin;

impl UzuLlmPlugin {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UzuLlmPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Plugin for UzuLlmPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            id: "adi.llm.uzu".to_string(),
            name: "ADI Uzu LLM".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            plugin_type: PluginType::Extension,
            author: Some("ADI Team".to_string()),
            description: Some("Local LLM inference on Apple Silicon using Uzu engine".to_string()),
            category: None,
        }
    }

    async fn init(&mut self, _ctx: &PluginContext) -> PluginResult<()> {
        // Initialize models hashmap
        *MODELS.lock().unwrap() = Some(HashMap::new());
        Ok(())
    }

    async fn shutdown(&self) -> PluginResult<()> {
        // Clear loaded models
        if let Ok(mut models) = MODELS.lock() {
            *models = None;
        }
        Ok(())
    }

    fn provides(&self) -> Vec<&'static str> {
        vec![SERVICE_CLI_COMMANDS]
    }
}

#[async_trait]
impl CliCommands for UzuLlmPlugin {
    async fn list_commands(&self) -> Vec<CliCommand> {
        vec![
            CliCommand {
                name: "load".to_string(),
                description: "Load a model".to_string(),
                usage: "load <model-path>".to_string(),
                has_subcommands: false,
            },
            CliCommand {
                name: "unload".to_string(),
                description: "Unload a model".to_string(),
                usage: "unload <model-path>".to_string(),
                has_subcommands: false,
            },
            CliCommand {
                name: "list".to_string(),
                description: "List loaded models".to_string(),
                usage: "list".to_string(),
                has_subcommands: false,
            },
            CliCommand {
                name: "generate".to_string(),
                description: "Generate text".to_string(),
                usage: "generate <model-path> <prompt> [--max-tokens <n>] [--temperature <t>]".to_string(),
                has_subcommands: false,
            },
            CliCommand {
                name: "info".to_string(),
                description: "Show model info".to_string(),
                usage: "info <model-path>".to_string(),
                has_subcommands: false,
            },
        ]
    }

    async fn run_command(&self, ctx: &CliContext) -> PluginResult<CliResult> {
        let subcommand = ctx.subcommand.as_deref().unwrap_or("");
        let args: Vec<&str> = ctx.args.iter().map(|s| s.as_str()).collect();
        let options = ctx.options_as_json();

        let result = match subcommand {
            "load" => {
                if args.is_empty() {
                    Err("Usage: load <model-path>".to_string())
                } else {
                    load_model(args[0]).map(|_| format!("Model loaded: {}", args[0]))
                }
            }
            "unload" => {
                if args.is_empty() {
                    Err("Usage: unload <model-path>".to_string())
                } else {
                    unload_model(args[0]).map(|_| format!("Model unloaded: {}", args[0]))
                }
            }
            "list" => {
                let models = list_models();
                serde_json::to_string(&models).map_err(|e| e.to_string())
            }
            "generate" => {
                if args.len() < 2 {
                    Err("Usage: generate <model-path> <prompt> [--max-tokens <n>]".to_string())
                } else {
                    let path = args[0];
                    let prompt = args[1..].join(" ");
                    let max_tokens = options
                        .get("max-tokens")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok());
                    let temperature = options
                        .get("temperature")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok());
                    generate_text(path, &prompt, max_tokens, temperature)
                }
            }
            "info" => {
                if args.is_empty() {
                    Err("Usage: info <model-path>".to_string())
                } else {
                    get_model_info(args[0])
                }
            }
            "" | "help" => Ok(get_help()),
            _ => Err(format!("Unknown command: {}", subcommand)),
        };

        match result {
            Ok(output) => Ok(CliResult::success(output)),
            Err(e) => Ok(CliResult::error(e)),
        }
    }
}

/// Create the plugin instance (v3 entry point)
#[no_mangle]
pub fn plugin_create() -> Box<dyn Plugin> {
    Box::new(UzuLlmPlugin::new())
}

/// Create the CLI commands interface
#[no_mangle]
pub fn plugin_create_cli() -> Box<dyn CliCommands> {
    Box::new(UzuLlmPlugin::new())
}

// === Helper Functions ===

fn get_help() -> String {
    r#"ADI Uzu LLM - Local LLM inference on Apple Silicon

Commands:
  load <model-path>           Load a model
  unload <model-path>         Unload a model
  list                        List loaded models
  generate <path> <prompt>    Generate text
  info <model-path>           Show model info

Options:
  --max-tokens <n>            Maximum tokens to generate
  --temperature <t>           Sampling temperature

Examples:
  adi llm-uzu load models/llama-3.2-1b.gguf
  adi llm-uzu generate models/llama-3.2-1b.gguf "Tell me about Rust""#
        .to_string()
}

fn load_model(path: &str) -> Result<(), String> {
    let mut models = MODELS
        .lock()
        .map_err(|e| format!("Failed to lock models: {}", e))?;

    let models_map = models
        .as_mut()
        .ok_or_else(|| "Models not initialized".to_string())?;

    if models_map.contains_key(path) {
        return Ok(()); // Already loaded
    }

    let client = Client::new(PathBuf::from(path))
        .map_err(|e| format!("Failed to load model: {}", e))?;

    models_map.insert(path.to_string(), client);
    Ok(())
}

fn unload_model(path: &str) -> Result<(), String> {
    let mut models = MODELS
        .lock()
        .map_err(|e| format!("Failed to lock models: {}", e))?;

    let models_map = models
        .as_mut()
        .ok_or_else(|| "Models not initialized".to_string())?;

    models_map
        .remove(path)
        .ok_or_else(|| format!("Model not loaded: {}", path))?;

    Ok(())
}

fn list_models() -> Vec<String> {
    MODELS
        .lock()
        .ok()
        .and_then(|m| m.as_ref().map(|map| map.keys().cloned().collect()))
        .unwrap_or_default()
}

fn generate_text(
    path: &str,
    prompt: &str,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
) -> Result<String, String> {
    // Ensure model is loaded
    load_model(path)?;

    let mut models = MODELS
        .lock()
        .map_err(|e| format!("Failed to lock models: {}", e))?;

    let models_map = models
        .as_mut()
        .ok_or_else(|| "Models not initialized".to_string())?;

    let client = models_map
        .get_mut(path)
        .ok_or_else(|| format!("Model not loaded: {}", path))?;

    let mut request = GenerateRequest::new(prompt);
    if let Some(max) = max_tokens {
        request = request.max_tokens(max);
    }
    if let Some(temp) = temperature {
        request = request.temperature(temp);
    }

    let response = client
        .generate(request)
        .map_err(|e| format!("Generation failed: {}", e))?;

    let result = json!({
        "text": response.text,
        "tokens_generated": response.tokens_generated,
        "stopped": response.stopped,
        "stop_reason": response.stop_reason,
    });

    Ok(serde_json::to_string(&result).unwrap_or_default())
}

fn get_model_info(path: &str) -> Result<String, String> {
    // Ensure model is loaded
    load_model(path)?;

    let models = MODELS
        .lock()
        .map_err(|e| format!("Failed to lock models: {}", e))?;

    let models_map = models
        .as_ref()
        .ok_or_else(|| "Models not initialized".to_string())?;

    let client = models_map
        .get(path)
        .ok_or_else(|| format!("Model not loaded: {}", path))?;

    let info = client.model_info();

    let result = json!({
        "name": info.name,
        "size": info.size,
        "loaded": info.loaded,
    });

    Ok(serde_json::to_string(&result).unwrap_or_default())
}
