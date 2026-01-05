//! ADI Uzu LLM Plugin
//!
//! Provides local LLM inference on Apple Silicon using the Uzu engine.
//! Optimized for M1/M2/M3 chips with Metal acceleration.

use abi_stable::std_types::{ROption, RResult, RStr, RString, RVec};
use lib_plugin_abi::{
    PluginContext, PluginError, PluginInfo, PluginVTable, ServiceDescriptor, ServiceError,
    ServiceHandle, ServiceMethod, ServiceVTable, ServiceVersion,
};
use once_cell::sync::Mutex;
use serde_json::json;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;

use lib_client_uzu::{Client, GenerateRequest};

/// Plugin-specific CLI service ID
const SERVICE_CLI: &str = "adi.llm.uzu.cli";

/// Plugin-specific inference service ID
const SERVICE_INFERENCE: &str = "adi.llm.inference";

/// Loaded models (path -> Client)
static MODELS: Mutex<Option<HashMap<String, Client>>> = Mutex::new(None);

// === Plugin VTable Implementation ===

extern "C" fn plugin_info() -> PluginInfo {
    PluginInfo::new(
        "adi.llm.uzu",
        "ADI Uzu LLM",
        env!("CARGO_PKG_VERSION"),
        "extension",
    )
    .with_author("ADI Team")
    .with_description("Local LLM inference on Apple Silicon using Uzu engine")
    .with_min_host_version("0.8.0")
}

extern "C" fn plugin_init(ctx: *mut PluginContext) -> i32 {
    // Initialize models hashmap
    *MODELS.lock().unwrap() = Some(HashMap::new());

    unsafe {
        let host = (*ctx).host();

        // Register CLI service
        let cli_descriptor =
            ServiceDescriptor::new(SERVICE_CLI, ServiceVersion::new(1, 0, 0), "adi.llm.uzu")
                .with_description("CLI commands for Uzu LLM management");

        let cli_handle = ServiceHandle::new(
            SERVICE_CLI,
            ctx as *const c_void,
            &CLI_SERVICE_VTABLE as *const ServiceVTable,
        );

        if let Err(code) = host.register_svc(cli_descriptor, cli_handle) {
            host.error(&format!("Failed to register CLI service: {}", code));
            return code;
        }

        // Register inference service
        let inference_descriptor = ServiceDescriptor::new(
            SERVICE_INFERENCE,
            ServiceVersion::new(1, 0, 0),
            "adi.llm.uzu",
        )
        .with_description("LLM inference service (Apple Silicon only)");

        let inference_handle = ServiceHandle::new(
            SERVICE_INFERENCE,
            ctx as *const c_void,
            &INFERENCE_SERVICE_VTABLE as *const ServiceVTable,
        );

        if let Err(code) = host.register_svc(inference_descriptor, inference_handle) {
            host.error(&format!("Failed to register inference service: {}", code));
            return code;
        }

        host.info("ADI Uzu LLM plugin initialized");
    }

    0
}

extern "C" fn plugin_cleanup(_ctx: *mut PluginContext) {
    // Clear loaded models
    if let Ok(mut models) = MODELS.lock() {
        *models = None;
    }
}

// === Plugin Entry Point ===

static PLUGIN_VTABLE: PluginVTable = PluginVTable {
    info: plugin_info,
    init: plugin_init,
    update: ROption::RNone,
    cleanup: plugin_cleanup,
    handle_message: ROption::RNone,
};

#[no_mangle]
pub extern "C" fn plugin_entry() -> *const PluginVTable {
    &PLUGIN_VTABLE
}

// === CLI Service VTable ===

static CLI_SERVICE_VTABLE: ServiceVTable = ServiceVTable {
    invoke: cli_invoke,
    list_methods: cli_list_methods,
};

extern "C" fn cli_invoke(
    _handle: *const c_void,
    method: RStr<'_>,
    args: RStr<'_>,
) -> RResult<RString, ServiceError> {
    match method.as_str() {
        "run_command" => {
            let result = run_cli_command(args.as_str());
            match result {
                Ok(output) => RResult::ROk(RString::from(output)),
                Err(e) => RResult::RErr(ServiceError::invocation_error(e)),
            }
        }
        "list_commands" => {
            let commands = json!([
                {"name": "load", "description": "Load a model", "usage": "load <model-path>"},
                {"name": "unload", "description": "Unload a model", "usage": "unload <model-path>"},
                {"name": "list", "description": "List loaded models", "usage": "list"},
                {"name": "generate", "description": "Generate text", "usage": "generate <model-path> <prompt> [--max-tokens <n>] [--temperature <t>]"},
                {"name": "info", "description": "Show model info", "usage": "info <model-path>"}
            ]);
            RResult::ROk(RString::from(
                serde_json::to_string(&commands).unwrap_or_default(),
            ))
        }
        _ => RResult::RErr(ServiceError::method_not_found(method.as_str())),
    }
}

extern "C" fn cli_list_methods(_handle: *const c_void) -> RVec<ServiceMethod> {
    vec![
        ServiceMethod::new("run_command").with_description("Run a CLI command"),
        ServiceMethod::new("list_commands").with_description("List available commands"),
    ]
    .into_iter()
    .collect()
}

// === CLI Command Handler ===

fn run_cli_command(args: &str) -> Result<String, String> {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.is_empty() {
        return Err("No command provided".to_string());
    }

    match parts[0] {
        "load" => {
            if parts.len() < 2 {
                return Err("Usage: load <model-path>".to_string());
            }
            let path = parts[1];
            load_model(path)?;
            Ok(format!("Model loaded: {}", path))
        }
        "unload" => {
            if parts.len() < 2 {
                return Err("Usage: unload <model-path>".to_string());
            }
            let path = parts[1];
            unload_model(path)?;
            Ok(format!("Model unloaded: {}", path))
        }
        "list" => {
            let models = list_models();
            Ok(serde_json::to_string(&models).unwrap_or_default())
        }
        "generate" => {
            if parts.len() < 3 {
                return Err("Usage: generate <model-path> <prompt> [--max-tokens <n>]".to_string());
            }
            let path = parts[1];
            let prompt = parts[2..].join(" ");
            generate_text(path, &prompt, None, None)
        }
        "info" => {
            if parts.len() < 2 {
                return Err("Usage: info <model-path>".to_string());
            }
            let path = parts[1];
            get_model_info(path)
        }
        _ => Err(format!("Unknown command: {}", parts[0])),
    }
}

// === Inference Service VTable ===

static INFERENCE_SERVICE_VTABLE: ServiceVTable = ServiceVTable {
    invoke: inference_invoke,
    list_methods: inference_list_methods,
};

extern "C" fn inference_invoke(
    _handle: *const c_void,
    method: RStr<'_>,
    args: RStr<'_>,
) -> RResult<RString, ServiceError> {
    match method.as_str() {
        "generate" => {
            #[derive(serde::Deserialize)]
            struct GenerateArgs {
                model_path: String,
                prompt: String,
                #[serde(default)]
                max_tokens: Option<usize>,
                #[serde(default)]
                temperature: Option<f32>,
            }

            let args: GenerateArgs = match serde_json::from_str(args.as_str()) {
                Ok(a) => a,
                Err(e) => return RResult::RErr(ServiceError::invocation_error(e.to_string())),
            };

            match generate_text(
                &args.model_path,
                &args.prompt,
                args.max_tokens,
                args.temperature,
            ) {
                Ok(output) => RResult::ROk(RString::from(output)),
                Err(e) => RResult::RErr(ServiceError::invocation_error(e)),
            }
        }
        _ => RResult::RErr(ServiceError::method_not_found(method.as_str())),
    }
}

extern "C" fn inference_list_methods(_handle: *const c_void) -> RVec<ServiceMethod> {
    vec![ServiceMethod::new("generate")
        .with_description("Generate text using loaded model")]
    .into_iter()
    .collect()
}

// === Helper Functions ===

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
