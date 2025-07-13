/**
 * Real AI Model Manager using Transformers.js
 * Handles loading and inference for real Whisper and language models
 */

class ModelManager {
    constructor() {
        this.models = new Map();
        this.config = {
            whisper: {
                modelName: 'Xenova/whisper-tiny.en',
                task: 'automatic-speech-recognition',
                device: 'webgpu', // Try WebGPU first, fallback to WASM
                dtype: 'fp16'
            },
            textgen: {
                modelName: window.selectedModel || 'Xenova/TinyLlama-1.1B-Chat-v1.0', // Use selected model
                task: 'text-generation',
                device: 'webgpu',
                dtype: 'q4'
            }
        };
        
        this.supportedDevices = [];
        this.supportedBackends = [];
        this.currentBackend = null;
        this.isInitialized = false;
        
        // MediaPipe integration
        this.mediaPipeLLM = null;
        this.useMediaPipe = window.selectedBackend === 'mediapipe';
    }

    async initialize() {
        try {
            // Wait for Transformers.js to be available
            while (!window.transformers) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            // Configure Transformers.js environment
            const { env } = window.transformers;
            
            // Set cache directory
            env.allowRemoteModels = true;
            env.allowLocalModels = false;
            
            // Detect supported devices
            await this.detectSupportedDevices();
            
            // Initialize MediaPipe LLM if available
            await this.initializeMediaPipe();
            
            this.isInitialized = true;
            console.log('Real AI Model Manager initialized');
            console.log('Supported devices:', this.supportedDevices);
            console.log('MediaPipe available:', this.mediaPipeLLM !== null);
            
        } catch (error) {
            console.error('Model Manager initialization failed:', error);
            throw error;
        }
    }

    async detectSupportedDevices() {
        this.supportedDevices = ['cpu']; // CPU is always supported
        
        // Check WebAssembly support
        if (typeof WebAssembly === 'object') {
            this.supportedDevices.unshift('wasm');
        }
        
        // Check WebGPU support
        if (navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    this.supportedDevices.unshift('webgpu');
                    console.log('WebGPU adapter found:', adapter);
                }
            } catch (e) {
                console.log('WebGPU not available:', e);
            }
        }
        
        // Update config to use best available device
        const bestDevice = this.supportedDevices[0];
        this.config.whisper.device = bestDevice;
        this.config.textgen.device = bestDevice;
        
        // Set supported backends and current backend
        this.supportedBackends = [...this.supportedDevices];
        
        // Add MediaPipe backend if available
        if (typeof window.MediaPipeLLM !== 'undefined') {
            this.supportedBackends.push('mediapipe');
        }
        
        this.currentBackend = bestDevice;
    }

    async initializeMediaPipe() {
        try {
            // Check if MediaPipe LLM is available
            if (typeof window.MediaPipeLLM !== 'undefined') {
                console.log('MediaPipeLLM class found, attempting to initialize...');
                
                this.mediaPipeLLM = new window.MediaPipeLLM();
                
                // Test MediaPipe initialization with shorter timeout for faster feedback
                const initTimeout = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error('MediaPipe library not available')), 2000);
                });
                
                const initPromise = this.mediaPipeLLM.initialize();
                
                try {
                    await Promise.race([initPromise, initTimeout]);
                    console.log('MediaPipe LLM integration initialized successfully');
                } catch (error) {
                    console.warn('MediaPipe initialization failed during test:', error.message);
                    // Set to null to indicate MediaPipe not working
                    this.mediaPipeLLM = null;
                    this.useMediaPipe = false;
                    
                    // Don't throw here - let the calling code handle fallback
                }
            } else {
                console.log('MediaPipe LLM class not available - falling back to ONNX models');
                this.mediaPipeLLM = null;
                this.useMediaPipe = false;
            }
        } catch (error) {
            console.warn('MediaPipe initialization failed:', error.message);
            this.mediaPipeLLM = null;
            this.useMediaPipe = false;
        }
    }

    async loadModel(modelName, progressCallback = null) {
        try {
            const config = this.config[modelName];
            if (!config) {
                throw new Error(`Unknown model: ${modelName}`);
            }

            // Check if this should use MediaPipe
            if (modelName === 'textgen' && this.shouldUseMediaPipe(config.modelName)) {
                return await this.loadMediaPipeModel(config.modelName, progressCallback);
            }

            if (progressCallback) progressCallback(0, `Loading real ${modelName} model...`);

            const { pipeline } = window.transformers;
            
            // Configure options for the model
            let deviceToUse = config.device;
            
            // Handle quantization path for SmolLM2 models
            let modelPath = config.modelName;
            const options = {
                device: deviceToUse,
                dtype: config.dtype,
                progress_callback: (data) => {
                    if (progressCallback && data.progress !== undefined) {
                        const percentage = Math.round(data.progress * 100);
                        const message = `${data.status || 'Loading'} ${config.modelName}... ${percentage}%`;
                        progressCallback(percentage, message);
                    }
                }
            };
            
            if (config.modelName.includes('/q4f16')) {
                modelPath = config.modelName.replace('/q4f16', '');
                // Force quantization by setting dtype
                options.dtype = 'q4';
                console.log(`Using 4-bit quantized SmolLM2 model for faster inference`);
            }
            
            // Force CPU/WASM for SmolLM2 models due to WebGPU compatibility issues
            if (modelPath.toLowerCase().includes('smollm2') && deviceToUse === 'webgpu') {
                deviceToUse = this.supportedDevices.includes('wasm') ? 'wasm' : 'cpu';
                options.device = deviceToUse;
                console.log(`Forcing ${deviceToUse} for SmolLM2 model due to WebGPU compatibility issues`);
            }

            // Add device-specific optimizations to reduce warnings
            if (deviceToUse === 'webgpu') {
                options.executionProviders = ['webgpu'];
            } else if (deviceToUse === 'wasm') {
                options.executionProviders = ['wasm'];
            }

            console.log(`Loading real ${modelName} model: ${modelPath} on ${deviceToUse}`);
            
            // Load the actual Transformers.js pipeline
            const modelPipeline = await pipeline(config.task, modelPath, options);
            
            // Store the real model
            this.models.set(modelName, {
                pipeline: modelPipeline,
                config,
                device: deviceToUse,
                loaded: true,
                realModel: true,
                backend: 'onnx'
            });

            if (progressCallback) progressCallback(100, `${modelName} ready! (Real AI model loaded)`);
            console.log(`Real ${modelName} model loaded successfully on ${deviceToUse}`);
            return true;

        } catch (error) {
            console.error(`Failed to load real ${modelName} model:`, error);
            
            // Try fallback to CPU if WebGPU failed
            if (this.config[modelName].device === 'webgpu') {
                console.log(`Retrying ${modelName} with WASM/CPU...`);
                this.config[modelName].device = 'wasm';
                return this.loadModel(modelName, progressCallback);
            }
            
            if (progressCallback) progressCallback(-1, `Failed to load ${modelName}`);
            throw error;
        }
    }

    shouldUseMediaPipe(modelName) {
        // Check if model is better suited for MediaPipe
        const modelLower = modelName.toLowerCase();
        
        // Use MediaPipe for pure Gemma models (not ONNX versions)
        if (modelLower.includes('gemma') && !modelLower.includes('xenova')) {
            return true;
        }
        
        // Use MediaPipe if explicitly requested
        if (this.useMediaPipe) {
            return true;
        }
        
        return false;
    }

    async loadMediaPipeModel(modelName, progressCallback = null) {
        try {
            if (!this.mediaPipeLLM) {
                throw new Error('MediaPipe LLM not available - MediaPipe library not loaded');
            }

            if (progressCallback) progressCallback(5, 'Initializing MediaPipe LLM...');

            // Initialize MediaPipe if not already done - with shorter timeout
            if (!this.mediaPipeLLM.isInitialized) {
                if (progressCallback) progressCallback(10, 'Testing MediaPipe library availability...');
                
                // Reduced timeout for faster fallback
                const initTimeout = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error('MediaPipe library not found - falling back to ONNX models')), 2000);
                });
                
                try {
                    await Promise.race([this.mediaPipeLLM.initialize(), initTimeout]);
                    if (progressCallback) progressCallback(25, 'MediaPipe initialized successfully');
                } catch (error) {
                    console.warn('MediaPipe initialization failed quickly:', error.message);
                    throw new Error(`MediaPipe unavailable: ${error.message.includes('timeout') ? 'Library not loaded' : error.message}`);
                }
            }

            // Map model names to MediaPipe model IDs
            const mediaPipeModelId = this.mapToMediaPipeModel(modelName);
            if (progressCallback) progressCallback(30, `Loading MediaPipe model: ${mediaPipeModelId}...`);
            
            // Load the MediaPipe model with shorter timeout for faster fallback
            const loadTimeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('MediaPipe model loading timeout - falling back to ONNX')), 15000);
            });
            
            const loadPromise = this.mediaPipeLLM.loadModel(mediaPipeModelId, progressCallback);
            
            await Promise.race([loadPromise, loadTimeout]);

            // Store the MediaPipe model reference
            this.models.set('textgen', {
                mediaPipeLLM: this.mediaPipeLLM,
                modelId: mediaPipeModelId,
                config: this.config.textgen,
                loaded: true,
                realModel: true,
                backend: 'mediapipe'
            });

            this.useMediaPipe = true;
            if (progressCallback) progressCallback(100, `MediaPipe ${mediaPipeModelId} ready!`);
            console.log(`MediaPipe model ${mediaPipeModelId} loaded successfully`);
            return true;

        } catch (error) {
            console.warn('MediaPipe model loading failed:', error.message);
            
            // Always attempt fallback for MediaPipe issues
            console.log('Attempting fallback to ONNX models...');
            if (progressCallback) progressCallback(15, 'MediaPipe unavailable, switching to ONNX models...');
            
            // Reset MediaPipe usage flag
            this.useMediaPipe = false;
            
            // Try to load equivalent ONNX model instead
            const fallbackModels = this.getFallbackONNXModel(modelName);
            
            if (fallbackModels) {
                const modelList = Array.isArray(fallbackModels) ? fallbackModels : [fallbackModels];
                const originalModelName = this.config.textgen.modelName;
                
                for (let i = 0; i < modelList.length; i++) {
                    const fallbackModel = modelList[i];
                    console.log(`Trying ONNX fallback ${i + 1}/${modelList.length}: ${fallbackModel}`);
                    this.config.textgen.modelName = fallbackModel;
                    
                    try {
                        const result = await this.loadModel('textgen', (progress, message) => {
                            if (progressCallback) {
                                progressCallback(20 + (progress * 0.7), `Loading ${fallbackModel.split('/').pop()}... ${progress}%`);
                            }
                        });
                        
                        console.log(`Successfully loaded ONNX fallback: ${fallbackModel}`);
                        if (progressCallback) progressCallback(100, `${fallbackModel.split('/').pop()} ready! (ONNX fallback)`);
                        return result;
                    } catch (fallbackError) {
                        console.warn(`ONNX fallback ${i + 1} failed: ${fallbackError.message}`);
                        
                        // If this is the last attempt, continue to final fallback
                        if (i === modelList.length - 1) {
                            break;
                        }
                    }
                }
                
                // All primary fallbacks failed, try final emergency fallback
                console.warn('All primary fallbacks failed, trying emergency fallback...');
                const emergencyFallback = 'HuggingFaceTB/SmolLM2-135M-Instruct';
                this.config.textgen.modelName = emergencyFallback;
                
                try {
                    const result = await this.loadModel('textgen', (progress, message) => {
                        if (progressCallback) {
                            progressCallback(20 + (progress * 0.7), `Emergency fallback: ${message}`);
                        }
                    });
                    
                    console.log('Successfully loaded emergency fallback model');
                    if (progressCallback) progressCallback(100, `${emergencyFallback.split('/').pop()} ready! (Emergency fallback)`);
                    return result;
                } catch (emergencyError) {
                    // Restore original model name and re-throw
                    this.config.textgen.modelName = originalModelName;
                    throw new Error(`All fallback attempts failed. MediaPipe: ${error.message}, Final fallback: ${emergencyError.message}`);
                }
            } else {
                throw new Error(`MediaPipe model loading failed and no suitable ONNX fallback found: ${error.message}`);
            }
        }
    }

    getFallbackONNXModel(mediaPipeModelName) {
        // Map MediaPipe models to suitable ONNX alternatives that actually exist
        const modelLower = mediaPipeModelName.toLowerCase();
        
        if (modelLower.includes('gemma-3-1b') || modelLower.includes('gemma3-1b')) {
            // Since Gemma 3 1B ONNX versions are not widely available yet,
            // fallback to proven working models with similar capabilities
            return [
                'HuggingFaceTB/SmolLM2-1.7B-Instruct',     // Best size match, proven to work
                'Xenova/TinyLlama-1.1B-Chat-v1.0',        // Reliable fallback 
                'HuggingFaceTB/SmolLM2-135M-Instruct'      // Smaller but very fast
            ];
        } else if (modelLower.includes('gemma-2b') || modelLower.includes('gemma2b')) {
            // Legacy Gemma 2B references - use efficient fallback
            return 'HuggingFaceTB/SmolLM2-1.7B-Instruct';
        } else if (modelLower.includes('gemma-7b') || modelLower.includes('gemma7b')) {
            // Fallback to a capable model for Gemma 7B
            return 'Xenova/TinyLlama-1.1B-Chat-v1.0';
        } else if (modelLower.includes('gemma')) {
            // Default fallback for any Gemma model
            return [
                'HuggingFaceTB/SmolLM2-1.7B-Instruct',
                'Xenova/TinyLlama-1.1B-Chat-v1.0',
                'HuggingFaceTB/SmolLM2-135M-Instruct'
            ];
        }
        
        // No suitable fallback found
        return null;
    }

    mapToMediaPipeModel(modelName) {
        const modelLower = modelName.toLowerCase();
        
        // Prioritize Q4 quantized models for better performance
        if (modelLower.includes('gemma-3-1b') || modelLower.includes('gemma3-1b') || modelLower.includes('gemma-1b')) {
            return 'gemma-3-1b-it-q4'; // Use Q4 quantized version for speed
        } else if (modelLower.includes('gemma-2b') || modelLower.includes('gemma2b')) {
            // Legacy Gemma 2B references - redirect to Gemma 3 1B Q4
            return 'gemma-3-1b-it-q4';
        } else if (modelLower.includes('gemma-7b') || modelLower.includes('gemma7b')) {
            return 'gemma-7b-it';
        } else if (modelLower.includes('gemma')) {
            // Default to Gemma 3 1B Q4 for unknown Gemma variants
            return 'gemma-3-1b-it-q4';
        } else {
            // Default fallback for non-Gemma models
            return 'gemma-3-1b-it-q4';
        }
    }

    async runWhisperInference(audioData) {
        const model = this.models.get('whisper');
        if (!model || !model.loaded) {
            throw new Error('Whisper model not loaded');
        }

        try {
            console.log('Running real Whisper inference...');
            
            // Ensure audioData is a proper Float32Array
            let processedAudio;
            if (audioData instanceof Float32Array) {
                processedAudio = audioData;
            } else if (Array.isArray(audioData)) {
                processedAudio = new Float32Array(audioData);
            } else {
                throw new Error('Invalid audio data format');
            }
            
            // Run real speech recognition
            const result = await model.pipeline(processedAudio, {
                chunk_length_s: 30,
                stride_length_s: 5,
                return_timestamps: false
            });
            
            const transcript = result.text || result;
            console.log('Whisper transcript:', transcript);
            
            return transcript.trim();

        } catch (error) {
            console.error('Real Whisper inference failed:', error);
            throw error;
        }
    }

    async runTextGeneration(prompt, maxTokens = 100, streamCallback = null) {
        const model = this.models.get('textgen');
        if (!model || !model.loaded) {
            throw new Error('Text generation model not loaded');
        }

        try {
            console.log('Running real text generation...');
            
            // Check if using MediaPipe backend
            if (model.backend === 'mediapipe') {
                return await this.runMediaPipeGeneration(model, prompt, maxTokens, streamCallback);
            }
            
            // Original ONNX-based generation
            const formattedPrompt = this.formatPromptForModel(prompt, this.config.textgen.modelName);
            
            if (streamCallback) {
                // Streaming mode - generate tokens one by one
                return await this.runStreamingGeneration(model, formattedPrompt, maxTokens, streamCallback);
            } else {
                // Non-streaming mode (original)
                const result = await model.pipeline(formattedPrompt, {
                    max_new_tokens: maxTokens,
                    temperature: 0.7,
                    do_sample: true,
                    top_p: 0.9,
                    repetition_penalty: 1.1
                });
                
                // Extract the generated text
                let response = result[0]?.generated_text || result.generated_text || result;
                
                // Clean up response based on model type
                response = this.extractResponseFromModel(response, this.config.textgen.modelName);
                
                console.log('Generated response:', response);
                return response;
            }

        } catch (error) {
            console.error('Real text generation failed:', error);
            throw error;
        }
    }

    async runMediaPipeGeneration(model, prompt, maxTokens, streamCallback) {
        try {
            console.log('Using MediaPipe for text generation');
            
            const options = {
                maxTokens,
                temperature: 0.7,
                topK: 40,
                topP: 0.9,
                streamCallback
            };

            const response = await model.mediaPipeLLM.generateText(prompt, options);
            console.log('MediaPipe generated response:', response);
            return response;

        } catch (error) {
            console.error('MediaPipe generation failed:', error);
            throw error;
        }
    }

    async runStreamingGeneration(model, prompt, maxTokens, streamCallback) {
        // Note: True streaming requires model modifications, so we'll simulate it
        // by generating in chunks and calling the callback
        let fullResponse = '';
        let sentenceBuffer = '';
        
        try {
            // Generate the full response first
            // Adjust parameters for SmolLM2 models
            let genParams = {
                max_new_tokens: maxTokens,
                temperature: 0.7,
                do_sample: true,
                top_p: 0.9,
                repetition_penalty: 1.1
            };
            
            if (this.config.textgen.modelName.toLowerCase().includes('smollm2')) {
                genParams = {
                    max_new_tokens: Math.min(maxTokens, 50), // Shorter responses
                    temperature: 0.6, // Less randomness
                    do_sample: true,
                    top_p: 0.85,
                    repetition_penalty: 1.15 // Stronger repetition penalty
                };
            }
            
            const result = await model.pipeline(prompt, genParams);
            
            let response = result[0]?.generated_text || result.generated_text || result;
            
            // Clean up response based on model type  
            response = this.extractResponseFromModel(response, this.config.textgen.modelName);
            
            // Simulate streaming by sending words with delays
            const words = response.split(' ');
            
            for (let i = 0; i < words.length; i++) {
                const word = words[i];
                sentenceBuffer += word + ' ';
                fullResponse += word + ' ';
                
                // Check for sentence endings
                if (word.match(/[.!?]$/)) {
                    // Send complete sentence for TTS
                    if (streamCallback) {
                        await streamCallback({
                            type: 'sentence',
                            text: sentenceBuffer.trim(),
                            isComplete: false
                        });
                    }
                    sentenceBuffer = '';
                    
                    // Small delay between sentences
                    await new Promise(resolve => setTimeout(resolve, 100));
                } else {
                    // Send word update
                    if (streamCallback) {
                        await streamCallback({
                            type: 'word',
                            text: word,
                            fullText: fullResponse.trim(),
                            isComplete: false
                        });
                    }
                    
                    // Small delay between words to simulate streaming
                    await new Promise(resolve => setTimeout(resolve, 25));
                }
            }
            
            // Send final sentence if there's remaining text
            if (sentenceBuffer.trim()) {
                if (streamCallback) {
                    await streamCallback({
                        type: 'sentence',
                        text: sentenceBuffer.trim(),
                        isComplete: false
                    });
                }
            }
            
            // Send completion signal
            if (streamCallback) {
                await streamCallback({
                    type: 'complete',
                    text: fullResponse.trim(),
                    isComplete: true
                });
            }
            
            return fullResponse.trim();
            
        } catch (error) {
            console.error('Streaming generation failed:', error);
            throw error;
        }
    }

    formatPromptForModel(prompt, modelName) {
        // Auto-detect prompt format based on model name
        const model = modelName.toLowerCase();
        
        if (model.includes('tinyllama')) {
            return `<|system|>\nYou are a helpful AI assistant.<|user|>\n${prompt}<|assistant|>\n`;
        } else if (model.includes('smollm2') || model.includes('smolml2')) {
            return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
        } else if (model.includes('qwen')) {
            return `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
        } else if (model.includes('phi')) {
            return `<|user|>\n${prompt}<|end|>\n<|assistant|>\n`;
        } else if (model.includes('gemma')) {
            const systemPrompt = `You are an intelligent voice assistant. Respond naturally and conversationally as if speaking to a friend. Keep answers brief and to the point - aim for 1-2 sentences unless the user asks for more detail. Use simple, clear language without technical jargon. Avoid using markdown, special formatting, or numbered lists in your responses. Be helpful, friendly, and direct.`;
            return `<bos><start_of_turn>system\n${systemPrompt}<end_of_turn>\n<start_of_turn>user\n${prompt}<end_of_turn>\n<start_of_turn>model\n`;
        } else if (model.includes('llama')) {
            return `<s>[INST] ${prompt} [/INST]`;
        } else if (model.includes('gpt2') || model.includes('distilgpt2')) {
            return `Human: ${prompt}\nAI:`;
        } else {
            // Generic format for unknown models
            return `Human: ${prompt}\nAI:`;
        }
    }

    extractResponseFromModel(response, modelName) {
        // Extract clean response based on model type
        const model = modelName.toLowerCase();
        
        if (model.includes('tinyllama')) {
            if (response.includes('<|assistant|>\n')) {
                response = response.split('<|assistant|>\n')[1];
            }
            return response.split('<|user|>')[0].split('<|system|>')[0].trim();
        } else if (model.includes('smollm2') || model.includes('smolml2')) {
            if (response.includes('<|im_start|>assistant\n')) {
                response = response.split('<|im_start|>assistant\n')[1];
            }
            // Clean up any remaining system or user tags
            response = response.split('<|im_end|>')[0]
                              .split('<|im_start|>')[0]
                              .split('system ')[0]
                              .split('user ')[0]
                              .trim();
            return response;
        } else if (model.includes('qwen')) {
            if (response.includes('<|im_start|>assistant\n')) {
                response = response.split('<|im_start|>assistant\n')[1];
            }
            return response.split('<|im_end|>')[0].trim();
        } else if (model.includes('phi')) {
            if (response.includes('<|assistant|>\n')) {
                response = response.split('<|assistant|>\n')[1];
            }
            return response.split('<|end|>')[0].trim();
        } else if (model.includes('gemma')) {
            if (response.includes('<start_of_turn>model\n')) {
                response = response.split('<start_of_turn>model\n')[1];
            }
            return response.split('<end_of_turn>')[0].trim();
        } else if (model.includes('llama')) {
            // Response is usually after [/INST]
            return response.split('[/INST]').pop().trim();
        } else if (model.includes('gpt2') || model.includes('distilgpt2')) {
            // GPT-2 models - extract after "AI:"
            if (response.includes('AI:')) {
                response = response.split('AI:')[1];
            }
            return response.split('Human:')[0].split('\n\n')[0].trim();
        } else {
            // Generic cleanup
            if (response.includes('AI:')) {
                response = response.split('AI:')[1];
            }
            return response.split('Human:')[0].split('\n\n')[0].trim();
        }
    }

    // Backwards compatibility methods (now using real models)
    async runPhi3Inference(prompt, maxTokens = 100) {
        return this.runTextGeneration(prompt, maxTokens);
    }

    async switchBackend(newBackend) {
        if (!this.supportedBackends.includes(newBackend)) {
            throw new Error(`Backend ${newBackend} is not supported`);
        }

        const oldBackend = this.currentBackend;
        this.currentBackend = newBackend;

        // Reload models with new backend
        const loadedModels = Array.from(this.models.keys());
        
        for (const modelName of loadedModels) {
            this.unloadModel(modelName);
        }

        console.log(`Switched from ${oldBackend} to ${newBackend} backend`);
        return true;
    }

    unloadModel(modelName) {
        const model = this.models.get(modelName);
        if (model && model.session) {
            // Clean up the session
            model.session = null;
            this.models.delete(modelName);
            console.log(`Model ${modelName} unloaded`);
        }
    }

    getModelStatus(modelName) {
        const model = this.models.get(modelName);
        return {
            loaded: model ? model.loaded : false,
            backend: model ? model.backend : null,
            fallbackMode: model ? model.fallbackMode : false,
            config: this.config[modelName] || null
        };
    }

    getAllModelsStatus() {
        const status = {};
        for (const modelName of Object.keys(this.config)) {
            status[modelName] = this.getModelStatus(modelName);
        }
        status.currentBackend = this.currentBackend;
        status.supportedBackends = this.supportedBackends;
        status.mediaPipeAvailable = this.mediaPipeLLM !== null;
        status.usingMediaPipe = this.useMediaPipe;
        
        if (this.mediaPipeLLM) {
            status.mediaPipeInfo = this.mediaPipeLLM.getPerformanceInfo();
        }
        
        return status;
    }

    async benchmarkModel(modelName, iterations = 5) {
        const model = this.models.get(modelName);
        if (!model || !model.loaded) {
            throw new Error(`Model ${modelName} not loaded`);
        }

        const times = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            
            try {
                if (modelName === 'whisper') {
                    // Benchmark with dummy audio data
                    const dummyAudio = new Float32Array(16000 * 5); // 5 seconds
                    await this.runWhisperInference(dummyAudio);
                } else if (modelName === 'phi3') {
                    // Benchmark with test prompt
                    await this.runPhi3Inference("Hello, how are you?");
                }
                
                const end = performance.now();
                times.push(end - start);
                
            } catch (error) {
                console.error(`Benchmark iteration ${i} failed:`, error);
            }
        }

        const avgTime = times.reduce((sum, time) => sum + time, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);

        return {
            modelName,
            backend: this.currentBackend,
            iterations,
            averageTime: avgTime,
            minTime,
            maxTime,
            times
        };
    }

    cleanup() {
        // Clean up all models
        for (const modelName of this.models.keys()) {
            this.unloadModel(modelName);
        }
        
        // Clean up MediaPipe if available
        if (this.mediaPipeLLM) {
            this.mediaPipeLLM.cleanup();
            this.mediaPipeLLM = null;
        }
        
        console.log('Model Manager cleaned up');
    }
}

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelManager;
} else {
    window.ModelManager = ModelManager;
}