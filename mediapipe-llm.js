/**
 * MediaPipe LLM Inference Integration
 * Provides high-performance LLM inference using Google's MediaPipe framework
 * Optimized for Gemma models and edge deployment
 */

class MediaPipeLLM {
    constructor() {
        this.llmInference = null;
        this.isInitialized = false;
        this.currentModel = null;
        this.supportedModels = {
            'gemma-3-1b-it': {
                name: 'Gemma 3 1B Instruct',
                size: '~1.0GB',
                description: 'Google\'s latest Gemma 3 1B instruction-tuned model',
                performance: '⚡⚡⚡⚡⚡',
                quality: '⭐⭐⭐⭐'
            },
            'gemma-7b-it': {
                name: 'Gemma 7B Instruct', 
                size: '~5.2GB',
                description: 'Google\'s Gemma 7B instruction-tuned model',
                performance: '⚡⚡⚡',
                quality: '⭐⭐⭐⭐⭐'
            }
        };
    }

    async initialize() {
        try {
            // Check if MediaPipe LLM Inference is available
            console.log('Checking for MediaPipe LLM Inference library...');
            
            // Add timeout for initialization to prevent hanging
            const initPromise = this.initializeMediaPipe();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('MediaPipe initialization timeout')), 5000);
            });
            
            await Promise.race([initPromise, timeoutPromise]);
            
            this.isInitialized = true;
            console.log('MediaPipe LLM inference initialized successfully');
            
            return true;
        } catch (error) {
            console.error('MediaPipe LLM initialization failed:', error);
            console.warn('Falling back to ONNX models. MediaPipe features will not be available.');
            throw error;
        }
    }

    async initializeMediaPipe() {
        // Check for MediaPipe LLM library availability
        if (typeof window !== 'undefined' && typeof window.LlmInference !== 'undefined') {
            console.log('Found MediaPipe LLM Inference library');
            
            // Create LLM inference instance
            this.llmInference = await window.LlmInference.createFromOptions({
                baseOptions: {
                    modelAssetPath: '', // Will be set when loading specific model
                    delegate: 'GPU' // Use GPU acceleration when available
                }
            });
            
            return true;
        } else {
            // Check if MediaPipe Tasks GenAI is available (alternative approach)
            if (typeof window !== 'undefined' && 
                window.MediaPipeTasksGenAI && 
                window.MediaPipeTasksGenAI.LlmInference) {
                
                console.log('Found MediaPipe Tasks GenAI library');
                this.llmInference = await window.MediaPipeTasksGenAI.LlmInference.createFromOptions({
                    baseOptions: {
                        modelAssetPath: '',
                        delegate: 'GPU'
                    }
                });
                
                return true;
            } else {
                // Fail immediately if MediaPipe is not available - don't simulate
                console.log('MediaPipe LLM library not detected - no simulation, immediate fallback');
                throw new Error('MediaPipe LLM Inference library not found. The real MediaPipe library needs to be included via CDN.');
            }
        }
    }

    async loadModel(modelId, progressCallback = null) {
        try {
            if (!this.isInitialized) {
                await this.initialize();
            }

            if (!this.supportedModels[modelId]) {
                throw new Error(`Unsupported model: ${modelId}`);
            }

            const modelInfo = this.supportedModels[modelId];
            
            if (progressCallback) {
                progressCallback(0, `Loading ${modelInfo.name}...`);
            }

            // MediaPipe model loading
            // Note: In a real implementation, you would load the actual model file
            // For now, we'll simulate the loading process
            const modelUrl = this.getModelUrl(modelId);
            
            if (progressCallback) {
                progressCallback(25, `Downloading ${modelInfo.name}...`);
            }

            // Simulate download progress
            for (let progress = 30; progress <= 90; progress += 20) {
                await new Promise(resolve => setTimeout(resolve, 200));
                if (progressCallback) {
                    progressCallback(progress, `Loading ${modelInfo.name}... ${progress}%`);
                }
            }

            // Configure the model
            await this.llmInference.setOptions({
                baseOptions: {
                    modelAssetPath: modelUrl,
                    delegate: 'GPU'
                }
            });

            this.currentModel = modelId;
            
            if (progressCallback) {
                progressCallback(100, `${modelInfo.name} ready!`);
            }

            console.log(`MediaPipe model ${modelId} loaded successfully`);
            return true;

        } catch (error) {
            console.error(`Failed to load MediaPipe model ${modelId}:`, error);
            if (progressCallback) {
                progressCallback(-1, `Failed to load ${this.supportedModels[modelId]?.name || modelId}`);
            }
            throw error;
        }
    }

    getModelUrl(modelId) {
        // In a real implementation, these would be actual URLs to the model files
        // For demo purposes, we'll use placeholder URLs
        const baseUrl = 'https://storage.googleapis.com/mediapipe-models/llm_inference/';
        
        switch (modelId) {
            case 'gemma-3-1b-it':
                return `${baseUrl}gemma-3-1b-it/latest/gemma-3-1b-it.task`;
            case 'gemma-7b-it':
                return `${baseUrl}gemma-7b-it/latest/gemma-7b-it.task`;
            default:
                throw new Error(`Unknown model ID: ${modelId}`);
        }
    }

    async generateText(prompt, options = {}) {
        try {
            if (!this.isInitialized || !this.currentModel) {
                throw new Error('MediaPipe LLM not initialized or no model loaded');
            }

            const {
                maxTokens = 100,
                temperature = 0.7,
                topK = 40,
                topP = 0.9,
                streamCallback = null
            } = options;

            console.log('Generating text with MediaPipe LLM...');

            // Format prompt for Gemma model
            const formattedPrompt = this.formatPromptForGemma(prompt);

            if (streamCallback) {
                return await this.generateStreamingText(formattedPrompt, {
                    maxTokens,
                    temperature,
                    topK,
                    topP,
                    streamCallback
                });
            } else {
                // Non-streaming generation
                const result = await this.llmInference.generateResponse(formattedPrompt);
                return this.extractResponseText(result);
            }

        } catch (error) {
            console.error('MediaPipe text generation failed:', error);
            throw error;
        }
    }

    async generateStreamingText(prompt, options) {
        const { maxTokens, temperature, topK, topP, streamCallback } = options;
        
        try {
            // For demonstration, we'll simulate streaming by generating the full response
            // and then streaming it word by word. In a real implementation, 
            // MediaPipe would provide true streaming capabilities.
            
            const fullResponse = await this.llmInference.generateResponse(prompt, {
                maxOutputTokens: maxTokens,
                temperature,
                topK,
                topP
            });

            const responseText = this.extractResponseText(fullResponse);
            
            // Simulate streaming by sending words with delays
            const words = responseText.split(' ');
            let currentSentence = '';
            let fullText = '';

            for (let i = 0; i < words.length; i++) {
                const word = words[i];
                currentSentence += word + ' ';
                fullText += word + ' ';

                // Check for sentence endings
                if (word.match(/[.!?]$/)) {
                    // Send complete sentence
                    if (streamCallback) {
                        await streamCallback({
                            type: 'sentence',
                            text: currentSentence.trim(),
                            isComplete: false
                        });
                    }
                    currentSentence = '';
                    
                    // Small delay between sentences
                    await new Promise(resolve => setTimeout(resolve, 50));
                } else {
                    // Send word update
                    if (streamCallback) {
                        await streamCallback({
                            type: 'word',
                            text: word,
                            fullText: fullText.trim(),
                            isComplete: false
                        });
                    }
                    
                    // Small delay between words
                    await new Promise(resolve => setTimeout(resolve, 20));
                }
            }

            // Send final sentence if there's remaining text
            if (currentSentence.trim()) {
                if (streamCallback) {
                    await streamCallback({
                        type: 'sentence',
                        text: currentSentence.trim(),
                        isComplete: false
                    });
                }
            }

            // Send completion signal
            if (streamCallback) {
                await streamCallback({
                    type: 'complete',
                    text: fullText.trim(),
                    isComplete: true
                });
            }

            return fullText.trim();

        } catch (error) {
            console.error('MediaPipe streaming generation failed:', error);
            throw error;
        }
    }

    formatPromptForGemma(prompt) {
        // Optimized system prompt for voice assistant interaction with Gemma 3 1B
        const systemPrompt = `You are an intelligent voice assistant. Respond naturally and conversationally as if speaking to a friend. Keep answers brief and to the point - aim for 1-2 sentences unless the user asks for more detail. Use simple, clear language without technical jargon. Avoid using markdown, special formatting, or numbered lists in your responses. Be helpful, friendly, and direct.`;
        
        // Use proper Gemma 3 chat format with system instruction
        return `<bos><start_of_turn>system\n${systemPrompt}<end_of_turn>\n<start_of_turn>user\n${prompt}<end_of_turn>\n<start_of_turn>model\n`;
    }

    extractResponseText(result) {
        // Extract clean response from MediaPipe result
        let responseText = '';
        
        if (result && result.generatedText) {
            responseText = result.generatedText;
        } else if (typeof result === 'string') {
            responseText = result;
        } else {
            throw new Error('Invalid response format from MediaPipe');
        }

        // Clean up Gemma model response
        responseText = responseText.split('<end_of_turn>')[0]
                                 .split('<start_of_turn>')[0]
                                 .trim();

        return responseText;
    }

    getSupportedModels() {
        return this.supportedModels;
    }

    getCurrentModel() {
        return this.currentModel;
    }

    isModelLoaded() {
        return this.isInitialized && this.currentModel !== null;
    }

    getPerformanceInfo() {
        return {
            backend: 'MediaPipe',
            acceleration: 'GPU',
            currentModel: this.currentModel,
            modelInfo: this.currentModel ? this.supportedModels[this.currentModel] : null
        };
    }

    async cleanup() {
        try {
            if (this.llmInference) {
                await this.llmInference.close();
                this.llmInference = null;
            }
            
            this.isInitialized = false;
            this.currentModel = null;
            
            console.log('MediaPipe LLM cleaned up');
        } catch (error) {
            console.error('MediaPipe cleanup error:', error);
        }
    }
}

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MediaPipeLLM;
} else {
    window.MediaPipeLLM = MediaPipeLLM;
}