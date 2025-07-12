#!/usr/bin/env node

/**
 * Node.js test for MediaPipe integration
 * Validates the MediaPipe LLM integration without browser dependencies
 */

// Mock global objects for Node.js environment
global.window = global;

// Mock browser APIs
global.navigator = { gpu: null };
global.WebAssembly = { object: true };
global.transformers = { pipeline: () => {}, env: {} };
global.selectedModel = 'gemma-3-1b-it';
global.selectedBackend = 'mediapipe';

// Load the modules using require
const MediaPipeLLM = require('./mediapipe-llm.js');
const ModelManager = require('./model-manager.js');

// Make them globally available
global.MediaPipeLLM = MediaPipeLLM;
global.ModelManager = ModelManager;

// Test runner
class MediaPipeTestRunner {
    constructor() {
        this.tests = [];
        this.results = [];
    }

    addTest(name, testFunc) {
        this.tests.push({ name, testFunc });
    }

    async runTests() {
        console.log('🚀 MediaPipe Integration Tests\n');

        for (const test of this.tests) {
            try {
                const result = await test.testFunc();
                this.results.push({ name: test.name, passed: true, message: result });
                console.log(`✅ ${test.name}: ${result}`);
            } catch (error) {
                this.results.push({ name: test.name, passed: false, message: error.message });
                console.log(`❌ ${test.name}: ${error.message}`);
            }
        }

        this.printSummary();
    }

    printSummary() {
        const passed = this.results.filter(r => r.passed).length;
        const total = this.results.length;
        
        console.log('\n📊 Test Summary:');
        console.log(`${passed}/${total} tests passed`);
        
        if (passed === total) {
            console.log('🎉 All tests passed!');
        } else {
            console.log('⚠️  Some tests failed');
        }
    }
}

// Create test runner
const runner = new MediaPipeTestRunner();

// Test 1: MediaPipe class loading
runner.addTest('MediaPipe Class Loading', () => {
    if (typeof MediaPipeLLM === 'undefined') {
        throw new Error('MediaPipeLLM class not found');
    }

    const mediaPipe = new MediaPipeLLM();
    
    const requiredMethods = ['initialize', 'loadModel', 'generateText', 'getSupportedModels'];
    const missingMethods = requiredMethods.filter(method => typeof mediaPipe[method] !== 'function');
    
    if (missingMethods.length > 0) {
        throw new Error(`Missing methods: ${missingMethods.join(', ')}`);
    }

    return 'MediaPipeLLM class loaded with all required methods';
});

// Test 2: Supported models configuration
runner.addTest('Supported Models Configuration', () => {
    const mediaPipe = new MediaPipeLLM();
    const supportedModels = mediaPipe.getSupportedModels();
    
    if (!supportedModels || Object.keys(supportedModels).length === 0) {
        throw new Error('No supported models found');
    }

    const hasGemma3_1B = 'gemma-3-1b-it' in supportedModels;
    const hasGemma7B = 'gemma-7b-it' in supportedModels;
    
    if (!hasGemma3_1B || !hasGemma7B) {
        throw new Error('Missing Gemma models in configuration');
    }

    return `Found ${Object.keys(supportedModels).length} supported models including Gemma 3 1B and 7B`;
});

// Test 3: Model Manager integration
runner.addTest('Model Manager Integration', () => {
    if (typeof ModelManager === 'undefined') {
        throw new Error('ModelManager class not found');
    }

    // Mock transformers
    global.transformers = { pipeline: () => {}, env: {} };
    global.selectedModel = 'gemma-3-1b-it';
    global.selectedBackend = 'mediapipe';

    const modelManager = new ModelManager();
    
    if (modelManager.mediaPipeLLM === undefined) {
        throw new Error('MediaPipe integration not found in ModelManager');
    }

    const requiredMethods = ['initializeMediaPipe', 'shouldUseMediaPipe', 'loadMediaPipeModel', 'mapToMediaPipeModel'];
    const missingMethods = requiredMethods.filter(method => typeof modelManager[method] !== 'function');
    
    if (missingMethods.length > 0) {
        throw new Error(`Missing ModelManager methods: ${missingMethods.join(', ')}`);
    }

    return 'ModelManager has proper MediaPipe integration';
});

// Test 4: Backend selection logic
runner.addTest('Backend Selection Logic', () => {
    global.transformers = { pipeline: () => {}, env: {} };
    global.selectedBackend = undefined; // Reset to test natural selection
    const modelManager = new ModelManager();
    
    // Test Gemma model detection
    const gemmaModels = ['gemma-3-1b-it', 'gemma-7b-it'];
    const xenovaGemmaModel = 'Xenova/gemma-3-1b'; // Should use ONNX, not MediaPipe
    const nonGemmaModel = 'Xenova/TinyLlama-1.1B-Chat-v1.0';
    
    const gemmaDetected = gemmaModels.every(model => modelManager.shouldUseMediaPipe(model));
    const xenovaGemmaCorrect = !modelManager.shouldUseMediaPipe(xenovaGemmaModel);
    const nonGemmaCorrect = !modelManager.shouldUseMediaPipe(nonGemmaModel);
    
    if (!gemmaDetected) {
        throw new Error('Pure Gemma models not properly detected for MediaPipe');
    }

    if (!xenovaGemmaCorrect) {
        throw new Error('Xenova Gemma model incorrectly assigned to MediaPipe (should use ONNX)');
    }

    if (!nonGemmaCorrect) {
        throw new Error('Non-Gemma model incorrectly assigned to MediaPipe');
    }

    return 'Backend selection logic working correctly';
});

// Test 5: Model mapping
runner.addTest('Model Mapping Logic', () => {
    global.transformers = { pipeline: () => {}, env: {} };
    const modelManager = new ModelManager();
    
    const testCases = [
        { input: 'gemma-3-1b-it', expected: 'gemma-3-1b-it' },
        { input: 'gemma-7b-it', expected: 'gemma-7b-it' },
        { input: 'Xenova/gemma-3-1b', expected: 'gemma-3-1b-it' },
        { input: 'unknown-gemma', expected: 'gemma-3-1b-it' }
    ];

    for (const testCase of testCases) {
        const result = modelManager.mapToMediaPipeModel(testCase.input);
        if (result !== testCase.expected) {
            throw new Error(`Mapping failed: ${testCase.input} -> ${result}, expected ${testCase.expected}`);
        }
    }

    return 'Model mapping logic working correctly';
});

// Test 6: MediaPipe instance methods
runner.addTest('MediaPipe Instance Methods', () => {
    const mediaPipe = new MediaPipeLLM();
    
    // Test basic methods
    if (typeof mediaPipe.formatPromptForGemma !== 'function') {
        throw new Error('formatPromptForGemma method not found');
    }

    if (typeof mediaPipe.extractResponseText !== 'function') {
        throw new Error('extractResponseText method not found');
    }

    // Test prompt formatting
    const testPrompt = 'Hello, how are you?';
    const formattedPrompt = mediaPipe.formatPromptForGemma(testPrompt);
    
    if (!formattedPrompt.includes('<bos><start_of_turn>system') || 
        !formattedPrompt.includes('<start_of_turn>user') ||
        !formattedPrompt.includes('<start_of_turn>model')) {
        throw new Error('Prompt formatting for Gemma model incorrect - missing system prompt or proper structure');
    }

    return 'MediaPipe instance methods working correctly';
});

// Run all tests
runner.runTests().catch(console.error);