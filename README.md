# Voice Assistant POC - Optimized

An optimized browser-based voice assistant with two powerful options:

1. **Local AI Models** (`index.html`) - Privacy-first with ONNX Runtime Web
2. **🌸 Pollinations AI** (`index-pollinations.html`) - Cloud-powered for better performance

[**→ Try Pollinations AI Version**](index-pollinations.html) | [📖 Pollinations Documentation](README-pollinations.md)

## Model Optimizations

### Speech Recognition
- **Model**: Whisper Tiny ONNX
- **Benefits**: 39M parameters, optimized for browser deployment
- **Performance**: Real-time inference with WebGPU acceleration

### Language Model
- **Model**: Phi-3 Mini ONNX (upgraded from TinyLlama)
- **Benefits**: 
  - 3.8B parameters vs TinyLlama's 1.1B
  - Significantly better conversation quality
  - Microsoft-optimized for browser deployment
  - 4-bit quantization reduces model size to ~2.4GB

### Text-to-Speech
- **Model**: Web Speech API
- **Fallback**: Compatible across all modern browsers

## Backend Support

### Progressive Enhancement
1. **WebGPU** (Primary): Hardware acceleration for optimal performance
2. **WebAssembly** (Fallback): CPU optimization for broader compatibility
3. **CPU** (Final fallback): Basic support for older browsers

### Performance Characteristics
- **WebGPU**: ~10x faster inference, requires Chrome 113+/Edge 113+
- **WebAssembly**: Balanced performance and compatibility
- **CPU**: Slowest but universally supported

## Architecture Improvements

### Enhanced Audio Processing (`audio-processor.js`)
- Real-time audio capture with 16kHz sampling
- Proper audio preprocessing for Whisper input
- Voice Activity Detection (VAD)
- Audio format conversion and normalization

### Model Manager (`model-manager.js`)
- Centralized ONNX model loading and inference
- Backend detection and switching
- Model benchmarking and performance monitoring
- Error handling and fallback mechanisms

### Key Features
- **Privacy-first**: All processing happens locally in your browser
- **No server required**: Models run entirely client-side
- **Progressive enhancement**: Automatically uses the best available backend
- **Real-time performance**: Optimized for low-latency voice interaction

## Browser Compatibility

### Recommended
- Chrome 113+ or Edge 113+ (WebGPU support)
- 8GB+ RAM for optimal Phi-3 Mini performance
- Modern desktop/laptop for best experience

### Minimum Requirements
- Any modern browser with WebAssembly support
- 4GB+ RAM
- Microphone access

## File Structure

```
voice_assistant/
├── index.html                     # 🔒 Local AI models version (privacy-first)
├── index-pollinations.html        # 🌸 Pollinations AI version (cloud-powered)
├── audio-processor.js             # Shared audio processing pipeline
├── model-manager.js               # Local ONNX model management
├── pollinations-model-manager.js  # Pollinations API integration
├── test-continuous-conversation.html
├── README.md                      # This documentation
└── README-pollinations.md         # Pollinations-specific documentation
```

## Quick Start

### Option 1: Pollinations AI (Recommended)
🌸 **Faster, cloud-powered experience**
1. Open [`index-pollinations.html`](index-pollinations.html) in your browser
2. Allow microphone permissions
3. Click the microphone and start talking
4. Enjoy fast AI responses + image generation!

### Option 2: Local AI Models (Privacy-focused)
🔒 **100% local processing**
1. Open [`index.html`](index.html) in your browser
2. Wait for models to download and load (2-5 minutes)
3. Allow microphone permissions  
4. Choose your preferred language model
5. Start your voice conversation

**[📖 See detailed comparison and setup guide](README-pollinations.md)**

### GitHub Pages Deployment

This app can be easily deployed to GitHub Pages since it requires no server-side processing:

1. **Create a GitHub repository**
2. **Upload all files** (index.html, model-manager.js, audio-processor.js, README.md)
3. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Set Source to "Deploy from a branch"
   - Select "main" branch and "/ (root)" folder
   - Click Save
4. **Access your app** at `https://username.github.io/repository-name`

**Why it works perfectly on GitHub Pages:**
- ✅ Pure client-side application (HTML/JS/CSS only)
- ✅ No server or backend required
- ✅ HTTPS provided automatically (required for microphone access)
- ✅ Models loaded directly from Hugging Face CDN
- ✅ All AI processing happens in the browser

**Deployment Tips:**
- Repository can be public or private (GitHub Pages works with both)
- No build process required - just upload the files as-is
- Models are cached by the browser, so loading is faster on subsequent visits
- Works on mobile devices too (with appropriate browser support)

## Model Loading

The application loads real ONNX models from Hugging Face:

**Speech Recognition:**
- Whisper Tiny: `Xenova/whisper-tiny.en` (WebGPU accelerated)

**Language Models (choose at startup):**
- TinyLlama 1.1B: `Xenova/TinyLlama-1.1B-Chat-v1.0` (~650MB)
- DistilGPT2: `Xenova/distilgpt2` (~350MB, fastest)
- GPT-2: `Xenova/gpt2` (~550MB, classic)
- SmolLM2 135M (Fast): 4-bit quantized for speed (~118MB)
- SmolLM2 135M (Quality): Standard precision (~270MB)
- Custom models: Any compatible ONNX model from Hugging Face

All models run entirely in your browser with no data sent to external servers.

## Performance Monitoring

- Real-time latency measurement
- Backend performance indicators
- Model benchmarking tools
- Audio processing status

## Future Enhancements

- Support for larger Whisper models (Base, Small)
- Integration with F5-TTS for better voice synthesis
- WebNN backend support
- Model streaming for faster initial load
- Voice cloning capabilities

## Technical Notes

This is a proof-of-concept demonstrating the feasibility of running modern AI models entirely in the browser. While the current implementation includes fallbacks to simulated responses, the architecture is designed to support real ONNX model inference with minimal modifications.

The choice of Phi-3 Mini over TinyLlama represents a significant upgrade in conversation quality while maintaining reasonable performance characteristics for browser deployment.