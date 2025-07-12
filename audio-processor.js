/**
 * Enhanced Audio Processing Pipeline for Voice Assistant
 * Handles real audio capture, preprocessing, and format conversion for ONNX models
 */

class AudioProcessor {
    constructor() {
        this.audioContext = null;
        this.mediaRecorder = null;
        this.sourceNode = null;
        this.processorNode = null;
        this.recordedChunks = [];
        this.isRecording = false;
        
        // Audio configuration for Whisper
        this.sampleRate = 16000;
        this.channels = 1;
        this.bufferSize = 4096;
        
        // Audio buffer for real-time processing
        this.audioBuffer = new Float32Array(16000 * 30); // 30 seconds max
        this.bufferPosition = 0;
    }

    async initialize(mediaStream) {
        try {
            // Create audio context with target sample rate
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });

            // Resume context if suspended (required by some browsers)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Create source from media stream
            this.sourceNode = this.audioContext.createMediaStreamSource(mediaStream);
            
            // Create audio worklet processor for real-time processing
            await this.setupAudioWorklet();
            
            // Setup MediaRecorder for fallback
            this.setupMediaRecorder(mediaStream);
            
            console.log(`Audio processor initialized: ${this.sampleRate}Hz, ${this.channels} channel(s)`);
            
        } catch (error) {
            console.error('Audio processor initialization failed:', error);
            throw error;
        }
    }

    async setupAudioWorklet() {
        try {
            // Try to use AudioWorkletNode (modern approach)
            if (this.audioContext.audioWorklet) {
                try {
                    // Create a simple worklet inline for audio processing
                    const workletCode = `
                        class AudioRecorderProcessor extends AudioWorkletProcessor {
                            process(inputs, outputs, parameters) {
                                const input = inputs[0];
                                if (input.length > 0) {
                                    const inputData = input[0];
                                    this.port.postMessage({
                                        type: 'audiodata',
                                        data: inputData
                                    });
                                }
                                return true;
                            }
                        }
                        registerProcessor('audio-recorder', AudioRecorderProcessor);
                    `;
                    
                    const blob = new Blob([workletCode], { type: 'application/javascript' });
                    const workletUrl = URL.createObjectURL(blob);
                    
                    await this.audioContext.audioWorklet.addModule(workletUrl);
                    
                    this.processorNode = new AudioWorkletNode(this.audioContext, 'audio-recorder');
                    this.processorNode.port.onmessage = (event) => {
                        if (event.data.type === 'audiodata' && this.isRecording) {
                            this.processAudioData(event.data.data);
                        }
                    };
                    
                    // Connect the audio graph
                    this.sourceNode.connect(this.processorNode);
                    console.log('Using modern AudioWorkletNode');
                    
                } catch (workletError) {
                    console.warn('AudioWorkletNode failed, falling back to ScriptProcessorNode:', workletError);
                    this.setupLegacyProcessor();
                }
            } else {
                this.setupLegacyProcessor();
            }
            
        } catch (error) {
            console.error('Audio worklet setup failed:', error);
            throw error;
        }
    }

    setupLegacyProcessor() {
        // Fallback to ScriptProcessorNode for compatibility
        this.processorNode = this.audioContext.createScriptProcessor(
            this.bufferSize, 
            this.channels, 
            this.channels
        );

        this.processorNode.onaudioprocess = (event) => {
            if (this.isRecording) {
                this.processAudioBuffer(event.inputBuffer);
            }
        };

        // Connect the audio graph
        this.sourceNode.connect(this.processorNode);
        this.processorNode.connect(this.audioContext.destination);
        console.log('Using legacy ScriptProcessorNode');
    }

    processAudioData(inputData) {
        // Process audio data from AudioWorkletNode
        for (let i = 0; i < inputData.length; i++) {
            if (this.bufferPosition < this.audioBuffer.length) {
                this.audioBuffer[this.bufferPosition] = inputData[i];
                this.bufferPosition++;
            }
        }
    }

    setupMediaRecorder(mediaStream) {
        try {
            // Setup MediaRecorder as fallback for audio capture
            const options = {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 128000
            };

            if (MediaRecorder.isTypeSupported(options.mimeType)) {
                this.mediaRecorder = new MediaRecorder(mediaStream, options);
            } else {
                this.mediaRecorder = new MediaRecorder(mediaStream);
            }

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };

        } catch (error) {
            console.warn('MediaRecorder setup failed, using audio worklet only:', error);
        }
    }

    processAudioBuffer(inputBuffer) {
        // Get audio data from the first channel
        const inputData = inputBuffer.getChannelData(0);
        
        // Copy to our main buffer
        for (let i = 0; i < inputData.length; i++) {
            if (this.bufferPosition < this.audioBuffer.length) {
                this.audioBuffer[this.bufferPosition] = inputData[i];
                this.bufferPosition++;
            }
        }
    }

    startRecording() {
        this.isRecording = true;
        this.recordedChunks = [];
        this.bufferPosition = 0;
        
        // Clear the audio buffer
        this.audioBuffer.fill(0);
        
        // Start MediaRecorder if available
        if (this.mediaRecorder && this.mediaRecorder.state === 'inactive') {
            this.mediaRecorder.start(100); // Collect data every 100ms
        }
        
        console.log('Audio recording started');
    }

    stopRecording() {
        this.isRecording = false;
        
        // Stop MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        console.log('Audio recording stopped');
    }

    getAudioData() {
        // Return the processed audio data
        return {
            audioBuffer: this.audioBuffer.slice(0, this.bufferPosition),
            sampleRate: this.sampleRate,
            channels: this.channels,
            duration: this.bufferPosition / this.sampleRate
        };
    }

    async getAudioBlob() {
        // Return recorded audio as blob for fallback processing
        if (this.recordedChunks.length === 0) {
            return null;
        }
        
        return new Blob(this.recordedChunks, { 
            type: this.mediaRecorder?.mimeType || 'audio/webm' 
        });
    }

    prepareWhisperInput(audioData = null) {
        // Prepare audio data in the format expected by Whisper ONNX model
        const data = audioData || this.getAudioData();
        
        if (data.audioBuffer.length === 0) {
            throw new Error('No audio data available');
        }
        
        // Whisper expects exactly 30 seconds of audio at 16kHz (480,000 samples)
        const targetLength = 30 * this.sampleRate;
        const processedAudio = new Float32Array(targetLength);
        
        if (data.audioBuffer.length > targetLength) {
            // Truncate if too long
            processedAudio.set(data.audioBuffer.slice(0, targetLength));
        } else {
            // Pad with zeros if too short
            processedAudio.set(data.audioBuffer);
        }
        
        // Apply preprocessing (normalize, etc.)
        this.normalizeAudio(processedAudio);
        
        return processedAudio;
    }

    normalizeAudio(audioData) {
        // Normalize audio to [-1, 1] range
        let max = 0;
        for (let i = 0; i < audioData.length; i++) {
            max = Math.max(max, Math.abs(audioData[i]));
        }
        
        if (max > 0) {
            const scale = 1.0 / max;
            for (let i = 0; i < audioData.length; i++) {
                audioData[i] *= scale;
            }
        }
    }

    calculateRMS(audioData) {
        // Calculate Root Mean Square for volume level
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        return Math.sqrt(sum / audioData.length);
    }

    detectSpeech(threshold = 0.01) {
        // Simple speech detection based on RMS
        const recentData = this.audioBuffer.slice(
            Math.max(0, this.bufferPosition - this.sampleRate), // Last 1 second
            this.bufferPosition
        );
        
        const rms = this.calculateRMS(recentData);
        return rms > threshold;
    }

    async convertBlobToFloat32Array(blob) {
        // Convert audio blob to Float32Array for processing
        try {
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Get first channel and resample if necessary
            let audioData = audioBuffer.getChannelData(0);
            
            // Resample to 16kHz if needed
            if (audioBuffer.sampleRate !== this.sampleRate) {
                audioData = this.resample(audioData, audioBuffer.sampleRate, this.sampleRate);
            }
            
            return audioData;
            
        } catch (error) {
            console.error('Audio conversion failed:', error);
            throw error;
        }
    }

    resample(audioData, fromSampleRate, toSampleRate) {
        // Simple linear interpolation resampling
        if (fromSampleRate === toSampleRate) {
            return audioData;
        }
        
        const ratio = fromSampleRate / toSampleRate;
        const newLength = Math.round(audioData.length / ratio);
        const resampled = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
            const fraction = srcIndex - srcIndexFloor;
            
            resampled[i] = audioData[srcIndexFloor] * (1 - fraction) + 
                          audioData[srcIndexCeil] * fraction;
        }
        
        return resampled;
    }

    cleanup() {
        // Clean up resources
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode = null;
        }
        
        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.recordedChunks = [];
        this.isRecording = false;
        
        console.log('Audio processor cleaned up');
    }

    // Utility methods for debugging and monitoring
    getStatus() {
        return {
            isRecording: this.isRecording,
            bufferPosition: this.bufferPosition,
            bufferDuration: this.bufferPosition / this.sampleRate,
            audioContextState: this.audioContext?.state,
            mediaRecorderState: this.mediaRecorder?.state
        };
    }

    async getVolumeLevel() {
        // Get current volume level for UI feedback
        if (!this.isRecording || this.bufferPosition === 0) {
            return 0;
        }
        
        const recentData = this.audioBuffer.slice(
            Math.max(0, this.bufferPosition - 1024), // Last ~64ms at 16kHz
            this.bufferPosition
        );
        
        return this.calculateRMS(recentData);
    }
}

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioProcessor;
} else {
    window.AudioProcessor = AudioProcessor;
}