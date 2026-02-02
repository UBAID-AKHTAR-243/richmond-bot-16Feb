// static/js/api-integration.js
// ===== API INTEGRATION =====

// Configuration
let API_BASE_URL = '';
let userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
let isAPIConnected = false;

// Determine API URL based on environment
function determineAPIBaseURL() {
    const hostname = window.location.hostname;
   // const protocol = window.location.protocol;

    // Local development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:8000';
    }

    // GitHub Pages - you need to update this with your actual backend URL
    if (hostname.includes('github.io')) {
        // Example: return 'https://your-backend.herokuapp.com';
        // Example: return 'https://your-backend.onrender.com';
        return ''; // Empty for now - you'll configure this later
    }

    return '';
}

// STT API Function
async function transcribeAudioAPI(audioBlob) {
    try {
        console.log('Sending audio to STT API...');

        if (!API_BASE_URL) {
            console.log('No API URL configured, using simulated transcription');
            return "This is a simulated transcription. Configure API_BASE_URL in api-integration.js";
        }

        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
        //ubaid    throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('STT Result:', result);

        if (result.success) {
            return result.transcription;
        } else {
        //ubaid    throw new Error(result.error || 'Transcription failed');
        }
    } catch (error) {
        console.error('STT API Error:', error);
        return null;
    }
}

// Chat API Function
async function sendToChatAPI(text) {
    try {
        console.log('Sending to Chat API:', text);

        if (!API_BASE_URL) {
            console.log('No API URL configured, using simulated response');
            return {
                success: true,
                bot_response: {
                    text: "This is a simulated response. Configure API_BASE_URL in api-integration.js"
                }
            };
        }

        const response = await fetch(`${API_BASE_URL}/api/chat/send`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                type: 'text',
                user_id: userId
            })
        });

        if (!response.ok) {
        //ubaid    throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Chat API Result:', result);
        return result;
    } catch (error) {
        console.error('Chat API Error:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

// Check API Health
async function checkAPIHealth() {
    if (!API_BASE_URL) {
        isAPIConnected = false;
        return false;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/health`, {
            method: 'GET',
            timeout: 5000
        });
        isAPIConnected = response.ok;
        console.log(`API Health: ${isAPIConnected ? '✅ Connected' : '❌ Not responding'}`);
        return isAPIConnected;
    } catch (error) {
        isAPIConnected = false;
        console.log('❌ Cannot reach API:', error.message);
        return false;
    }
}

// Add API Status Indicator
function addAPIStatusIndicator() {
    // Remove existing indicator if present
    const existingIndicator = document.getElementById('apiStatusIndicator');
    if (existingIndicator) {
        existingIndicator.remove();
    }

    const statusIndicator = document.createElement('div');
    statusIndicator.id = 'apiStatusIndicator';
    statusIndicator.style.cssText = `
        position: fixed;
        bottom: 15px;
        right: 15px;
        background: ${isAPIConnected ? '#00a884' : '#ff3b30'};
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        cursor: pointer;
        opacity: 0.9;
    `;

    statusIndicator.innerHTML = `
        <i class="fas fa-circle" style="font-size: 8px;"></i>
        <span>${isAPIConnected ? 'API Connected' : 'API Disconnected'}</span>
    `;

    statusIndicator.title = `API URL: ${API_BASE_URL || 'Not configured'}\nClick for details`;
    statusIndicator.onclick = function() {
        alert(`API Status: ${isAPIConnected ? 'Connected' : 'Disconnected'}\nURL: ${API_BASE_URL || 'Not configured'}\nUser ID: ${userId}`);
    };

    document.body.appendChild(statusIndicator);

    // Auto-hide after 10 seconds if connected
    if (isAPIConnected) {
        setTimeout(() => {
            statusIndicator.style.opacity = '0.5';
        }, 10000);
    }

    return statusIndicator;
}

// Update Status Indicator
function updateAPIStatusIndicator() {
    const indicator = document.getElementById('apiStatusIndicator');
    if (indicator) {
        indicator.style.background = isAPIConnected ? '#00a884' : '#ff3b30';
        const span = indicator.querySelector('span');
        if (span) {
            span.textContent = isAPIConnected ? 'API Connected' : 'API Disconnected';
        }
    }
}

// Initialize API
async function initAPI() {
    // Determine API URL
    API_BASE_URL = determineAPIBaseURL();
    console.log('API Base URL:', API_BASE_URL || 'Not configured');

    // Check health if API URL is configured
    if (API_BASE_URL) {
        await checkAPIHealth();
    } else {
        isAPIConnected = false;
        console.log('API URL not configured. Running in standalone mode.');
    }

    // Add status indicator
    addAPIStatusIndicator();

    // Periodic health check
    if (API_BASE_URL) {
        setInterval(async () => {
            const wasConnected = isAPIConnected;
            await checkAPIHealth();
            if (wasConnected !== isAPIConnected) {
                updateAPIStatusIndicator();
            }
        }, 30000);
    }

    return isAPIConnected;
}

// Enhanced functions that work with existing app.js
async function enhancedStopVoiceRecording() {
    // Get the original function
    const originalStopVoiceRecording = window.stopVoiceRecording;

    // Call original function
    if (typeof originalStopVoiceRecording === 'function') {
        originalStopVoiceRecording();
    }

    // If we have audio chunks and recording was long enough
    if (window.audioChunks && window.audioChunks.length > 0 && window.recordingDuration >= 0.5) {
        const audioBlob = new Blob(window.audioChunks, { type: 'audio/wav' });

        // Show processing indicator
        const listeningIndicator = document.getElementById('listeningIndicator');
        if (listeningIndicator) {
            listeningIndicator.textContent = 'Transcribing...';
            listeningIndicator.classList.add('active');
        }

        try {
            // Call STT API
            const transcribedText = await transcribeAudioAPI(audioBlob);

            if (transcribedText) {
                // Send to chat API
                const chatResult = await sendToChatAPI(transcribedText);

                if (chatResult.success && chatResult.bot_response) {
                    // Use existing function to add bot message
                    if (window.addMessageToUI) {
                        const botMessage = {
                            id: Date.now() + 1,
                            type: 'incoming',
                            text: chatResult.bot_response.text,
                            time: window.getCurrentTime ? window.getCurrentTime() : new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
                            isTextMessage: true
                        };
                        window.addMessageToUI(botMessage);

                        // Scroll to bottom
                        const messagesContainer = document.getElementById('messagesContainer');
                        if (messagesContainer) {
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        }
                    }
                }
            }
        } catch (error) {
            console.error('API processing failed:', error);
            // Fallback to simulated reply
            if (window.simulateReply) {
                setTimeout(() => {
                    window.simulateReply(false, false);
                }, 1000);
            }
        } finally {
            // Hide processing indicator
            if (listeningIndicator) {
                listeningIndicator.classList.remove('active');
            }
        }
    }
}

async function enhancedSendMessage() {
    // Get original function
    const originalSendMessage = window.sendMessage;
    const messageInput = document.getElementById('messageInput');
    const text = messageInput ? messageInput.value.trim() : '';

    if (!text) {
        return originalSendMessage();
    }

    // First call original function for UI updates
    originalSendMessage();

    // Then send to API for bot response
    try {
        const chatResult = await sendToChatAPI(text);

        if (chatResult.success && chatResult.bot_response) {
            // Add bot message after a delay
            setTimeout(() => {
                if (window.addMessageToUI) {
                    const botMessage = {
                        id: Date.now() + 1,
                        type: 'incoming',
                        text: chatResult.bot_response.text,
                        time: window.getCurrentTime ? window.getCurrentTime() : new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
                        isTextMessage: true
                    };
                    window.addMessageToUI(botMessage);

                    // Scroll to bottom
                    const messagesContainer = document.getElementById('messagesContainer');
                    if (messagesContainer) {
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }
                }
            }, 1000);
        }
    } catch (error) {
        console.error('Failed to get bot response:', error);
        // Fallback already handled by original simulateReply
    }
}

// Initialize API Integration
async function initAPIIntegration() {
    // Wait for DOM and app.js to load
    await new Promise(resolve => {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', resolve);
        } else {
            resolve();
        }
    });

    // Wait a bit more for app.js to initialize
    await new Promise(resolve => setTimeout(resolve, 100));

    // Initialize API
    await initAPI();

    // Replace functions with enhanced versions if they exist
    if (typeof window.stopVoiceRecording === 'function') {
      //ubaid  const originalStop = window.stopVoiceRecording;
        window.stopVoiceRecording = async function() {
            await enhancedStopVoiceRecording();
        };
    }

    if (typeof window.sendMessage === 'function') {
      //ubaid  const originalSend = window.sendMessage;
        window.sendMessage = async function() {
            await enhancedSendMessage();
        };
    }

    console.log('✅ API Integration initialized');
}

// Start API Integration
initAPIIntegration();