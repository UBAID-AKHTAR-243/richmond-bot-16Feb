from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import os
import json
from typing import List, Optional
from datetime import datetime
import uuid

# Try to import stt, but provide fallback if not available
try:
    from stt import transcribe

    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("Warning: stt module not available. Voice transcription will not work.")

# Initialize FastAPI app
app = FastAPI(title="Voice Chat API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # For development only
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static/js", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory chat storage
chat_messages = []


# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main HTML page"""
    try:
        # Check if file exists
        if not os.path.exists("bot.html"):
            print("bot.html not found, creating default page")
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Voice Chat Bot</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        padding: 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-align: center;
                        min-height: 100vh;
                    }
                    .container {
                        max-width: 800px;
                        margin: 50px auto;
                        background: rgba(255,255,255,0.1);
                        padding: 40px;
                        border-radius: 20px;
                        backdrop-filter: blur(10px);
                    }
                    h1 {
                        font-size: 2.5em;
                        margin-bottom: 20px;
                    }
                    .btn {
                        display: inline-block;
                        background: #4f46e5;
                        color: white;
                        padding: 12px 24px;
                        border-radius: 8px;
                        text-decoration: none;
                        margin: 10px;
                        transition: transform 0.3s;
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        background: #4338ca;
                    }
                    .api-list {
                        background: white;
                        color: #333;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 30px 0;
                        text-align: left;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎤 Voice Chat Bot</h1>
                    <p>Richmond Pharmaceutical Assistant</p>
                    <p>Server is running successfully!</p>

                    <div class="api-list">
                        <h3>Available Endpoints:</h3>
                        <p><strong>GET</strong> <a href="/api/health" style="color: #4f46e5;">/api/health</a> - Health check</p>
                        <p><strong>GET</strong> <a href="/api/chat/history" style="color: #4f46e5;">/api/chat/history</a> - Chat history</p>
                        <p><strong>POST</strong> /api/transcribe - Upload audio for transcription</p>
                        <p><strong>POST</strong> /api/chat/send - Send chat message</p>
                        <p><strong>WebSocket</strong> /ws/chat - Real-time chat</p>
                    </div>

                    <a class="btn" href="/api/health">Test API Health</a>
                    <a class="btn" href="/api/chat/history">View Chat History</a>
                </div>
            </body>
            </html>
            """
            # Create the bot.html file for next time
            try:
                with open("bot.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
                print("Created bot.html file")
            except Exception as e:
                print(f"Could not create bot.html: {e}")

            return HTMLResponse(content=html_content)

        # File exists, try to read it
        print(f"Reading bot.html file")
        with open("bot.html", "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Successfully read {len(content)} bytes from bot.html")
            return HTMLResponse(content=content)

    except UnicodeDecodeError:
        # Try with different encoding
        print("UTF-8 failed, trying latin-1")
        try:
            with open("bot.html", "r", encoding="latin-1") as f:
                content = f.read()
                return HTMLResponse(content=content)
        except:
            pass
    except Exception as e:
        print(f"Error reading bot.html: {e}")
        # Fallback HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error loading page</h1>
            <p>Error: {str(e)}</p>
            <p><a href="/api/health">Test API</a></p>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=500)


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio using Whisper"""
    if not STT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Speech-to-text module not available")

    try:
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be audio")

        result = await transcribe(audio)

        return {
            "success": True,
            "transcription": result["text"],
            "language": {
                "code": result["language_code"],
                "name": result["language_name"]
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/chat/send")
async def send_message(message: dict):
    """Send chat message and get response"""
    try:
        text = message.get("text", "")
        user_id = message.get("user_id", "anonymous")

        if not text:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Store user message
        user_msg = {
            "id": str(uuid.uuid4()),
            "text": text,
            "sender": "user",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        chat_messages.append(user_msg)

        # Generate bot response
        bot_response = await generate_bot_response(text)

        bot_msg = {
            "id": str(uuid.uuid4()),
            "text": bot_response,
            "sender": "bot",
            "timestamp": datetime.now().isoformat()
        }
        chat_messages.append(bot_msg)

        # Broadcast to WebSocket
        await manager.broadcast(json.dumps({
            "type": "new_message",
            "message": bot_msg
        }))

        return {
            "success": True,
            "user_message": user_msg,
            "bot_response": bot_msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


async def generate_bot_response(text: str) -> str:
    """Generate bot response"""
    text_lower = text.lower()

    responses = {
        "hello": "Hello! Welcome to Richmond Pharmaceutical. How can I assist you today?",
        "hi": "Hi there! How can I help you with our pharmaceutical services?",
        "price": "Our pricing varies by product. Could you specify which product you're interested in?",
        "delivery": "We offer delivery in 3-5 business days. Free shipping for orders over $100.",
        "product": "We have a wide range of pharmaceutical products. What specific need do you have?",
        "contact": "Contact us at support@richmondpharma.com or call (555) 123-4567.",
        "help": "I can help with product information, pricing, delivery, and general inquiries. What do you need?",
        "thanks": "You're welcome! Is there anything else I can help you with?",
        "bye": "Goodbye! Thank you for visiting Richmond Pharmaceutical.",
        "hours": "Our customer service hours are Monday to Friday, 9 AM to 5 PM."
    }

    for keyword, response in responses.items():
        if keyword in text_lower:
            return response

    # Default responses
    default_responses = [
        "Thank you for your message. How can I assist you further?",
        "I understand. Could you provide more details about your inquiry?",
        "That's interesting! Tell me more about what you need.",
        "I'll help you with that. What specific information are you looking for?",
        "Thanks for reaching out. Let me know how else I can assist you."
    ]

    import random
    return random.choice(default_responses)


@app.get("/api/chat/history")
async def get_chat_history(limit: int = 50, user_id: Optional[str] = None):
    """Get chat history"""
    try:
        if user_id:
            filtered = [msg for msg in chat_messages if msg.get("user_id") == user_id or msg.get("sender") == "bot"]
            messages = filtered[-limit:] if limit > 0 else filtered
        else:
            messages = chat_messages[-limit:] if limit > 0 else chat_messages

        return {"success": True, "messages": messages, "total": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@app.post("/api/chat/clear")
async def clear_chat_history(user_id: Optional[str] = None):
    """Clear chat history"""
    try:
        global chat_messages
        if user_id:
            original_count = len(chat_messages)
            chat_messages = [msg for msg in chat_messages if msg.get("user_id") != user_id]
            count = original_count - len(chat_messages)
        else:
            count = len(chat_messages)
            chat_messages = []

        return {"success": True, "message": "Chat cleared", "cleared_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time chat"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)

                if message_data.get("type") == "message":
                    text = message_data.get("text", "")
                    user_id = message_data.get("user_id", "anonymous")

                    if text:
                        # Generate response
                        response = await generate_bot_response(text)

                        # Send back to client
                        await websocket.send_text(json.dumps({
                            "type": "message",
                            "message": {
                                "id": str(uuid.uuid4()),
                                "text": response,
                                "sender": "bot",
                                "timestamp": datetime.now().isoformat()
                            }
                        }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "voice-chat-api",
        "timestamp": datetime.now().isoformat(),
        "stt_available": STT_AVAILABLE,
        "total_messages": len(chat_messages),
        "active_connections": len(manager.active_connections),
        "endpoints": [
            "/api/transcribe",
            "/api/chat/send",
            "/api/chat/history",
            "/ws/chat"
        ]
    }


@app.get("/app.js")
async def serve_app_js():
    """Serve app.js directly"""
    try:
        if os.path.exists("static/js/app.js"):
            with open("static/js/app.js", "r", encoding="utf-8") as f:
                return Response(content=f.read(), media_type="application/javascript")
        else:
            # Create default app.js if it doesn't exist
            js_content = """
            // Default app.js
            console.log('Voice Chat Bot loaded');
            """
            os.makedirs("static/js", exist_ok=True)
            with open("static/js/app.js", "w", encoding="utf-8") as f:
                f.write(js_content)
            return Response(content=js_content, media_type="application/javascript")
    except Exception as e:
        return Response(content=f"console.error('Error: {e}');", media_type="application/javascript")


@app.get("/api-integration.js")
async def serve_api_integration_js():
    """Serve api-integration.js directly"""
    try:
        if os.path.exists("static/js/api-integration.js"):
            with open("static/js/api-integration.js", "r", encoding="utf-8") as f:
                return Response(content=f.read(), media_type="application/javascript")
        else:
            # Create default api-integration.js
            js_content = """
            // API Integration utilities
            const API = {
                async healthCheck() {
                    const response = await fetch('/api/health');
                    return await response.json();
                }
            };
            window.API = API;
            """
            os.makedirs("static/js", exist_ok=True)
            with open("static/js/api-integration.js", "w", encoding="utf-8") as f:
                f.write(js_content)
            return Response(content=js_content, media_type="application/javascript")
    except Exception as e:
        return Response(content=f"console.error('Error: {e}');", media_type="application/javascript")


@app.get("/test-simple")
async def test_simple():
    """Simple test endpoint"""
    return HTMLResponse(content="<h1>Simple Test</h1><p>If you see this, server is working.</p>")


@app.get("/debug")
async def debug():
    """Debug endpoint"""
    import os
    files = os.listdir('.')
    bot_exists = os.path.exists('bot.html')
    bot_size = os.path.getsize('bot.html') if bot_exists else 0

    return {
        "current_directory": os.getcwd(),
        "files": files,
        "bot.html_exists": bot_exists,
        "bot.html_size": bot_size,
        "python_version": os.sys.version
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)