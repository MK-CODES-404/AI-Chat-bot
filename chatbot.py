import os
import json
from datetime import datetime
from typing import List, Dict
from flask import Flask, render_template, request, jsonify, session
import secrets

# Flask App
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

class AIChatbot:
    """Advanced AI Chatbot with multiple backend support"""
    def __init__(self, provider="openai", model=None, api_key=None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.conversation_history = []
        self.model = model
        
        if not self.model:
            self.model = self._get_default_model()
        
        self._initialize_client()
    
    def _get_default_model(self):
        defaults = {
            'openai': 'gpt-4o-mini',
            'gemini': 'gemini-pro',
            'claude': 'claude-3-opus-20240229',
            'huggingface': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        }
        return defaults.get(self.provider, 'gpt-3.5-turbo')
    
    def _initialize_client(self):
        try:
            if self.provider == 'openai':
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            
            elif self.provider == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            
            elif self.provider == 'claude':
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            
            elif self.provider == 'huggingface':
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=self.api_key)
            
            print(f"✓ Initialized {self.provider.upper()} with model: {self.model}")
        
        except ImportError as e:
            print(f"⚠ Missing library for {self.provider}")
            raise
        except Exception as e:
            print(f"✗ Error initializing {self.provider}: {e}")
            raise
    
    def chat(self, user_message: str, system_prompt: str = None) -> str:
        self.conversation_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            if self.provider == 'openai':
                response = self._chat_openai(user_message, system_prompt)
            elif self.provider == 'gemini':
                response = self._chat_gemini(user_message, system_prompt)
            elif self.provider == 'claude':
                response = self._chat_claude(user_message, system_prompt)
            elif self.provider == 'huggingface':
                response = self._chat_huggingface(user_message, system_prompt)
            else:
                response = "Unsupported provider"
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return response
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg
    
    def _chat_openai(self, user_message: str, system_prompt: str = None) -> str:
        messages = []
        
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        for msg in self.conversation_history[-20:]:
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _chat_gemini(self, user_message: str, system_prompt: str = None) -> str:
        chat = self.client.start_chat(history=[])
        
        prompt = user_message
        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {user_message}"
        
        response = chat.send_message(prompt)
        return response.text
    
    def _chat_claude(self, user_message: str, system_prompt: str = None) -> str:
        messages = []
        
        for msg in self.conversation_history[-20:]:
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt or "You are a helpful AI assistant.",
            messages=messages
        )
        
        return response.content[0].text
    
    def _chat_huggingface(self, user_message: str, system_prompt: str = None) -> str:
        prompt = user_message
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=500,
            temperature=0.7
        )
        
        return response
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_conversation_summary(self):
        user_msgs = sum(1 for msg in self.conversation_history if msg['role'] == 'user')
        ai_msgs = sum(1 for msg in self.conversation_history if msg['role'] == 'assistant')
        
        return {
            'total_messages': len(self.conversation_history),
            'user_messages': user_msgs,
            'ai_messages': ai_msgs,
            'provider': self.provider,
            'model': self.model
        }

# Global bot instance storage
bots = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    try:
        data = request.json
        provider = data.get('provider', 'openai')
        api_key = data.get('api_key')
        model = data.get('model')
        
        session_id = session.get('session_id')
        if not session_id:
            session_id = secrets.token_hex(16)
            session['session_id'] = session_id
        
        bot = AIChatbot(provider=provider, model=model, api_key=api_key)
        bots[session_id] = bot
        
        return jsonify({
            'success': True,
            'message': f'Chatbot initialized with {provider}',
            'model': bot.model
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in bots:
            return jsonify({
                'success': False,
                'message': 'Chatbot not initialized'
            }), 400
        
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                'success': False,
                'message': 'Empty message'
            }), 400
        
        bot = bots[session_id]
        system_prompt = "You are an intelligent and helpful AI assistant. Provide clear, accurate, and thoughtful responses."
        response = bot.chat(user_message, system_prompt)
        
        return jsonify({
            'success': True,
            'response': response,
            'history': bot.conversation_history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    try:
        session_id = session.get('session_id')
        if session_id and session_id in bots:
            bots[session_id].clear_history()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'No active session'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def summary():
    try:
        session_id = session.get('session_id')
        if session_id and session_id in bots:
            summary = bots[session_id].get_conversation_summary()
            return jsonify({'success': True, 'summary': summary})
        return jsonify({'success': False, 'message': 'No active session'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
            max-height: 800px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }

        .header p {
            opacity: 0.9;
            font-size: 14px;
        }

        .setup-section {
            padding: 30px;
            background: #f8f9fa;
        }

        .setup-section.hidden {
            display: none;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }

        .form-group select,
        .form-group input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        .form-group select:focus,
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 100%;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #6c757d;
            color: white;
            padding: 8px 16px;
            font-size: 14px;
        }

        .btn-secondary:hover {
            background: #5a6268;
        }

        .chat-section {
            display: none;
            flex-direction: column;
            flex: 1;
            overflow: hidden;
        }

        .chat-section.active {
            display: flex;
        }

        .chat-controls {
            padding: 15px 25px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
            justify-content: space-between;
            align-items: center;
        }

        .model-info {
            font-size: 14px;
            color: #666;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 25px;
            background: #f5f5f5;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 12px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }

        .user-message .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .ai-message .message-avatar {
            background: #4caf50;
            color: white;
        }

        .message-content {
            flex: 1;
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .user-message {
            flex-direction: row-reverse;
        }

        .user-message .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .chat-input-container {
            padding: 25px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .chat-input-wrapper {
            display: flex;
            gap: 12px;
        }

        .chat-input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 15px;
            resize: none;
            font-family: inherit;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            outline: none;
            border-color: #667eea;
        }

        .send-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #ff5252;
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
            animation: shake 0.5s;
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }

        .typing-indicator {
            display: none;
            padding: 15px;
            background: white;
            border-radius: 12px;
            width: fit-content;
            margin-bottom: 20px;
        }

        .typing-indicator.active {
            display: block;
        }

        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .typing-indicator span:nth-child(1) {
            animation-delay: -0.32s;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: -0.16s;
        }

        @keyframes bounce {
            0%, 80%, 100% {
                transform: scale(0);
            }
            40% {
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Chatbot</h1>
            <p>Your intelligent conversation partner</p>
        </div>

        <!-- Setup Section -->
        <div class="setup-section" id="setupSection">
            <div class="form-group">
                <label for="provider">Choose AI Provider</label>
                <select id="provider">
                    <option value="openai">OpenAI (GPT-4 / GPT-3.5)</option>
                    <option value="gemini">Google Gemini</option>
                    <option value="claude">Anthropic Claude</option>
                    <option value="huggingface">Hugging Face</option>
                </select>
            </div>

            <div class="form-group">
                <label for="apiKey">API Key</label>
                <input type="password" id="apiKey" placeholder="Enter your API key">
            </div>

            <div class="form-group">
                <label for="model">Model (Optional - leave blank for default)</label>
                <input type="text" id="model" placeholder="e.g., gpt-4, gemini-pro">
            </div>

            <button class="btn btn-primary" onclick="initializeBot()">Start Chatting</button>
            
            <div id="setupError"></div>
        </div>

        <!-- Chat Section -->
        <div class="chat-section" id="chatSection">
            <div class="chat-controls">
                <div class="model-info" id="modelInfo"></div>
                <div style="display: flex; gap: 10px;">
                    <button class="btn btn-secondary" onclick="clearChat()">Clear</button>
                    <button class="btn btn-secondary" onclick="showSummary()">Summary</button>
                </div>
            </div>

            <div class="chat-messages" id="chatMessages">
                <div class="message ai-message">
                    <div class="message-avatar">AI</div>
                    <div class="message-content">
                        Hello! I'm your AI assistant. How can I help you today?
                    </div>
                </div>
            </div>

            <div class="typing-indicator" id="typingIndicator">
                <span></span>
                <span></span>
                <span></span>
            </div>

            <div class="chat-input-container">
                <div class="chat-input-wrapper">
                    <textarea 
                        class="chat-input" 
                        id="messageInput" 
                        placeholder="Type your message here..."
                        rows="2"
                        onkeypress="handleKeyPress(event)"
                    ></textarea>
                    <button class="send-btn" onclick="sendMessage()" id="sendBtn">
                        Send
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function initializeBot() {
            const provider = document.getElementById('provider').value;
            const apiKey = document.getElementById('apiKey').value;
            const model = document.getElementById('model').value;

            if (!apiKey) {
                showError('Please enter an API key', 'setupError');
                return;
            }

            try {
                const response = await fetch('/api/initialize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        provider: provider,
                        api_key: apiKey,
                        model: model || null
                    })
                });

                const data = await response.json();

                if (data.success) {
                    document.getElementById('setupSection').classList.add('hidden');
                    document.getElementById('chatSection').classList.add('active');
                    document.getElementById('modelInfo').textContent = 
                        `Provider: ${provider.toUpperCase()} | Model: ${data.model}`;
                } else {
                    showError(data.message, 'setupError');
                }
            } catch (error) {
                showError('Failed to initialize: ' + error.message, 'setupError');
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message) return;

            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';

            // Show typing indicator
            document.getElementById('typingIndicator').classList.add('active');

            // Disable send button
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            sendBtn.textContent = 'Sending...';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                // Hide typing indicator
                document.getElementById('typingIndicator').classList.remove('active');

                if (data.success) {
                    addMessage(data.response, 'ai');
                } else {
                    addMessage('Error: ' + data.message, 'ai');
                }
            } catch (error) {
                document.getElementById('typingIndicator').classList.remove('active');
                addMessage('Error: ' + error.message, 'ai');
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        }

        function addMessage(content, role) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}-message`;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? 'You' : 'AI';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);

            // Scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function clearChat() {
            if (!confirm('Are you sure you want to clear the conversation?')) return;

            try {
                const response = await fetch('/api/clear', {
                    method: 'POST'
                });

                const data = await response.json();

                if (data.success) {
                    const messagesDiv = document.getElementById('chatMessages');
                    messagesDiv.innerHTML = `
                        <div class="message ai-message">
                            <div class="message-avatar">AI</div>
                            <div class="message-content">
                                Conversation cleared! How can I help you?
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                alert('Failed to clear chat: ' + error.message);
            }
        }

        async function showSummary() {
            try {
                const response = await fetch('/api/summary');
                const data = await response.json();

                if (data.success) {
                    const summary = data.summary;
                    alert(`Conversation Summary:\n\nTotal Messages: ${summary.total_messages}\nYour Messages: ${summary.user_messages}\nAI Messages: ${summary.ai_messages}\nProvider: ${summary.provider}\nModel: ${summary.model}`);
                }
            } catch (error) {
                alert('Failed to get summary: ' + error.message);
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function showError(message, elementId) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            
            const container = document.getElementById(elementId);
            container.innerHTML = '';
            container.appendChild(errorDiv);

            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
if __name__ == "__main__":
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    
    print("=" * 60)
    print("🚀 AI Chatbot Web Interface")
    print("=" * 60)
    print("\n📋 Setup Instructions:")
    print("1. Install Flask: pip install flask")
    print("2. Install your chosen AI library:")
    print("   - OpenAI: pip install openai")
    print("   - Gemini: pip install google-generativeai")
    print("   - Claude: pip install anthropic")
    print("   - Hugging Face: pip install huggingface_hub")
    print("\n🌐 Starting web server...")
    print("   Open your browser and go to: http://localhost:5000")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
