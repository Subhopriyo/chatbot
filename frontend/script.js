// ============================================
// CONFIGURATION & STATE
// ============================================

const API_URL = 'http://localhost:5000/api';
let currentConversationId = null;
let isTyping = false;
let recognition = null;
let speechSynthesis = window.speechSynthesis;
let currentTypewriterTimeout = null;
let uploadedFiles = [];

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    setupVoiceRecognition();
});

async function initializeApp() {
    await checkBackendStatus();
    await loadConversations();
    await loadSuggestions();
    setupDragAndDrop();
    loadTheme();
}

function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }
}

// ============================================
// BACKEND CONNECTION
// ============================================

async function checkBackendStatus() {
    try {
        const response = await fetch('http://localhost:5000/health');
        if (response.ok) {
            const data = await response.json();
            updateStatus(true, 'Connected');
            console.log('✅ Backend connected:', data);
        } else {
            throw new Error('Backend not responding');
        }
    } catch (error) {
        console.error('❌ Backend connection failed:', error);
        updateStatus(false, 'Disconnected');
    }
}

function updateStatus(connected, text) {
    const statusText = document.querySelector('.status-text');
    const statusDot = document.querySelector('.status-dot');
    
    if (statusText) {
        statusText.textContent = text;
    }
    
    if (statusDot) {
        statusDot.style.background = connected ? 'var(--success-color)' : 'var(--error-color)';
    }
}

// ============================================
// THEME TOGGLE
// ============================================

function toggleTheme() {
    const body = document.body;
    const themeIcon = document.getElementById('themeIcon');
    const currentTheme = body.getAttribute('data-theme');
    
    if (currentTheme === 'dark') {
        body.removeAttribute('data-theme');
        themeIcon.className = 'fas fa-moon';
        localStorage.setItem('theme', 'light');
    } else {
        body.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-sun';
        localStorage.setItem('theme', 'dark');
    }
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    const themeIcon = document.getElementById('themeIcon');
    
    if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        if (themeIcon) themeIcon.className = 'fas fa-sun';
    }
}

// ============================================
// SIDEBAR MANAGEMENT
// ============================================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('active');
}

function switchTab(tab) {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    // Here you can add logic to switch between files and chats view
    console.log(`Switched to ${tab} tab`);
}

// ============================================
// FILE UPLOAD HANDLING
// ============================================

function setupDragAndDrop() {
    const chatArea = document.getElementById('chatArea');
    
    ['dragover', 'dragleave', 'drop'].forEach(eventName => {
        chatArea.addEventListener(eventName, handleDragEvents);
    });
}

function handleDragEvents(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const chatArea = document.getElementById('chatArea');
    
    if (e.type === 'dragover') {
        chatArea.style.opacity = '0.5';
    } else if (e.type === 'dragleave') {
        chatArea.style.opacity = '1';
    } else if (e.type === 'drop') {
        chatArea.style.opacity = '1';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload({ target: { files: files } });
        }
    }
}

async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files || !files.length) return;
    
    hideWelcome();
    
    // Create conversation if none exists
    if (!currentConversationId) {
        console.log('🆕 Creating new conversation for file upload...');
        try {
            await createNewConversation();
        } catch (error) {
            addMessageToUI('❌ Failed to create conversation for file upload', 'assistant');
            return;
        }
    }
    
    // Show upload modal
    showUploadProgress(files);
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        await uploadSingleFile(file, i);
    }
    
    // Hide modal after a delay
    setTimeout(() => {
        hideUploadProgress();
    }, 1500);
    
    // Reset file input
    event.target.value = '';
    
    // Reload suggestions
    await loadSuggestions();
}

function showUploadProgress(files) {
    const modal = document.getElementById('uploadModal');
    const progressContainer = document.getElementById('progressContainer');
    
    if (!modal || !progressContainer) return;
    
    progressContainer.innerHTML = '';
    
    Array.from(files).forEach((file, index) => {
        const progressItem = document.createElement('div');
        progressItem.className = 'progress-item';
        progressItem.id = `progress-${index}`;
        progressItem.innerHTML = `
            <div class="progress-info">
                <span class="progress-name">${file.name}</span>
                <span class="progress-percent">0%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: 0%"></div>
            </div>
        `;
        progressContainer.appendChild(progressItem);
    });
    
    modal.classList.add('active');
}

function updateUploadProgress(index, percent) {
    const progressItem = document.getElementById(`progress-${index}`);
    if (!progressItem) return;
    
    const percentSpan = progressItem.querySelector('.progress-percent');
    const progressFill = progressItem.querySelector('.progress-bar-fill');
    
    if (percentSpan) percentSpan.textContent = `${percent}%`;
    if (progressFill) progressFill.style.width = `${percent}%`;
}

function hideUploadProgress() {
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

async function uploadSingleFile(file, index) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (currentConversationId) {
        formData.append('conversation_id', currentConversationId);
    }
    
    console.log(`📎 Uploading ${file.name} to conversation ${currentConversationId}`);
    
    // Add file to UI immediately
    addFileToSidebar(file);
    addMessageToUI(`📎 Uploading: ${file.name}`, 'user');
    
    try {
        // Simulate progress
        updateUploadProgress(index, 30);
        
        const response = await fetch(`${API_URL}/files/upload`, {
            method: 'POST',
            body: formData
        });
        
        updateUploadProgress(index, 70);
        
        const result = await response.json();
        
        updateUploadProgress(index, 100);
        
        if (response.ok) {
            let message = `✅ ${result.message || 'File uploaded successfully'}`;
            if (result.columns) {
                message += `\n📊 Columns: ${result.columns.join(', ')}`;
            }
            await addMessageToUI(message, 'assistant', {}, true);
            
            // Store file info
            uploadedFiles.push({
                name: file.name,
                size: formatFileSize(file.size),
                type: file.type,
                uploadedAt: new Date()
            });
        } else {
            addMessageToUI(`❌ Error: ${result.error || 'Upload failed'}`, 'assistant');
        }
    } catch (error) {
        updateUploadProgress(index, 100);
        addMessageToUI(`❌ Upload failed: ${error.message}`, 'assistant');
    }
}

function addFileToSidebar(file) {
    const filesList = document.getElementById('filesList');
    const emptyState = document.getElementById('emptyState');
    
    if (emptyState) {
        emptyState.style.display = 'none';
    }
    
    const fileIcon = getFileIcon(file.name);
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.innerHTML = `
        <div class="file-icon ${fileIcon.class}">
            <i class="fas ${fileIcon.icon}"></i>
        </div>
        <div class="file-info">
            <div class="file-name">${file.name}</div>
            <div class="file-size">${formatFileSize(file.size)}</div>
        </div>
        <div class="file-actions">
            <button class="file-action-btn" onclick="removeFile(this)" title="Remove">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    filesList.appendChild(fileItem);
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    
    const icons = {
        'pdf': { icon: 'fa-file-pdf', class: 'pdf' },
        'doc': { icon: 'fa-file-word', class: 'word' },
        'docx': { icon: 'fa-file-word', class: 'word' },
        'csv': { icon: 'fa-file-csv', class: 'csv' },
        'txt': { icon: 'fa-file-alt', class: 'txt' }
    };
    
    return icons[ext] || { icon: 'fa-file', class: 'default' };
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function removeFile(button) {
    const fileItem = button.closest('.file-item');
    if (fileItem) {
        fileItem.remove();
    }
    
    // Check if files list is empty
    const filesList = document.getElementById('filesList');
    const emptyState = document.getElementById('emptyState');
    const items = filesList.querySelectorAll('.file-item');
    
    if (items.length === 0 && emptyState) {
        emptyState.style.display = 'block';
    }
}

// ============================================
// CONVERSATION MANAGEMENT
// ============================================

async function loadConversations() {
    try {
        const response = await fetch(`${API_URL}/chat/conversations`);
        const data = await response.json();
        
        displayConversations(data.conversations);
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

function displayConversations(conversations) {
    // This would populate the "Recent Chats" tab
    console.log('Conversations:', conversations);
}

async function createNewConversation() {
    try {
        const response = await fetch(`${API_URL}/chat/conversations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New Chat' })
        });
        
        const data = await response.json();
        currentConversationId = data.conversation.id;
        
        console.log(`✨ New conversation created with ID: ${currentConversationId}`);
        
    } catch (error) {
        console.error('Error creating conversation:', error);
        throw error;
    }
}

async function clearChat() {
    if (!confirm('Clear current chat and start fresh?')) return;
    
    if (currentConversationId) {
        try {
            await fetch(`${API_URL}/chat/conversations/${currentConversationId}`, {
                method: 'DELETE'
            });
            console.log(`🗑️ Deleted conversation ${currentConversationId}`);
        } catch (error) {
            console.error('Error deleting conversation:', error);
        }
    }
    
    currentConversationId = null;
    document.getElementById('chatArea').innerHTML = '';
    
    // Show welcome again
    showWelcome();
    
    await loadConversations();
    await createNewConversation();
}

// ============================================
// MESSAGE HANDLING
// ============================================

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message || isTyping) return;
    
    input.value = '';
    hideWelcome();
    
    // Add user message to UI
    addMessageToUI(message, 'user');
    showTyping();
    
    try {
        const response = await fetch(`${API_URL}/chat/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                message: message
            })
        });
        
        const result = await response.json();
        hideTyping();
        
        if (response.ok) {
            // Update current conversation ID if new
            if (!currentConversationId) {
                currentConversationId = result.conversation_id;
                await loadConversations();
                console.log(`📝 New conversation created: ${currentConversationId}`);
            }
            
            console.log(`🤖 Response mode: ${result.mode || 'unknown'}`);
            
            // Add AI response with metadata and typewriter effect
            const metadata = {
                chart_url: result.chart_url,
                chart_info: result.chart_info,
                sources: result.sources,
                confidence: result.confidence,
                mode: result.mode
            };
            
            await addMessageToUI(result.answer, 'assistant', metadata, true);
            
        } else {
            addMessageToUI(`❌ Error: ${result.error || 'Unknown error'}`, 'assistant');
        }
        
    } catch (error) {
        hideTyping();
        addMessageToUI(`❌ Connection error: ${error.message}`, 'assistant');
    }
}

// ============================================
// TYPEWRITER EFFECT
// ============================================

function typewriterEffect(element, text, speed = 15) {
    return new Promise((resolve) => {
        element.classList.add('typing');
        let index = 0;
        
        if (currentTypewriterTimeout) {
            clearTimeout(currentTypewriterTimeout);
        }
        
        function type() {
            if (index < text.length) {
                // Handle HTML tags
                if (text.charAt(index) === '<') {
                    const closingIndex = text.indexOf('>', index);
                    if (closingIndex !== -1) {
                        element.innerHTML += text.substring(index, closingIndex + 1);
                        index = closingIndex + 1;
                        currentTypewriterTimeout = setTimeout(type, 0);
                        return;
                    }
                }
                
                element.innerHTML += text.charAt(index);
                index++;
                currentTypewriterTimeout = setTimeout(type, speed);
            } else {
                element.classList.remove('typing');
                resolve();
            }
        }
        
        type();
    });
}

function formatText(text) {
    // Remove asterisks for bold/emphasis
    text = text.replace(/\*\*/g, '');
    text = text.replace(/\*/g, '');
    
    // Format numbered lists
    text = text.replace(/^(\d+)\.\s+([^\n]+)/gm, '<div class="insight-item"><span class="insight-number">$1.</span> <span class="insight-text">$2</span></div>');
    
    // Format sections with headings
    if (text.includes('Possible Insights:')) {
        text = text.replace('Possible Insights:', '<div class="insights-title">Possible Insights:</div>');
    }
    
    // Convert newlines to <br>
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

async function addMessageToUI(content, role, metadata = {}, useTypewriter = false) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role === 'user' ? 'user' : 'bot'}`;
    
    const avatar = role === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    let messageHTML = `
        <div class="avatar">${avatar}</div>
        <div class="message-content">
            <span class="message-text"></span>
        </div>
    `;
    
    messageDiv.innerHTML = messageHTML;
    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    
    const messageText = messageDiv.querySelector('.message-text');
    const formattedContent = formatText(content);
    
    if (useTypewriter && role === 'assistant') {
        isTyping = true;
        await typewriterEffect(messageText, formattedContent, 15);
        isTyping = false;
    } else {
        messageText.innerHTML = formattedContent;
    }
    
    chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
    const chatArea = document.getElementById('chatArea');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatArea.appendChild(typingDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function hideTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

function hideWelcome() {
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'none';
}

function showWelcome() {
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'flex';
}

// ============================================
// VOICE INPUT
// ============================================

function setupVoiceRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById('messageInput').value = transcript;
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
        };
    }
}

function toggleVoiceInput() {
    if (!recognition) {
        alert('Voice recognition not supported in your browser');
        return;
    }
    
    const voiceBtn = document.querySelector('.voice-btn i');
    
    if (recognition.isListening) {
        recognition.stop();
        voiceBtn.className = 'fas fa-microphone';
    } else {
        recognition.start();
        voiceBtn.className = 'fas fa-microphone-slash';
        recognition.isListening = true;
        
        recognition.onend = () => {
            voiceBtn.className = 'fas fa-microphone';
            recognition.isListening = false;
        };
    }
}

// ============================================
// SUGGESTIONS
// ============================================

async function loadSuggestions() {
    try {
        const response = await fetch(`${API_URL}/analysis/suggestions`);
        const data = await response.json();
        
        // Handle suggestions if needed
        console.log('Suggestions:', data.suggestions);
    } catch (error) {
        console.error('Error loading suggestions:', error);
    }
}

// ============================================
// EVENT HANDLERS
// ============================================

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    
    return date.toLocaleDateString();
}

// ============================================
// CLEANUP
// ============================================

window.addEventListener('beforeunload', () => {
    if (currentTypewriterTimeout) {
        clearTimeout(currentTypewriterTimeout);
    }
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
    }
});