import { useState, useRef, useEffect } from 'react';
import { postChatMessage, postAudioForTranscription, fetchTextToSpeech } from './apiService';
import AccessibilityToggle from './AccessibilityToggle';
import ThinkingBubble from './ThinkingBubble';

function App() {
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: 'Hello! How can I help you find a book today? You can ask me for recommendations like "a book about magic and friendship" or "something for a fan of war stories".',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const chatWindowRef = useRef(null);

  const [isAccessibilityMode, setIsAccessibilityMode] = useState(false);

// Automatically scroll down when new messages are added
  useEffect(() => {
      if (chatWindowRef.current) {
          chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
      }
  }, [messages]);

  useEffect(() => {
    if (isAccessibilityMode) {
      document.body.classList.add('accessibility-theme');
    } else {
      document.body.classList.remove('accessibility-theme');
    }
    // Optional: Save preference to localStorage
    localStorage.setItem('accessibilityMode', isAccessibilityMode);
  }, [isAccessibilityMode]);

  // Optional: Load preference on initial render
  useEffect(() => {
    const savedMode = localStorage.getItem('accessibilityMode') === 'true';
    setIsAccessibilityMode(savedMode);
  }, []);

  const handleToggleAccessibility = () => {
    setIsAccessibilityMode(prevMode => !prevMode);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: inputValue };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const botResponse = await postChatMessage(inputValue);
      const newBotMessage = {
        sender: 'bot',
        text: botResponse.answer,
        imageUrl: botResponse.imageUrl,
      };
      setMessages((prev) => [...prev, newBotMessage]);
    } catch (error) {
      console.error('Error fetching chat response:', error);
      const errorMessage = { sender: 'bot', text: 'Sorry, I encountered an error. Please try again.' };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleListen = async (text) => {
    try {
      const audioBlob = await fetchTextToSpeech(text);
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      console.error('Error fetching TTS audio:', error);
    }
  };

  const handleRecord = async () => {
    if (isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder.current = new MediaRecorder(stream);
        audioChunks.current = [];

        mediaRecorder.current.ondataavailable = (event) => {
          audioChunks.current.push(event.data);
        };

        mediaRecorder.current.onstop = async () => {
          const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
          setIsLoading(true);
          try {
            const { text } = await postAudioForTranscription(audioBlob);
            setInputValue(text); // Populate input field with transcribed text
          } catch (error) {
            console.error('Error transcribing audio:', error);
          } finally {
            setIsLoading(false);
          }
        };

        mediaRecorder.current.start();
        setIsRecording(true);
      } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access the microphone. Please check your browser permissions.');
      }
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Smart Librarian</h1>
        <AccessibilityToggle
          isEnabled={isAccessibilityMode}
          onToggle={handleToggleAccessibility}
        />
      </header>
      <div className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <p>{msg.text}</p>
            {msg.imageUrl && (
              <img src={msg.imageUrl} alt="Book cover" className="bot-image" />
            )}
            {msg.sender === 'bot' && !msg.imageUrl && index > 0 && (
              <button onClick={() => handleListen(msg.text)} className="tts-button">
                Listen
              </button>
            )}
          </div>
        ))}
        {isLoading && <ThinkingBubble />}
      </div>
      <form className="input-area" onSubmit={handleSubmit}>
        <input
          type="text"
          className="text-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask for a book recommendation..."
          disabled={isLoading || isRecording}
        />
        <button
          type="button"
          onClick={handleRecord}
          className={`action-button mic-button ${isRecording ? 'recording' : ''}`}
          disabled={isLoading}
        >
          {isRecording ? 'Stop' : 'Mic'}
        </button>
        <button type="submit" className="action-button" disabled={isLoading || isRecording}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;