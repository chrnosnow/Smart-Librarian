/**
 * Sends a chat query to the backend.
 * @param {string} query The user's text message.
 * @returns {Promise<object>} The response data { answer, imageUrl }.
 */
export const postChatMessage = async (query) => {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error('Network response was not ok.');
  return response.json();
};

/**
 * Sends an audio file for transcription.
 * @param {Blob} audioBlob The recorded audio blob.
 * @returns {Promise<object>} The response data { text }.
 */
export const postAudioForTranscription = async (audioBlob) => {
  const formData = new FormData();
  formData.append('audio_file', audioBlob, 'user_recording.wav');

  const response = await fetch('/api/stt', {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error('Network response was not ok.');
  return response.json();
};

/**
 * Requests audio for a given text.
 * @param {string} text The text to convert to speech.
 * @returns {Promise<Blob>} The audio data as a blob.
 */
export const fetchTextToSpeech = async (text) => {
  const response = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error('Network response was not ok.');
  return response.blob();
};