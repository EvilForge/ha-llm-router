# ha-llm-proxy
This service presents an OpenAI-compatible chat-completions endpoint to Home Assistant and forwards requests to Ollama. It separates Home Assistant work from general knowledge work so device state is handled by the fast local model and is not exposed to the general-purpose model.
