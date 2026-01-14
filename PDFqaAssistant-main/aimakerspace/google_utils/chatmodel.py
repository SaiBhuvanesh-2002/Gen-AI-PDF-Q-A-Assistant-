import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


class ChatGemini:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if self.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY is not set")
        
        genai.configure(api_key=self.google_api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def run(self, messages, text_only: bool = True, **kwargs):
        """
        Adapts the OpenAI-style 'messages' list to Gemini's format.
        OpenAI format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        
        # Simple extraction of the last user message and context from system prompts.
        # Gemini enables 'system_instruction' at model init or we can prepend it.
        # For simplicity in this drop-in replacement, we will concatenate system prompt + user context.
        
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                user_content += msg["content"] + "\n"
        
        full_prompt = system_content + user_content
        
        start_chat = self.model.start_chat()
        response = start_chat.send_message(full_prompt, **kwargs)

        if text_only:
            return response.text

        return response
    
    async def astream(self, messages, **kwargs):
        """
        Async streaming support.
        """
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
            
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                user_content += msg["content"] + "\n"
        
        full_prompt = system_content + user_content
        
        # Note: google-generativeai's async support might be limited depending on version.
        # We will use the async method if available, otherwise synchronous iterator wrapped in async.
        # The library supports async_send_message in newer versions.
        
        response = await self.model.generate_content_async(full_prompt, stream=True, **kwargs)
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
