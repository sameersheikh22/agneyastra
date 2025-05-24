# API Keys for various services
# Copy this file to config.py and fill in your actual API keys.
# Ensure config.py is in your .gitignore file!

API_KEYS = {
    "GOOGLE_AI_API_KEY": "YOUR_GOOGLE_AI_KEY_HERE",
    "GROQ_API_KEY": "YOUR_GROQ_KEY_HERE",
    "TAVILY_API_KEY": "YOUR_TAVILY_KEY_HERE",
    "SERPER_DEV_API_KEY": "YOUR_SERPER_DEV_KEY_HERE"
}

# Example function to access a key
def get_api_key(service_name):
    return API_KEYS.get(service_name)
