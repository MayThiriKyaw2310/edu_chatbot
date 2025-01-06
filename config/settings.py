import os

GPT_API_KEY = os.getenv("GPT_API_KEY")

# Application settings
APP_NAME = "Education Chatbot"
APP_VERSION = "1.0.0"
DEFAULT_LANGUAGE = "en" 

# Debug mode
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
