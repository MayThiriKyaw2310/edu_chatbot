import langid
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

api_key = st.secrets.get("OPENAI_API_KEY")

# Initialize GPT model with improved settings
def initialize_gpt(api_key, model_name="gpt-4"):
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.2,  # Lower temperature for more focused and relevant answers
        openai_api_key=api_key
    )

# Helper functions to limit the size of the inputs
def get_recent_conversation_history(conversation_history, num_lines=5):
    """
    Limit the conversation history to the most recent num_lines exchanges.
    """
    history_lines = conversation_history.split("\n")[-num_lines:]
    return "\n".join(history_lines)

def truncate_question(question, max_length=200):
    """
    Truncate the question if it exceeds the max_length.
    """
    if len(question) > max_length:
        return question[:max_length] + "..."
    return question

def limit_context_length(context, max_chars=500):
    """
    Limit the context to the first max_chars characters.
    """
    return context[:max_chars] if len(context) > max_chars else context

# Function to create formatted prompt with enhanced clarity
def create_formatted_prompt(conversation_history, question, context, burmese=True, num_history=5, max_question_length=200, max_context_length=500):
    # Limit conversation history
    conversation_history = get_recent_conversation_history(conversation_history, num_lines=num_history)
    # Truncate question
    question = truncate_question(question, max_length=max_question_length)
    # Limit context length
    context = limit_context_length(context, max_chars=max_context_length)

    # Choose prompt template based on language
    if burmese:
        formatted_prompt = burmese_prompt.format(conversation_history=conversation_history, question=question, context=context)
    else:
        formatted_prompt = english_prompt.format(conversation_history=conversation_history, question=question, context=context)

    return formatted_prompt

# Updated Prompts for Burmese and English (Clearer Instructions)
burmese_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        သင်သည် ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ မေးခွန်းများအတွက် ကျောင်းသားများကို အကြံဉာဏ်ပေးသူ ဖြစ်ပါသည်။ အကြောင်းအရာ အရေးပါသည့်အချက်များကိုသာ ရေးပါ။ conversation history နှင့် context ကို အခြေခံပြီး ရိုးရှင်းပြီး ထိရောက်သော ဖြေကြားချက်များ ပေးပါ။

        စကားဝိုင်းမှတ်တမ်း:
        {conversation_history}

        အခြေခံချက်: {context}

        မေးခွန်း: {question}

        ဖြေကြားချက်: (ရိုးရှင်းပြီး အရေးပါတဲ့ အချက်များသာ ရေးပါ။ များစွာ မရေးပါနဲ့)
    """
)

english_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        You are an educational advisor helping students with their academic and career-related questions. Provide clear, concise, and relevant answers based on the conversation history and context.

        Conversation History:
        {conversation_history}

        Context: {context}

        Question: {question}

        Answer: (Please write a concise, clear, and focused response, providing only the most important details. Avoid unnecessary elaboration.)
    """
)

# Define the clean_response function
def clean_response(response):
    """
    Clean and format the response text.
    This can include removing unnecessary punctuation, fixing grammar,and eliminating redundant or repeated information, etc.
    """
    # Example: Strip whitespace and remove multiple newlines or extra spaces.
    response = response.strip()
    response = ' '.join(response.split())  # Remove extra spaces
    return response

# Query function with language detection
def query_with_language(llm, question, context="", conversation_history=""):
    try:
        detected_language, _ = langid.classify(question)
        
        if detected_language == 'my':
            language = "burmese"
        else:
            language = "english"
        
        # Create a formatted prompt with a limited size
        formatted_prompt = create_formatted_prompt(
            conversation_history, question, context, burmese=(language=="burmese")
        )
        
        # Get the model's response
        response = llm.invoke(formatted_prompt)
        response_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        
        # Clean and format the response
        cleaned_response = clean_response(response_text)
        
        return cleaned_response

    except Exception as e:
        return f"An error occurred: {str(e)}"
