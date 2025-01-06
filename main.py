import langid
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

api_key = st.secrets.get("OPENAI_API_KEY")

# Initialize GPT model
def initialize_gpt(api_key, model_name="gpt-4"):
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        openai_api_key=api_key
    )
def get_recent_conversation_history(conversation_history, num_lines=5):
    """
    Limit the conversation history to the most recent num_lines exchanges.
    """
    history_lines = conversation_history.split("\n")[-num_lines:]
    return "\n".join(history_lines)

# Helper functions to limit the size of the inputs
def truncate_question(question, max_length=200):
    """
    Truncate the question if it exceeds the max_length.
    If truncating, append "..." to indicate the question is shortened.
    """
    if len(question) > max_length:
        return question[:max_length] + "..."
    return question

def limit_context_length(context, max_chars=500):
    """
    Limit the context to the first max_chars characters.
    If truncating, ensure the cut-off maintains important details.
    """
    if len(context) > max_chars:
        return context[:max_chars] + "..."
    return context

def create_formatted_prompt(conversation_history, question, context, burmese=True, num_history=5, max_question_length=200, max_context_length=500):
    """
    Create the formatted prompt for the language model based on the conversation, question, and context.
    """
    # Limit conversation history to the most recent num_lines exchanges
    conversation_history = get_recent_conversation_history(conversation_history, num_lines=num_history)
    
    # Truncate question and context based on the maximum allowed lengths
    question = truncate_question(question, max_length=max_question_length)
    context = limit_context_length(context, max_chars=max_context_length)

    # Choose prompt template based on language
    if burmese:
        formatted_prompt = burmese_prompt.format(conversation_history=conversation_history, question=question, context=context)
    else:
        formatted_prompt = english_prompt.format(conversation_history=conversation_history, question=question, context=context)

    return formatted_prompt

# Prompts for Burmese and English
burmese_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        သင်သည် ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ မေးခွန်းများအတွက် ကျောင်းသားများကို အကြံဉာဏ်ပေးသူ ဖြစ်ပါသည်။ အတိုချုံးပြီး အရေးပါသည့် အချက်များကိုသာ ပေးပါ။ conversation history နှင့် context အပေါ် အခြေခံပြီး တိုတို ရိုးရှင်းသော ဖြေကြားချက်ပေးပါ။

        စကားဝိုင်းမှတ်တမ်း:
        {conversation_history}

        အခြေခံချက်: {context}

        မေးခွန်း: {question}

        ဖြေကြားချက်: (တိုတောင်းပြီး အရေးပါသော အချက်များပဲ ရေးပါ။ များစွာ မရေးပါနဲ့)
    """
)

english_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        You are an educational advisor helping students with their academic and career-related questions. Provide concise and relevant answers based on the conversation history and context.

        Conversation History:
        {conversation_history}

        Context: {context}

        Question: {question}

        Answer:  
    """
)
def finish_line_fix(response_text):
    """
    Ensure the response ends with a suitable punctuation mark.
    If it doesn't, add a period.
    """
    # Check if the response text ends with a suitable ending punctuation
    if response_text.endswith(('.', '!', '?','။')):
        return response_text
    else:
        # If the response doesn't end properly, add a period or any suitable ending
        return response_text + "."
        
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

        seen = set()
        unique_response = []
        for word in response_text.split():
            if word not in seen:
                seen.add(word)
                unique_response.append(word)
        
        cleaned_response = ' '.join(unique_response)
        final_response = finish_line_fix(cleaned_response)
        return final_response

    except Exception as e:
        return f"An error occurred: {str(e)}"
