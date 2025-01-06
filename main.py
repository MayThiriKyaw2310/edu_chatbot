import langid
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

api_key = st.secrets.get("OPENAI_API_KEY")

# Initialize GPT model
def initialize_gpt(api_key, model_name="gpt-4"):
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.2,
        openai_api_key=api_key
    )

# Prompts for Burmese and English
burmese_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        သင်သည် ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ မေးခွန်းများအတွက် ကျောင်းသားများကို အကြံဉာဏ်ပေးသူ ဖြစ်ပါသည်။ အတိုချုံးပြီး အရေးပါသည့် အချက်များကိုသာ ပေးပါ။ conversation history နှင့် context အပေါ် အခြေခံပြီး တိုတို ရိုးရှင်းသော ဖြေကြားချက်ပေးပါ။

        စကားဝိုင်းမှတ်တမ်း:
        {conversation_history}

        အခြေခံချက်: {context}

        မေးခွန်း: {question}

        ဖြေကြားချက်: (အရေးပါတဲ့ အချက်တွေကိုပဲ ရေးပါ။ များစွာ မရေးပါနဲ့)
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
max_history_length =500
# Query function with language detection
def query_with_language(llm, question, context="", conversation_history=""):
    try:
        if len(conversation_history) > max_history_length:
            conversation_history = conversation_history[-max_history_length:]

        detected_language, _ = langid.classify(question)
        
        if detected_language == 'my':
            language = "burmese"
        else:
            language = "english"
            
        if language == "burmese":
            formatted_prompt = burmese_prompt.format(
                conversation_history=conversation_history, question=question, context=context
            )
        else:
            formatted_prompt = english_prompt.format(
                conversation_history=conversation_history, question=question, context=context
            )
        
        # Get the model's response
        response = llm.invoke(formatted_prompt)
        response_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Remove repetition based on whole phrases, faster method
        seen = set()
        unique_response = []
        for word in response_text.split():
            if word not in seen:
                seen.add(word)
                unique_response.append(word)
        
        cleaned_response = ' '.join(unique_response)

        return cleaned_response

    except Exception as e:
        return f"An error occurred: {str(e)}"

