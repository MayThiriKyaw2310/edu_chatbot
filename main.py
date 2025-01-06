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

# Prompts for Burmese and English
burmese_prompt = PromptTemplate(
    input_variables=["conversation_history", "question", "context"],
    template="""
        သင်သည် ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ မေးခွန်းများအတွက် ကျောင်းသားများကို အကြံဉာဏ်ပေးသူ ဖြစ်ပါသည်။ မေးခွန်းအပေါ် အခြေခံပြီး အကြောင်းအရာကို အောက်ပါအတိုင်း တိုတို ရိုးရှင်းသော ဖြေကြားချက်ပေးပါ။ မေးခွန်း၏ အကြောင်းအရာနှင့် အခြေအနေများကို ရှင်းလင်းပြီး သေချာတဲ့ အကြောင်းအရာကို ထည့်သွင်းပါ။

        **အကြောင်းအရာ**:
        1. **အမည်နှင့် အကြောင်းအရာ**: 
           - သင်၏ အမည်နှင့် လျှောက်လွှာ၏ အကြောင်းအရာကို ဖော်ပြပါ။
        
        2. **ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ အတွေ့အကြုံများ**:
           - သင်၏ ပညာရေးနှင့် အလုပ်အကိုင်ဆိုင်ရာ အတွေ့အကြုံများကို ဖော်ပြပါ။
        
        3. **ရည်ရွယ်ချက်နှင့် အကြံပေးချက်များ**:
           - သင်၏ ရည်ရွယ်ချက်နှင့် အကြံပေးချက်များကို ဖော်ပြပါ။
        
        4. **ဘယ်လို အကောင်အထည်ဖော်နိုင်သည်ကို ဖော်ပြပါ**:
           - သင်၏ ရည်ရွယ်ချက်နှင့် ရေရှည်အကျိုးရှိသော လုပ်ဆောင်ချက်များကို ဖော်ပြပါ။

        5. **အကောင်အထည်ဖော်နိုင်သည့် အချက်အလက်များ**:
           - သင်၏ လက်ရှိလုပ်ငန်းများ၊ စီမံကိန်းများနှင့် အရည်အချင်းများကို ဖော်ပြပါ။

        စကားဝိုင်းမှတ်တမ်း:
        {conversation_history}

        အခြေခံချက်: {context}

        မေးခွန်း: {question}

        ဖြေကြားချက်: (အကျိုးပြုသော အကြောင်းအရာများကို ရေးပါ။)
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
# Query function with language detection
def query_with_language(llm, question, context="", conversation_history=""):
    try:
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

