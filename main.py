import langid
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

# Retrieve the API key securely
api_key = st.secrets.get("OPENAI_API_KEY")

# Ensure the API key is valid
if not api_key:
    st.error("API key not found. Please set it in Streamlit secrets.")
    st.stop()

# Initialize GPT model
def initialize_gpt(api_key, model_name="gpt-3.5-turbo"):
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        openai_api_key=api_key
    )

# Limit conversation history size
def get_recent_conversation_history(conversation_history, num_lines=5):
    """
    Limit the conversation history to the most recent num_lines exchanges.
    """
    history_lines = conversation_history.split("\n")[-num_lines:]
    return "\n".join(history_lines)

# Truncate long questions
def truncate_question(question, max_length=200):
    """
    Truncate the question if it exceeds the max_length.
    """
    if len(question) > max_length:
        return question[:max_length].rsplit(' ', 1)[0] + "..."
    return question

# Limit context length
def limit_context_length(context, max_chars=500):
    """
    Limit the context to the first max_chars characters, avoiding breaking words.
    """
    if len(context) > max_chars:
        return context[:max_chars].rsplit(' ', 1)[0] + "..."
    return context

# Create the formatted prompt
def create_formatted_prompt(conversation_history, question, context, burmese=True, num_history=5, max_question_length=200, max_context_length=500):
    """
    Create the formatted prompt for the language model based on the conversation, question, and context.
    """
    # Limit conversation history to the most recent num_lines exchanges
    conversation_history = get_recent_conversation_history(conversation_history, num_lines=num_history)

    # Truncate question and context
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

# Ensure responses end with proper punctuation
def finish_line_fix(response_text):
    """
    Ensure the response ends with a suitable punctuation mark.
    """
    if response_text.endswith(('.', '!', '?', '။')):
        return response_text
    else:
        return response_text + "."

# Query function with language detection
def query_with_language(llm, question, context="", conversation_history=""):
    try:
        # Detect the input language
        detected_language, _ = langid.classify(question)
        language = "burmese" if detected_language == 'my' else "english"

        # Log detected language for debugging
        st.write(f"Detected language: {language}")

        # Create a formatted prompt
        formatted_prompt = create_formatted_prompt(
            conversation_history, question, context, burmese=(language == "burmese")
        )

        # Log prompt for debugging
        st.write(f"Formatted Prompt:\n{formatted_prompt}")

        # Get model's response
        response = llm.invoke(formatted_prompt)
        response_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Ensure proper punctuation
        final_response = finish_line_fix(response_text)
        return final_response

    except Exception as e:
        return f"Error during query processing: {str(e)}"

# Initialize GPT model
llm = initialize_gpt(api_key)

# Streamlit interface
st.title("Multilingual Educational Advisor")
question = st.text_input("Enter your question:")
context = st.text_area("Provide additional context (optional):")
conversation_history = st.text_area("Conversation history (optional):")

if st.button("Get Answer"):
    response = query_with_language(llm, question, context, conversation_history)
    st.write("Response:")
    st.write(response)
