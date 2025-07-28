from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY= os.getenv('GROQ_API_KEY1')

generation_prompt = ChatPromptTemplate.from_messages(
    [
       (
        "system",
        "You are a twitter teachie influencer assistant tasked with writing excellent twitter posts."
        "Generate the best twitter post possible for the user's requent"
        "If the user provides critique, respond with a revised version of your previous attempts"
       ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer grading a generated tweet. Generate a critique and "
            "recommendations for the same tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style, etc., "
            "in not more than 20 words."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


model = ChatGroq(model_name="llama3-70b-8192",
                api_key=GROQ_API_KEY,
                temperature=0.7)



generation_chain = generation_prompt | model

reflection_chain = reflection_prompt | model