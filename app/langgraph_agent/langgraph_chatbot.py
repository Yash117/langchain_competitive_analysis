from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from typing import List, Dict

class LangGraphChatbot:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.initial_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant guiding a user through a competitive analysis of startups."),
            ("human", "I'd like to analyze competitors in the {industry} industry, specifically these companies: {competitors}.\nMy main focus areas are: {focus_areas}.\nCan you help me refine my search for relevant competitive data?"),
        ])
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant guiding a user through a competitive analysis of startups."),
            ("human", "Here are my initial requirements: {initial_requirements}\n\nBased on the initial results and feedback, how can I improve my search to get more relevant competitive insights? Should I expand the scope, focus on specific keywords, or adjust my criteria in any way?"),
        ])

    def get_refined_query(self, messages: List[Dict], industry: str, competitors: str, focus_areas: List[str]) -> str:
        if not messages:
            return self.initial_prompt.format_messages(
                industry=industry, competitors=competitors, focus_areas=", ".join(focus_areas)
            )[0].content
        else:
            initial_requirements = f"Industry: {industry}\nCompetitors: {competitors}\nFocus Areas: {', '.join(focus_areas)}"
            messages = messages + [
                {"role": "human", "content": self.refine_prompt.format_messages(initial_requirements=initial_requirements)[0].content}
            ]
            return self.llm(messages).content
