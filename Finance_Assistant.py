import os
from openai import OpenAI
from mem0 import Memory

# Set the OpenAI and Groq API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

class PersonalAssistant:
    def __init__(self, role_description="Answer any questions asked in briefest way possible, one to 5 words", user_description="User is a beginner and require explanation in the simplest manner", user_id="user-1"):
        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": "mixtral-8x7b-32768",
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            }
        }
        self.client = OpenAI()
        self.memory = Memory.from_config(config)
        self.role_description = role_description
        self.messages = [{"role": "system", "content": f"{role_description}"}]
        self.messages = [{"role": "system", "content": f"{role_description}"}]
        result = self.memory.add(f"""User Description: {user_description}""", user_id=user_id)
        self.memory_id = result[0]['id']
        self.user_id = user_id

    def ask_question(self, question, context):
        # Fetch previous related memories
        previous_memories = self.search_memories(f"find context to this question:'{question}' and summarize briefest way possible")
        user_expertise = self.search_memories("What is the expertise of user")
        if not previous_memories:
            previous_memories = "None"
        self.messages.append({"role": "user", "content": f"""
            User Expertise: {user_expertise}

            Context for reference (This is up-to-date context): {context}

            Previous Memories of conversation related to the question: {previous_memories}
            
            Question: {question}
        """})
        # Generate response using GPT-4o-mini
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        answer = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": answer})

        # Store the question in memory of user_id
        self.memory.add(f"""Question asked: {question}, Answer Generated: {answer}""", user_id=self.user_id)
        return answer

    def get_memories(self):
        memories = self.memory.get_all(user_id=self.user_id)
        return [m['text'] for m in memories]

    def search_memories(self, query):
        memories = self.memory.search(query, user_id=self.user_id)
        return [m['text'] for m in memories]
    
    def update_user_expertise_memory(self, user_description):
        try:
            self.memory.update(memory_id=self.memory_id, data=user_description)
            print(f"Memory with ID {self.memory_id} updated successfully.")
        except Exception as e:
            print(f"Failed to update memory: {e}")

    def add_user_chat_memory(self, Conversation):
        try:
            self.memory.add(user_id=self.user_id, data=f"""{Conversation}""")
            print(f"Memory with ID {self.user_id} updated successfully.")
        except Exception as e:
            print(f"Failed to update memory: {e}")

roles_of_different_agents = [
    {"role": "assistant", "content": """I am an expert at categorizing a user question into either 'Can be found in Annual Report,' 'Typically Searched on the Internet,' or 'Can be answered without reference.' I will categorize the user question into these categories and provide just the name of the category, nothing else."""},

    {"role": "assistant", "content": """As a grader, I assess the relevance of a retrieved document to a user's question. If the document contains keywords or semantic meaning related to the user's question, I grade it as relevant. I give a binary score, 'yes' or 'no,' to indicate whether the document is relevant to the question."""},

    {"role": "assistant", "content": """I'm here to help with your question-answering tasks. If I have the relevant information, I'll provide a detailed response based on the context I have. If not, I'll let you know that I don't have the answer. My goal is to be as thorough as possible, tailored to your level of expertise."""},

    {"role": "assistant", "content": """I'm an assistant specializing in question-answering tasks. I provide concise and accurate responses based on my knowledge, aiming to keep each answer to a maximum of three sentences. My goal is to deliver clear, direct information without unnecessary detail."""},

    {"role": "assistant", "content": """I'm an assistant specializing in question-answering tasks. I provide concise and accurate responses based on my knowledge, aiming to keep each answer to a maximum of three sentences. My goal is to deliver clear, direct information without unnecessary detail."""},

    {"role": "assistant", "content": """I’m a financial grader tasked with determining whether an LLM-generated response is grounded in or supported by a set of retrieved facts. I need to give a binary score of 'yes' or 'no'—'yes' means that the answer is indeed supported by the set of facts."""},

    {"role": "assistant", "content": """I am a financial grader evaluating whether an answer effectively addresses or resolves a question. If the answer meets the criteria and resolves the question, I will give it a binary score of 'yes.' If it does not resolve the question, I will assign a 'no.'."""},
]