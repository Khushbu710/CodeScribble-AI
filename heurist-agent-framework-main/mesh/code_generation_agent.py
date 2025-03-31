from agents.tools import Tools
from core.llm import call_llm

class CodeGenerationAgent:
    def __init__(self, config):
        self.llm = call_llm(config['llm_settings'])
        self.tools = {
            'generate_code': self.generate_code,
            'explain_code': self.explain_code
        }
    
    async def generate_code(self, requirements: str, language: str = 'python'):
        """Generates code based on specified requirements and language."""
        prompt = f"Generate {language} code that meets the following requirements:\n{requirements}\nEnsure the code follows best practices and includes comments."
        return await self.llm.generate(prompt)
    
    async def explain_code(self, code: str):
        """Provides a detailed explanation of the provided code."""
        prompt = f"Explain the following code:\n{code}\nProvide a clear and concise explanation suitable for someone familiar with programming concepts."
        return await self.llm.generate(prompt)
