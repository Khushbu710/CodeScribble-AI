import logging
import os
import threading
import asyncio
import dotenv

from agents.core_agent import CoreAgent
from interfaces.api import FlaskAgent
from interfaces.farcaster_post import FarcasterAgent
from interfaces.telegram import TelegramAgent
from interfaces.twitter_post import TwitterAgent
from mesh.code_generation_agent import CodeGenerationAgent


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_flask(flask_agent):
    """Runs the (blocking) Flask API agent in a separate thread."""
    try:
        logger.info("Starting Flask API agent...")
        flask_agent.run(host="0.0.0.0", port=5005)
    except Exception as e:
        logger.error(f"Flask API agent error: {str(e)}")


def run_telegram(telegram_agent):
    """Run the Telegram agent"""
    try:
        logger.info("Starting Telegram agent...")
        telegram_agent.run()
    except Exception as e:
        logger.error(f"Telegram agent error: {str(e)}")


def run_twitter(twitter_agent):
    """Run the Twitter agent"""
    try:
        logger.info("Starting Twitter agent...")
        twitter_agent.run()
    except Exception as e:
        logger.error(f"Twitter agent error: {str(e)}")


# def reload_environment():
#     """Reload environment variables"""
#     os.environ.clear()
#     dotenv.load_dotenv(override=True)
#     logger.info("Environment variables reloaded")

def reload_environment():
    """Reload environment variables without clearing them."""
    dotenv.load_dotenv(override=True)
    logger.info("Environment variables reloaded")


async def test_code_generation(code_generation_agent):
    """Test the Code Generation agent by generating and explaining sample code."""
    try:
        logger.info("Testing Code Generation agent...")

        # Sample inputs
        requirements = "Create a Python function that returns the Fibonacci sequence up to n."
        code = "def fibonacci(n):\n    a, b = 0, 1\n    sequence = []\n    while a < n:\n        sequence.append(a)\n        a, b = b, a + b\n    return sequence"

        # Generate code
        generated_code = await code_generation_agent.generate_code(requirements)
        logger.info(f"Generated Code:\n{generated_code}")

        # Explain code
        explanation = await code_generation_agent.explain_code(code)
        logger.info(f"Code Explanation:\n{explanation}")

        # Generate README
        readme = await code_generation_agent.generate_readme(code, "Fibonacci Generator")
        logger.info(f"Generated README:\n{readme}")

    except Exception as e:
        logger.error(f"Code Generation agent error: {str(e)}")


def main():
    """Main entry point"""
    try:
        # Initial load of environment variables
        reload_environment()

        # Create shared core agent and interfaces
        core_agent = CoreAgent()
        flask_agent = FlaskAgent(core_agent)
        telegram_agent = TelegramAgent(core_agent)
        twitter_agent = TwitterAgent(core_agent)
        farcaster_agent = FarcasterAgent(core_agent)  # noqa: F841

        # Create Code Generation Agent
        config = {"llm_settings": {"model": "gpt-4"}}  # Example config
        code_generation_agent = CodeGenerationAgent(config)

        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask, args=(flask_agent,), daemon=True)
        flask_thread.start()

        # Start Twitter in a separate thread
        twitter_thread = threading.Thread(target=run_twitter, args=(twitter_agent,), daemon=True)
        twitter_thread.start()

        # Run Code Generation Agent Tests
        asyncio.run(test_code_generation(code_generation_agent))

        # Run Telegram in the main thread
        run_telegram(telegram_agent)

        # Wait for other threads
        flask_thread.join()
        twitter_thread.join()

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
