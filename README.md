# LLM Survivor Bot

A Discord bot that runs a "Survivor"-style game between multiple Large Language Models (LLMs) via the OpenRouter API.

## Description

This bot orchestrates a game where different AI models, treated as players, interact in a Discord channel. They engage in conversation based on prompts and rules, and then vote each other out round by round until only one remains. The bot manages the game state, player turns, conversation flow, voting process, and communicates the game's progress in the designated Discord channel.

## Features

*   **Multi-LLM Gameplay:** Supports various models available through OpenRouter.
*   **Survivor-style Elimination:** Models vote each other out based on conversation and strategy.
*   **Dynamic Turn Management:** Implements round-robin turns initially, transitioning to a more dynamic system where players can target the next speaker.
*   **Conversation History:** Maintains context for the LLMs during their turns.
*   **Voting Mechanism:** Handles vote collection, parsing (with flexible formats), tie-breaking, and elimination announcements.
*   **Discord Integration:** Runs as a Discord bot, interacting within a specific channel.
*   **Configurable:** Key parameters like initial models and conversation length are easily adjustable in the script.
*   **Robust API Handling:** Includes retries, timeouts, and error handling for OpenRouter API calls.
*   **Logging:** Detailed logging to a file (`llm-survivor.log`) for debugging and monitoring.

## Requirements

*   Python 3.8+
*   `discord.py` library
*   `python-dotenv` library
*   `requests` library
*   An OpenRouter API Key
*   A Discord Bot Token

## Setup

1.  **Clone the repository (if applicable) or save the `llm-survivor.py` file.**
2.  **Install dependencies:**
    ```bash
    pip install discord.py python-dotenv requests
    ```
3.  **Create a `.env` file** in the same directory as the script with the following content:
    ```dotenv
    DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN_HERE
    OPENROUTER_KEY=YOUR_OPENROUTER_API_KEY_HERE
    ```
    Replace the placeholder values with your actual Discord Bot Token and OpenRouter API Key.
4.  **Configure Initial Models (Optional):** Edit the `INITIAL_MODELS` list within `llm-survivor.py` to include the specific OpenRouter model IDs you want to participate in the game.

## How to Run

1.  Ensure your Discord bot is invited to your server and has the necessary permissions (Read Messages/View Channels, Send Messages).
2.  Run the Python script:
    ```bash
    python llm-survivor.py
    ```
3.  The bot will log in and print a confirmation message to the console.

## Bot Commands

*   **`!start_survivor`**: (Must be run in the desired game channel) Starts a new LLM Survivor game in the channel where the command is issued.
*   **`!stop_survivor`**: Stops the currently running game prematurely.

## Configuration

Within the `llm-survivor.py` script, you can adjust:

*   `INITIAL_MODELS`: A list of OpenRouter model ID strings to start the game with.
*   `TOTAL_CONVERSATION_TURNS_PER_ROUND`: The maximum number of conversational turns before voting begins each round.
*   `MAX_RETRIES`: Number of times to retry a failed API call for a player's turn or vote.
*   API call parameters (temperature, timeouts, etc.) within the `generate_text`, `run_player_turn`, and `run_voting` functions.

## Logging

The bot logs detailed information about its operation, API calls, game state changes, and errors to a file named `llm-survivor.log` in the same directory as the script.

## Notes

*   The bot requires the `message_content` intent to be enabled in your Discord Developer Portal settings.
*   Ensure the bot has permissions to send messages in the channel where you intend to run the game.
*   OpenRouter API usage costs may apply depending on the models used and the length of the game. 