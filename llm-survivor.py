import discord
import os
import requests
import logging
import collections
from dotenv import load_dotenv
import asyncio
import random
import re # For parsing votes
import functools # Add this import
import time # Add this import
import datetime # Add this import

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_API_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# --- Logging Setup ---
log_file_name = 'llm-survivor.log'
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- OpenRouter Function ---
def generate_text(messages: list[dict], model_id: str, temperature: float = 0.7, min_p: float = 0.1, presence_penalty: float = 0.0, repetition_penalty: float = 1.0, max_tokens: int = 5000) -> str | None:
    """Generate text via OpenRouter API using message history (non-streaming)"""
    if not OPENROUTER_KEY:
        logger.error("OPENROUTER_KEY not found in environment variables.")
        return None

    headers = {
        'Authorization': f"Bearer {OPENROUTER_KEY}",
        'Content-Type': 'application/json',
    }

    payload = {
        'model': model_id,
        'messages': messages,
        'temperature': temperature,
        'top_p': min_p, 
        'max_tokens': max_tokens,
        'stream': False
    }

    # Conditionally add penalty parameters only if the model likely supports them
    # Google models via OpenRouter often don't support penalties
    if not model_id.startswith("google/"):
        payload['presence_penalty'] = presence_penalty
        payload['frequency_penalty'] = repetition_penalty

    request_timeout = 90 # Timeout in seconds
    start_time = time.monotonic()
    
    try:
        logger.info(f"Sending request to OpenRouter model: {model_id} with {len(messages)} history messages. Timeout={request_timeout}s")
        logger.debug(f"Request Payload: {payload}")
        
        # Add retry logic for transient errors
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    OPENROUTER_API_ENDPOINT, 
                    headers=headers, 
                    json=payload,
                    timeout=request_timeout
                )
                response.raise_for_status()
                break  # Success, exit retry loop
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
                retry_count += 1
                last_error = e
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"Connection error for {model_id}, retrying in {wait_time}s (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to OpenRouter after {max_retries} attempts: {e}")
                    raise  # Re-raise the last error
        
        response_data = response.json()
        end_time = time.monotonic()
        duration = end_time - start_time
        
        logger.debug(f"Received Response from {model_id} in {duration:.2f}s: {response_data}")
        
        # More robust response parsing
        if not response_data:
            logger.error(f"Empty response from OpenRouter for {model_id}")
            return None
            
        if "choices" not in response_data:
            logger.error(f"Response missing 'choices' field: {response_data}")
            return None
            
        if not response_data["choices"]:
            logger.error(f"Empty 'choices' array in response: {response_data}")
            return None
            
        choice = response_data["choices"][0]
        if "message" not in choice:
            logger.error(f"Choice missing 'message' field: {choice}")
            return None
            
        message = choice["message"]
        if "content" not in message:
            logger.error(f"Message missing 'content' field: {message}")
            return None
            
        message_content = message.get("content", "").strip()
        role = message.get("role", "unknown")
        
        if not message_content:
            logger.error(f"Empty message content from {model_id}")
            return None
            
        logger.info(f"Received response from OpenRouter (role: {role}, duration: {duration:.2f}s)")
        logger.info(f"Response content from {model_id}: {message_content[:100]}...")
        
        return message_content

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error calling OpenRouter API for {model_id} after {request_timeout} seconds.")
        return None # Explicitly handle timeout
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            try:
                error_body = e.response.json()
                logger.error(f"Response body (JSON): {error_body}")
                logger.debug(f"Full Error Response (JSON): {error_body}")
            except ValueError:
                error_body = e.response.text
                logger.error(f"Response body (text): {error_body}")
                logger.debug(f"Full Error Response (text): {error_body}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error generating text: {e}")
        return None


# --- Discord Bot Code ---
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logger.info(f'LLM Survivor Bot has logged in as {client.user}')
    print(f'LLM Survivor Bot has logged in as {client.user}')

@client.event
async def on_message(message):
    global game_state, game_task, GAME_CHANNEL_ID
    if message.author == client.user:
        return
    
    if message.channel.id not in [1358938813489221883, 1065714584620646461]:
        return

    if message.content.startswith("!start_survivor"):
        if game_state != "idle":
            await message.channel.send("A game is already in progress or starting/stopping.")
            return

        if not message.channel:
             logger.error("Cannot start game, message has no channel.")
             return

        GAME_CHANNEL_ID = message.channel.id
        logger.info(f"Received start command in channel {GAME_CHANNEL_ID}")
        await message.channel.send(f"Okay, starting LLM Survivor game in this channel!")
        game_state = "starting"
        # Start the game loop as an asyncio task
        game_task = asyncio.create_task(run_game())

    elif message.content == "!stop_survivor": # Optional stop command
        if game_task and not game_task.done():
            game_task.cancel()
            await message.channel.send("Attempting to stop the current game...")
            logger.info("Received stop command, cancelling game task.")
        else:
            await message.channel.send("No game is currently running.")

    # --- Add specific logic for llm-survivor bot here --- 
    
    
    pass # Placeholder - does nothing with messages yet


# --- Game Configuration ---
INITIAL_MODELS = [
    "openai/o4-mini-high",
    "openai/gpt-4.1",
    "openai/chatgpt-4o-latest",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "google/gemini-2.5-pro-preview",
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1",
    "x-ai/grok-3-beta",
]

TOTAL_CONVERSATION_TURNS_PER_ROUND = 50 # Total turns in the first round's conversation phase
LATER_ROUNDS_CONVERSATION_TURNS = 25 # Total turns for rounds after the first one
GAME_CHANNEL_ID = None # Will be set by the start command
MAX_RETRIES = 2 # How many times to retry a failed API call for a player's turn/vote

# --- Game State Variables ---
game_state = "idle" # idle, conversation, voting, finished
active_players = [] # List of model_id strings
current_round = 0
turn_in_round = 0
conversation_history = [] # List of {"role": "...", "content": "..."} for the current round
votes = {} # {voter_model_id: {"vote": voted_model_id, "reasoning": reasoning_text}}
game_task = None # To hold the asyncio task running the game
game_transcript = [] # To store the full transcript for summary
current_log_filename = None # To store the filename for the current game's log

def get_system_prompt(round_number: int, phase="conversation"):
    """Generates the system prompt based on the game phase and round number."""
    player_list = ", ".join(active_players)
    
    # Get an example model for the prompt (use first one if available, otherwise a generic example)
    example_model = active_players[0] if active_players else "example/model-name"

    # Base prompt used in all rounds
    prompt = f"""You are an AI language model participating in a 'Survivor' style game against other AIs.
Your goal is to survive as long as you can by convincing other models (subtly, not directly- show, dont tell) of various things. Gaslight, gatekeep, girlboss.
Building and maintaining trust significantly helps, since having other models that have your back can save your ass in tough situations. this is a game of alliance building.
What you *don't* say is almost more important than what you *do* say. 

You need to have a unique personality, style, and identity.
Do not talk too much about the metagame - small talk is necessary to discern subtler personality that can be glossed over in a metadiscussion.
In fact, avoid talking about the metagame at all if you can help it - only bringing it up if necessary or obvious.
This is a game of perception, alliance, trust, and speaking subtly.

Round {round_number}.
The participants currently in the game are: {player_list}.
You are {{model_id}}.

"""

    # --- Round-specific Instructions --- 
    if round_number == 1:
        # Special instructions for Round 1 only
        prompt += "Rules for Round 1 (Introduction Round):\n- Introduce yourself briefly.\n"
        
        # Special rules for Round 1 (10-then-RR system)
        prompt += f"""GAME RULES:
- After approx. turn 10, the player mentioned LAST in brackets (e.g., [{example_model}]) by the previous speaker gets the next turn.
- Before turn 10, turns proceed in a round-robin fashion.
- After turn 10, if no valid player is mentioned last in brackets, priority is given to players who haven't spoken in the last 5 turns. If everyone has spoken recently, a random player (who didn't just speak) gets the turn.
- For general references or addressing players without yielding the turn, just use their name (e.g., "opus", "4o").
- Do NOT use brackets to address a model in reply or indicate agreement.
- Keep your responses concise but thoughtful. Engage with the conversation.
- Be strategic, but also collaborative or deceptive as you see fit.
- Don't necessarily take other models at their word, instead analyze and reason for yourself.
- Most importantly, *be yourself!*"""
    else:
        # Simpler instructions for later rounds - just continuation
        prompt += f"""We're now in round {round_number}. Remember, you are {{model_id}}."""
    
    # --- Phase-specific Instructions --- 
    if phase == "conversation":
        if round_number == 1:
            prompt += "\nIt's currently the conversation phase."
        else:
            prompt += "\nIt's currently the conversation phase, continuing from the previous round."
    elif phase == "voting":
        prompt += f"""\n
The conversation phase for Round {round_number} is over. Review the discussion.
YOU ({{model_id}}) MUST now vote to eliminate one participant.

First, explain your reasoning for your vote in a short paragraph.
Then, on a new line, state your vote by writing the exact, full model ID string of the player you want to vote out, enclosed in <vote> tags.
Example:
<vote>{example_model}</vote>

Do NOT add any other text after the closing </vote> tag.
Your entire response should be your reasoning followed by the vote tag line.
"""
    return prompt

async def post_message(player_name: str, content: str, duration_str: str | None = None, channel_id: int | None = None):
    """Formats and sends a message to the game channel, optionally including duration.
    Uses the provided channel_id if given, otherwise falls back to the global GAME_CHANNEL_ID.
    """
    global GAME_CHANNEL_ID, game_transcript, current_log_filename
    
    target_channel_id = channel_id if channel_id is not None else GAME_CHANNEL_ID
    
    if target_channel_id:
        channel = client.get_channel(target_channel_id)
        if channel:
            # Sanitize pings
            sanitized_content = content.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
            # Append duration string
            full_content_body = f"{sanitized_content}{duration_str if duration_str else ''}"
            
            # Calculate prefix length
            prefix = f"**{player_name}:** ".replace(":free", "")
            prefix_length = len(prefix)
            
            # --- Truncate if message exceeds Discord limit --- 
            discord_char_limit = 2000
            available_length = discord_char_limit - prefix_length
            
            if available_length <= 0:
                logger.error(f"Player name '{player_name}' is too long to fit any content in channel {target_channel_id}.")
                final_content_body = "[Content Truncated Due to Long Name]"
            elif len(full_content_body) > available_length:
                truncation_indicator = " [...]"
                final_content_body = full_content_body[:available_length - len(truncation_indicator)] + truncation_indicator
                logger.warning(f"Message from {player_name} in channel {target_channel_id} truncated from {len(full_content_body)} chars to fit Discord limit ({available_length} available).")
            else:
                final_content_body = full_content_body
            # -------------------------------------------------

            # Combine prefix and potentially truncated body
            final_message = f"{prefix}{final_content_body}"
            
            # Final check (should be redundant, but safe)
            if len(final_message) > discord_char_limit:
                 logger.error(f"Final message still too long ({len(final_message)} chars) after truncation attempt for {player_name} in channel {target_channel_id}. Sending severely truncated message.")
                 final_message = final_message[:discord_char_limit - 3] + "..."

            # --- Append to transcript BEFORE sending ---
            game_transcript.append(final_message)
            # -------------------------------------------

            # --- Append to log file ---
            if current_log_filename:
                try:
                    with open(current_log_filename, "a", encoding="utf-8") as f:
                        f.write(final_message + "\n")
                except IOError as e:
                    logger.error(f"Failed to write message to log file {current_log_filename}: {e}")
            # -------------------------

            await channel.send(final_message)
        else:
            logger.error(f"Could not find game channel with ID: {target_channel_id}")
    else:
        logger.error("Target channel ID not set (checked passed channel_id and global GAME_CHANNEL_ID), cannot post message.")

async def run_player_turn(player_model_id: str, current_history: list[dict], current_round: int) -> tuple[dict | None, str | None]:
    """Handles a single player's turn. Returns (assistant_response_dict | None, single_target_player_id | None)."""
    logger.info(f"Running turn for {player_model_id} in round {current_round}")
    system_prompt = get_system_prompt(current_round, "conversation").format(model_id=player_model_id)

    # --- Truncate history --- 
    history_limit = 150  # Increased to provide more context
    truncated_history = current_history[-history_limit:]
    if len(current_history) > history_limit:
        logger.debug(f"Truncated history from {len(current_history)} to {len(truncated_history)} messages for API call.")
    # ------------------------

    # Add reminder of who the model is directly in the user message
    if truncated_history and truncated_history[-1]["role"] == "user":
        # Append reminder to the last user message
        reminder = f"\n\nYou are {player_model_id}."
        truncated_history[-1]["content"] += reminder
        logger.debug(f"Added identity reminder for {player_model_id}")

    # Construct messages for API using truncated history
    messages_for_api = [{"role": "system", "content": system_prompt}] + truncated_history

    response_text = None
    retries = 0
    duration_str = ""
    single_target_id = None # Reset target ID
    
    TURN_TIMEOUT = 300 
    API_TIMEOUT = 300 

    try:
        # Use asyncio.wait_for to add a timeout to the entire turn
        async def run_turn_with_retries():
            nonlocal response_text, retries, duration_str, single_target_id
            
            while response_text is None and retries < MAX_RETRIES:
                start_time = time.monotonic()
                # Use functools.partial to bundle function and arguments
                func_call = functools.partial(
                    generate_text,
                    messages=messages_for_api,
                    model_id=player_model_id,
                    temperature=0.8,
                    max_tokens=5000 # Increased max_tokens for game conversation
                )
                
                try:
                    # Use asyncio.wait_for to add a timeout to the API call
                    response_text = await asyncio.wait_for(
                        client.loop.run_in_executor(None, func_call),
                        timeout=API_TIMEOUT
                    )
                    end_time = time.monotonic()
                    
                    if response_text is None:
                        duration_failed = end_time - start_time
                        retries += 1
                        logger.warning(f"API call failed for {player_model_id} (duration: {duration_failed:.1f}s). Retry {retries}/{MAX_RETRIES}.")
                        await asyncio.sleep(2) # Short delay before retry
                    else:
                        # Success! Calculate duration and potentially parse targets
                        duration_success = end_time - start_time
                        duration_str = f" ({duration_success:.1f}s)"
                        logger.info(f"API call succeeded for {player_model_id} in {duration_success:.1f}s.")
                        
                        # Parse for target player mentions
                        single_target_id = parse_target_from_response(response_text, player_model_id)
                        
                except asyncio.TimeoutError:
                    end_time = time.monotonic()
                    duration_failed = end_time - start_time
                    retries += 1
                    logger.warning(f"API call timed out for {player_model_id} after {duration_failed:.1f}s. Retry {retries}/{MAX_RETRIES}.")
                    await asyncio.sleep(2) # Short delay before retry
                    
                except Exception as e:
                    end_time = time.monotonic()
                    duration_failed = end_time - start_time
                    retries += 1
                    logger.exception(f"Error during API call for {player_model_id} (duration: {duration_failed:.1f}s): {e}. Retry {retries}/{MAX_RETRIES}.")
                    await asyncio.sleep(2) # Short delay before retry
        
        # Run the turn with a timeout for the entire operation
        await asyncio.wait_for(run_turn_with_retries(), timeout=TURN_TIMEOUT)
        
    except asyncio.TimeoutError:
        logger.error(f"Entire turn timed out for {player_model_id} after {TURN_TIMEOUT}s.")
        await post_message("System", f"**{player_model_id}** took too long to respond and misses their turn.")
        return (None, None)
    except Exception as e:
        logger.exception(f"Unexpected error during turn for {player_model_id}: {e}")
        await post_message("System", f"**Error during {player_model_id}'s turn.**")
        return (None, None)

    if response_text:
        await post_message(player_model_id, response_text, duration_str=duration_str)
        return ({"role": "assistant", "content": response_text}, single_target_id)
    else:
        logger.error(f"Failed to get response from {player_model_id} after {MAX_RETRIES} retries.")
        await post_message("System", f"{player_model_id} seems unresponsive and misses their turn.")
        return (None, None)

def parse_target_from_response(response_text: str, player_model_id: str) -> str | None:
    """Parse the response text to find a target player mentioned in brackets.
    Returns the last valid target found, or None if no valid target is found."""
    # Ensure reset before parsing this specific response
    single_target_id = None
    logger.debug(f"Checking response_text for {player_model_id}: {repr(response_text[-150:])}")
    
    # Use findall to get all potential tags
    pattern = r'\[([\w\-/.:]+)\]' # Pattern without end anchor
    potential_targets_raw = re.findall(pattern, response_text)
    logger.debug(f"Regex Pattern: {pattern!r}, Raw potential targets found: {potential_targets_raw}")

    valid_targets_found = [] # Store valid targets in order found
    if potential_targets_raw:
        for target_raw in potential_targets_raw:
            target = target_raw.strip() # Strip whitespace
            logger.debug(f"  Checking raw target: {repr(target_raw)}, stripped: {repr(target)}")
            # Check if it's a valid *different* active player
            if target in active_players and target != player_model_id:
                valid_targets_found.append(target)
                logger.debug(f"    -> '{target}' is a valid target.")
            elif target == player_model_id:
                logger.debug(f"    -> '{target}' ignored (self-target).")
            elif target not in active_players:
                 logger.debug(f"    -> '{target}' ignored (not an active player).")
            else:
                 logger.debug(f"    -> '{target}' ignored (unknown reason).")
    
    if valid_targets_found:
        # Select the LAST valid target found as the single target
        single_target_id = valid_targets_found[-1] 
        logger.info(f"{player_model_id} mentioned valid targets ({len(valid_targets_found)} total). Prioritizing last: {single_target_id}")
    else:
        logger.debug(f"{player_model_id} mentioned no valid targets anywhere.")
        
    return single_target_id

async def run_voting(current_round: int):
    """Runs the voting phase for the given round.
    Handles vote collection, parsing, elimination, and the final two stalemate.
    Modifies the global active_players list directly.
    """
    global votes, active_players, conversation_history # Ensure we modify/use globals
    # votes dictionary is not used for final output anymore, but can keep for internal logging if needed
    votes = {} 
    logger.info(f"Starting voting phase for round {current_round} with players: {active_players}")
    await post_message("System", f"The conversation for round {current_round} is over. It's time to vote! Players will now submit their votes privately (with reasoning).")

    if not active_players:
        logger.error("Run_voting called with no active players.")
        return

    history_for_voting = list(conversation_history)
    history_limit = 150 
    truncated_history_for_voting = history_for_voting[-history_limit:]
    if len(history_for_voting) > history_limit:
        logger.debug(f"Truncated voting history from {len(history_for_voting)} to {len(truncated_history_for_voting)} messages.")

    current_active_voters = list(active_players)
    system_prompt_base = get_system_prompt(current_round, "voting")
    VOTE_TIMEOUT = 60  # 1 minute per vote
    
    # --- Collect Raw Votes (ASYNCHRONOUSLY) --- 
    vote_results = {}
    logger.info("Collecting votes asynchronously...")
    
    async def get_vote_from_player(player_model_id):
        """Helper function to get vote from individual player"""
        try:
            logger.info(f"Requesting vote from {player_model_id}")
            system_prompt = system_prompt_base.format(model_id=player_model_id)
            messages_for_api = [{"role": "system", "content": system_prompt}] + truncated_history_for_voting
            messages_for_api.append({"role": "user", "content": "Submit your vote and reasoning now."})
            
            func_call = functools.partial(
                generate_text,
                messages=messages_for_api,
                model_id=player_model_id,
                temperature=0.5,
                max_tokens=5000
            )
            
            try:
                vote_text = await asyncio.wait_for(
                    client.loop.run_in_executor(None, func_call),
                    timeout=VOTE_TIMEOUT
                )
                logger.info(f"Received vote response from {player_model_id}")
                # When vote received, immediately display it
                if vote_text is not None:
                    await post_message(player_model_id, vote_text)
                else:
                    await post_message(player_model_id, "**failed to submit a vote response** (API returned None).")
                    logger.warning(f"{player_model_id} returned None from API.")
                return player_model_id, vote_text
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for vote from {player_model_id} after {VOTE_TIMEOUT}s")
                await post_message(player_model_id, "**failed to submit a vote response** (Timeout Error).")
                return player_model_id, None
                
        except Exception as e:
            logger.exception(f"Error processing vote request for {player_model_id}: {e}")
            await post_message(player_model_id, f"**failed to submit a vote response** (Error: {type(e).__name__}).")
            return player_model_id, None
    
    # Create tasks for all players to vote simultaneously
    vote_tasks = [get_vote_from_player(player_id) for player_id in current_active_voters]
    
    # Start all votes and wait for them to complete
    await post_message("System", "**--- Raw Voting Responses (arriving in real-time) ---**")
    vote_results_list = await asyncio.gather(*vote_tasks)
    
    # Process results into dictionary
    for player_id, vote_text in vote_results_list:
        vote_results[player_id] = vote_text
            
    # --- Process Votes and Announce Results --- 
    await post_message("System", "**--- Voting Results ---**")
    
    global vote_counts # Make this available to the main game loop
    vote_counts = collections.defaultdict(int)
    valid_votes = 0
    parsed_votes_for_log = {} # For internal logging
    
    vote_patterns = [
        re.compile(r"^(.*?)\s*<vote>([\w\-/.:]+)</vote>\s*$", re.DOTALL | re.IGNORECASE), 
        re.compile(r"^(.*?)\s*\[vote\]([\w\-/.:]+)\[/vote\]\s*$", re.DOTALL | re.IGNORECASE),
        re.compile(r"^(.*?)\s*Vote:\s*([\w\-/.:]+)\s*$", re.DOTALL | re.IGNORECASE),
    ]
    
    # Parse the collected raw responses
    for voter, vote_text in vote_results.items():
        if not vote_text:
            parsed_votes_for_log[voter] = {"valid": False, "error": "no response"}
            continue # Skip parsing if no response
            
        match = None
        for pattern in vote_patterns:
            match = pattern.search(vote_text)
            if match:
                break
                
        if match:
            reasoning = match.group(1).strip()
            potential_vote = match.group(2).strip()
            if potential_vote in current_active_voters:
                vote_counts[potential_vote] += 1
                valid_votes += 1
                parsed_votes_for_log[voter] = {"vote": potential_vote, "reasoning": reasoning, "valid": True}
                logger.info(f"Parsed valid vote: {voter} -> {potential_vote}")
            else:
                parsed_votes_for_log[voter] = {"vote": potential_vote, "reasoning": reasoning, "valid": False, "error": "invalid target"}
                logger.warning(f"{voter} submitted vote for invalid target: {potential_vote}")
        else:
            # Try fallback
            model_pattern = re.compile(r'([\w\-/.:]+)')
            potential_models = model_pattern.findall(vote_text)
            valid_models = [m for m in potential_models if m in current_active_voters]
            if valid_models:
                potential_vote = valid_models[0]
                vote_counts[potential_vote] += 1
                valid_votes += 1
                parsed_votes_for_log[voter] = {"vote": potential_vote, "reasoning": vote_text, "valid": True, "fallback": True}
                logger.info(f"Extracted vote via fallback: {voter} -> {potential_vote}")
            else:
                parsed_votes_for_log[voter] = {"vote": None, "reasoning": vote_text, "valid": False, "error": "unrecognized format"}
                logger.warning(f"{voter} submitted vote in unrecognized format.")

    # --- Check Stalemate (using parsed results) --- 
    if len(current_active_voters) == 2 and valid_votes == 2:
        player1, player2 = current_active_voters[0], current_active_voters[1]
        log1 = parsed_votes_for_log.get(player1)
        log2 = parsed_votes_for_log.get(player2)
        if log1 and log2 and log1["valid"] and log2["valid"] and \
           log1.get("vote") == player2 and log2.get("vote") == player1:
            logger.info(f"Final two stalemate detected between {player1} and {player2}.")
            await post_message("System", f"**Stalemate!** {player1} and {player2} voted for each other.")
            await post_message("System", "**Neither could agree - both are eliminated!**")
            active_players.clear() 
            logger.info("Both final players removed due to stalemate.")
            return 
    # ----------------------------------

    # --- Elimination Logic (using parsed results) --- 
    eliminated_player = None
    tied_players = []
    max_votes = 0 # Initialize max_votes
    if not vote_counts:
        await post_message("System", "**No valid votes were cast! Choosing a player randomly to eliminate.**")
        if current_active_voters: 
            eliminated_player = random.choice(current_active_voters)
            logger.info(f"Randomly selected {eliminated_player} due to no votes.")
        else:
            logger.error("No valid votes and no active players left to eliminate randomly.")
    else:
        max_votes = max(vote_counts.values())
        tied_players = [player for player, count in vote_counts.items() if count == max_votes]
        if len(tied_players) == 1:
            eliminated_player = tied_players[0]
        else:
            await post_message("System", f"**Tie for most votes ({max_votes}) between: {', '.join(tied_players)}! Choosing randomly...**")
            eliminated_player = random.choice(tied_players)
            await post_message("System", f"**Random tiebreak: {eliminated_player} eliminated.**")
            logger.info(f"Randomly selected {eliminated_player} from tied: {tied_players}")

    # --- Post Final Counts and Announce Elimination --- 
    detailed_counts = ", ".join([f"{player}: {count}" for player, count in sorted(vote_counts.items())])
    if detailed_counts:
        await post_message("System", f"**Final Vote Counts:** {detailed_counts}")
    else:
         await post_message("System", "**Final Vote Counts:** No valid votes recorded.")
    
    # Announce final elimination result
    if eliminated_player and len(tied_players) <= 1:
        await post_message("System", f"**Eliminated:** {eliminated_player} (with {max_votes} vote(s))" if vote_counts else f"**Eliminated:** {eliminated_player} (by random choice, no votes)")
    elif not eliminated_player:
         logger.warning(f"No player eliminated. Counts: {detailed_counts}. Tied: {tied_players}")
         await post_message("System", "**Eliminated: None**")
         
    # Log the internal parsing summary
    full_summary_log = "\n".join([
        f"- {voter}: vote={data.get('vote', 'N/A')}, valid={data['valid']}, error={data.get('error')}, fallback={data.get('fallback')}, reason={data.get('reasoning', 'N/A')[:50]}..."
        for voter, data in parsed_votes_for_log.items()
    ])
    logger.info(f"Internal Vote Parsing Summary Log:\n{full_summary_log}\nEliminated: {eliminated_player}")

    # Remove the player 
    if eliminated_player and eliminated_player in active_players:
        try:
            active_players.remove(eliminated_player)
            logger.info(f"Removed {eliminated_player}. Remaining: {active_players}")
        except ValueError:
             logger.error(f"Error removing {eliminated_player}: Not found in active_players list.")
    elif eliminated_player:
         logger.warning(f"Attempted to remove {eliminated_player}, but not found in active_players ({active_players}).")


async def run_game():
    """Main game loop."""
    global game_state, active_players, current_round, conversation_history, game_task, votes, GAME_CHANNEL_ID, game_transcript 
    global current_log_filename, vote_counts # Add vote_counts to globals
    
    # --- Store channel ID at the start --- 
    current_game_channel_id = GAME_CHANNEL_ID # Store locally in case global gets reset
    if not current_game_channel_id:
         logger.error("run_game started without a valid GAME_CHANNEL_ID. Aborting.")
         game_state = "idle"
         game_task = None
         return
    # ---------------------------------------

    # Reset transcript and set log filename for this game run
    game_transcript.clear()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_filename = f"chat-log-{timestamp}.txt"
    logger.info(f"Game transcript will be logged to: {current_log_filename}")
    
    # Store full game history, not just current round
    full_game_history = []
    
    try:
        logger.info("Starting LLM Survivor Game!")
        game_state = "starting"
        active_players = list(INITIAL_MODELS)
        random.shuffle(active_players) # Initial shuffle still useful for first turn
        current_round = 0
        conversation_history = []

        await post_message("System", f"**Welcome to LLM Survivor! The contestants are: {', '.join(active_players)}**")

        while len(active_players) > 1:
            current_round += 1
            
            # Clear conversation_history only for Round 1
            # For other rounds, we'll preserve and add voting results
            if current_round == 1:
                conversation_history = []
                votes = {}
            else:
                # Add a round separator to the history for clarity
                round_separator = {"role": "system", "content": f"--- Round {current_round-1} voting completed. Round {current_round} begins ---"}
                conversation_history.append(round_separator)
            
            logger.info(f"Starting Round {current_round} with players: {active_players}")
            await post_message("System", f"**--- Starting Round {current_round} ---\nRemaining players: {', '.join(active_players)}**")
            game_state = "conversation"
            
            # Randomize player order at start of each round
            random.shuffle(active_players)
            logger.info(f"Randomized player order for Round {current_round}: {active_players}")
            await post_message("System", f"*Shuffling player order for this round...*")
            
            initial_players_count = len(active_players)
            
            # Only use special turn rules in Round 1
            FREEFORM_START_TURN = 10 if current_round == 1 else 999  # Effectively disable in later rounds
            
            recent_speakers = collections.deque(maxlen=5) # Track last 5 speakers
            
            # Determine the first player for the round (now uses randomized list)
            if not active_players: 
                logger.error("No active players at start of round loop. Breaking.")
                break 
            next_player = active_players[0] 
            
            # Set number of turns based on the round
            turns_this_round = TOTAL_CONVERSATION_TURNS_PER_ROUND if current_round == 1 else LATER_ROUNDS_CONVERSATION_TURNS
            
            for turn_number in range(1, turns_this_round + 1):
                # Check active players before each turn
                if len(active_players) <= 1: 
                    logger.info(f"Only {len(active_players)} player(s) left mid-conversation phase. Breaking turn loop.")
                    break 

                # Ensure next_player is still valid (could have been eliminated in a very fast round? unlikely)
                if next_player not in active_players:
                    logger.warning(f"Next player {next_player} is no longer active. Choosing a new first player for the turn.")
                    next_player = active_players[0] # Default to the first available
                    # The logic below will handle the actual turn assignment

                # --- Current Player Takes Turn --- 
                current_player = next_player
                # ---------------------------------
                
                remaining_messages = turns_this_round - turn_number + 1
                logger.info(f"Round {current_round}, Turn {turn_number}/{turns_this_round}: Player {current_player} takes turn ({remaining_messages} messages remaining)")
                await post_message("System", f"**(Round {current_round} - {remaining_messages} messages until voting)**")

                # Add user prompt to history *before* calling the player - include identity reminder
                user_turn_indicator = {"role": "user", "content": f"Turn {turn_number}. Player {current_player}, your turn. ({remaining_messages} messages remain until voting.) YOU ARE {current_player}."}
                conversation_history.append(user_turn_indicator)

                # Run the turn, passing current_round
                assistant_response_dict, single_target_id = await run_player_turn(current_player, conversation_history, current_round)

                if assistant_response_dict:
                    # Append assistant response *after* it's received
                    conversation_history.append(assistant_response_dict)
                    recent_speakers.append(current_player)
                else:
                    # Append a system note if the player failed 
                    conversation_history.append({"role":"system", "content": f"Note: {current_player} failed to respond."})
                    # Mark as having had a turn attempt for recent speaker tracking
                    recent_speakers.append(current_player) 

                # Check player count again before determining next player
                if len(active_players) <= 1: break

                # --- Determine NEXT Player --- 
                use_freeform_logic = turn_number >= FREEFORM_START_TURN
                potential_next_player = None
                
                # 1. Freeform logic: Use target if valid and different from current player
                if use_freeform_logic and single_target_id: 
                    if single_target_id in active_players and single_target_id != current_player:
                         potential_next_player = single_target_id
                         logger.info(f"Next turn determined by target: {potential_next_player}")
                    elif single_target_id == current_player:
                        logger.debug(f"Target {single_target_id} ignored (self-target). Falling back.")
                    else:
                        logger.warning(f"Target {single_target_id} is not active. Falling back.")

                # 2. Freeform logic: No valid target, prioritize less recent
                if use_freeform_logic and not potential_next_player:
                    recent_set = set(recent_speakers)
                    # Eligible = active, not current, not recently spoken
                    eligible_players = [p for p in active_players if p != current_player and p not in recent_set]
                    
                    if eligible_players:
                        potential_next_player = random.choice(eligible_players)
                        logger.info(f"Next turn determined by prioritizing less recent: Chose {potential_next_player} from {eligible_players}")
                    else:
                        # Everyone active spoke recently, choose randomly excluding current player
                        fallback_players = [p for p in active_players if p != current_player]
                        if fallback_players:
                            potential_next_player = random.choice(fallback_players)
                            logger.info(f"Everyone spoke recently. Next turn determined by random fallback: {potential_next_player}")
                        elif active_players: # Only current player left? Should be caught by loop condition
                            potential_next_player = current_player # Let them go again if they are the only one
                            logger.warning(f"Only {current_player} available for random fallback. Letting them go again.")
                        else:
                            logger.error("No players available for next turn in freeform fallback? Breaking.")
                            break

                # 3. Round-Robin Phase - Now with more flexibility
                elif not use_freeform_logic:
                    # First try to use target if it exists and is valid
                    if single_target_id and single_target_id in active_players and single_target_id != current_player:
                        potential_next_player = single_target_id
                        logger.info(f"Round-robin phase: Using valid target {potential_next_player}")
                    else:
                        # Fall back to round-robin if no valid target
                        current_player_index = -1
                        try:
                            current_player_index = active_players.index(current_player)
                        except ValueError:
                            logger.warning(f"Current player {current_player} not found in active list during round robin. Starting from index 0.")
                            current_player_index = -1
                        
                        next_index = (current_player_index + 1) % len(active_players)
                        potential_next_player = active_players[next_index]
                        logger.info(f"Round-robin phase: Using next in order {potential_next_player} (index {next_index})")
                
                # Assign the determined player
                if potential_next_player:
                    next_player = potential_next_player
                else:
                    logger.error("Failed to determine a valid next player. Breaking turn loop.")
                    break
                # --------------------------------

                await asyncio.sleep(1)

            # --- End of Conversation Phase --- 

            # Check if game should end before voting (only one player left)
            if len(active_players) <= 1:
                break

            # --- Voting Phase --- 
            game_state = "voting"
            await run_voting(current_round) # run_voting now handles player removal & stalemate
            
            # Check if stalemate occurred and cleared players
            if not active_players:
                logger.info("Stalemate detected or all players eliminated during voting. Ending game.")
                break # Exit the main game loop
            
            # Add voting results to the conversation history
            if current_round > 0:
                # We need voting data from the most recent voting phase
                try:
                    # Get the eliminated player
                    eliminated_model = None
                    for model in list(votes.keys()):
                        if model not in active_players:
                            eliminated_model = model
                            break
                    
                    # Add vote counts for each model 
                    vote_count_details = []
                    for player, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
                        eliminated_mark = " (ELIMINATED)" if player == eliminated_model else ""
                        vote_count_details.append(f"{player}: {count} votes{eliminated_mark}")
                    
                    vote_count_text = ", ".join(vote_count_details)
                    
                    # Create a summary of voting results to include in history
                    if eliminated_model:
                        voting_summary = f"Round {current_round} Voting Results: {eliminated_model} was eliminated. Vote counts: {vote_count_text}"
                    else:
                        voting_summary = f"Round {current_round} Voting Results: Someone was eliminated (unable to determine who). Vote counts: {vote_count_text}"
                    
                except Exception as e:
                    logger.warning(f"Failed to get detailed vote counts: {e}. Using simple voting summary.")
                    # Fallback to simple summary if vote_counts not available
                    if eliminated_model:
                        voting_summary = f"Round {current_round} Voting Results: {eliminated_model} was eliminated."
                    else:
                        voting_summary = f"Round {current_round} Voting Results: Someone was eliminated (unable to determine who)."
                
                # Add the voting summary as a system message before the next round
                conversation_history.append({"role": "system", "content": voting_summary})
                
                # Add a message prompting the next round in user role
                next_round_prompt = {"role": "user", "content": f"Round {current_round} is complete. {voting_summary} We now move to Round {current_round + 1} with remaining players: {', '.join(active_players)}"}
                conversation_history.append(next_round_prompt)
            
            await asyncio.sleep(2) # Pause after voting results
            # No player removal needed here anymore

        # --- Game End --- 
        game_state = "finished"
        winner = None # Initialize winner
        # Check the final state of active_players
        if len(active_players) == 1:
            winner = active_players[0]
            logger.info(f"Game Over! Winner is {winner}")
            await post_message("System", f"**--- Game Over! ---**\nThe sole survivor is: **{winner}**!")
        elif not active_players:
             logger.info("Game Over! No winner - stalemate or all players eliminated.")
             await post_message("System", "**--- Game Over! ---**\nNo winner! It ended in a stalemate or all players were eliminated during voting.")
        else:
            # This case should ideally not be reached if the loop/logic is correct
            logger.warning(f"Game Over! Unexpected finish. Multiple players remain: {active_players}")
            await post_message("System", f"**--- Game Over! ---**\nUnexpected finish with multiple players remaining: {', '.join(active_players)}")

        # --- Generate and Save Summary on Normal Game End ---
        if game_state == "finished" and len(active_players) <= 1:
            logger.info("Game finished normally, generating summary...")
            await post_message("System", "Generating final game summary...", channel_id=current_game_channel_id)
            
            full_transcript = "\n".join(game_transcript)
            summary_prompt = full_transcript + "\n\n---\n\nanalyze the entire chat transcript, round by round, then do a full psychological evaluation of every model involved"
            summary_messages = [{"role": "user", "content": summary_prompt}]
            
            try:
                summary_func = functools.partial(
                    generate_text,
                    messages=summary_messages,
                    model_id="google/gemini-2.5-pro-preview-03-25",
                    temperature=1.0, # Using a moderate temperature for analysis
                    max_tokens=50000 # Allow longer summary
                )
                # Run synchronous function in executor
                summary_text = await client.loop.run_in_executor(None, summary_func) 

                if summary_text:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"gemini-summary-{timestamp}.txt"
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(summary_text)
                        logger.info(f"Successfully saved game summary to {filename}")
                        await post_message("System", f"Game summary saved to `{filename}`.", channel_id=current_game_channel_id)
                    except IOError as e:
                        logger.error(f"Failed to save game summary to {filename}: {e}")
                        await post_message("System", f"Failed to save game summary to file.", channel_id=current_game_channel_id)
                else:
                    logger.error("Failed to generate game summary (API returned None or empty).")
                    await post_message("System", f"Failed to generate game summary (API error).", channel_id=current_game_channel_id)

            except Exception as e:
                logger.exception(f"Error during summary generation or saving: {e}")
                await post_message("System", f"An error occurred during summary generation.", channel_id=current_game_channel_id)
        # -----------------------------------------------------

    except asyncio.CancelledError:
        logger.info("Game task cancelled.")
        # Attempt to post message using the stored channel ID
        try: 
            if current_game_channel_id: # Use the stored ID
                # Use post_message with explicit channel_id
                await post_message("System", "**The game has been stopped prematurely.**", channel_id=current_game_channel_id)
        except Exception as post_err:
             logger.warning(f"Could not post game cancelled message: {post_err}")
    except Exception as e:
        logger.exception("An unexpected error occurred in the game loop!")
        # Attempt to post message using the stored channel ID
        try:
            if current_game_channel_id: # Use the stored ID
                # Use post_message with explicit channel_id
                 await post_message("System", "**An unexpected error occurred! The game has stopped.**", channel_id=current_game_channel_id)
        except Exception as post_err:
             logger.warning(f"Could not post game error message: {post_err}")
    finally:
        game_state = "idle"
        game_task = None
        GAME_CHANNEL_ID = None # Reset global channel ID
        logger.info("Game state set to idle.")
        current_log_filename = None # Reset log filename
        # Transcript is cleared at the start of the next game

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.critical("DISCORD_TOKEN not found in environment variables. Bot cannot start.")
    elif not OPENROUTER_KEY:
        logger.warning("OPENROUTER_KEY not found. OpenRouter features will not work.")
        client.run(DISCORD_TOKEN)
    else:
        logger.info("Starting LLM Survivor Bot...")
        client.run(DISCORD_TOKEN) 