import aiohttp
import asyncio
import sys
import time
import os
from case_closed_game import Game, Direction, GameResult
import random

# Enable ANSI color support on Windows
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
    LIGHT_RED = '\033[38;5;210m'  # Even lighter red/pink for head
    LIGHT_BLUE = '\033[38;5;117m'  # Lighter blue for head
    RESET = '\033[0m'
    BOLD = '\033[1m'

class RandomPlayer:
    def __init__(self, player_id=1):
        self.player_id = player_id
    
    def get_possible_moves(self):
        """Returns list of all possible directions for agent."""
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        
    def get_best_move(self):
        """Returns a random valid direction."""
        possible_moves = self.get_possible_moves()
        return random.choice(possible_moves)

TIMEOUT = 4  # time for each move

class PlayerAgent:
    def __init__(self, participant, agent_name):
        self.participant = participant
        self.agent_name = agent_name
        self.latency = None

class Judge:
    def __init__(self, p1_url, p2_url):
        self.p1_url = p1_url
        self.p2_url = p2_url
        self.game = Game()
        self.p1_agent = None
        self.p2_agent = None
        self.game_str = ""  # Track game moves as string
        self.session = None  # Will hold aiohttp ClientSession
    
    def display_colored_board(self):
        """Display board with colored directional arrows for each agent."""
        board = self.game.board
        
        # Build direction maps for each agent
        agent1_dir_map = self._build_direction_map(self.game.agent1.get_trail_positions())
        agent2_dir_map = self._build_direction_map(self.game.agent2.get_trail_positions())
        
        # Get head positions
        agent1_trail = list(self.game.agent1.get_trail_positions())
        agent2_trail = list(self.game.agent2.get_trail_positions())
        agent1_head = agent1_trail[-1] if agent1_trail else None
        agent2_head = agent2_trail[-1] if agent2_trail else None
        
        # Arrow symbols for directions
        arrow_map = {
            (0, -1): "↑",   # UP
            (0, 1): "↓",    # DOWN
            (-1, 0): "←",   # LEFT
            (1, 0): "→",    # RIGHT
        }
        
        board_str = ""
        for y in range(board.height):
            for x in range(board.width):
                pos = (x, y)
                # Check for agent heads first (display as H in lighter color)
                if pos == agent1_head:
                    board_str += f"{Colors.LIGHT_RED}{Colors.BOLD}H{Colors.RESET} "
                elif pos == agent2_head:
                    board_str += f"{Colors.LIGHT_BLUE}{Colors.BOLD}H{Colors.RESET} "
                # Then check for trail positions (display as arrows)
                elif pos in agent1_dir_map:
                    direction = agent1_dir_map[pos]
                    arrow = arrow_map.get(direction, "●")
                    board_str += f"{Colors.RED}{Colors.BOLD}{arrow}{Colors.RESET} "
                elif pos in agent2_dir_map:
                    direction = agent2_dir_map[pos]
                    arrow = arrow_map.get(direction, "●")
                    board_str += f"{Colors.BLUE}{Colors.BOLD}{arrow}{Colors.RESET} "
                else:
                    board_str += ". "
            board_str += "\n"
        
        print(board_str, end="")
    
    def _build_direction_map(self, trail_positions):
        """Build a map of position -> direction for a trail."""
        direction_map = {}
        trail = list(trail_positions)
        
        if len(trail) < 2:
            return direction_map
        
        board = self.game.board
        
        # For each position (except the last), calculate direction to next position
        for i in range(len(trail) - 1):
            curr_pos = trail[i]
            next_pos = trail[i + 1]
            
            # Calculate direction (with torus wrapping consideration)
            dx = next_pos[0] - curr_pos[0]
            dy = next_pos[1] - curr_pos[1]
            
            # Handle torus wrapping (if movement is > half board width/height, it wrapped)
            if abs(dx) > board.width // 2:
                dx = -1 if dx > 0 else 1
            if abs(dy) > board.height // 2:
                dy = -1 if dy > 0 else 1
            
            # Normalize to unit direction
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
            
            direction_map[curr_pos] = (dx, dy)
        
        # For the last position (head), use the same direction as the previous segment
        if len(trail) >= 2:
            direction_map[trail[-1]] = direction_map[trail[-2]]
        
        return direction_map
    
    async def create_session(self):
        """Create persistent aiohttp session with connection pooling."""
        # Configure connector for connection pooling and keep-alive
        connector = aiohttp.TCPConnector(
            limit=10,  # Max number of connections
            limit_per_host=5,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            keepalive_timeout=30,  # Keep connections alive
        )
        
        # Create timeout configuration
        timeout = aiohttp.ClientTimeout(total=TIMEOUT, connect=2, sock_read=TIMEOUT)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True
        )
    
    async def close_session(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

    async def check_latency_single(self, url, player_num):
        """Check latency for a single player."""
        try:
            start_time = time.time()
            async with self.session.get(url) as response:
                end_time = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    agent = PlayerAgent(
                        data.get("participant", f"Participant{player_num}"), 
                        data.get("agent_name", f"Agent{player_num}")
                    )
                    agent.latency = (end_time - start_time)
                    return agent
                else:
                    return None
                    
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
            return None

    async def check_latency(self):
        """Check latency for both players in parallel."""
        # Check both players concurrently
        p1_task = self.check_latency_single(self.p1_url, 1)
        p2_task = self.check_latency_single(self.p2_url, 2)
        
        results = await asyncio.gather(p1_task, p2_task, return_exceptions=True)
        
        self.p1_agent = results[0] if not isinstance(results[0], Exception) and results[0] else None
        self.p2_agent = results[1] if not isinstance(results[1], Exception) and results[1] else None
        
        return self.p1_agent is not None and self.p2_agent is not None

    async def send_state_single(self, player_num):
        """Send current game state to a single player via POST."""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        state_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "player_number": player_num,
        }
        
        try:
            async with self.session.post(f"{url}/send-state", json=state_data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print(f"  Player {player_num} returned status {response.status}")
                    return False
                return True
        except asyncio.TimeoutError:
            print(f"  Player {player_num} timed out (>5s)")
            return False
        except aiohttp.ClientError as e:
            print(f"  Player {player_num} connection error: {e}")
            return False
        except Exception as e:
            print(f"  Player {player_num} error: {e}")
            return False
    
    async def send_state(self, player_num):
        """Send state to a specific player (for compatibility)."""
        return await self.send_state_single(player_num)
    
    async def send_state_both(self):
        """Send current game state to both players concurrently."""
        results = await asyncio.gather(
            self.send_state_single(1),
            self.send_state_single(2),
            return_exceptions=True
        )
        return all(r is True for r in results if not isinstance(r, Exception))

    async def get_move(self, player_num, attempt_number, random_moves_left):
        """Request a move from a player via GET with query parameters."""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        # Build query parameters for GET request
        params = {
            "player_number": player_num,
            "attempt_number": attempt_number,
            "random_moves_left": random_moves_left,
            "turn_count": self.game.turns,
        }
        
        try:
            start_time = time.time()
            async with self.session.get(f"{url}/send-move", params=params) as response:
                end_time = time.time()
                
                if player_num == 1:
                    self.p1_agent.latency = (end_time - start_time)
                else:
                    self.p2_agent.latency = (end_time - start_time)
                
                if response.status == 200:
                    move_data = await response.json()
                    return move_data.get('move')
                else:
                    return None
                    
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
            return None

    async def end_game(self, result):
        """End the game and notify both players concurrently."""
        end_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "result": result.name if isinstance(result, GameResult) else str(result),
        }
        
        async def notify_player(url):
            try:
                async with self.session.post(f"{url}/end", json=end_data) as response:
                    return response.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
                return False
        
        # Notify both players concurrently
        await asyncio.gather(
            notify_player(self.p1_url),
            notify_player(self.p2_url),
            return_exceptions=True
        )
        
        # Print result with colors
        if isinstance(result, GameResult):
            if result == GameResult.AGENT1_WIN:
                print(f"{Colors.RED}Winner: Agent 1 ({self.p1_agent.agent_name}){Colors.RESET}")
            elif result == GameResult.AGENT2_WIN:
                print(f"{Colors.BLUE}Winner: Agent 2 ({self.p2_agent.agent_name}){Colors.RESET}")
            else:
                print("Game ended in a draw")
        else:
            print(f"Game ended: {result}")

    def handle_move(self, move, player_num, is_random=False):
        """Validate and execute a move. Returns 'forfeit' or tuple (valid, boost_flag, direction)"""
        
        # Validate move format
        if not isinstance(move, str):
            print(f"Invalid move format by Player {player_num}: move must be a string")
            return "forfeit"
        
        # Parse move - can be "DIRECTION" or "DIRECTION:BOOST"
        move_parts = move.upper().split(':')
        direction_str = move_parts[0]
        use_boost = len(move_parts) > 1 and move_parts[1] == 'BOOST'
        
        # Convert move string to Direction
        direction_map = {
            'UP': Direction.UP,
            'DOWN': Direction.DOWN,
            'LEFT': Direction.LEFT,
            'RIGHT': Direction.RIGHT,
        }
        
        if direction_str not in direction_map:
            print(f"Invalid direction by Player {player_num}: {direction_str}")
            return "forfeit"
        
        direction = direction_map[direction_str]
        
        # Check if move is opposite to current direction (invalid move)
        agent = self.game.agent1 if player_num == 1 else self.game.agent2
        current_dir = agent.direction
        
        # Check if requested direction is opposite to current
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            print(f"Player {player_num} attempted invalid move (opposite direction). Using current direction instead.")
            direction = current_dir
            direction_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', 
                           Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        
        print(f"Player {player_num}'s move: {direction_str}{' (BOOST)' if use_boost else ''}{' (RANDOM)' if is_random else ''}")
        
        # Record move in game string with improved format
        move_abbrev = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R'}
        boost_marker = 'B' if use_boost else ''
        random_marker = 'R' if is_random else ''
        self.game_str += f"{player_num}{move_abbrev[direction_str]}{boost_marker}{random_marker}-"
        
        return (True, use_boost, direction)  # Return tuple: (valid, boost_flag, direction)

async def get_player_move(judge, player_num, random_moves_left):
    """Get move from a player with retry logic. Returns (direction, boost, used_random)"""
    
    # Try to get move from player (2 attempts)
    move = None
    validation = None
    
    for attempt in range(1, 3):  # 2 attempts
        move = await judge.get_move(player_num, attempt, random_moves_left)
        if move:
            validation = judge.handle_move(move, player_num, is_random=False)
            if validation == "forfeit":
                return ("forfeit", False, False)
            elif validation:
                boost = validation[1]
                direction = validation[2]
                return (direction, boost, False)
        print(f"  Player {player_num} Attempt {attempt} failed")
    
    # If both attempts failed, use random move or forfeit
    if not move or not validation:
        if random_moves_left > 0:
            print(f"Using random move for Player {player_num} ({random_moves_left} random moves left)")
            random_agent = RandomPlayer(player_num)
            direction = random_agent.get_best_move()
            # Convert Direction to string for handle_move
            dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
            validation = judge.handle_move(dir_to_str[direction], player_num, is_random=True)
            return (direction, False, True)  # Random moves don't use boost
        else:
            print(f"Player {player_num} has no random moves left. Forfeiting.")
            return ("forfeit", False, False)
    
    return ("forfeit", False, False)

async def main():
    print("Judge engine (FAST + COLORED) starting up, waiting for agents...")
    time.sleep(5)

    # Get agent URLs from environment variables
    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")

    # Creating judge
    print(f"Creating judge for {PLAYER1_URL} and {PLAYER2_URL}...")
    judge = Judge(PLAYER1_URL, PLAYER2_URL)
    
    # Create persistent session for connection pooling
    await judge.create_session()

    try:
        # Check connectivity and latency (in parallel)
        if not await judge.check_latency():
            print("Failed to connect to one or both players")
            return
            
        print(f"{Colors.RED}Player 1{Colors.RESET}: {judge.p1_agent.agent_name} ({judge.p1_agent.participant})")
        print(f"{Colors.BLUE}Player 2{Colors.RESET}: {judge.p2_agent.agent_name} ({judge.p2_agent.participant})")
        print(f"Initial latencies - P1: {judge.p1_agent.latency:.3f}s, P2: {judge.p2_agent.latency:.3f}s")
        
        # Send initial state to both players (in parallel)
        print("Sending initial game state...")
        if not await judge.send_state_both():
            print("Failed to send initial state")
            return

        # Random moves left for p1 and p2
        p1_random = 5
        p2_random = 5

        # Game loop
        while True:
            print(f"\n=== Turn {judge.game.turns + 1} ===")
            
            # Request moves from both players concurrently
            print("Requesting moves from both players...")
            p1_task = get_player_move(judge, 1, p1_random)
            p2_task = get_player_move(judge, 2, p2_random)
            
            # Wait for both moves concurrently
            results = await asyncio.gather(p1_task, p2_task)
            
            p1_direction, p1_boost, p1_used_random = results[0]
            p2_direction, p2_boost, p2_used_random = results[1]
            
            # Check for forfeits
            if p1_direction == "forfeit":
                print("Player 1 forfeited")
                await judge.end_game(GameResult.AGENT2_WIN)
                print("Game String:", judge.game_str)
                return
            
            if p2_direction == "forfeit":
                print("Player 2 forfeited")
                await judge.end_game(GameResult.AGENT1_WIN)
                print("Game String:", judge.game_str)
                return
            
            # Update random move counters
            if p1_used_random:
                p1_random -= 1
            if p2_used_random:
                p2_random -= 1
            
            # Execute both moves simultaneously
            result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
            
            # Send updated state to both players (in parallel)
            await judge.send_state_both()
            
            # Display current board state with colored trails
            judge.display_colored_board()
            print(f"{Colors.RED}Agent 1{Colors.RESET}: Trail Length={judge.game.agent1.length}, Alive={judge.game.agent1.alive}, Boosts={judge.game.agent1.boosts_remaining}")
            print(f"{Colors.BLUE}Agent 2{Colors.RESET}: Trail Length={judge.game.agent2.length}, Alive={judge.game.agent2.alive}, Boosts={judge.game.agent2.boosts_remaining}")
            
            # Check for game end
            if result is not None:
                await judge.end_game(result)
                print("Game String:", judge.game_str)
                break
            
            # Check for max turns (safety)
            if judge.game.turns >= 500:
                print("Maximum turns reached")
                await judge.end_game(GameResult.DRAW)
                print("Game String:", judge.game_str)
                break
    
    finally:
        # Clean up session
        await judge.close_session()


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)

