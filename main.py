import pygame
import os
import random
import math
import json
import numpy as np
import unicodedata
import sys
from datetime import datetime
from huggingface_hub import InferenceClient

DEBUG = False
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (128, 128, 128)

class GameObject(pygame.sprite.Sprite):
    def __init__(self, x, y, image, center=True):
        super().__init__()
        self.image = image
        if center is True:
            self.rect = self.image.get_rect(midtop=(x, y))
            self.collision_rect = self.calculate_collision_rect()
        else:
            self.collision_rect = self.rect = self.image.get_rect(topleft=(x, y))
        self.depth = self.rect.bottom

    def calculate_collision_rect(self):
        width_reduction = int(self.rect.width * 0.2)
        height_reduction = int(self.rect.height * 0.5)
        return pygame.Rect(
            self.rect.left + width_reduction // 2,
            self.rect.bottom - height_reduction,
            self.rect.width - width_reduction,
            height_reduction
        )

    def update(self):
        self.depth = self.rect.bottom

class Player(GameObject):
    def __init__(self, x, y, frames):
        self.frames = frames
        super().__init__(x, y, self.frames['down'][1])
        self.speed = 2
        self.facing = 'down'
        self.animation_time = 0
        self.interaction_rect = pygame.Rect(0, 0, self.rect.width*2, self.rect.width*2)

    def update(self, dt, obstacles):
        keys = pygame.key.get_pressed()
        dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]

        if dx != 0 or dy != 0:
            self.facing = 'right' if dx > 0 else 'left' if dx < 0 else 'up' if dy < 0 else 'down'
            self.animation_time = (self.animation_time + dt * 10) % len(self.frames[self.facing])
            self.image = self.frames[self.facing][int(self.animation_time)]

            new_rect = self.collision_rect.move(dx * self.speed, dy * self.speed)
            if not any(new_rect.colliderect(obstacle.collision_rect) for obstacle in obstacles):
                self.rect = self.rect.move(dx * self.speed, dy * self.speed)
                self.collision_rect = new_rect
                self.rect.clamp_ip(pygame.display.get_surface().get_rect())
                self.collision_rect.clamp_ip(pygame.display.get_surface().get_rect())

        self.update_interaction_rect()
        super().update()

    def update_interaction_rect(self):
        if self.facing == 'up':
            self.interaction_rect.midbottom = self.rect.midbottom
        elif self.facing == 'down':
            self.interaction_rect.midtop = self.rect.midtop
        elif self.facing == 'left':
            self.interaction_rect.midright = self.rect.midright
        elif self.facing == 'right':
            self.interaction_rect.midleft = self.rect.midleft

class NPCArea:
    def __init__(self, npcs):
        if not npcs:
            self.rect = pygame.Rect(0, 0, 0, 0)
            return

        min_x = min(npc.interaction_box.left for npc in npcs)
        min_y = min(npc.interaction_box.top for npc in npcs)
        max_x = max(npc.interaction_box.right for npc in npcs)
        max_y = max(npc.interaction_box.bottom for npc in npcs)

        self.rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

class NPC(GameObject):
    def __init__(self, x, y, name, frames):
        self.frames = frames
        super().__init__(x, y, self.frames['down'][1])
        self.name = name
        self.memory_manager = MemoryManager(self)
        self.speech_manager = SpeechManager(self)
        self.movement_manager = MovementManager(self)
        self.interaction_manager = InteractionManager(self)
        self.interaction_box_size = 50
        self.interaction_box = pygame.Rect(0, 0, self.interaction_box_size, self.interaction_box_size)
        self.update_interaction_box()

    def update(self, dt, obstacles, npcs, npc_area):
        super().update()
        self.speech_manager.update(dt)
        self.movement_manager.update(dt, obstacles, npc_area)
        self.interaction_manager.update(dt, npcs)
        self.update_interaction_box()

    def update_interaction_box(self):
        self.interaction_box.center = self.rect.center

    def start_conversation(self):
        initial_prompt = self.get_initial_prompt()
        response = mistral_api(initial_prompt, self.memory_manager.conversation_history, self.name)
        self.speech_manager.set_speech(response['responses'])
        self.memory_manager.update_conversation_history(response['history'])

    def respond(self, player_input):
        prompt = self.get_response_prompt(player_input)
        response = mistral_api(prompt, self.memory_manager.conversation_history, self.name)
        self.speech_manager.set_speech(response['responses'])
        self.memory_manager.update_conversation_history(response['history'])

    def interact_with_npc(self, other_npc):
        self.interaction_manager.current_interacting_npc = other_npc
        self.start_npc_conversation()

    def start_npc_conversation(self):
        prompt = self.get_npc_conversation_prompt()
        response = mistral_api(prompt, self.memory_manager.conversation_history, self.name)
        self.speech_manager.set_speech(response['responses'])
        self.memory_manager.update_conversation_history(response['history'])
        self.interaction_manager.set_cooldown()
        self.interaction_manager.current_interacting_npc.respond_to_npc(self)

    def respond_to_npc(self, initiator_npc):
        prompt = self.get_npc_response_prompt(initiator_npc)
        response = mistral_api(prompt, self.memory_manager.conversation_history, self.name)
        self.speech_manager.set_speech(response['responses'])
        self.memory_manager.update_conversation_history(response['history'])
        self.interaction_manager.set_cooldown()
        self.memory_manager.summarize_and_store_interaction(initiator_npc)

    def end_conversation(self):
        self.memory_manager.end_conversation()

    def get_initial_prompt(self):
        return f"You are {self.name}, an NPC in a 2D RPG game. {self.memory_manager.get_conversation_summary()} {self.memory_manager.get_gossip_prompt()} Greet the player in a friendly manner, keeping your response concise (max 10 words) and referencing past interactions or gossip if relevant."

    def get_response_prompt(self, player_input):
        return f"As {self.name}, respond to the player's message: '{player_input}'. {self.memory_manager.get_gossip_prompt()} Keep your response concise (max 10 words) and in character."

    def get_npc_conversation_prompt(self):
        return f"You are {self.name}, an NPC in a 2D RPG game. You're having a brief conversation with {self.interaction_manager.current_interacting_npc.name}. {self.memory_manager.get_gossip_prompt()} Start a short, casual conversation (max 10 words). You may choose to share gossip or not."

    def get_npc_response_prompt(self, initiator_npc):
        return f"You are {self.name}, responding to {initiator_npc.name}'s comment: '{initiator_npc.speech_manager.full_message}'. {self.memory_manager.get_gossip_prompt()} Give a brief, casual response (max 10 words). You may choose to respond to their gossip or share your own."

class SpeechManager:
    def __init__(self, npc):
        self.npc = npc
        self.speech_bubble = None
        self.speech_timer = 0
        self.full_message = ""

    def set_speech(self, message):
        self.full_message = f"{self.npc.name}: {message}"
        preview = self.full_message[:15] + "..." if len(self.full_message) > 15 else self.full_message
        self.speech_bubble = SpeechBubble(self.npc.rect.centerx, self.npc.rect.top, preview)
        self.speech_timer = 5

    def update(self, dt):
        if self.speech_bubble:
            self.speech_bubble.x = self.npc.rect.centerx
            self.speech_bubble.y = self.npc.rect.top
            self.speech_bubble.update_rect()

            self.speech_timer -= dt
            if self.speech_timer <= 0:
                self.speech_bubble = None
                self.full_message = ""

class MovementManager:
    def __init__(self, npc):
        self.npc = npc
        self.speed = 1
        self.direction = random.choice(['up', 'down', 'left', 'right'])
        self.move_timer = 0
        self.move_duration = random.uniform(1, 3)
        self.pause_duration = random.uniform(1, 3)
        self.is_moving = False
        self.animation_time = 0

    def update(self, dt, obstacles, npc_area):
        self.move_timer += dt
        if self.is_moving:
            if self.move_timer >= self.move_duration:
                self.stop_moving()
            else:
                self.move(dt, obstacles, npc_area)
        else:
            if self.move_timer >= self.pause_duration:
                self.start_moving()

        self.animate(dt)

    def move(self, dt, obstacles, npc_area):
        dx, dy = self.get_movement_vector()
        new_rect = self.npc.rect.move(dx, dy)
        new_collision_rect = self.npc.collision_rect.move(dx, dy)

        if self.is_valid_movement(new_collision_rect, obstacles, npc_area):
            self.npc.rect = new_rect
            self.npc.collision_rect = new_collision_rect
            self.npc.depth = self.npc.rect.bottom
            self.npc.update_interaction_box()
        else:
            self.change_direction()

    def get_movement_vector(self):
        dx, dy = 0, 0
        if self.direction == 'up':
            dy = -self.speed
        elif self.direction == 'down':
            dy = self.speed
        elif self.direction == 'left':
            dx = -self.speed
        elif self.direction == 'right':
            dx = self.speed
        return dx, dy

    def is_valid_movement(self, new_collision_rect, obstacles, npc_area):
        return (npc_area.rect.contains(new_collision_rect) and
                not any(new_collision_rect.colliderect(obstacle.collision_rect) for obstacle in obstacles))

    def start_moving(self):
        self.is_moving = True
        self.move_timer = 0
        self.move_duration = random.uniform(1, 3)
        self.change_direction()

    def stop_moving(self):
        self.is_moving = False
        self.move_timer = 0
        self.pause_duration = random.uniform(1, 3)

    def change_direction(self):
        self.direction = random.choice(['up', 'down', 'left', 'right'])

    def animate(self, dt):
        if self.is_moving:
            self.animation_time = (self.animation_time + dt * 5) % 3
            self.npc.image = self.npc.frames[self.direction][int(self.animation_time)]

class InteractionManager:
    def __init__(self, npc):
        self.npc = npc
        self.interaction_cooldown = 0
        self.current_interacting_npc = None

    def update(self, dt, npcs):
        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= dt

        if not self.npc.speech_manager.speech_bubble and self.interaction_cooldown <= 0:
            self.check_npc_interactions(npcs)

    def check_npc_interactions(self, npcs):
        for other_npc in npcs:
            if other_npc != self.npc and self.npc.interaction_box.colliderect(other_npc.interaction_box):
                self.npc.interact_with_npc(other_npc)
                break

    def set_cooldown(self):
        self.interaction_cooldown = 20

class MemoryManager:
    all_memories = {}

    def __init__(self, npc):
        self.npc = npc
        self.conversation_history = None
        if npc.name not in MemoryManager.all_memories:
            MemoryManager.all_memories[npc.name] = {
                "conversation_summary": "",
                "interactions": {
                    "player": [],
                    "npcs": {}
                }
            }
        self.memory = MemoryManager.all_memories[npc.name]

    def get_conversation_summary(self):
        return f"Here's a summary of your past interactions with the player: {self.memory['conversation_summary']}" if self.memory['conversation_summary'] else ""

    def get_gossip_prompt(self):
        gossip = self.get_gossip()
        return f"You have some information you could potentially share: {', '.join(gossip)}. " if gossip else ""

    def get_gossip(self):
        gossip = [interaction['summary'] for interaction in self.memory['interactions']['player']]
        for npc, interactions in self.memory['interactions']['npcs'].items():
            gossip += [interaction['summary'] for interaction in interactions]
        return gossip

    def update_conversation_history(self, history):
        self.conversation_history = history

    def end_conversation(self):
        if self.conversation_history:
            summary = self.summarize_conversation()
            self.memory['conversation_summary'] += f" {summary}"
            self.add_interaction("player", summary)
        self.conversation_history = None

    def summarize_conversation(self):
        summarize_prompt = f"Summarize the conversation so far between {self.npc.name} and the player in 2-3 sentences, highlighting key points and any important information shared."
        return mistral_api(summarize_prompt, self.conversation_history)['responses']

    def summarize_and_store_interaction(self, other_npc):
        summarize_prompt = f"Summarize the conversation so far between {self.npc.name} and {other_npc.name} in 1 sentence, highlighting key points and any important information shared."
        summary = mistral_api(summarize_prompt, self.conversation_history)['responses']
        self.add_interaction(other_npc.name, summary)

    def add_interaction(self, other_name, interaction):
        timestamp = datetime.now().isoformat()
        interaction_data = {
            "timestamp": timestamp,
            "summary": interaction
        }
        if other_name == "player":
            self.memory['interactions']["player"].append(interaction_data)
        else:
            if other_name not in self.memory['interactions']["npcs"]:
                self.memory['interactions']["npcs"][other_name] = []
            self.memory['interactions']["npcs"][other_name].append(interaction_data)

    @classmethod
    def save_all_memories(cls):
        filename = "npc_memories.json"
        with open(filename, 'w') as f:
            json.dump(cls.all_memories, f, indent=2)
        print(f"All NPC memories saved to {filename}")

    @classmethod
    def initialize_memories(cls, npcs):
        cls.load_all_memories()
        for npc in npcs:
            if npc.name not in cls.all_memories:
                cls.all_memories[npc.name] = {
                    "conversation_summary": "",
                    "interactions": {
                        "player": [],
                        "npcs": {}
                    }
                }
        cls.save_all_memories()

    @classmethod
    def load_all_memories(cls):
        filename = "npc_memories.json"
        try:
            with open(filename, 'r') as f:
                cls.all_memories = json.load(f)
            print("All NPC memories loaded successfully.")
        except FileNotFoundError:
            print("No existing memory file found. Starting with empty memories for all NPCs.")

class SpeechBubble:
    def __init__(self, x, y, text):
        self.font = pygame.font.Font(None, 20)
        self.padding = 10
        self.x = x
        self.y = y
        self.set_text(text)

    def set_text(self, text):
        self.text = text
        self.render_text()
        self.update_rect()

    def render_text(self):
        self.text_surface = self.font.render(self.text, True, BLACK)
        self.text_rect = self.text_surface.get_rect()

    def update_rect(self):
        width = self.text_rect.width + self.padding * 2
        height = self.text_rect.height + self.padding * 2
        self.rect = pygame.Rect(self.x - width // 2, self.y - height - 10, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        surface.blit(self.text_surface, (self.rect.x + self.padding, self.rect.y + self.padding))

        point_list = [
            (self.x, self.y),
            (self.x - 10, self.y - 10),
            (self.x + 10, self.y - 10)
        ]
        pygame.draw.polygon(surface, WHITE, point_list)
        pygame.draw.polygon(surface, BLACK, point_list, 2)

class ChatBox:
    def __init__(self, width, height):

        self.chat_height = int(height * 0.17)
        self.input_height = int(height * 0.05)
        self.padding = int(width * 0.01)

        self.rect = pygame.Rect(
            self.padding,
            height - self.chat_height - self.input_height - self.padding,
            width - 2 * self.padding,
            self.chat_height
        )
        self.input_rect = pygame.Rect(
            self.padding,
            height - self.input_height - self.padding,
            width - 2 * self.padding,
            self.input_height
        )

        self.text = ""
        self.chat_history = []

        self.font_size = int(height * 0.05)
        pygame.font.init()
        self.font = pygame.font.SysFont(None, self.font_size)

        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.line_height = int(self.font_size * 1.03)
        self.max_lines = self.chat_height // self.line_height
        self.wrap_width = self.rect.width - 2 * self.padding

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                temp = self.text
                self.text = ""
                return temp
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode
        return None

    def update(self, dt):
        self.cursor_timer += dt
        if self.cursor_timer >= 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

    def add_message(self, message):
        wrapped_lines = self.wrap_text(message)
        self.chat_history.extend(wrapped_lines)
        while len(self.chat_history) > self.max_lines:
            self.chat_history.pop(0)

    def wrap_text(self, text):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_surface = self.font.render(word, True, (0, 0, 0))
            word_width = word_surface.get_width()

            if current_width + word_width <= self.wrap_width:
                current_line.append(word)
                current_width += word_width + self.font.size(' ')[0]
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def draw(self, surface):
        if self.active:
            pygame.draw.rect(surface, (200, 200, 200), self.rect)
            pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)

            for i, line in enumerate(self.chat_history):
                text_surface = self.font.render(line, True, (0, 0, 0))
                surface.blit(text_surface, (self.rect.x + self.padding, self.rect.y + self.padding + i * self.line_height))

            pygame.draw.rect(surface, (255, 255, 255), self.input_rect)
            pygame.draw.rect(surface, (0, 0, 0), self.input_rect, 2)

            display_text = self.text
            if self.cursor_visible:
                display_text += "|"

            text_surface = self.font.render(display_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(self.input_rect.x + self.padding, self.input_rect.y + self.padding))

            if text_rect.width > self.input_rect.width - 2 * self.padding:
                surface.set_clip(self.input_rect.inflate(-2 * self.padding, -2 * self.padding))
                surface.blit(text_surface, text_rect)
                surface.set_clip(None)
            else:
                surface.blit(text_surface, text_rect)

def mistral_api(prompt, history, role = None):
    if DEBUG:
        return {'responses':"Hi 😃 😘 😎 🤨 😱", 'history':'bye'}
    history = '<s>' if history is None else history
    history += f"[INST] {prompt} [/INST]"
    if role is not None:
        history += f" {role}:"
    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token = os.environ.get('HF_READ_TOKEN', False))
    generate_kwargs = dict(
        temperature=0.9,
        max_new_tokens=256,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
        stream=False,
        details=False,
        return_full_text=False,
    )
    result = client.text_generation(history, **generate_kwargs)
    result = result.strip()
    history += f" {result}</s> "

    print(f'### Prompt ###\n{history}\n### Output ###\n{result}')
    return {'responses':result, 'history':history}

class GameEngine:
    def __init__(self, foreground, tile_size=64, tile_scale=.5, char_scale=1):
        self.initialize_game(foreground, tile_size, tile_scale)
        self.setup_pygame_window()
        self.load_assets(tile_size, tile_scale, char_scale)
        self.generate_game_layers(foreground)

    def initialize_game(self, foreground, tile_size, tile_scale):
        background = self.generate_background(foreground)
        elevation = self.generate_elevation(foreground, background)
        width, height = len(background[0]), len(background)
        self.TILE_SIZE = int(tile_size * tile_scale)
        self.WIDTH = width * self.TILE_SIZE
        self.HEIGHT = height * self.TILE_SIZE

    def setup_pygame_window(self):
        pygame.init()
        info = pygame.display.Info()
        self.monitor_width, self.monitor_height = info.current_w, info.current_h
        self.calculate_scale_factor()
        self.create_game_window()

    def calculate_scale_factor(self):
        scale_x = self.monitor_width / self.WIDTH
        scale_y = (self.monitor_height - 100) / self.HEIGHT
        self.scale_factor = min(scale_x, scale_y, 1)
        self.game_width = int(self.WIDTH * self.scale_factor)
        self.game_height = int(self.HEIGHT * self.scale_factor)

    def create_game_window(self):
        self.screen = pygame.display.set_mode((self.game_width, self.game_height))
        self.game_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("2D RPG with Chat System")
        self.clock = pygame.time.Clock()

    def load_assets(self, tile_size, tile_scale, char_scale):
        self.load_tiles(tile_size, tile_scale)
        self.load_objects(tile_scale)
        self.load_characters(char_scale)

    def load_tiles(self, tile_size, tile_scale):
        self.tileset = self.load_image('Tilemap_Flat.png')
        self.tiles = self.split_tiles(self.tileset, tile_size)
        self.elev_tileset = self.load_image('Tilemap_Elevation.png')
        self.elev_tiles = self.split_tiles(self.elev_tileset, tile_size)
        self.tiles += self.elev_tiles
        self.tiles = [self.scale_image(i, tile_scale) for i in self.tiles]

    def load_objects(self, tile_scale):
        self.tree_image = self.scale_image(self.crop_image(self.load_image('Tree.png')), tile_scale)
        self.house_image = self.scale_image(self.crop_image(self.load_image('House_Blue.png')), tile_scale)
        self.castle_image = self.scale_image(self.crop_image(self.load_image('Castle_Blue.png')), tile_scale)
        self.tower_image = self.scale_image(self.crop_image(self.load_image('Tower_Blue.png')), tile_scale)

    def load_characters(self, char_scale):
        self.sprite_sheet = self.load_image('Small-8-Direction-Characters_by_AxulArt.png')
        self.player_frames = self.create_frames(9, char_scale)
        self.npc_frames = self.create_frames(5, char_scale)

    def generate_game_layers(self, foreground):
        background = self.generate_background(foreground)
        elevation = self.generate_elevation(foreground, background)
        self.foreground_surface = self.draw_layer(foreground)
        self.background_surface = self.draw_layer(background)
        self.elevation_surface = self.draw_layer(elevation)
        self.elevation_objects = self.create_elevation_objects(elevation)

    def run(self, dict_objects=None):
        game_objects = self.initialize_game_objects(dict_objects)
        chat_box, npcs, environment_objects, all_objects, npc_area = game_objects

        running = True
        active_npc = None
        try:
            while running:
                dt = self.clock.tick(60) / 1000
                running, active_npc = self.handle_events(chat_box, npcs, active_npc)
                self.update_game_state(chat_box, dt, environment_objects, npcs, npc_area)
                self.render_game(all_objects, npcs, chat_box, npc_area)
        finally:
            MemoryManager.save_all_memories()
        pygame.quit()

    def initialize_game_objects(self, dict_objects):
        chat_box = ChatBox(self.WIDTH, self.HEIGHT)
        npcs, environment_objects = self.generate_environment(dict_objects)
        MemoryManager.initialize_memories(npcs)
        all_objects = pygame.sprite.Group(environment_objects + npcs + [self.player])
        npc_area = NPCArea(npcs)
        return chat_box, npcs, environment_objects, all_objects, npc_area

    def start_conversation(self, npcs, chat_box):
        for npc in npcs:
            if self.player.interaction_rect.colliderect(npc.collision_rect):
                npc.start_conversation()
                chat_box.active = True
                chat_box.add_message(npc.speech_manager.full_message)
                return npc
        return None

    def handle_events(self, chat_box, npcs, active_npc):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, active_npc
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not chat_box.active:
                    active_npc = self.start_conversation(npcs, chat_box)
                elif chat_box.active:
                    active_npc = self.handle_chat_input(event, chat_box, active_npc)
        return True, active_npc

    def handle_chat_input(self, event, chat_box, active_npc):
        player_input = chat_box.handle_event(event)
        if player_input is not None:
            if player_input == "":
                chat_box.active = False
                active_npc.end_conversation()
                active_npc = None
                chat_box.chat_history = []
            elif active_npc:
                chat_box.add_message(f"You: {player_input}")
                active_npc.respond(player_input)
                chat_box.add_message(active_npc.speech_manager.full_message)
        return active_npc

    def update_game_state(self, chat_box, dt, environment_objects, npcs, npc_area):
        if not chat_box.active:
            self.player.update(dt, environment_objects + self.elevation_objects + npcs)
            for obj in environment_objects:
                obj.update()
            for npc in npcs:
                npc.update(dt, environment_objects + self.elevation_objects + [self.player], npcs, npc_area)
        else:
            chat_box.update(dt)

    def render_game(self, all_objects, npcs, chat_box, npc_area):
        all_objects = sorted(all_objects, key=lambda obj: obj.depth)
        self.draw_game_layers()
        self.draw_game_objects(all_objects, npcs)
        chat_box.draw(self.game_surface)
        if DEBUG:
            self.draw_debug_info(all_objects, npcs, npc_area)
        self.draw_scaled_surface()

    def draw_game_layers(self):
        self.game_surface.fill((0,128,128))
        self.game_surface.blit(self.background_surface, (0, 0))
        self.game_surface.blit(self.elevation_surface, (0, 0))
        self.game_surface.blit(self.foreground_surface, (0, 0))

    def draw_game_objects(self, all_objects, npcs):
        for obj in all_objects:
            self.game_surface.blit(obj.image, obj.rect)
        for npc in npcs:
            if npc.speech_manager.speech_bubble:
                npc.speech_manager.speech_bubble.draw(self.game_surface)

    def draw_debug_info(self, all_objects, npcs, npc_area):
        for obj in all_objects:
            pygame.draw.rect(self.game_surface, (255, 0, 0), obj.collision_rect, 1)
        for obs in self.elevation_objects:
            pygame.draw.rect(self.game_surface, (255, 0, 0), obs.collision_rect, 1)
        for npc in npcs:
            pygame.draw.rect(self.game_surface, (0, 0, 255), npc.interaction_box, 1)
        pygame.draw.rect(self.game_surface, (0, 255, 0), self.player.interaction_rect, 1)
        pygame.draw.rect(self.game_surface, (255, 255, 0), npc_area.rect, 1)

    def draw_scaled_surface(self):
        scaled_surface = pygame.transform.smoothscale(self.game_surface, (self.game_width, self.game_height))
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    def create_frames(self, offset=9, scale=1):
        frame_rects = {
            'up': self.get_frames(offset, 0),
            'right': self.get_frames(offset, 2),
            'down': self.get_frames(offset, 4),
            'left': self.get_frames(offset, 6),
        }

        frames = {}
        for direction, rects in frame_rects.items():
            frames[direction] = []
            for rect in rects:
                frame = self.sprite_sheet.subsurface(rect)

                if scale != 1:
                    new_width = int(rect.width * scale)
                    new_height = int(rect.height * scale)

                    frame = pygame.transform.smoothscale(frame, (new_width, new_height))

                frames[direction].append(frame)

        return frames

    def load_image(self, filename):
        return pygame.image.load(os.path.join('assets', filename)).convert_alpha()

    def get_frames(self, row, col, sprite_width=16, sprite_height=24):
        return [pygame.Rect(col * sprite_width, (row+i) * sprite_height, sprite_width, sprite_height) for i in range(3)]

    def split_tiles(self, image, tile_size):
        tiles = []
        for y in range(0, image.get_height(), tile_size):
            for x in range(0, image.get_width(), tile_size):
                tiles.append(image.subsurface((x, y, tile_size, tile_size)))
        return tiles

    def crop_image(self, image):
        rect = pygame.mask.from_surface(image).get_bounding_rects()[0]
        cropped_image = pygame.Surface(rect.size, pygame.SRCALPHA)
        cropped_image.blit(image, (0, 0), rect)
        return cropped_image

    def generate_environment(self, dict_objects):
        npcs = []
        objects = []

        for obj_type, obj_data in dict_objects.items():
            positions = np.array(obj_data['positions']) * self.TILE_SIZE

            for xy in positions:
                if obj_type == 'player':
                    self.player = Player(*xy, self.player_frames)
                else:
                    image = getattr(self, f"{obj_type}_image")
                    new_object = GameObject(*xy, image)
                    objects.append(new_object)

                    if 'npc_names' in obj_data:
                        npc_name = obj_data['npc_names'].pop(0) if obj_data['npc_names'] else f"NPC_{obj_type}"
                        npcs.append(NPC(*new_object.rect.midbottom, npc_name, self.npc_frames))

        return npcs, objects

    def create_elevation_objects(self, elevation):
        elevation_objects = []
        for y, row in enumerate(elevation):
            for x, tile in enumerate(row):
                if tile in [40,41,42, 48,50, 52, 53, 54]:
                    obj = GameObject(x * self.TILE_SIZE, y * self.TILE_SIZE, self.tiles[tile], center=False)
                    if tile in [48]:
                        obj.collision_rect = pygame.Rect(obj.collision_rect.left,obj.collision_rect.top, 1, obj.collision_rect.height)
                    if tile in [50]:
                        obj.collision_rect = pygame.Rect(obj.collision_rect.right,obj.collision_rect.top, 1, obj.collision_rect.height)
                    elevation_objects.append(obj)
        return elevation_objects

    def draw_layer(self, layer):
        surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for y, row in enumerate(layer):
            for x, tile_index in enumerate(row):
                if tile_index > -1:
                    surface.blit(self.tiles[tile_index], (x * self.TILE_SIZE, y * self.TILE_SIZE))
        return surface

    def generate_background(self, foreground, n_empty=3):
        w = max([len(i) for i in foreground])
        h = len(foreground)
        background = [[-1]*w]*n_empty
        background += [[5] + [6] * (w-2) + [7]]
        background += [[15] + [16] * (w-2) + [17]] * (h-n_empty)
        background += [[25] + [26] * (w-2) + [27]]
        return background

    def generate_elevation(self, foreground, background):
        elevation = [[-1 for _ in range(len(row))] for row in background]
        for y in range(len(foreground)):
            for x, tile in enumerate(foreground[y]):
                if tile in [20, 21, 22]:
                    elevation[y][x] = tile+28
                    elevation[y+1][x] = tile+32
                elif tile in [10,12]:
                    elevation[y][x] = tile + 38
                elif tile in [0,1,2]:
                    elevation[y][x] = tile + 40
                elif tile < -1 and elevation[y][x] == 53:
                    edge = foreground[y][x-1] - foreground[y][x+1]
                    edge = np.sign(edge)
                    elevation[y][x] += 16 - edge
        return elevation

    @staticmethod
    def scale_image(image, scale_factor):
        new_w = int(image.get_width() * scale_factor)
        new_h = int(image.get_height() * scale_factor)
        return pygame.transform.scale(image, (new_w, new_h))

if __name__ == '__main__':
    foreground = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2, -1, -1],
        [-1,  0, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12,  2, -1],
        [-1, 10, 10, 11, 11, 11, 21, 21, 21, 21, 11, 11, 11, 12, 12, -1],
        [-1, 10, 20, 21, 21, 22, -2, -2, -2, -2, 20, 21, 21, 22, 12, -1],
        [-1, 10, -1, -1, -1, -1, 11, 11, 11, 11, -1, -1, -1, -1, 12, -1],
        [-1, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, -1],
        [-1, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, -1],
        [-1, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, -1],
        [-1, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, -1],
        [-1, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, -1],
        [-1, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, -1],
        [-1, 10, 11, 11, 11, 21, 21, 21, 21, 21, 21, 11, 11, 11, 12, -1],
        [-1, 20, 21, 21, 22, -2, -2, -2, -2, -2, -2, 20, 21, 21, 22, -1]
    ]

    dict_objects = {
        'player': {'positions': [(8, 14)]},
        'castle': {'positions': [(8, 0)]},
        'tower': {'positions': [(4, 1), (12, 1)]},
        'house': {
            'positions': [(4, 6), (4, 9), (12, 7)],
            'npc_names': ["Alice", "Bob", "Charlie"]
        }
    }

    game = GameEngine(foreground)
    game.run(dict_objects)
