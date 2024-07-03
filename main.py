import pygame
import os
import random
import math
import json
import numpy as np
import asyncio
import sys
from datetime import datetime
from time import sleep
from huggingface_hub import InferenceClient

DEBUG = False
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (128, 128, 128)

class GameObject(pygame.sprite.Sprite):
    def __init__(self, x, y, image, center=True):
        super().__init__()
        self.image = image
        self.center = center
        if self.center is True:
            self.rect = self.image.get_rect(midtop=(x, y))
        else:
            self.rect = self.image.get_rect(topleft=(x, y))
        self.collision_rect = self.calculate_collision_rect()
        self.depth = self.rect.bottom

    def calculate_collision_rect(self):
        if self.center is True:
            width_reduction = int(self.rect.width * 0.2)
            height_reduction = int(self.rect.height * 0.5)
            return pygame.Rect(
                self.rect.left + width_reduction // 2,
                self.rect.bottom - height_reduction,
                self.rect.width - width_reduction,
                height_reduction
            )
        return self.rect

    def update(self):
        self.depth = self.rect.bottom
        self.collision_rect = self.calculate_collision_rect()

class Player(GameObject):
    def __init__(self, x, y, frames, area):
        self.frames = frames
        super().__init__(x, y, self.frames['down'][1])
        self.speed = 2
        self.facing = 'down'
        self.animation_time = 0
        self.interaction_rect = pygame.Rect(0, 0, self.rect.width*2, self.rect.width*2)
        self.area = area

    def update(self, dt, obstacles):
        keys = pygame.key.get_pressed()
        dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]

        if dx != 0 or dy != 0:
            self.facing = 'right' if dx > 0 else 'left' if dx < 0 else 'up' if dy < 0 else 'down'
            self.animation_time = (self.animation_time + dt * 5 * self.speed) % len(self.frames[self.facing])
            self.image = self.frames[self.facing][int(self.animation_time)]

            new_rect = self.collision_rect.move(dx * self.speed, dy * self.speed)
            if not any(new_rect.colliderect(obstacle.collision_rect) for obstacle in obstacles if obstacle is not self):
                self.rect = self.rect.move(dx * self.speed, dy * self.speed)
                self.rect.clamp_ip(self.area)

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

class NPC(GameObject):
    def __init__(self, x, y, name, frames):
        super().__init__(x, y, frames['down'][1])
        self.name = name
        self.frames = frames
        self.long_term_memory = MemoryManager(self)
        self.speed = 1
        self.direction = random.choice(['up', 'down', 'left', 'right'])
        self.move_cooldown = 3
        self.is_moving = True
        self.animation_time = 0
        self.interaction_rect = pygame.Rect(0, 0, self.rect.height*3, self.rect.height*3)
        self.interaction_rect.center = self.rect.center
        self.speech_bubble = None
        self.interaction_cooldown = 0
        self.is_interacting = False
        self.working_memory = ''
        self.prompt_history = None

    async def interact_with_npc(self, other_npc):
        prompt = self.get_prompt("initial", name=other_npc.name)
        greet = await asyncio.to_thread(mistral_api, prompt, None, self.name)
        greet = greet['responses']
        self.speech_bubble = SpeechBubble(self, greet)
        prompt = other_npc.get_prompt("response", self.name).format(message=greet)
        response = await asyncio.to_thread(mistral_api, prompt, None, other_npc.name)
        response = response['responses']
        other_npc.speech_bubble = SpeechBubble(other_npc, response)
        self.speech_bubble.bump = other_npc.speech_bubble
        self.working_memory = f'{self.name}: {greet}\n{other_npc.name}: {response}\n'
        await self.end_conversation(other_npc)

    async def end_conversation(self, other_npc = None):
        if len(self.working_memory) > 0:
            prompt = f"Summarize the following conversation in 1 past-tense sentence (max 10 words).\n\n{self.working_memory}"
            summary = await asyncio.to_thread(mistral_api, prompt, None)
            summary = summary['responses']
            self.long_term_memory.add_memory(summary)
            self.working_memory = ''
            self.is_interacting = False
            if other_npc is not None:
                other_npc.long_term_memory.add_memory(summary)
                other_npc.is_interacting = False

    def start_conversation(self):
        self.is_interacting = True
        self.interaction_cooldown = 10
        prompt = self.get_prompt("initial", 'Joe the player')
        result = mistral_api(prompt, None, self.name)
        self.prompt_history = result['history']
        response = result['responses']
        self.speech_bubble = SpeechBubble(self, response)
        self.working_memory = f'{self.name}: {response}\n'
        return response

    def respond(self, prompt):
        result = mistral_api(prompt, self.prompt_history, self.name)
        self.prompt_history = result['history']
        response = result['responses']
        self.speech_bubble = SpeechBubble(self, response)
        self.working_memory += f'Player: {prompt}\n{self.name}: {response}\n'
        return response

    def get_prompt(self, prompt_type, name):
        memory = self.long_term_memory.get_memory()
        if name.strip().split(' ')[0] not in memory:
            memory += f' You have no past interaction with {name}.'
        prompts = {
            "initial": f"{memory} You are now meeting {name} in a market square. "
                       f"Talk to {name} in a way that reflects your character. "
                       f"Keep it under 10 words.",

            "response": f"{memory} You're now talking to {name} in a market square. "
                        f'{name} just said: "{{message}}". '
                        f"Respond to {name} in a way that reflects your character and advances the conversation. "
                        f"Keep it under 10 words."
        }
        return prompts[prompt_type]

    def update(self, dt, obstacles, npcs, npc_area):
        super().update()
        self.update_movement(dt, obstacles, npc_area)
        self.update_interaction(dt, npcs)
        if self.speech_bubble and self.speech_bubble.update(dt):
            self.speech_bubble = None

    def update_movement(self, dt, obstacles, npc_area):
        self.move_cooldown -= dt

        if self.is_moving and not self.is_interacting:
            if self.move_cooldown <= 0:
                self.is_moving = False
                self.move_cooldown = random.uniform(1, 3)
            else:
                self.move(obstacles, npc_area)
                self.animation_time = (self.animation_time + dt * 5 * self.speed) % len(self.frames[self.direction])
                self.image = self.frames[self.direction][int(self.animation_time)]
        else:
            if self.move_cooldown <= 0:
                self.is_moving = True
                self.move_cooldown = random.uniform(1, 3)
                self.direction = random.choice(['up', 'down', 'left', 'right'])

    def move(self, obstacles, npc_area):
        dx, dy = {
            'up': (0, -self.speed),
            'down': (0, self.speed),
            'left': (-self.speed, 0),
            'right': (self.speed, 0)
        }[self.direction]

        new_rect = self.rect.move(dx, dy)
        new_collision_rect = self.collision_rect.move(dx, dy)

        if self.is_valid_movement(new_collision_rect, obstacles, npc_area):
            self.rect = new_rect
            self.collision_rect = new_collision_rect
            self.depth = self.rect.bottom
        else:
            self.direction = random.choice(['up', 'down', 'left', 'right'])
        self.interaction_rect.center = self.rect.center

    def update_interaction(self, dt, npcs):
        self.interaction_cooldown = max(0, self.interaction_cooldown - dt)
        if self.is_available:
            for other_npc in npcs:
                if other_npc != self and other_npc.is_available and self.interaction_rect.colliderect(other_npc.collision_rect):
                    self.is_interacting = other_npc.is_interacting = True
                    self.interaction_cooldown = other_npc.interaction_cooldown = 10
                    asyncio.create_task(self.interact_with_npc(other_npc))
                    break

    def is_valid_movement(self, new_rect, obstacles, npc_area):
        return npc_area is not None and npc_area.contains(new_rect) and not any(new_rect.colliderect(obstacle.collision_rect) for obstacle in obstacles if obstacle is not self)

    @property
    def is_available(self):
        return not self.speech_bubble and self.interaction_cooldown <= 0 and not self.is_interacting

class MemoryManager:
    all_memories = None

    def __init__(self, npc):
        if MemoryManager.all_memories is None:
            MemoryManager.load_all_memories()
        if npc.name not in MemoryManager.all_memories:
            MemoryManager.all_memories[npc.name] = f'You are {npc.name}, an NPC in a D&D game.'
        self.name = npc.name

    def get_memory(self):
        return MemoryManager.all_memories[self.name]

    def add_memory(self, summary):
        MemoryManager.all_memories[self.name] += f' {summary.strip()}'

    @classmethod
    def save_all_memories(cls):
        filename = "npc_memories.json"
        with open(filename, 'w') as f:
            json.dump(cls.all_memories, f, indent=2)
        print(f"All NPC memories saved to {filename}")

    @classmethod
    def load_all_memories(cls):
        filename = "npc_memories.json"
        try:
            with open(filename, 'r') as f:
                cls.all_memories = json.load(f)
            print("All NPC memories loaded successfully.")
        except FileNotFoundError:
            print("No existing memory file found. Starting with empty memories for all NPCs.")
            cls.all_memories = {}

class SpeechBubble:
    def __init__(self, npc, message, font_size=16, duration=3, preview_length=20):
        self.font = pygame.font.Font(None, font_size)
        self.padding = 4
        self.npc = npc
        self.x = npc.rect.centerx
        self.y = npc.rect.top
        self.timer = duration
        message = f"{self.npc.name.strip().split(' ')[0]}: {message}"
        self.text = message[:preview_length] + "..." if len(message) > preview_length+3 else message
        self.bump = None
        self.bumpy = 0
        self.render_text()
        self.update_rect()

    def render_text(self):
        self.text_surface = self.font.render(self.text, True, BLACK)
        self.text_rect = self.text_surface.get_rect()

    def update_rect(self):
        width = self.text_rect.width + self.padding * 2
        height = self.text_rect.height + self.padding
        self.rect = pygame.Rect(self.x - width // 2, self.y - height - self.padding//2, width, height)

    def update(self, dt):
        self.set_bump()
        self.x = self.npc.rect.centerx
        self.y = self.npc.rect.top - self.bumpy
        self.update_rect()
        self.timer -= dt
        return self.timer <= 0

    def set_bump(self):
        if self.bumpy != 0 and self.bump and self.bump.rect.colliderect(self.rect):
            self.bumpy = self.rect.height if self.bump.rect.top > self.rect.top else - self.rect.height

    def draw(self, surface):
        combined_points = [
            self.rect.topleft, self.rect.topright, self.rect.bottomright,
            (self.x + self.padding, self.y - self.padding//2),
            (self.x, self.y),
            (self.x - self.padding, self.y - self.padding//2),
            self.rect.bottomleft,
        ]

        pygame.draw.polygon(surface, WHITE, combined_points)
        pygame.draw.polygon(surface, BLACK, combined_points, 2)

        surface.blit(self.text_surface, (self.rect.x + self.padding, self.rect.y + self.padding))


class ChatBox:
    def __init__(self, width, height):

        self.chat_height = int(height * 0.18)
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

        self.font_size = int(height * 0.04)
        self.font = pygame.font.Font(None, self.font_size)

        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.line_height = int(self.font_size * 1.01)
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

def generate_random_names(count = 3):
    npc_names = ["Charlie", "Bob", "Alice", "Roy", "Emma", "Alex", "Sofia", "Liam", "Zoe", "Ethan", "Olivia", "Noah", "Mason", "Isabella", "Bill", "Mia", "Jim", "Charlotte", "Benjamin", "Elijah", "Harper", "Evelyn", "Abigail", "Henry"]
    npc_traits = ["gentle", "grumpy", "cheerful", "mysterious", "clumsy", "quirky", "wise", "naive", "brave", "cowardly", "ambitious", "lazy", "suspicious", "trusting", "eccentric", "reserved", "boisterous", "sarcastic", "loyal", "greedy", "forgetful", "perfectionist", "absent-minded", "superstitious"]
    npc_occupations = ["blacksmith", "merchant", "mage", "herbalist", "bard", "innkeeper", "guard", "farmer", "priest", "alchemist", "hunter", "fisherman", "carpenter", "tailor", "stable master", "baker", "brewer", "jeweler", "scribe", "healer", "librarian", "fortune teller", "miner", "cook"]

    unique_npcs = [f"{name} the {trait} {occupation}" for name, trait, occupation in
                zip(random.sample(npc_names, count),
                    random.sample(npc_traits, count),
                    random.sample(npc_occupations, count))]
    return unique_npcs

def mistral_api(prompt, history, role = None):
    history = '<s>' if history is None else history
    history += f"[INST] {prompt} [/INST]"
    if role is not None:
        history += f" {role}:"
    if DEBUG:
        result = f"Hi 😃 {datetime.now().isoformat()}"
        sleep(1)
        history = history + f" {result}</s> "
        print(f'### {role} ###\n{history}')
        return {'responses':result, 'history':history}
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
    print(f'\n\n### {role} ###\n{history}')
    return {'responses':result, 'history':history}

class GameEngine:
    def __init__(self, height_map, dict_objects=None, tile_size=64, tile_scale=.5, char_scale=1, tilemaps=('Tilemap_Sand.png', 'Tilemap_Elevation.png', 'Tilemap_Grass.png'), spritesheet='Small-8-Direction-Characters_by_AxulArt.png'):
        backgrounds = [self.generate_background(height_map, i) for i in (0,.5,1)]
        self.setup_pygame_window(backgrounds[-1], tile_size * tile_scale)
        tiles_2d = self.create_2d_tilemap(tilemaps, tile_size, tile_scale)
        self.surfaces = [self.draw_layer(x, y) for x, y in zip(backgrounds, tiles_2d)]
        self.border_objects = self.generate_border_objects(backgrounds[-1], tile_size * tile_scale)
        self.generate_environment(dict_objects, spritesheet, tile_size * tile_scale, char_scale)

    def setup_pygame_window(self, background, tile_size):
        w = len(background[0]) * tile_size
        h = len(background) * tile_size
        pygame.init()
        pygame.font.init()
        self.game_surface = pygame.Surface((w, h))
        info = pygame.display.Info()
        scale_x = info.current_w / w
        scale_y = (info.current_h - 100) / h
        scale_factor = min(scale_x, scale_y, 1)
        self.screen = pygame.display.set_mode((int(w * scale_factor), int(h * scale_factor)))
        pygame.display.set_caption("2D RPG with Mistral-Instruct-v0.3")

    def generate_environment(self, dict_objects, spritesheet, tile_size, char_scale):
        player_frames = self.create_frames(spritesheet, 9, char_scale)
        npc_frames = self.create_frames(spritesheet, 5, char_scale)
        npcs = []
        objects = []
        for obj_type, obj_data in dict_objects.items():
            positions = np.array(obj_data['positions']) * tile_size
            if obj_type == 'player':
                player = Player(*positions[0], player_frames, self.game_surface.get_rect())
            else:
                image = GameEngine.scale_image(GameEngine.crop_image(GameEngine.load_image(obj_data['image'])), obj_data.get('scale', 0.5))
                for xy in positions:
                    new_object = GameObject(*xy, image)
                    objects.append(new_object)
                    if 'npc_names' in obj_data:
                        npc_name = obj_data['npc_names'].pop(0) if obj_data['npc_names'] else f"NPC_{obj_type}"
                        npcs.append(NPC(*new_object.rect.midbottom, npc_name, npc_frames))
        self.player = player
        self.npcs = npcs
        self.npc_area = GameEngine.get_npc_area(self.npcs)
        self.objects = self.border_objects + objects

    def __call__(self):
        asyncio.run(self.run())

    async def run(self):
        all_objects = pygame.sprite.Group(self.objects + self.npcs + [self.player])
        chat_box = ChatBox(*self.game_surface.get_size())
        running = True
        active_npc = None
        clock = pygame.time.Clock()
        try:
            while running:
                dt = clock.tick(60) / 1000
                running, active_npc = self.handle_events(chat_box, self.player, self.npcs, active_npc)
                if not chat_box.active:
                    self.player.update(dt, all_objects)
                    for npc in self.npcs:
                        npc.update(dt, all_objects, self.npcs, self.npc_area)
                else:
                    chat_box.update(dt)
                self.render_game(all_objects, chat_box)
                await asyncio.sleep(0.01)
        finally:
            MemoryManager.save_all_memories()
            pygame.quit()

    def handle_events(self, chat_box, player, npcs, active_npc):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, active_npc
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not chat_box.active:
                    active_npc = self.start_conversation(player, npcs, chat_box)
                elif chat_box.active:
                    active_npc = self.handle_chat_input(event, chat_box, active_npc)
        return True, active_npc

    def start_conversation(self, player, npcs, chat_box):
        for npc in npcs:
            if player.interaction_rect.colliderect(npc.collision_rect) and npc.is_available:
                chat_box.add_message(f"{npc.name.strip().split(' ')[0]}: {npc.start_conversation()}")
                chat_box.active = True
                return npc
        return None

    def handle_chat_input(self, event, chat_box, active_npc):
        player_input = chat_box.handle_event(event)
        if player_input is not None:
            if player_input == "":
                chat_box.active = False
                asyncio.ensure_future(active_npc.end_conversation())
                active_npc = None
                chat_box.chat_history = []
            elif active_npc:
                chat_box.add_message(f"You: {player_input}")
                chat_box.add_message(f"{active_npc.name.strip().split(' ')[0]}: {active_npc.respond(player_input)}")
        return active_npc

    def render_game(self, all_objects, chat_box):
        self.game_surface.fill((0,128,128))
        for bg in self.surfaces:
            self.game_surface.blit(bg, (0,0))
        for obj in sorted(all_objects, key=lambda obj: obj.depth):
            self.game_surface.blit(obj.image, obj.rect)
            if isinstance(obj, NPC) and obj.speech_bubble:
                obj.speech_bubble.draw(self.game_surface)
        chat_box.draw(self.game_surface)
        if DEBUG:
            for obj in all_objects:
                pygame.draw.rect(self.game_surface, (255, 0, 0), obj.collision_rect, 1)
                if hasattr(obj, 'interaction_rect'):
                    pygame.draw.rect(self.game_surface, (0,0,255), obj.interaction_rect, 1)
        scaled_surface = pygame.transform.smoothscale(self.game_surface, self.screen.get_size())
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    @staticmethod
    def create_2d_tilemap(tilemaps, tile_size, tile_scale):
        result = []
        for tilemap in tilemaps:
            image = GameEngine.load_image(tilemap)
            width, height = image.get_size()
            tiles_2d = [
                [image.subsurface((x, y, tile_size, tile_size))
                for x in range(0, width, tile_size)]
                for y in range(0, height, tile_size)
            ]
            scaled_tiles_2d = [
                [GameEngine.scale_image(tile, tile_scale) for tile in row]
                for row in tiles_2d
            ]
            result.append(scaled_tiles_2d)
        return result

    @staticmethod
    def get_npc_area(npcs):
        if not npcs:
            return None
        min_x = min(npc.interaction_rect.left for npc in npcs)
        min_y = min(npc.interaction_rect.top for npc in npcs)
        max_x = max(npc.interaction_rect.right for npc in npcs)
        max_y = max(npc.interaction_rect.bottom for npc in npcs)
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    @staticmethod
    def generate_background(height_map, min_height):
        def compare(x, elevate=True):
            if elevate:
                if not x[1].is_integer():
                    return 7
                if x[0] > x[1] and x[0].is_integer():
                    return 5
            return 2 * int(x[1] > x[2]) or int(x[0] >= x[1])

        height_map = np.pad(height_map, ((1,1), (1,1)), constant_values=-1)
        background = height_map.tolist()

        for y in range(1, len(height_map)-1):
            for x in range(1, len(height_map[y])-1):
                by = compare(height_map[y-1:y+2,x])
                if by != 5 and height_map[y][x] < min_height:
                    continue
                bx = compare(height_map[y,x-1:x+2], False)
                background[y][x] = (by, bx)
        return [i[1:-1] for i in background[:-1]]

    @staticmethod
    def draw_layer(layer, tiles, tile_interval=None):
        tile_w, tile_h = tiles[0][0].get_size() if tile_interval is None else tile_interval
        layer_w = len(layer[0]) * tile_w
        layer_h = len(layer) * tile_h
        surface = pygame.Surface((layer_w, layer_h), pygame.SRCALPHA)
        for y, row in enumerate(layer):
            for x, tile_index in enumerate(row):
                try:
                    surface.blit(tiles[tile_index[0]][tile_index[1]], (x * tile_w, y * tile_h))
                except:
                    pass
        return surface

    @staticmethod
    def generate_border_objects(elevation, tile_size):
        def is_border_tile(tile):
            if not isinstance(tile, tuple):
                return False
            x,y = tile
            return x == 0 or x == 5 or (x in {1, 2} and y != 1)
        border_objects = []
        for y, row in enumerate(elevation):
            for x, tile in enumerate(row):
                if is_border_tile(tile):
                    obj = GameObject(x * tile_size, y * tile_size, pygame.Surface((tile_size, tile_size), pygame.SRCALPHA), center=False)
                    if tile in [(0,1)]:
                        obj.collision_rect = pygame.Rect(obj.rect.left, obj.rect.top, obj.rect.width, 1)
                    if tile in [(1,0), (2,0)]:
                        obj.collision_rect = pygame.Rect(obj.rect.left,obj.rect.top, 1, obj.rect.height)
                    if tile in [(1,2), (2,2)]:
                        obj.collision_rect = pygame.Rect(obj.rect.right,obj.rect.top, 1, obj.rect.height)
                    border_objects.append(obj)
        return border_objects

    @staticmethod
    def get_frames(row, col, sprite_width=16, sprite_height=24, num_frames=3):
        return [pygame.Rect(col * sprite_width, (row+i) * sprite_height, sprite_width, sprite_height) for i in range(num_frames)]

    @staticmethod
    def create_frames(filename, offset=9, scale=1):
        frame_rects = {
            'up': GameEngine.get_frames(offset, 0),
            'right': GameEngine.get_frames(offset, 2),
            'down': GameEngine.get_frames(offset, 4),
            'left': GameEngine.get_frames(offset, 6),
        }
        sprite_sheet = pygame.image.load(os.path.join('assets', filename)).convert_alpha()
        frames = {}
        for direction, rects in frame_rects.items():
            frames[direction] = []
            for rect in rects:
                frame = sprite_sheet.subsurface(rect)
                if scale != 1:
                    new_width = int(rect.width * scale)
                    new_height = int(rect.height * scale)
                    frame = pygame.transform.smoothscale(frame, (new_width, new_height))
                frames[direction].append(frame)
        return frames

    @staticmethod
    def load_image(filename):
        return pygame.image.load(os.path.join('assets', filename)).convert_alpha()

    @staticmethod
    def crop_image(image):
        rect = pygame.mask.from_surface(image).get_bounding_rects()[0]
        cropped_image = pygame.Surface(rect.size, pygame.SRCALPHA)
        cropped_image.blit(image, (0, 0), rect)
        return cropped_image

    @staticmethod
    def scale_image(image, scale_factor):
        new_w = int(image.get_width() * scale_factor)
        new_h = int(image.get_height() * scale_factor)
        return pygame.transform.scale(image, (new_w, new_h))

if __name__ == '__main__':
    # `Create sea (altitude -1)
    height_map = np.array([[-1.]*16]*16)

    # `Create sand (altitude 0)
    height_map[2:16] += 1

    # `Create hills (altidues 1 and 2)
    height_map[1:14, 1:-1] = 1.
    height_map[0:4,2:14] = 2.

    # `Create stairs
    height_map[13, 5:11] = 0.5
    height_map[3, 6:10] = 1.5

    # `Add objects (trees, towers, houses with NPCs, etc)
    dict_objects = {
        'player': {
            'positions': [(8, 14)],
        },
        'castle': {
            'positions': [(8, 0)],
            'image': 'Castle_Blue.png',
        },
        'tower': {
            'positions': [(4, 1.5), (12, 1.5)],
            'image': 'Tower_Blue.png',
        },
        'house': {
            'positions': [(4, 6), (3, 10), (12, 7)],
            'image': 'House_Blue.png',
            'npc_names': generate_random_names(3)
        },
        'tree': {
            'positions': [(14,5), (12,10),(13,9),(13,11)],
            'image': 'Tree.png',
        }
    }

    # `Build game
    game = GameEngine(height_map, dict_objects)

    # `Run game
    game()
