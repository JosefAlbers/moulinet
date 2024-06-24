import pygame
import os
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D RPG with Natural Environment")

# Load sprite sheet
sprite_sheet = pygame.image.load("Small-8-Direction-Characters_by_AxulArt.png").convert_alpha()

# Define tile size
TILE_SIZE = 32  # Adjust this value as needed

# Load images
grass_image = pygame.image.load(os.path.join('TILESET VILLAGE TOP DOWN', 'GRASS TILE - DAY.png')).convert_alpha()
tree_image = pygame.image.load(os.path.join('TILESET VILLAGE TOP DOWN', 'TREE 1 - DAY.png')).convert_alpha()
house_image = pygame.image.load(os.path.join('TILESET VILLAGE TOP DOWN', 'HOUSE 1 - DAY.png')).convert_alpha()

# Scale grass image to match TILE_SIZE
grass_image = pygame.transform.scale(grass_image, (TILE_SIZE, TILE_SIZE))

# Calculate grid dimensions
GRID_WIDTH = WIDTH // TILE_SIZE
GRID_HEIGHT = HEIGHT // TILE_SIZE

# Function to generate natural object placement
def generate_natural_environment():
    objects = []

    # Create a small village area
    village_center_x = GRID_WIDTH // 2
    village_center_y = GRID_HEIGHT // 2

    # Minimum distance between houses (in grid units)
    MIN_HOUSE_DISTANCE = 4

    # Place houses
    for _ in range(3):  # Place 3 houses
        attempts = 0
        while attempts < 50:  # Limit attempts to avoid infinite loop
            x = village_center_x + random.randint(-5, 5)
            y = village_center_y + random.randint(-5, 5)

            # Check if the new house is far enough from existing houses
            if all(math.sqrt((x - ox)**2 + (y - oy)**2) >= MIN_HOUSE_DISTANCE
                   for obj, (ox, oy) in objects if obj == 'HOUSE'):
                objects.append(('HOUSE', (x, y)))
                break
            attempts += 1

    # Place trees
    for _ in range(15):  # Place 15 trees
        attempts = 0
        while attempts < 50:  # Limit attempts to avoid infinite loop
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            # Ensure trees are not too close to houses or other trees
            if all((abs(x - ox) > 1 or abs(y - oy) > 1) for _, (ox, oy) in objects):
                objects.append(('TREE', (x, y)))
                break
            attempts += 1

    return objects


# Generate natural environment
environment_objects = generate_natural_environment()

def draw_background():
    # Fill the screen with grass tiles
    for y in range(0, HEIGHT, TILE_SIZE):
        for x in range(0, WIDTH, TILE_SIZE):
            screen.blit(grass_image, (x, y))

    # Draw environment objects
    for obj_type, (x, y) in environment_objects:
        if obj_type == 'TREE':
            screen.blit(tree_image, (x * TILE_SIZE, y * TILE_SIZE - tree_image.get_height() + TILE_SIZE))
        elif obj_type == 'HOUSE':
            screen.blit(house_image, (x * TILE_SIZE, y * TILE_SIZE - house_image.get_height() + TILE_SIZE))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define frame rectangles for the characters
SPRITE_WIDTH = 16
SPRITE_HEIGHT = 24
frame_rects_down = [pygame.Rect(4*SPRITE_WIDTH, (9+i) * SPRITE_HEIGHT, SPRITE_WIDTH, SPRITE_HEIGHT) for i in range(3)]
frame_rects_up = [pygame.Rect(0, (9+i) * SPRITE_HEIGHT, SPRITE_WIDTH, SPRITE_HEIGHT) for i in range(3)]
frame_rects_right = [pygame.Rect(2*SPRITE_WIDTH, (9+i) * SPRITE_HEIGHT, SPRITE_WIDTH, SPRITE_HEIGHT) for i in range(3)]
frame_rects_left = [pygame.Rect(6*SPRITE_WIDTH, (9+i) * SPRITE_HEIGHT, SPRITE_WIDTH, SPRITE_HEIGHT) for i in range(3)]
frame_rects_npc = [pygame.Rect(4*SPRITE_WIDTH, (5+i) * SPRITE_HEIGHT, SPRITE_WIDTH, SPRITE_HEIGHT) for i in range(3)]

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.frames_up = [pygame.transform.scale(sprite_sheet.subsurface(rect), (32, 48)) for rect in frame_rects_up]
        self.frames_down = [pygame.transform.scale(sprite_sheet.subsurface(rect), (32, 48)) for rect in frame_rects_down]
        self.frames_right = [pygame.transform.scale(sprite_sheet.subsurface(rect), (32, 48)) for rect in frame_rects_right]
        self.frames_left = [pygame.transform.scale(sprite_sheet.subsurface(rect), (32, 48)) for rect in frame_rects_left]
        self.frames = self.frames_down
        self.current_frame = 0
        self.image = self.frames[self.current_frame]
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.speed = 5
        self.facing = 'down'
        self.animation_speed = 0.2
        self.animation_time = 0
        self.is_moving = False
        self.interact_range = 50
        self.speech_bubble = None

    def update(self):
        if self.is_moving:
            self.animation_time += self.animation_speed
            if self.animation_time >= len(self.frames):
                self.animation_time = 0
            self.current_frame = int(self.animation_time)
            self.image = self.frames[self.current_frame]
        else:
            self.image = self.frames[0]

    def move(self, dx, dy):
        if dx == 0 and dy == 0:
            self.is_moving = False
            return

        self.is_moving = True
        self.rect.x += dx * self.speed
        self.rect.y += dy * self.speed
        self.rect.clamp_ip(screen.get_rect())

        if dx > 0:
            self.facing = 'right'
            self.frames = self.frames_right
        elif dx < 0:
            self.facing = 'left'
            self.frames = self.frames_left
        elif dy < 0:
            self.facing = 'up'
            self.frames = self.frames_up
        elif dy > 0:
            self.facing = 'down'
            self.frames = self.frames_down

    def interact(self, npcs):
        for npc in npcs:
            if pygame.sprite.collide_circle_ratio(self.interact_range / 32)(self, npc):
                self.speech_bubble = SpeechBubble("Hello!", self.rect.midtop)
                npc.speech_bubble = SpeechBubble("Hi there!", npc.rect.midtop)
                return
        self.speech_bubble = None

class NPC(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.frames = [pygame.transform.scale(sprite_sheet.subsurface(rect), (32, 48)) for rect in frame_rects_npc]
        self.image = self.frames[0]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speech_bubble = None

class SpeechBubble:
    def __init__(self, text, position):
        self.font = pygame.font.Font(None, 24)
        self.text = self.font.render(text, True, BLACK)
        self.rect = self.text.get_rect()
        self.rect.midbottom = position

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect.inflate(10, 10))
        pygame.draw.rect(surface, BLACK, self.rect.inflate(10, 10), 2)
        surface.blit(self.text, self.rect)

# Create sprites
player = Player()
npcs = pygame.sprite.Group()
for _ in range(3):
    npcs.add(NPC(random.randint(0, WIDTH - 32), random.randint(0, HEIGHT - 32)))

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.interact(npcs)

    # Player movement
    keys = pygame.key.get_pressed()
    dx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
    dy = keys[pygame.K_DOWN] - keys[pygame.K_UP]
    player.move(dx, dy)

    # Update player
    player.update()

    # Draw everything
    draw_background()
    screen.blit(player.image, player.rect)
    npcs.draw(screen)

    # Draw speech bubbles
    if player.speech_bubble:
        player.speech_bubble.draw(screen)
    for npc in npcs:
        if npc.speech_bubble:
            npc.speech_bubble.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
