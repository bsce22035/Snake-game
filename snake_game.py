import pygame
import sys
import random
import json
import os
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

try:
    import numpy as np
except Exception:
    np = None

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

DEFAULT_GRID_SIZE = 20
DEFAULT_GRID_WIDTH = WINDOW_WIDTH // DEFAULT_GRID_SIZE
DEFAULT_GRID_HEIGHT = WINDOW_HEIGHT // DEFAULT_GRID_SIZE

# Colors
COLOR_BG = (18, 18, 18)
COLOR_GRID = (40, 40, 40)
COLOR_SNAKE_HEAD = (0, 180, 0)
COLOR_SNAKE_BODY = (0, 140, 0)
COLOR_FOOD = (200, 60, 60)
COLOR_TEXT = (230, 230, 230)
COLOR_TITLE = (255, 200, 60)
COLOR_HIGHLIGHT = (100, 200, 255)
COLOR_BTN = (40, 40, 40)
COLOR_BTN_HOVER = (70, 70, 70)
COLOR_INPUT_BG = (28, 28, 28)
COLOR_OBSTACLE = (120, 120, 120)
COLOR_OBSTACLE_MOVING = (200, 150, 80)

BASE_SPEED = 8.0
SPEED_INCREMENT = 0.5
SCORE_PER_FOOD = 10
SPEED_MILESTONE = 50

LEADERBOARD_FILE = "leaderboard.json"
STATS_FILE = "stats.json"
LEADERBOARD_SIZE = 7

FPS = 60

TITLE_FONT_SIZE = 52
UI_FONT_SIZE = 18
SMALL_FONT_SIZE = 14
SCORE_FONT_SIZE = 16

# Level config
LEVEL_SCORE_STEP = 100   # every 100 points -> next level
MAX_STATIC_OBSTACLES = 12
MAX_MOVING_OBSTACLES = 6
MOVING_OBSTACLE_BASE_SPEED = 0.6  # grid cells per second

#  Data Classes 

@dataclass
class Point:
    x: int
    y: int

    def copy(self):
        return Point(self.x, self.y)


@dataclass
class Snake:
    body: List[Point] = field(default_factory=list)
    direction: Point = field(default_factory=lambda: Point(1, 0))
    grow: int = 0

    def head(self):
        return self.body[0]

    def move_step(self):
        new_head = Point(self.head().x + self.direction.x, self.head().y + self.direction.y)
        self.body.insert(0, new_head)
        if self.grow > 0:
            self.grow -= 1
        else:
            self.body.pop()

    def set_direction(self, new_dir: Point):
        if len(self.body) > 1:
            second = self.body[1]
            dx = self.head().x - second.x
            dy = self.head().y - second.y
            if new_dir.x == -dx and new_dir.y == -dy:
                return
        self.direction = new_dir

    def occupies(self, p: Point) -> bool:
        return any(seg.x == p.x and seg.y == p.y for seg in self.body)


@dataclass
class Food:
    pos: Point


@dataclass
class Obstacle:
    pos: Point
    w: int
    h: int
    moving: bool = False
    vel: Point = field(default_factory=lambda: Point(0, 0))
    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass
class GameConfig:
    grid_size: int = DEFAULT_GRID_SIZE
    grid_width: int = DEFAULT_GRID_WIDTH
    grid_height: int = DEFAULT_GRID_HEIGHT
    snake_color_head: Tuple[int, int, int] = COLOR_SNAKE_HEAD
    snake_color_body: Tuple[int, int, int] = COLOR_SNAKE_BODY
    food_color: Tuple[int, int, int] = COLOR_FOOD
    show_grid: bool = True
    base_speed: float = BASE_SPEED


@dataclass
class GameState:
    snake: Snake
    food: Food
    score: int = 0
    speed: float = BASE_SPEED
    paused: bool = False
    game_over: bool = False
    start_time: float = field(default_factory=time.time)
    play_seconds: float = 0.0
    level: int = 1
    level_message_until: float = 0.0
    obstacles: List[Obstacle] = field(default_factory=list)


# Utility functions 

def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def now_timestamp():
    return int(time.time())


def load_json_file(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json_file(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Error saving JSON:", e)


# ------------------ Sound generation ------------------

def generate_sine_wave(freq=440, duration=0.2, volume=0.2, sample_rate=44100):
    if np is None:
        return None
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    wave = np.sin(freq * t * 2 * math.pi)
    envelope = np.linspace(1.0, 0.01, wave.size)
    wave = wave * envelope
    audio = (wave * (32767 * volume)).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return stereo


def make_sound_from_array(arr):
    if arr is None:
        return None
    try:
        return pygame.sndarray.make_sound(arr)
    except Exception:
        return None


def get_sounds():
    eat_arr = generate_sine_wave(freq=880, duration=0.12, volume=0.15) if np else None
    crash_arr = generate_sine_wave(freq=180, duration=0.4, volume=0.25) if np else None
    click_arr = generate_sine_wave(freq=660, duration=0.08, volume=0.12) if np else None
    eat_s = make_sound_from_array(eat_arr)
    crash_s = make_sound_from_array(crash_arr)
    click_s = make_sound_from_array(click_arr)
    return {"eat": eat_s, "crash": crash_s, "click": click_s}


# ------------------ Leaderboard & Stats ------------------

def load_leaderboard() -> List[Dict]:
    data = load_json_file(LEADERBOARD_FILE, [])
    if not isinstance(data, list):
        return []
    return data


def save_leaderboard(entries: List[Dict]):
    entries_sorted = sorted(entries, key=lambda e: e.get("score", 0), reverse=True)[:LEADERBOARD_SIZE]
    save_json_file(LEADERBOARD_FILE, entries_sorted)


def add_leaderboard_entry(name: str, score: int):
    entries = load_leaderboard()
    entries.append({"name": name, "score": int(score), "ts": now_timestamp()})
    save_leaderboard(entries)


def load_stats() -> Dict:
    default = {"games_played": 0, "total_play_time": 0.0, "highest_score": 0, "total_score": 0}
    s = load_json_file(STATS_FILE, default)
    for k in default:
        if k not in s:
            s[k] = default[k]
    return s


def save_stats(stats: Dict):
    save_json_file(STATS_FILE, stats)


def update_stats_with_game(stats: Dict, score: int, play_seconds: float):
    stats["games_played"] = int(stats.get("games_played", 0)) + 1
    stats["total_play_time"] = float(stats.get("total_play_time", 0.0)) + float(play_seconds)
    stats["highest_score"] = max(int(stats.get("highest_score", 0)), int(score))
    stats["total_score"] = int(stats.get("total_score", 0)) + int(score)
    save_stats(stats)


def get_stats_summary(stats: Dict) -> Dict:
    gp = stats.get("games_played", 0)
    avg = (stats.get("total_score", 0) / gp) if gp > 0 else 0
    total_time = stats.get("total_play_time", 0.0)
    return {
        "games_played": gp,
        "total_play_time": total_time,
        "highest_score": stats.get("highest_score", 0),
        "avg_score": int(avg)
    }


# ------------------ Game Logic (with obstacles) ------------------

def create_initial_snake(config: GameConfig) -> Snake:
    mid_x = config.grid_width // 2
    mid_y = config.grid_height // 2
    body = [Point(mid_x, mid_y), Point(mid_x - 1, mid_y), Point(mid_x - 2, mid_y), Point(mid_x - 3, mid_y)]
    return Snake(body=body, direction=Point(1, 0), grow=0)


def random_food_position_excluding(exclude: List[Point], config: GameConfig) -> Point:
    attempts = 0
    while True:
        x = random.randint(0, config.grid_width - 1)
        y = random.randint(0, config.grid_height - 1)
        p = Point(x, y)
        conflict = any(p.x == e.x and p.y == e.y for e in exclude)
        if not conflict:
            return p
        attempts += 1
        if attempts > 1000:
            for yy in range(config.grid_height):
                for xx in range(config.grid_width):
                    p2 = Point(xx, yy)
                    if not any(p2.x == e.x and p2.y == e.y for e in exclude):
                        return p2


def spawn_food_for_snake(sn: Snake, config: GameConfig, obstacles: List[Obstacle] = None) -> Food:
    exclude = [seg.copy() for seg in sn.body]
    if obstacles:
        for ob in obstacles:
            for oy in range(ob.h):
                for ox in range(ob.w):
                    exclude.append(Point(ob.pos.x + ox, ob.pos.y + oy))
    pos = random_food_position_excluding(exclude, config)
    return Food(pos=pos)


def increase_speed_for_score(base_speed: float, score: int) -> float:
    inc = score // SPEED_MILESTONE
    return base_speed + inc * SPEED_INCREMENT


def collision_with_wall(head: Point, config: GameConfig) -> bool:
    return not (0 <= head.x < config.grid_width and 0 <= head.y < config.grid_height)


def collision_with_body(sn: Snake) -> bool:
    head = sn.head()
    return any(seg.x == head.x and seg.y == head.y for seg in sn.body[1:])


def snake_collides_obstacles(sn: Snake, obstacles: List[Obstacle]) -> bool:
    h = sn.head()
    for ob in obstacles:
        for oy in range(ob.h):
            for ox in range(ob.w):
                if h.x == ob.pos.x + ox and h.y == ob.pos.y + oy:
                    return True
    return False


def generate_obstacles_for_level(level: int, config: GameConfig, snake: Snake) -> List[Obstacle]:
    obstacles = []
    static_count = min(MAX_STATIC_OBSTACLES, 2 + level)
    moving_count = min(MAX_MOVING_OBSTACLES, max(0, level - 2))

    reserved = set()
    midx = config.grid_width // 2
    midy = config.grid_height // 2
    for dx in range(-6, 7):
        for dy in range(-3, 4):
            reserved.add((midx + dx, midy + dy))

    def can_place(px, py, w, h):
        if px < 0 or py < 0 or px + w > config.grid_width or py + h > config.grid_height:
            return False
        for yy in range(h):
            for xx in range(w):
                if (px + xx, py + yy) in reserved:
                    return False
                for seg in snake.body:
                    if seg.x == px + xx and seg.y == py + yy:
                        return False
        for ex in obstacles:
            for yy in range(ex.h):
                for xx in range(ex.w):
                    for yy2 in range(h):
                        for xx2 in range(w):
                            if (ex.pos.x + xx == px + xx2) and (ex.pos.y + yy == py + yy2):
                                return False
        return True

    tries = 0
    while len(obstacles) < static_count and tries < static_count * 200:
        tries += 1
        w = random.choice([1, 1, 2, 3])
        h = random.choice([1, 1, 2])
        x = random.randint(0, config.grid_width - w)
        y = random.randint(0, config.grid_height - h)
        if can_place(x, y, w, h):
            obstacles.append(Obstacle(pos=Point(x, y), w=w, h=h, moving=False))

    tries = 0
    while len([o for o in obstacles if o.moving]) < moving_count and tries < moving_count * 200:
        tries += 1
        w = random.choice([1, 1, 2])
        h = random.choice([1, 1])
        x = random.randint(0, config.grid_width - w)
        y = random.randint(0, config.grid_height - h)
        if can_place(x, y, w, h):
            vx = random.choice([-1, 1])
            vy = random.choice([-1, 0, 1])
            ob = Obstacle(pos=Point(x, y), w=w, h=h, moving=True, vel=Point(vx, vy))
            obstacles.append(ob)
    return obstacles


def update_moving_obstacles(obstacles: List[Obstacle], config: GameConfig, dt: float, level: int):
    speed = MOVING_OBSTACLE_BASE_SPEED + (level - 1) * 0.1
    for ob in obstacles:
        if not ob.moving:
            continue
        dirx = 0 if ob.vel.x == 0 else (1 if ob.vel.x > 0 else -1)
        diry = 0 if ob.vel.y == 0 else (1 if ob.vel.y > 0 else -1)
        dx = dirx * speed * dt
        dy = diry * speed * dt
        ob.offset_x += dx
        ob.offset_y += dy
        while ob.offset_x >= 1.0:
            ob.pos.x += 1
            ob.offset_x -= 1.0
        while ob.offset_x <= -1.0:
            ob.pos.x -= 1
            ob.offset_x += 1.0
        while ob.offset_y >= 1.0:
            ob.pos.y += 1
            ob.offset_y -= 1.0
        while ob.offset_y <= -1.0:
            ob.pos.y -= 1
            ob.offset_y += 1.0
        bounced = False
        if ob.pos.x < 0:
            ob.pos.x = 0
            ob.vel.x *= -1
            ob.offset_x = 0
            bounced = True
        if ob.pos.y < 0:
            ob.pos.y = 0
            ob.vel.y *= -1
            ob.offset_y = 0
            bounced = True
        if ob.pos.x + ob.w > config.grid_width:
            ob.pos.x = config.grid_width - ob.w
            ob.vel.x *= -1
            ob.offset_x = 0
            bounced = True
        if ob.pos.y + ob.h > config.grid_height:
            ob.pos.y = config.grid_height - ob.h
            ob.vel.y *= -1
            ob.offset_y = 0
            bounced = True
        if bounced and ob.vel.x == 0 and ob.vel.y == 0:
            ob.vel.x = random.choice([-1, 1])
            ob.vel.y = random.choice([-1, 0, 1])


# ------------------ Rendering Helpers ------------------

def grid_to_pixels(p: Point, config: GameConfig) -> Tuple[int, int]:
    return p.x * config.grid_size, p.y * config.grid_size


def draw_grid(surface, config: GameConfig):
    if not config.show_grid:
        return
    for x in range(0, WINDOW_WIDTH, config.grid_size):
        pygame.draw.line(surface, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, config.grid_size):
        pygame.draw.line(surface, COLOR_GRID, (0, y), (WINDOW_WIDTH, y))


def draw_snake(surface, snake: Snake, config: GameConfig):
    for i, seg in enumerate(snake.body):
        px, py = grid_to_pixels(seg, config)
        rect = pygame.Rect(px, py, config.grid_size, config.grid_size)
        if i == 0:
            pygame.draw.rect(surface, config.snake_color_head, rect)
        else:
            pygame.draw.rect(surface, config.snake_color_body, rect)


def draw_food(surface, food: Food, config: GameConfig):
    px, py = grid_to_pixels(food.pos, config)
    rect = pygame.Rect(px + 2, py + 2, config.grid_size - 4, config.grid_size - 4)
    pygame.draw.ellipse(surface, config.food_color, rect)


def draw_obstacles(surface, obstacles: List[Obstacle], config: GameConfig):
    for ob in obstacles:
        px, py = grid_to_pixels(ob.pos, config)
        rect = pygame.Rect(px, py, ob.w * config.grid_size, ob.h * config.grid_size)
        color = COLOR_OBSTACLE_MOVING if ob.moving else COLOR_OBSTACLE
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (20, 20, 20), rect, 2)


def render_text_center(surface, text, font, color, y_offset=0):
    s = font.render(text, True, color)
    r = s.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + y_offset))
    surface.blit(s, r)


# ------------------ Simple UI Primitives ------------------

class Button:
    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font, action=None):
        self.rect = rect
        self.text = text
        self.font = font
        self.action = action
        self.hover = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()

    def draw(self, surf):
        color = COLOR_BTN_HOVER if self.hover else COLOR_BTN
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        t = self.font.render(self.text, True, COLOR_TEXT)
        tr = t.get_rect(center=self.rect.center)
        surf.blit(t, tr)


class Slider:
    def __init__(self, rect: pygame.Rect, min_val, max_val, value, font, label=""):
        self.rect = rect
        self.min = min_val
        self.max = max_val
        self.value = value
        self.font = font
        self.label = label
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.set_by_pos(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_by_pos(event.pos[0])

    def set_by_pos(self, x):
        rel = clamp((x - self.rect.x) / self.rect.w, 0.0, 1.0)
        self.value = self.min + rel * (self.max - self.min)

    def draw(self, surf):
        pygame.draw.rect(surf, COLOR_BTN, self.rect, border_radius=6)
        rel = (self.value - self.min) / (self.max - self.min)
        kx = int(self.rect.x + rel * self.rect.w)
        ky = self.rect.centery
        pygame.draw.circle(surf, COLOR_HIGHLIGHT, (kx, ky), 8)
        lbl = f"{self.label}: {int(self.value)}"
        t = self.font.render(lbl, True, COLOR_TEXT)
        surf.blit(t, (self.rect.x, self.rect.y - 22))


class Toggle:
    def __init__(self, rect: pygame.Rect, value: bool, font, label=""):
        self.rect = rect
        self.value = value
        self.font = font
        self.label = label

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value

    def draw(self, surf):
        pygame.draw.rect(surf, COLOR_BTN, self.rect, border_radius=6)
        if self.value:
            pygame.draw.rect(surf, COLOR_HIGHLIGHT, self.rect.inflate(-6, -6), border_radius=6)
        t = self.font.render(f"{self.label}: {'On' if self.value else 'Off'}", True, COLOR_TEXT)
        surf.blit(t, (self.rect.x + 6, self.rect.y + 6))


# ------------------ Title & Settings Screen (fixed centered layout) ------------------

def title_and_settings_screen(screen, clock, fonts, sounds, config: GameConfig):
    title_font, ui_font, small_font = fonts

    # controls
    button_width = 220
    button_height = 40

    # Create UI objects positioned relative to center (we will compute exact positions in the loop)
    # We store button objects to manage hover/click (we'll set rects each loop)
    start_btn = Button(pygame.Rect(0, 0, button_width, button_height), "Start Game", ui_font, action=lambda: None)
    leaderboard_btn = Button(pygame.Rect(0, 0, button_width, button_height), "Show Leaderboard", ui_font, action=lambda: None)
    quit_btn = Button(pygame.Rect(0, 0, button_width, button_height), "Quit", ui_font, action=lambda: quit_program())

    grid_slider = Slider(pygame.Rect(0, 0, 320, 14), 12, 36, config.grid_size, small_font, label="Grid Size")
    speed_slider = Slider(pygame.Rect(0, 0, 320, 14), 4, 18, config.base_speed, small_font, label="Base Speed")
    show_grid_toggle = Toggle(pygame.Rect(0, 0, 160, 34), config.show_grid, small_font, label="Gridlines")

    color_options = [
        ("Green", (0, 180, 0), (0, 140, 0)),
        ("Cyan", (0, 190, 180), (0, 150, 140)),
        ("Red", (200, 60, 60), (180, 40, 40)),
        ("Purple", (170, 90, 200), (140, 60, 170))
    ]
    color_idx = 0
    color_btn = Button(pygame.Rect(0, 0, 160, 34), f"Color: {color_options[color_idx][0]}", small_font, action=lambda: None)

    show_leaderboard = False

    while True:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_program()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                                 pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d):
                    return key_to_direction(event.key), config
            # forward events to UI briefly (we'll update rects below per frame)
            grid_slider.handle_event(event)
            speed_slider.handle_event(event)
            show_grid_toggle.handle_event(event)
            start_btn.handle_event(event)
            leaderboard_btn.handle_event(event)
            quit_btn.handle_event(event)
            color_btn.handle_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if color_btn.rect.collidepoint(event.pos):
                    color_idx = (color_idx + 1) % len(color_options)
                    color_btn.text = f"Color: {color_options[color_idx][0]}"
                    if sounds.get("click"):
                        sounds["click"].play()

        # compute centered layout
        screen.fill(COLOR_BG)
        # title
        title_surf = title_font.render("SNAKE+(Obstacles & Levels)", True, COLOR_TITLE)
        screen.blit(title_surf, title_surf.get_rect(center=(WINDOW_WIDTH // 2, 60)))

        # info lines centered
        info_lines = [
            "New: levels and obstacles! Avoid static and moving blocks.",
            "Controls: Arrow keys / WASD to move. P to pause. R to restart.",
            "Eat food to score points. Levels increase every 100 points.",
            "Leaderboard and stats are persistent in JSON files."
        ]
        y = 110
        for line in info_lines:
            s = small_font.render(line, True, COLOR_TEXT)
            screen.blit(s, s.get_rect(center=(WINDOW_WIDTH // 2, y)))
            y += 22

        # settings region
        settings_y = y + 20
        slider_w = 360
        center_x = WINDOW_WIDTH // 2 - slider_w // 2

        # set rects so UI objects respond correctly
        grid_slider.rect = pygame.Rect(center_x, settings_y, slider_w, 14)
        grid_slider.draw(screen)
        settings_y += 56

        speed_slider.rect = pygame.Rect(center_x, settings_y, slider_w, 14)
        speed_slider.draw(screen)
        settings_y += 56

        show_grid_toggle.rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, settings_y, 160, 34)
        show_grid_toggle.draw(screen)
        settings_y += 56

        color_btn.rect = pygame.Rect(WINDOW_WIDTH // 2 - 80, settings_y, 160, 34)
        color_btn.draw(screen)
        settings_y += 64

        # Buttons: centered below
        bx = WINDOW_WIDTH // 2 - button_width // 2
        by = settings_y
        start_btn.rect = pygame.Rect(bx, by, button_width, button_height)
        start_btn.draw(screen)
        by += 56
        leaderboard_btn.rect = pygame.Rect(bx, by, button_width, button_height)
        leaderboard_btn.draw(screen)
        by += 56
        quit_btn.rect = pygame.Rect(bx, by, button_width, button_height)
        quit_btn.draw(screen)

        # handle mouse clicks manually for start/leaderboard/quit
        mx, my = pygame.mouse.get_pos()
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:
            if start_btn.rect.collidepoint((mx, my)):
                # commit settings from sliders/toggles
                config.grid_size = int(grid_slider.value)
                config.grid_width = WINDOW_WIDTH // config.grid_size
                config.grid_height = WINDOW_HEIGHT // config.grid_size
                config.base_speed = float(speed_slider.value)
                config.show_grid = bool(show_grid_toggle.value)
                chead, cbody = color_options[color_idx][1], color_options[color_idx][2]
                config.snake_color_head = chead
                config.snake_color_body = cbody
                if sounds.get("click"):
                    sounds["click"].play()
                pygame.time.wait(100)
                return Point(1, 0), config
            if leaderboard_btn.rect.collidepoint((mx, my)):
                show_leaderboard = not show_leaderboard
                if sounds.get("click"):
                    sounds["click"].play()
                pygame.time.wait(120)
            if quit_btn.rect.collidepoint((mx, my)):
                if sounds.get("click"):
                    sounds["click"].play()
                quit_program()

        if show_leaderboard:
            draw_leaderboard_panel(screen, ui_font, small_font)

        pygame.display.flip()


def draw_leaderboard_panel(surface, ui_font, small_font):
    entries = load_leaderboard()
    panel_w = 420
    panel_h = 260
    panel_rect = pygame.Rect((WINDOW_WIDTH - panel_w) // 2, (WINDOW_HEIGHT - panel_h) // 2, panel_w, panel_h)
    pygame.draw.rect(surface, COLOR_BTN, panel_rect, border_radius=8)
    pygame.draw.rect(surface, COLOR_HIGHLIGHT, panel_rect, 2, border_radius=8)
    t = ui_font.render("Leaderboard", True, COLOR_TITLE)
    surface.blit(t, (panel_rect.x + 18, panel_rect.y + 12))
    stats = load_stats()
    summary = get_stats_summary(stats)
    st = small_font.render(f"Games: {summary['games_played']}  Time: {int(summary['total_play_time'])}s  High: {summary['highest_score']}  Avg: {summary['avg_score']}", True, COLOR_TEXT)
    surface.blit(st, (panel_rect.x + 18, panel_rect.y + 44))
    for idx, e in enumerate(entries[:LEADERBOARD_SIZE]):
        s = small_font.render(f"{idx + 1}. {e.get('name','?')} - {e.get('score',0)}", True, COLOR_TEXT)
        surface.blit(s, (panel_rect.x + 18, panel_rect.y + 80 + idx * 26))


# ------------------ Input helpers ------------------

def key_to_direction(key):
    if key in (pygame.K_UP, pygame.K_w):
        return Point(0, -1)
    if key in (pygame.K_DOWN, pygame.K_s):
        return Point(0, 1)
    if key in (pygame.K_LEFT, pygame.K_a):
        return Point(-1, 0)
    if key in (pygame.K_RIGHT, pygame.K_d):
        return Point(1, 0)
    return Point(1, 0)


# ------------------ Main Game Loop (with obstacles & levels) ------------------

def run_game(start_dir: Optional[Point], config: GameConfig, fonts, sounds):
    title_font, ui_font, small_font = fonts
    snake = create_initial_snake(config)
    if start_dir:
        snake.direction = start_dir
    obstacles = generate_obstacles_for_level(1, config, snake)
    food = spawn_food_for_snake(snake, config, obstacles)
    state = GameState(snake=snake, food=food, score=0, speed=config.base_speed, paused=False, game_over=False,
                      start_time=time.time(), play_seconds=0.0, level=1, level_message_until=0.0, obstacles=obstacles)

    key_dir_map = {
        pygame.K_UP: Point(0, -1),
        pygame.K_w: Point(0, -1),
        pygame.K_DOWN: Point(0, 1),
        pygame.K_s: Point(0, 1),
        pygame.K_LEFT: Point(-1, 0),
        pygame.K_a: Point(-1, 0),
        pygame.K_RIGHT: Point(1, 0),
        pygame.K_d: Point(1, 0)
    }

    clock = pygame.time.Clock()
    move_accumulator = 0.0

    while True:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_program()
            if event.type == pygame.KEYDOWN:
                if event.key in key_dir_map:
                    snake.set_direction(key_dir_map[event.key])
                elif event.key == pygame.K_p:
                    if not state.game_over:
                        state.paused = not state.paused
                        if sounds.get("click"):
                            sounds["click"].play()
                elif event.key == pygame.K_r:
                    if state.game_over:
                        update_stats_after_game(state, config)
                        return "restart"
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_program()

        if state.paused:
            screen = pygame.display.get_surface()
            screen.fill(COLOR_BG)
            draw_grid(screen, config)
            draw_obstacles(screen, state.obstacles, config)
            draw_food(screen, state.food, config)
            draw_snake(screen, state.snake, config)
            render_text_center(screen, "PAUSED", title_font, COLOR_HIGHLIGHT, y_offset=-40)
            render_text_center(screen, "Press P to resume", ui_font, COLOR_TEXT, y_offset=0)
            pygame.display.flip()
            continue

        if state.game_over:
            player_name = prompt_for_name_and_save(state.score, state.play_seconds, fonts, sounds)
            update_stats_after_game(state, config)
            res = game_over_menu(fonts, state.score)
            if res == "restart":
                return "restart"
            else:
                quit_program()

        state.speed = increase_speed_for_score(config.base_speed, state.score)
        timestep = 1.0 / state.speed
        move_accumulator += dt

        update_moving_obstacles(state.obstacles, config, dt, state.level)

        while move_accumulator >= timestep:
            move_accumulator -= timestep
            snake.move_step()
            if collision_with_wall(snake.head(), config) or collision_with_body(snake):
                state.game_over = True
                if sounds.get("crash"):
                    sounds["crash"].play()
                break
            if snake_collides_obstacles(snake, state.obstacles):
                state.game_over = True
                if sounds.get("crash"):
                    sounds["crash"].play()
                break
            if snake.head().x == state.food.pos.x and snake.head().y == state.food.pos.y:
                snake.grow += 1
                state.score += SCORE_PER_FOOD
                new_level = (state.score // LEVEL_SCORE_STEP) + 1
                if new_level > state.level:
                    state.level = new_level
                    state.level_message_until = time.time() + 1.4
                    state.obstacles = generate_obstacles_for_level(state.level, config, snake)
                    state.food = spawn_food_for_snake(snake, config, state.obstacles)
                    if sounds.get("click"):
                        sounds["click"].play()
                else:
                    state.food = spawn_food_for_snake(snake, config, state.obstacles)
                if sounds.get("eat"):
                    sounds["eat"].play()

        state.play_seconds = time.time() - state.start_time

        screen = pygame.display.get_surface()
        screen.fill(COLOR_BG)
        draw_grid(screen, config)
        draw_obstacles(screen, state.obstacles, config)
        draw_food(screen, state.food, config)
        draw_snake(screen, state.snake, config)

        score_s = small_font.render(f"Score: {state.score}", True, COLOR_TEXT)
        screen.blit(score_s, (8, 8))
        level_s = small_font.render(f"Level: {state.level}", True, COLOR_TEXT)
        screen.blit(level_s, (8, 30))
        speed_s = small_font.render(f"Speed: {state.speed:.1f}", True, COLOR_TEXT)
        screen.blit(speed_s, (8, 52))
        time_s = small_font.render(f"Time: {int(state.play_seconds)}s", True, COLOR_TEXT)
        screen.blit(time_s, (8, 74))
        hint_s = small_font.render("P:Pause  R:Restart(on game over)  Esc:Quit", True, COLOR_TEXT)
        screen.blit(hint_s, (8, WINDOW_HEIGHT - 22))

        if state.level_message_until > time.time():
            msg = title_font.render(f"Level {state.level}", True, COLOR_HIGHLIGHT)
            r = msg.get_rect(center=(WINDOW_WIDTH // 2, 40))
            screen.blit(msg, r)

        pygame.display.flip()


def update_stats_after_game(state: GameState, config: GameConfig):
    stats = load_stats()
    update_stats_with_game(stats, state.score, state.play_seconds)


# ------------------ Game Over / Name prompt / Quit ------------------

def game_over_menu(fonts, score):
    title_font, ui_font, small_font = fonts
    clock = pygame.time.Clock()
    while True:
        dt = clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_program()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "restart"
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_program()
        screen = pygame.display.get_surface()
        screen.fill(COLOR_BG)
        render_text_center(screen, "Game Over", title_font, COLOR_TITLE, y_offset=-80)
        render_text_center(screen, f"Score: {score}", ui_font, COLOR_TEXT, y_offset=-20)
        render_text_center(screen, "Press R to restart or Esc to quit", small_font, COLOR_TEXT, y_offset=40)
        entries = load_leaderboard()
        for idx, e in enumerate(entries[:5]):
            s = small_font.render(f"{idx+1}. {e.get('name','?')} - {e.get('score',0)}", True, COLOR_TEXT)
            screen.blit(s, (WINDOW_WIDTH // 2 - 80, WINDOW_HEIGHT // 2 + 80 + idx * 22))
        pygame.display.flip()


def prompt_for_name_and_save(score, play_seconds, fonts, sounds):
    title_font, ui_font, small_font = fonts
    clock = pygame.time.Clock()
    input_rect = pygame.Rect((WINDOW_WIDTH // 2 - 180, WINDOW_HEIGHT // 2 + 20, 360, 36))
    text = ""
    placeholder = "Enter your name (press Enter to save)"
    while True:
        dt = clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_program()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    name = text.strip() or "Player"
                    add_leaderboard_entry(name, score)
                    if sounds.get("click"):
                        sounds["click"].play()
                    return name
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    if len(event.unicode) == 1 and len(text) < 20:
                        text += event.unicode
        screen = pygame.display.get_surface()
        screen.fill(COLOR_BG)
        render_text_center(screen, "Game Over", title_font, COLOR_TITLE, y_offset=-120)
        render_text_center(screen, f"Score: {score}", ui_font, COLOR_TEXT, y_offset=-60)
        pygame.draw.rect(screen, COLOR_INPUT_BG, input_rect, border_radius=6)
        display_text = text if text else placeholder
        t = small_font.render(display_text, True, COLOR_TEXT if text else (120, 120, 120))
        screen.blit(t, (input_rect.x + 8, input_rect.y + 8))
        pygame.display.flip()


def quit_program():
    pygame.quit()
    sys.exit(0)


# ------------------ Main ------------------

def main():
    pygame.init()
    try:
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
    except Exception:
        pass
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Plus - Obstacles & Levels (Fixed UI)")
    clock = pygame.time.Clock()

    try:
        title_font = pygame.font.SysFont("consolas", TITLE_FONT_SIZE, bold=True)
        ui_font = pygame.font.SysFont("consolas", UI_FONT_SIZE)
        small_font = pygame.font.SysFont("consolas", SMALL_FONT_SIZE)
    except Exception:
        title_font = pygame.font.Font(None, TITLE_FONT_SIZE)
        ui_font = pygame.font.Font(None, UI_FONT_SIZE)
        small_font = pygame.font.Font(None, SMALL_FONT_SIZE)

    fonts = (title_font, ui_font, small_font)
    sounds = get_sounds()

    config = GameConfig()
    config.grid_size = DEFAULT_GRID_SIZE
    config.grid_width = WINDOW_WIDTH // config.grid_size
    config.grid_height = WINDOW_HEIGHT // config.grid_size
    config.base_speed = BASE_SPEED

    # main loop: title/settings -> game -> repeat on restart
    while True:
        start_dir, config = title_and_settings_screen(screen, clock, fonts, sounds, config)
        res = run_game(start_dir, config, fonts, sounds)
        if res == "restart":
            continue
        else:
            break


# ------------------ Entry ------------------

if __name__ == "__main__":
    main()




