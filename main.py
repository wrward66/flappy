import pygame
from pygame.locals import QUIT
import random
import numpy as np
from itertools import count
import math

# -----------------------------
# Optimized Neural Network
# -----------------------------
class NeuralNetwork:
    def __init__(self, input_size=7, hidden_size=10, output_size=1, weights=None):
        if weights is not None:
            w1, b1, w2, b2 = weights
            self.w1 = w1.copy()
            self.b1 = b1.copy()
            self.w2 = w2.copy()
            self.b2 = b2.copy()
        else:
            # He initialization for ReLU-like activations
            self.w1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.random.randn(hidden_size, 1) * 0.01
            self.w2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.random.randn(output_size, 1) * 0.01

    def predict(self, inputs):
        x = np.array(inputs, dtype=np.float32).reshape(-1, 1)
        z1 = np.dot(self.w1, x) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.sigmoid(z2)
        return float(a2[0, 0])

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    def get_weights(self):
        return [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]

    def set_weights(self, weights):
        w1, b1, w2, b2 = weights
        self.w1 = w1.copy()
        self.b1 = b1.copy()
        self.w2 = w2.copy()
        self.b2 = b2.copy()

    def mutate(self, rate=0.1, strength=0.2):
        """Conservative mutation for stable learning"""
        def mutate_array(arr):
            mask = np.random.rand(*arr.shape) < rate
            noise = np.random.randn(*arr.shape) * strength
            return arr + noise * mask

        self.w1 = mutate_array(self.w1)
        self.b1 = mutate_array(self.b1)
        self.w2 = mutate_array(self.w2)
        self.b2 = mutate_array(self.b2)

    def crossover(self, other):
        """Uniform crossover"""
        def cross(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)

        return NeuralNetwork(weights=[
            cross(self.w1, other.w1),
            cross(self.b1, other.b1),
            cross(self.w2, other.w2),
            cross(self.b2, other.b2)
        ])

# -----------------------------
# Pygame Setup - ADJUSTED DIFFICULTY
# -----------------------------
pygame.init()
clock = pygame.time.Clock()
fps = 60
screen_width = 864
screen_height = 936
GROUND_Y = 768
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird AI')
font = pygame.font.SysFont('Bauhaus 93', 60)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)

# Load images (using placeholder surfaces if files not found)
try:
    bg = pygame.image.load('bg.png').convert()
except:
    bg = pygame.Surface((screen_width, screen_height))
    bg.fill((100, 150, 255))

try:
    ground = pygame.image.load('ground.png').convert_alpha()
except:
    ground = pygame.Surface((screen_width, 100))
    ground.fill((100, 60, 10))

try:
    restart_img = pygame.image.load('restart.png').convert_alpha()
except:
    restart_img = pygame.Surface((100, 50))
    restart_img.fill((200, 0, 0))

# Create bird frames if not available
_bird_frames = []
for i in range(3):
    try:
        _bird_frames.append(pygame.image.load(f'bird{i+1}.png').convert_alpha())
    except:
        bird_surf = pygame.Surface((40, 30))
        bird_surf.fill((255, 255, 0) if i == 0 else (220, 220, 0) if i == 1 else (200, 200, 0))
        _bird_frames.append(bird_surf)

scroll = 0
scroll_speed = 4.5

pipe_gap = 160
pipe_frequency = 1700
last_pipe = pygame.time.get_ticks() - pipe_frequency
class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.images = _bird_frames[:]
        self.index = 0
        self.counter = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect(center=(x, y))
        self.vel = 0.0
        self.alive = True
        self.score = 0
        self.frames_alive = 0
        self.passed_pipes = set()

    def physics(self):
        self.vel = min(self.vel + 0.5, 9)
        self.rect.y += int(self.vel)
        
        if self.rect.top <= 0:
            self.alive = False
            self.kill()
            
        if self.rect.bottom >= GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.alive = False
            self.kill()

    def animate(self):
        self.counter += 1
        if self.counter > 5:
            self.counter = 0
            self.index = (self.index + 1) % len(self.images)
            
        rotated = pygame.transform.rotate(self.images[self.index], -2 * self.vel)
        center = self.rect.center
        self.image = rotated
        self.rect = self.image.get_rect(center=center)

    def update(self):
        if not self.alive:
            return
            
        self.physics()
        self.animate()
        self.frames_alive += 1

class AIBird(Bird):
    def __init__(self, x, y, brain=None):
        super().__init__(x, y)
        self.nn = NeuralNetwork(weights=brain.get_weights()) if isinstance(brain, NeuralNetwork) else NeuralNetwork()
        self.next_pipe_distance = 0
        self.gap_center = screen_height // 2
        self.next_pipe_x = screen_width
        self.last_flap_time = 0
        self.total_gap_distance = 0

    def decide(self, pipes):
        if not self.alive:
            return
            
        next_top, next_bottom, min_dx = None, None, float('inf')
        for p in pipes:
            if p.position == 1 and p.rect.centerx > self.rect.centerx - 50:
                dx = p.rect.centerx - self.rect.centerx
                if dx < min_dx:
                    min_dx = dx
                    next_top, next_bottom = p, p.partner
                    
        if next_top and next_bottom:
            gap_center = (next_top.rect.bottom + next_bottom.rect.top) / 2.0
            self.next_pipe_distance = min_dx
            self.gap_center = gap_center
            self.next_pipe_x = next_top.rect.centerx
            
            gap_distance = abs(self.rect.centery - gap_center)
            self.total_gap_distance += gap_distance
            
            time_since_flap = pygame.time.get_ticks() - self.last_flap_time
            inputs = [
                (self.rect.centery - 50) / (GROUND_Y - 100),
                (gap_center - 50) / (GROUND_Y - 100),
                
                np.tanh(self.vel / 5.0),

                min_dx / 400.0,
                
                np.tanh((self.rect.centery - gap_center) / 100.0),
                
                min(time_since_flap / 200.0, 1.0),
                
                np.tanh((self.rect.centery + self.vel * 10 - gap_center) / 100.0)
            ]
            
            if self.nn.predict(inputs) > 0.5:
                self.flap()
        else:
            self.next_pipe_distance = screen_width
            self.gap_center = screen_height // 2
            self.next_pipe_x = screen_width

    def flap(self):
        self.vel = -9.8
        self.last_flap_time = pygame.time.get_ticks()

    def update(self):
        super().update()
        # 1. Survival bonus - linear reward for staying alive
        survival_points = self.frames_alive * 2.0
        
        # 2. Score bonus - exponential reward for clearing pipes
        score_points = self.score * 500 + (self.score ** 2) * 200
        
        # 3. Gap centering bonus - reward staying near gap centers
        avg_gap_distance = self.total_gap_distance / max(1, self.frames_alive)
        centering_bonus = max(0, 100 - avg_gap_distance) * 2
        
        # 4. Progress bonus - reward moving toward pipes
        if self.next_pipe_distance < 300:
            progress_bonus = (300 - self.next_pipe_distance) * 0.5
        else:
            progress_bonus = 0
            
        # 5. Penalty for being too high/low (avoid ceiling/ground camping)
        if self.rect.centery < 100 or self.rect.centery > GROUND_Y - 100:
            boundary_penalty = 50
        else:
            boundary_penalty = 0
            
        self.fitness = survival_points + score_points + centering_bonus + progress_bonus - boundary_penalty

_pipe_id_counter = count(1)

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        super().__init__()
        try:
            self.image = pygame.image.load('pipe.png').convert_alpha()
        except:
            self.image = pygame.Surface((80, 500))
            self.image.fill((0, 180, 0))
            
        self.rect = self.image.get_rect()
        self.id = next(_pipe_id_counter)
        self.position = position
        self.partner = None
        
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = (x, y - pipe_gap // 2)
        else:
            self.rect.topleft = (x, y + pipe_gap // 2)

    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right < 0:
            self.kill()

def roulette_selection(population):
    """Fitness-proportionate selection"""
    fitnesses = [max(1.0, getattr(b, 'fitness', 0)) for b in population]
    total_fitness = sum(fitnesses)
    
    if total_fitness == 0:
        return random.choice(population)
        
    pick = random.uniform(0, total_fitness)
    current = 0
    
    for bird, fitness in zip(population, fitnesses):
        current += fitness
        if current >= pick:
            return bird
            
    return population[-1]

def evolve(population, top_n=60, elite_cap=25):
    if not population:
        return [AIBird(100, screen_height // 2) for _ in range(population_size)]
        
    # Sort by fitness
    population.sort(key=lambda b: getattr(b, 'fitness', 0), reverse=True)
    breeders = population[:max(5, top_n)]  # Larger breeding pool
    
    new_birds = []
    
    elite_count = min(elite_cap, len(breeders))
    for i in range(elite_count):
        elite_brain = NeuralNetwork(weights=breeders[i].nn.get_weights())
        if i > 5:
            elite_brain.mutate(rate=0.05, strength=0.1)
        new_birds.append(AIBird(100, screen_height // 2, brain=elite_brain))
        
    # Fill rest with offspring
    while len(new_birds) < len(population):
        if random.random() < 0.7:
            parent1 = roulette_selection(breeders[:20])  # Favor top performers
            parent2 = roulette_selection(breeders[:20])
        else:
            parent1 = random.choice(breeders)  # Sometimes pick from wider pool
            parent2 = random.choice(breeders)
            
        child = parent1.nn.crossover(parent2.nn)
        # Adaptive mutation based on generation diversity
        child.mutate(rate=0.1, strength=0.2)
        new_birds.append(AIBird(100, screen_height // 2, brain=child))
        
    return new_birds
class Button:
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect(topleft=(x, y))
        
    def draw(self, surf):
        surf.blit(self.image, self.rect)
        return self.rect.collidepoint(pygame.mouse.get_pos())

def draw_text(text, font, color, x, y):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))


population_size = 150
generation = 1
max_score_ever = 0
generation_scores = []
best_fitness_ever = 0
birds = [AIBird(100, screen_height // 2) for _ in range(population_size)]
pipe_group = pygame.sprite.Group()
restart_button = Button(screen_width // 2 - 50, screen_height // 2 - 100, restart_img)

running = True

while running:
    clock.tick(fps)
    
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            
    screen.blit(bg, (0, 0))
    time_now = pygame.time.get_ticks()

    if time_now - last_pipe > pipe_frequency:
        pipe_height = random.randint(-150, 150)
        mid_y = screen_height // 2 + pipe_height
        
        top_pipe = Pipe(screen_width, mid_y, position=1)
        btm_pipe = Pipe(screen_width, mid_y, position=-1)
        top_pipe.partner, btm_pipe.partner = btm_pipe, top_pipe
        
        pipe_group.add(top_pipe, btm_pipe)
        last_pipe = time_now
        
    pipe_group.update()
    pipe_group.draw(screen)
    
    screen.blit(ground, (scroll, GROUND_Y))
    scroll -= scroll_speed
    if abs(scroll) > 35:
        scroll = 0
        
    alive_count, current_max_score = 0, 0
    for b in birds:
        if b.alive:
            b.decide(pipe_group.sprites())
            b.update()
            alive_count += 1
            
    # Draw birds
    drawn = 0
    for b in birds:
        if b.alive:
            screen.blit(b.image, b.rect)
            pygame.draw.line(screen, (255, 255, 0), 
                            (b.rect.centerx, b.rect.centery), 
                            (b.next_pipe_x, b.rect.centery), 1)
            vel_end_y = b.rect.centery + int(b.vel * 10)
            pygame.draw.line(screen, (255, 0, 255), 
                            (b.rect.centerx, b.rect.centery), 
                            (b.rect.centerx, vel_end_y), 2)
            drawn += 1
            if drawn >= 10:
                break
                
    # Collision and scoring
    for b in birds:
        if not b.alive:
            continue
            
        for p in pipe_group:
            if p.position == 1 and p.rect.centerx < b.rect.centerx and p.id not in b.passed_pipes:
                b.score += 1
                b.passed_pipes.add(p.id)
                if b.score > max_score_ever:
                    max_score_ever = b.score
                    
            if pygame.sprite.spritecollideany(b, pipe_group):
                b.alive = False
                b.kill()
                
        if b.score > current_max_score:
            current_max_score = b.score
            
    # Generation completion
    if alive_count == 0:
        if birds:
            best_fitness = max(getattr(b, 'fitness', 0) for b in birds)
            best_score = max(b.score for b in birds)
            avg_score = sum(b.score for b in birds) / len(birds)
            avg_fitness = sum(getattr(b, 'fitness', 0) for b in birds) / len(birds)
            generation_scores.append(avg_score)
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness

            trend_info = ""
            if len(generation_scores) >= 5:
                recent_avg = sum(generation_scores[-5:]) / 5
                if len(generation_scores) >= 10:
                    older_avg = sum(generation_scores[-10:-5]) / 5
                    trend = recent_avg - older_avg
                    trend_info = f" │ Δ{trend:+.2f}"
                else:
                    trend_info = f" │ Recent: {recent_avg:.2f}"
                    
            progress_bar = "█" * min(20, best_score) + "░" * max(0, 20 - best_score)
            print(f"Gen {generation:3d} │ Fitness: {best_fitness:8.0f} │ Best: {best_score:2d} │ Avg: {avg_score:5.2f} │ [{progress_bar}]{trend_info}")
            
        generation += 1
        birds = evolve(birds, top_n=60, elite_cap=25)
        pipe_group.empty()
        scroll = 0
        last_pipe = pygame.time.get_ticks() - pipe_frequency
        continue
        
    draw_text(f"Gen: {generation}", font, white, 10, 10)
    draw_text(f"Alive: {alive_count}", font, white, 10, 60)
    draw_text(f"Best: {current_max_score}", font, green, 10, 110)
    draw_text(f"Record: {max_score_ever}", font, red, 10, 160)
    
    pygame.display.update()
    
pygame.quit()

print(f"\n Training completed!")
print(f"Best score achieved: {max_score_ever}")
print(f"Best fitness achieved: {best_fitness_ever:.0f}")

if len(generation_scores) >= 10:
    final_trend = sum(generation_scores[-5:]) / 5 - sum(generation_scores[-10:-5]) / 5
    print(f"Final learning trend: {final_trend:+.2f} pipes/generation")