import time
import random
import os
import tensorflow as tf
import numpy as np
import pygame

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 191, 255)
YELLOW = (255, 255, 0)
LIGHT_GREEN = (181, 230, 29)
BROWN = (139, 69, 19)
VIOLET = (148, 0, 211)
WIDTH, HEIGHT = 800, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 72)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


def main():
    tetris_model = build_model()
    tetris_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    plane, piece, altitude, score, lost, lockIn, next_piece = [[]], Tetromino(0), 1, 0, False, False, Tetromino(0)
    for i in range(0, 23):
        plane[0].append((1, WHITE))
    for i in range(1, 12):
        plane.append([])
        plane[i].append((1, WHITE))
        for j in range(1, 22):
            plane[i].append((0, BLACK))
        plane[i].append((1, WHITE))
    plane.append([])
    for i in range(0, 23):
        plane[12].append((1, WHITE))
    piece = Tetromino(rand(0, 7))

    run = True
    while run:
        plane[piece.left][piece.top] = (1, piece.color)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (1, piece.color)
        update_display(plane, score)
        plane[piece.left][piece.top] = (0, BLACK)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (0, BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        if lost or not run:
            continue
        next_piece = Tetromino(rand(0, 7))
        prep_input_temp = prepared_input_final(plane, piece, False, next_piece)
        action = tf.random.categorical(tf.expand_dims(tf.squeeze(tetris_model(prep_input_temp)), axis=0), num_samples=1).numpy().flatten()[0] // 44

        column = (action % 11) + 1
        rotation = action // 11

        # <handling rotation>
        if rotation == 1:
            piece.grid = ((-piece.grid[0][1], piece.grid[0][0]), (-piece.grid[1][1], piece.grid[1][0]), (-piece.grid[2][1], piece.grid[2][0]))
        elif rotation == 2:
            piece.grid = ((-piece.grid[0][0], -piece.grid[0][1]), (-piece.grid[1][0], -piece.grid[1][1]), (-piece.grid[2][0], -piece.grid[2][1]))
        elif rotation == 3:
            piece.grid = ((piece.grid[0][1], -piece.grid[0][0]), (piece.grid[1][1], -piece.grid[1][0]), (piece.grid[2][1], -piece.grid[2][0]))
        for i in range(0, 3):
            if piece.top + piece.grid[i][1] == -1:
                piece.top = 3
                break
            if piece.top + piece.grid[i][1] == 0:
                piece.top = piece.top + 1

        plane[piece.left][piece.top] = (1, piece.color)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (1, piece.color)
        update_display(plane, score)
        plane[piece.left][piece.top] = (0, BLACK)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (0, BLACK)

        # </handling rotation>
        # <handling sideways movement>
        piece.left = column
        if piece.left + piece.grid[0][0] == -1:
            piece.left = 3
        elif piece.left + piece.grid[0][0] == 13:
            piece.left = 9
        else:
            for i in range(0, 3):
                if piece.left + piece.grid[i][0] == 0 or piece.left + piece.grid[i][0] == 12:
                    piece.left = piece.left - piece.grid[i][0]
                    break

        plane[piece.left][piece.top] = (1, piece.color)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (1, piece.color)
        update_display(plane, score)
        plane[piece.left][piece.top] = (0, BLACK)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (0, BLACK)

        # </handling sideways movement>
        altitude = update_altitude(plane, piece)
        piece.top = piece.top + altitude
        plane[piece.left][piece.top] = (1, piece.color)
        for i in range(0, 3):
            plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = (1, piece.color)

        update_display(plane, score)

        levels = [piece.top]
        for i in range(0, 3):
            there = False
            for j in range(0, len(levels)):
                if piece.top + piece.grid[i][1] == levels[j]:
                    there = True
                    break
            if there:
                continue
            levels.append(piece.top + piece.grid[i][1])
        for i in levels:
            num_blocks = 0
            for j in range(1, 12):
                if plane[j][i][0] == 1:
                    num_blocks += 1

        holes_new = 0
        for i in range(1, 12):
            hasARoof = False
            for j in range(1, 22):
                if plane[i][j][0] == 1:
                    hasARoof = True
                if plane[i][j][0] == 0 and hasARoof:
                    holes_new += 1

        if piece.top < 7:  # max 7
            lost = True

        lines = []
        for i in range(1, 22):
            filled = True
            for j in range(1, 12):
                if plane[j][i][0] != 1:
                    filled = False
                    break
            if not filled:
                continue
            lines.append(i)
        for i in lines:
            for j in range(1, 12):
                for k in reversed(range(1, i + 1)):
                    plane[j][k] = plane[j][k - 1]
                plane[j][1] = (0, BLACK)

        piece = Tetromino(rand(0, 7))

        if len(lines) == 1:
            score += 1
        elif len(lines) == 2:
            score += 3
        elif len(lines) == 3:
            score += 5
        elif len(lines) == 4:
            score += 8
    pygame.quit()


def update_display(plane, score):
    for i in range(0, 13):
        for j in range(0, 23):
            pygame.draw.rect(WIN, plane[i][j][1], pygame.Rect(100 + i * 21, 100 + j * 21, 20, 20))
    # </printing the plane>
    scoreDisplay = font.render("Score: " + str(score), False, WHITE)
    cover = WIN.blit(scoreDisplay, (400, 364))
    pygame.display.update()
    pygame.draw.rect(WIN, BLACK, pygame.Rect(cover.left + 205, cover.top, cover.width, cover.height))


class Tetromino:
    def __init__(self, types):
        self.type = types
        self.left = 6
        self.top = 1
        self.grid = ((0, 0), (0, 0), (0, 0))
        self.color = BLACK
# types of blocks: 0 = '0'; 1 = 'T'; 2 = 'L'; 3 = 'J'; 4 = 'Z'; 5 = 'S'; 6 = 'I'
        if self.type == 0:
            self.grid = ((-1, 0), (0, 1), (-1, 1))
            self.color = BROWN
        elif self.type == 1:
            self.grid = ((-1, 0), (1, 0), (0, 1))
            self.color = VIOLET
        elif self.type == 2:
            self.grid = ((-1, 0), (1, 0), (-1, 1))
            self.color = RED
        elif self.type == 3:
            self.grid = ((-1, 0), (1, 0), (1, 1))
            self.color = YELLOW
        elif self.type == 4:
            self.grid = ((-1, 0), (0, 1), (1, 1))
            self.color = GREEN
        elif self.type == 5:
            self.grid = ((1, 0), (0, 1), (-1, 1))
            self.color = LIGHT_GREEN
        elif self.type == 6:
            self.grid = ((-2, 0), (-1, 0), (1, 0))
            self.color = BLUE


def update_altitude(plane_local, piece_local):
    altitude_local = 1
    end = False
    while True:
        if plane_local[piece_local.left][piece_local.top + altitude_local][0] == 1:
            altitude_local = altitude_local - 1
            break
        for i in range(0, 3):
            if plane_local[piece_local.left + piece_local.grid[i][0]][piece_local.top + piece_local.grid[i][1] + altitude_local][0] == 1:
                end = True
                altitude_local = altitude_local - 1
                break
        if end:
            break
        altitude_local = altitude_local + 1
    return altitude_local


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=16, activation=tf.nn.swish),
        tf.keras.layers.Dense(units=1, activation=None)
    ])
    return model


def prepared_input_final(plane_local, piece_local, inside, next_piece_local=Tetromino(0), score_so_far=(0, 0, 0, 0)):
    if inside:
        prepared_plane = np.zeros((44, 4))
    else:
        prepared_plane = np.zeros((1936, 4))
    for rotation in range(0, 4):
        for column in range(1, 12):
            newPiece = Tetromino(piece_local.type)
            newPlane = [[]]
            for i in range(0, 23):
                newPlane[0].append((1, BLACK))
            for i in range(1, 12):
                newPlane.append([])
                newPlane[i].append((1, BLACK))
                for j in range(1, 22):
                    if plane_local[i][j][0] == 0:
                        newPlane[i].append((0, BLACK))
                    else:
                        newPlane[i].append((1, BLACK))
                newPlane[i].append((1, BLACK))
            newPlane.append([])
            for i in range(0, 23):
                newPlane[12].append((1, BLACK))
            # <handling rotation>
            if rotation == 1:
                newPiece.grid = ((-newPiece.grid[0][1], newPiece.grid[0][0]), (-newPiece.grid[1][1], newPiece.grid[1][0]), (-newPiece.grid[2][1], newPiece.grid[2][0]))
            elif rotation == 2:
                newPiece.grid = ((-newPiece.grid[0][0], -newPiece.grid[0][1]), (-newPiece.grid[1][0], -newPiece.grid[1][1]), (-newPiece.grid[2][0], -newPiece.grid[2][1]))
            elif rotation == 3:
                newPiece.grid = ((newPiece.grid[0][1], -newPiece.grid[0][0]), (newPiece.grid[1][1], -newPiece.grid[1][0]), (newPiece.grid[2][1], -newPiece.grid[2][0]))
            for i in range(0, 3):
                if newPiece.top + newPiece.grid[i][1] == -1:
                    newPiece.top = 3
                    break
                if newPiece.top + newPiece.grid[i][1] == 0:
                    newPiece.top = newPiece.top + 1
            # </handling rotation>
            # <handling sideways movement>
            newPiece.left = column
            if newPiece.left + newPiece.grid[0][0] == -1:
                newPiece.left = 3
            elif newPiece.left + newPiece.grid[0][0] == 13:
                newPiece.left = 9
            else:
                for i in range(0, 3):
                    if newPiece.left + newPiece.grid[i][0] == 0 or newPiece.left + newPiece.grid[i][0] == 12:
                        newPiece.left = newPiece.left - newPiece.grid[i][0]
                        break
            # </handling sideways movement>
            # <altitude>
            newAltitude = 1
            end = False
            while True:
                if newPlane[newPiece.left][newPiece.top + newAltitude][0] == 1:
                    newAltitude -= 1
                    break
                for i in range(0, 3):
                    if newPlane[newPiece.left + newPiece.grid[i][0]][newPiece.top + newPiece.grid[i][1] + newAltitude][0] == 1:
                        end = True
                        newAltitude -= 1
                        break
                if end:
                    break
                newAltitude += 1
            newPiece.top = newPiece.top + newAltitude
            # </altitude>
            newPlane[newPiece.left][newPiece.top] = (1, WHITE)
            for i in range(0, 3):
                newPlane[newPiece.left + newPiece.grid[i][0]][newPiece.top + newPiece.grid[i][1]] = (1, WHITE)
            # <preparing the output>
            reward_out = 0
            holes_out = 0
            bumpiness_out = 0
            levels = [newPiece.top]
            for i in range(0, 3):
                there = False
                for j in levels:
                    if newPiece.top + newPiece.grid[i][1] == j:
                        there = True
                        break
                if there:
                    continue
                levels.append(newPiece.top + newPiece.grid[i][1])
            for i in levels:
                num_blocks = 0
                for j in range(1, 12):
                    if newPlane[j][i][0] == 1:
                        num_blocks += 1
                reward_out += num_blocks
            lines = []
            for i in range(1, 22):
                filled = True
                for j in range(1, 12):
                    if newPlane[j][i][0] != 1:
                        filled = False
                        break
                if not filled:
                    continue
                lines.append(i)
            for i in lines:
                for j in range(1, 12):
                    for k in reversed(range(1, i + 1)):
                        newPlane[j][k] = newPlane[j][k - 1]
                    newPlane[j][1] = (0, BLACK)
            if len(lines) == 1:
                reward_out += 10
            elif len(lines) == 2:
                reward_out += 30
            elif len(lines) == 3:
                reward_out += 50
            elif len(lines) == 4:
                reward_out += 80

            for i in range(1, 12):
                hasARoof = False
                for j in range(1, 22):
                    if newPlane[i][j][0] == 1:
                        hasARoof = True
                    if newPlane[i][j][0] == 0 and hasARoof:
                        holes_out += 1
            max_height = 22
            height = 22
            for i in range(1, 22):
                if newPlane[1][i][0] == 1:
                    height = i
                    break
            for i in range(2, 12):
                for j in range(1, 23):
                    if newPlane[i][j][0] == 1:
                        bumpiness_out += abs(height - j)
                        height = j
                        max_height = min(max_height, j)
                        break
            # </preparing the output>
            # <the output>
            if inside:
                prepared_plane[column - 1 + 11 * rotation][0] = reward_out + score_so_far[0]
                prepared_plane[column - 1 + 11 * rotation][1] = holes_out + score_so_far[1]
                prepared_plane[column - 1 + 11 * rotation][2] = bumpiness_out + score_so_far[2]
                prepared_plane[column - 1 + 11 * rotation][3] = max_height + score_so_far[3]
            else:
                inside_prepared_plane = prepared_input_final(newPlane, next_piece_local, True, score_so_far=(reward_out, holes_out, bumpiness_out, max_height))
                for i in range(0, 44):
                    prepared_plane[(column - 1 + 11 * rotation) * 44 + i] = inside_prepared_plane[i]
            # </the output>
    return prepared_plane


def rand(a, b):
    random.seed(int(100000000 * (time.time() - int(time.time()))))
    return random.randint(a, b - 1)


if __name__ == "__main__":
    main()
