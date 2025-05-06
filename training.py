import time
import random
import os
import tensorflow as tf
import numpy as np

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate)


def main():
    tetris_model = build_model()
    memory = Memory()
    for i_episode in range(0, 1000):
        print('Episode ', i_episode)
        plane, piece, altitude, score, lost, lockIn, next_piece = [[]], Tetromino(0), 1, 0, False, False, Tetromino(0)
        memory.clear()
        for i in range(0, 23):
            plane[0].append(1)
        for i in range(1, 12):
            plane.append([])
            plane[i].append(1)
            for j in range(1, 22):
                plane[i].append(0)
            plane[i].append(1)
        plane.append([])
        for i in range(0, 23):
            plane[12].append(1)
        piece = Tetromino(rand(0, 7))

        while not lost:
            next_piece = Tetromino(rand(0, 7))
            prep_input_temp = prepared_input_final(plane, piece, False, next_piece)
            memory.observations.append(prep_input_temp)
            action = tf.random.categorical(tf.expand_dims(tf.squeeze(tetris_model(prep_input_temp)), axis=0), num_samples=1).numpy().flatten()[0]
            memory.actions.append(action)

            column = ((action // 44) % 11) + 1
            rotation = (action // 44) // 11

            reward = 1
            # <handling rotation>
            if rotation == 1:
                piece.grid = ((-piece.grid[0][1], piece.grid[0][0]), (-piece.grid[1][1], piece.grid[1][0]),
                              (-piece.grid[2][1], piece.grid[2][0]))
            elif rotation == 2:
                piece.grid = ((-piece.grid[0][0], -piece.grid[0][1]), (-piece.grid[1][0], -piece.grid[1][1]),
                              (-piece.grid[2][0], -piece.grid[2][1]))
            elif rotation == 3:
                piece.grid = ((piece.grid[0][1], -piece.grid[0][0]), (piece.grid[1][1], -piece.grid[1][0]),
                              (piece.grid[2][1], -piece.grid[2][0]))
            for i in range(0, 3):
                if piece.top + piece.grid[i][1] == -1:
                    piece.top = 3
                    break
                if piece.top + piece.grid[i][1] == 0:
                    piece.top = piece.top + 1
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
            # </handling sideways movement>
            altitude = update_altitude(plane, piece)
            piece.top = piece.top + altitude
            plane[piece.left][piece.top] = 1
            for i in range(0, 3):
                plane[piece.left + piece.grid[i][0]][piece.top + piece.grid[i][1]] = 1

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
                    if plane[j][i] == 1:
                        num_blocks += 1

            holes_new = 0
            for i in range(1, 12):
                hasARoof = False
                for j in range(1, 22):
                    if plane[i][j] == 1:
                        hasARoof = True
                    if plane[i][j] == 0 and hasARoof:
                        holes_new += 1

            if piece.top < 7:  # max 7
                lost = True

            lines = []
            for i in range(1, 22):
                filled = True
                for j in range(1, 12):
                    if plane[j][i] != 1:
                        filled = False
                        break
                if not filled:
                    continue
                lines.append(i)
            for i in lines:
                for j in range(1, 12):
                    for k in reversed(range(1, i + 1)):
                        plane[j][k] = plane[j][k - 1]
                    plane[j][1] = 0

            piece = Tetromino(rand(0, 7))

            if lost:
                reward -= 5
                memory.rewards.append(reward)
                break
            if len(lines) == 1:
                score += 1
                reward += 7
                print('IT MADE IT TO 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            elif len(lines) == 2:
                score += 3
                reward += 21
                print('IT MADE IT TO 2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            elif len(lines) == 3:
                score += 5
                reward += 35
                print('IT MADE IT TO 3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            elif len(lines) == 4:
                score += 8
                reward += 56
                print('IT MADE IT TO 4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(reward)
            memory.rewards.append(reward)
        batch_size = min(len(memory), 300)
        i = np.random.choice(len(memory), batch_size, replace=False)
        train_step(tetris_model, observations=np.array(memory.observations)[i], actions=np.array(memory.actions)[i], discounted_rewards=normalize(discount_rewards(memory.rewards))[i])
        print('rewards: ', np.sum(memory.rewards))
        memory.clear()
        if i_episode % 50 == 0:
            tetris_model.save_weights(checkpoint_prefix)
    tetris_model.save_weights(checkpoint_prefix)


class Tetromino:
    def __init__(self, types):
        self.type = types
        self.left = 6
        self.top = 1
        self.grid = ((0, 0), (0, 0), (0, 0))
# types of blocks: 0 = '0'; 1 = 'T'; 2 = 'L'; 3 = 'J'; 4 = 'Z'; 5 = 'S'; 6 = 'I'
        if self.type == 0:
            self.grid = ((-1, 0), (0, 1), (-1, 1))
        elif self.type == 1:
            self.grid = ((-1, 0), (1, 0), (0, 1))
        elif self.type == 2:
            self.grid = ((-1, 0), (1, 0), (-1, 1))
        elif self.type == 3:
            self.grid = ((-1, 0), (1, 0), (1, 1))
        elif self.type == 4:
            self.grid = ((-1, 0), (0, 1), (1, 1))
        elif self.type == 5:
            self.grid = ((1, 0), (0, 1), (-1, 1))
        elif self.type == 6:
            self.grid = ((-2, 0), (-1, 0), (1, 0))


def update_altitude(plane_local, piece_local):
    altitude_local = 1
    end = False
    while True:
        if plane_local[piece_local.left][piece_local.top + altitude_local] == 1:
            altitude_local = altitude_local - 1
            break
        for i in range(0, 3):
            if plane_local[piece_local.left + piece_local.grid[i][0]][piece_local.top + piece_local.grid[i][1] + altitude_local] == 1:
                end = True
                altitude_local = altitude_local - 1
                break
        if end:
            break
        altitude_local = altitude_local + 1
    return altitude_local


class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.clear()

    def clear(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.rewards)


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
                newPlane[0].append(1)
            for i in range(1, 12):
                newPlane.append([])
                newPlane[i].append(1)
                for j in range(1, 22):
                    if plane_local[i][j] == 0:
                        newPlane[i].append(0)
                    else:
                        newPlane[i].append(1)
                newPlane[i].append(1)
            newPlane.append([])
            for i in range(0, 23):
                newPlane[12].append(1)
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
                if newPlane[newPiece.left][newPiece.top + newAltitude] == 1:
                    newAltitude -= 1
                    break
                for i in range(0, 3):
                    if newPlane[newPiece.left + newPiece.grid[i][0]][newPiece.top + newPiece.grid[i][1] + newAltitude] == 1:
                        end = True
                        newAltitude -= 1
                        break
                if end:
                    break
                newAltitude += 1
            newPiece.top = newPiece.top + newAltitude
            # </altitude>
            newPlane[newPiece.left][newPiece.top] = 1
            for i in range(0, 3):
                newPlane[newPiece.left + newPiece.grid[i][0]][newPiece.top + newPiece.grid[i][1]] = 1
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
                    if newPlane[j][i] == 1:
                        num_blocks += 1
                reward_out += num_blocks
            lines = []
            for i in range(1, 22):
                filled = True
                for j in range(1, 12):
                    if newPlane[j][i] != 1:
                        filled = False
                        break
                if not filled:
                    continue
                lines.append(i)
            for i in lines:
                for j in range(1, 12):
                    for k in reversed(range(1, i + 1)):
                        newPlane[j][k] = newPlane[j][k - 1]
                    newPlane[j][1] = 0
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
                    if newPlane[i][j] == 1:
                        hasARoof = True
                    if newPlane[i][j] == 0 and hasARoof:
                        holes_out += 1
            max_height = 22
            height = 22
            for i in range(1, 22):
                if newPlane[1][i] == 1:
                    height = i
                    break
            for i in range(2, 12):
                for j in range(1, 23):
                    if newPlane[i][j] == 1:
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


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss


def train_step(model, observations, actions, discounted_rewards):
    predictions = [0] * len(observations)
    with tf.GradientTape() as tape:
        for i in range(0, len(observations)):
            predictions[i] = tf.squeeze(model(observations[i]))
        loss = compute_loss(predictions, actions, discounted_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 2)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def normalize(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x.astype(np.float32)


def discount_rewards(rewards, gamma=0.8):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
#    print('rewards {')
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
#        print(R)
#    print('} rewards')
    return discounted_rewards


def rand(a, b):
    random.seed(int(100000000 * (time.time() - int(time.time()))))
    return random.randint(a, b - 1)


if __name__ == "__main__":
    main()
