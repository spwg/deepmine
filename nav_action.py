import numpy as np


CAM_AMT = 20

def interpret_probs_one(probs, env):
    moves = ["attack", "back", "forward", "jump", "left", "right", 'cam_left', 'cam_right', 'cam_up', 'cam_down']
    print('probs: ', probs)
    to_return = env.action_space.noop()
    act_idx = np.random.choice(range(len(probs)), p=probs.numpy())
    move = moves[act_idx]
    print('move: ', move)
    if move == 'cam_left':
        to_return['camera'] = [0, -CAM_AMT]
    elif move == 'cam_right':
        to_return['camera'] = [0, CAM_AMT]
    elif move == 'cam_down':
        to_return['camera'] = [-CAM_AMT, 0]
    elif move == 'cam_up':
        to_return['camera'] = [CAM_AMT, 0]
    else:
        to_return[move] = 1
    return to_return, act_idx