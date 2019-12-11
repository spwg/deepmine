moves = ["attack", "back", "forward", "jump", "left", "right", "sneak", "sprint"]

def new_action_treechop(choices, action_idx):
    """Given a dict for env.step() and chosen action for MineRLTreechop-v0,
  	returns the dict with the probabilities set."""
    for i in range(len(moves)):
        action = moves[i]
        if action_idx == i:
            choices[action] = 1.
    return choices
