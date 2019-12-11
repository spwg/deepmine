import sys
import matplotlib.pyplot as plt


def main():
    with open("rewards" + str(int(sys.argv[1])) + ".txt") as f:
        lines = f.readlines()
        episodes = [i for i in range(len(lines)) if "Finished in" in lines[i]]
        rewards = [float(lines[ep].split()[-1]) for ep in episodes]
        plt.plot(rewards)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("DQN on MineRLNavigateDense-v0")
        plt.show()


if __name__ == '__main__':
    main()
