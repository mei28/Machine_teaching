import os
import sys


def main():
    args = sys.argv

    if len(args) == 2:
        base_path = args[1]
    else:
        print('args error')
        return
    os.chdir('result')
    os.mkdir(base_path)
    os.chdir(base_path)

    os.makedirs('Omniscient')
    os.makedirs('Surrogate')
    os.makedirs('Random')
    os.makedirs('Without_teacher/majority')
    # os.makedirs('Without_teacher/mix')
    # os.makedirs('Without_teacher/prob')
    os.makedirs('Without_teacher/y')
    os.makedirs('Without_teacher/w_star')


if __name__ == "__main__":
    main()
