from state import State


class Puzzle:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state

        self.state_history = None
        self.action_history = None

    def reset(self):
        self.state_history = [self.start_state]
        self.action_history = []
        return self._step_result()

    def apply_action(self, a):
        new_state = self.state_history[-1].apply_action(a)
        self.state_history.append(new_state)
        self.action_history.append(a)
        return self._step_result()

    def _goal_reached(self):
        return self.goal_state.is_same(self.state_history[-1])

    def _step_result(self):
        current_state = self.state_history[-1]
        return current_state, current_state.get_actions(), self._goal_reached()


if __name__ == '__main__':
    initial_state = State()
    print('this is the initial state')
    print(initial_state.to_string())
    goal_state = initial_state.apply_action('r')
    print('this is the goal state')
    print(goal_state.to_string())
    puzzle = Puzzle(initial_state, goal_state)
    current_state, valid_actions, is_goal = puzzle.reset()
    print('current state right after reset() method')
    print(current_state.to_string())
    print('valid actions from this state {}, is in goal? {}'.format(valid_actions, is_goal))
    current_state, valid_actions, is_goal = puzzle.apply_action('r')
    print('current state after applying action "r"')
    print(current_state.to_string())
    print('valid actions from this state {}, is in goal? {}'.format(valid_actions, is_goal))
