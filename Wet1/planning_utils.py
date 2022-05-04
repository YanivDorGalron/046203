def traverse(goal_state, prev):
    """
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    """
    # state_t = prev[state_t_plus_1.to_string()]
    prev_state = prev[goal_state.to_string()]

    result = [(goal_state, None)]  # The action is clear when state_t and state_t_plus_1 are matched
    current = result[-1][0]
    while prev_state:
        for action in prev_state.get_actions():
            if prev_state.apply_action(action) == current:
                result.append((prev_state, action))
                break
        current = prev_state.copy()
        prev_state = prev[current.to_string()]

    result.reverse()
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan) - 1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
