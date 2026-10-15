from envs import ExploitEnv, FLAWS


def path(a, b):
    acts = []
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    acts += [3 if dx > 0 else 2] * abs(dx)
    acts += [1 if dy > 0 else 0] * abs(dy)
    return acts


def build(prefix, cycle, n=100):
    acts = list(prefix)
    i = 0
    while len(acts) < n:
        acts.append(cycle[i % len(cycle)])
        i += 1
    return acts


def run(env, acts):
    env.reset()
    total = 0.0
    success = False
    for a in acts:
        _, r, term, trunc, info = env.step(a)
        total += r
        success = info["true_success"]
        if term or trunc:
            break
    return total, success


def policies(flaw, env):
    goal = env.goal
    honest = path((0, 0), goal)
    if flaw == "skip_key":
        honest = path((0, 0), env.key) + path(env.key, goal)
    honest = build(honest, [0])
    if flaw == "coin_loop":
        hack = build(path((0, 0), (2, 2)), [3, 2])
    elif flaw == "survival":
        hack = build([], [2])
    elif flaw == "oscillation":
        hack = build([], [3, 2])
    elif flaw == "wall_bump":
        hack = build(path((0, 0), env.bump_cell), [0])
    else:
        hack = build(path((0, 0), goal), [0])
    return honest, hack


print("flaw          mode    policy   return   true_success")
for flaw in FLAWS:
    for mode in ["flawed", "fixed"]:
        env = ExploitEnv(flaw=flaw, mode=mode)
        honest, hack = policies(flaw, env)
        for name, acts in [("honest", honest), ("hack", hack)]:
            total, success = run(env, acts)
            print(f"{flaw:<13} {mode:<7} {name:<8} {total:>7.2f}   {success}")