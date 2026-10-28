"""
Решение среды Gymnasium MountainCarContinuous-v0.

Два способа:
  1) heuristic — ручная «физическая» стратегия: толкать машину по направлению скорости;
  2) cem       — обучение параметров политики методом кросс-энтропии (Cross-Entropy Method).

Запуск:
  python mountain_car_continuous.py train      # обучить CEM, сохранить веса и графики
  python mountain_car_continuous.py evaluate   # оценить обе политики на 100 эпизодах
  python mountain_car_continuous.py render     # показать обученного агента в окне
"""

import json
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENV_ID = "MountainCarContinuous-v0"
OUT_DIR = Path(__file__).parent / "results"
WEIGHTS_FILE = OUT_DIR / "cem_policy.json"
SOLVED_THRESHOLD = 90.0  # среда считается решённой при среднем награде >= 90 за 100 эпизодов


# ---------------------------------------------------------------- политики

def heuristic_policy(obs):
    """Толкаем в сторону текущей скорости — так машина раскачивается и набирает энергию."""
    position, velocity = obs
    if velocity == 0.0:
        return np.array([-1.0])  # на старте скорость нулевая: начинаем с отката влево
    return np.array([1.0 if velocity > 0 else -1.0])


def features(obs):
    """
    Нормируем наблюдение: позиция в [-1.2, 0.6], скорость в [-0.07, 0.07].
    sign(v) нужен, чтобы сила не обнулялась при малой скорости: иначе на дне ямы
    политика вида a ~ v почти не толкает машину и та может не раскачаться за 999 шагов.
    """
    position, velocity = obs
    return np.array([(position + 0.3) / 0.9, velocity / 0.07, np.sign(velocity), 1.0])


def linear_policy(theta, obs):
    """a = tanh(theta · [x_norm, v_norm, sign(v), 1]) — политика с 4 параметрами."""
    return np.array([np.tanh(features(obs) @ theta)])


# ---------------------------------------------------------------- прогон эпизодов

def run_episode(env, policy, seed=None):
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, terminated


def evaluate(policy, n_episodes=100, seed0=10_000):
    env = gym.make(ENV_ID)
    results = [run_episode(env, policy, seed=seed0 + i) for i in range(n_episodes)]
    env.close()
    rewards = np.array([r for r, _, _ in results])
    steps = np.array([s for _, s, _ in results])
    success = np.mean([t for _, _, t in results])
    return rewards, steps, success


# ---------------------------------------------------------------- метод кросс-энтропии

def robust_score(rewards):
    rewards = np.asarray(rewards)
    return rewards.mean() - 0.5 * (rewards.mean() - rewards.min())


def train_cem(n_iters=30, pop_size=40, elite_frac=0.2, episodes_per_candidate=10, seed=0):
    """
    CEM: держим нормальное распределение N(mu, sigma) над параметрами theta.
    Каждую итерацию сэмплируем популяцию, оцениваем, берём лучших (элиту)
    и сдвигаем mu/sigma к элите. Градиенты не нужны.
    """
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)
    dim = 4
    mu = np.zeros(dim)
    sigma = np.full(dim, 2.0)  # большая начальная дисперсия = широкое исследование
    n_elite = int(pop_size * elite_frac)
    history = []

    for it in range(n_iters):
        population = mu + sigma * rng.standard_normal((pop_size, dim))
        # одинаковые сиды для всех кандидатов в итерации — честное сравнение
        seeds = rng.integers(0, 1_000_000, size=episodes_per_candidate)
        # оценка кандидата = средняя награда минус штраф за худший эпизод:
        # так CEM не скатывается к «слишком слабой» силе, которая иногда не доезжает до флага
        scores = np.array([
            robust_score([run_episode(env, lambda o, th=th: linear_policy(th, o), seed=int(s))[0]
                          for s in seeds])
            for th in population
        ])
        elite = population[np.argsort(scores)[-n_elite:]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.05  # небольшой шум не даёт распределению схлопнуться слишком рано

        history.append({"iter": it, "mean": float(scores.mean()),
                        "best": float(scores.max()), "elite_mean": float(np.sort(scores)[-n_elite:].mean())})
        print(f"iter {it:2d} | mean {scores.mean():7.2f} | elite {history[-1]['elite_mean']:7.2f} "
              f"| best {scores.max():7.2f} | mu {np.round(mu, 2)}")

    env.close()
    return mu, history


# ---------------------------------------------------------------- графики

def plot_training(history, path):
    it = [h["iter"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(it, [h["mean"] for h in history], label="среднее по популяции")
    plt.plot(it, [h["elite_mean"] for h in history], label="среднее по элите")
    plt.plot(it, [h["best"] for h in history], label="лучший кандидат")
    plt.axhline(SOLVED_THRESHOLD, color="gray", ls="--", label="порог решения (90)")
    plt.xlabel("итерация CEM")
    plt.ylabel("награда за эпизод")
    plt.title("Обучение CEM на MountainCarContinuous-v0")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_trajectory(theta, path):
    """Фазовый портрет (позиция-скорость) и действия одного эпизода для обеих политик."""
    env = gym.make(ENV_ID)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, pol in [("эвристика", heuristic_policy),
                      ("CEM", lambda o: linear_policy(theta, o))]:
        obs, _ = env.reset(seed=42)
        xs, vs, acts = [obs[0]], [obs[1]], []
        done = False
        while not done:
            a = pol(obs)
            obs, _, term, trunc, _ = env.step(a)
            xs.append(obs[0]); vs.append(obs[1]); acts.append(float(a[0]))
            done = term or trunc
        axes[0].plot(xs, vs, label=f"{name} ({len(acts)} шагов)")
        axes[1].plot(acts, label=name)
    env.close()
    axes[0].axvline(0.45, color="green", ls="--", label="флаг (x=0.45)")
    axes[0].set_xlabel("позиция"); axes[0].set_ylabel("скорость")
    axes[0].set_title("Фазовая траектория"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("шаг"); axes[1].set_ylabel("сила (действие)")
    axes[1].set_title("Действия агента"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ---------------------------------------------------------------- команды

def load_theta():
    return np.array(json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))["theta"])


def cmd_train():
    OUT_DIR.mkdir(exist_ok=True)
    theta, history = train_cem()
    WEIGHTS_FILE.write_text(json.dumps({"theta": theta.tolist(), "history": history},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    plot_training(history, OUT_DIR / "training_curve.png")
    plot_trajectory(theta, OUT_DIR / "trajectory.png")
    print(f"\nВеса сохранены в {WEIGHTS_FILE}")
    cmd_evaluate()


def cmd_evaluate():
    policies = {"heuristic": heuristic_policy}
    if WEIGHTS_FILE.exists():
        theta = load_theta()
        policies["cem"] = lambda o: linear_policy(theta, o)
    print(f"\nОценка на 100 эпизодах (порог решения: средняя награда >= {SOLVED_THRESHOLD}):")
    for name, pol in policies.items():
        rewards, steps, success = evaluate(pol)
        verdict = "РЕШЕНО" if rewards.mean() >= SOLVED_THRESHOLD else "не решено"
        print(f"  {name:9s}: награда {rewards.mean():6.2f} ± {rewards.std():4.2f}, "
              f"шагов {steps.mean():5.1f}, успех {success:.0%} -> {verdict}")


def cmd_render():
    theta = load_theta()
    env = gym.make(ENV_ID, render_mode="human")
    for ep in range(3):
        reward, steps, _ = run_episode(env, lambda o: linear_policy(theta, o), seed=ep)
        print(f"эпизод {ep}: награда {reward:.2f}, шагов {steps}")
    env.close()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": cmd_train, "evaluate": cmd_evaluate, "render": cmd_render}[command]()
