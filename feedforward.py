from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Literal, Callable, Dict

# --------- Types ---------
Vector = List[float]
Matrix = List[Vector]
Sample = Tuple[float, float, float]     # (x1, x2, y)  -- y used only for display in this feedforward demo
Dataset = List[Sample]
ActivationName = Literal["sigmoid", "relu", "identity"]

# --------- Activations ---------
def sigmoid(z: float) -> float:
    if z >= 0.0:
        ez: float = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def relu(z: float) -> float:
    return z if z > 0.0 else 0.0

def identity(z: float) -> float:
    return z

def get_activation(name: ActivationName) -> Callable[[float], float]:
    if name == "sigmoid":
        return sigmoid
    if name == "relu":
        return relu
    return identity

# --------- Math helpers ---------
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x*y for x, y in zip(a, b)))

# --------- Model params ---------
@dataclass
class LayerParams:
    W: Matrix   # shape: (n_units, n_prev)
    b: Vector   # shape: (n_units,)
    act: ActivationName

@dataclass
class MLPParams:
    # Here we build a 2–2–1 network for clarity
    hidden: LayerParams    # 2 units, act = "sigmoid" or "relu"
    output: LayerParams    # 1 unit, act = "identity"

# --------- Initialization ---------
def init_layer(n_units: int, n_prev: int, act: ActivationName, scale: float = 0.5, seed: Optional[int] = None) -> LayerParams:
    if seed is not None:
        random.seed(seed)
    W: Matrix = [[random.uniform(-scale, scale) for _ in range(n_prev)] for _ in range(n_units)]
    b: Vector = [0.0 for _ in range(n_units)]
    return LayerParams(W=W, b=b, act=act)

def init_mlp_2_2_1(hidden_act: ActivationName = "sigmoid", seed: Optional[int] = 0) -> MLPParams:
    # input size = 2
    hidden = init_layer(n_units=2, n_prev=2, act=hidden_act, scale=0.5, seed=seed)
    # output layer: 1 unit, identity activation (for value prediction)
    output = init_layer(n_units=1, n_prev=2, act="identity", scale=0.5, seed=(seed + 1) if seed is not None else None)
    return MLPParams(hidden=hidden, output=output)

# --------- Feedforward ---------
@dataclass
class ForwardCache:
    # Store all intermediates for teaching/printing
    x: Vector                    # input [x1, x2]
    z_hidden: Vector            # length 2
    a_hidden: Vector            # length 2
    z_out: Vector               # length 1
    a_out: Vector               # length 1 (the prediction)

def layer_forward(x: Vector, layer: LayerParams) -> Tuple[Vector, Vector]:
    act_fn: Callable[[float], float] = get_activation(layer.act)
    z_list: Vector = [dot(w_row, x) + b for w_row, b in zip(layer.W, layer.b)]
    a_list: Vector = [act_fn(z) for z in z_list]
    return z_list, a_list

def feedforward(x: Vector, params: MLPParams) -> Tuple[float, ForwardCache]:
    # Hidden layer
    z_hid, a_hid = layer_forward(x, params.hidden)
    # Output layer (identity activation for regression-like output)
    z_out, a_out = layer_forward(a_hid, params.output)
    # a_out is length 1
    cache = ForwardCache(x=x, z_hidden=z_hid, a_hidden=a_hid, z_out=z_out, a_out=a_out)
    return a_out[0], cache

# --------- Pretty trace for teaching ---------
def print_forward_trace(cache: ForwardCache) -> None:
    x1, x2 = cache.x
    print("---- FEEDFORWARD TRACE ----")
    print(f"Input: x = [{x1:.4f}, {x2:.4f}]")

    # Hidden layer details
    print("\nHidden layer (2 units):")
    for j, (zj, aj) in enumerate(zip(cache.z_hidden, cache.a_hidden)):
        print(f"  z_hidden[{j}] = {zj:.6f}  ->  a_hidden[{j}] = {aj:.6f}")

    # Output layer details
    print("\nOutput layer (1 unit, identity):")
    print(f"  z_out[0] = {cache.z_out[0]:.6f}  ->  a_out[0] = {cache.a_out[0]:.6f}")
    print("---------------------------\n")

# --------- Demo dataset helpers (addition) ---------
def addition_samples(min_v: int, max_v: int) -> Dataset:
    data: Dataset = []
    for x1 in range(min_v, max_v + 1):
        for x2 in range(min_v, max_v + 1):
            y: float = float(x1 + x2)
            data.append((float(x1), float(x2), y))
    return data

# --------- Script: feedforward-only demo ---------
if __name__ == "__main__":
    # 1) Init a tiny 2–2–1 MLP (choose "sigmoid" or "relu" for hidden)
    params: MLPParams = init_mlp_2_2_1(hidden_act="sigmoid", seed=1)

    # 2) Build a small addition set to test feedforward behavior
    data_small: Dataset = [(0.0, 0.0, 0.0),
                           (0.0, 1.0, 1.0),
                           (1.0, 0.0, 1.0),
                           (1.0, 1.0, 2.0)]

    # 3) Just FEEDFORWARD (no learning). Show predictions and a detailed trace for one sample.
    print("=== FEEDFORWARD ONLY: predictions (no weight updates) ===")
    for (x1, x2, y) in data_small:
        y_hat, cache = feedforward([x1, x2], params)
        print(f"x=({x1:.0f},{x2:.0f}) | target y={y:.1f} | prediction y_hat={y_hat:.6f}")

    # 4) Trace a single example to show all intermediate values (great for paper walkthrough)
    _, cache_one = feedforward([1.0, 1.0], params)
    print_forward_trace(cache_one)
