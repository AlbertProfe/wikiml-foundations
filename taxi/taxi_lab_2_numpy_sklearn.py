"""Taxi Lab 2 — a pickup fee, learned with NumPy, checked with scikit-learn.

More receipts, and a base fare: price = w * km + b.
True values: w = 1.50 (€/km), b = 2.00 (the flag fall).
Gradient descent starts at w = 0, b = 0, learning rate 0.01, 2000 steps.

Companion walkthrough:
https://ml.albertprofe.dev/foundations/mlab01-2.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

km = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 12.0])
price = np.array([3.50, 5.00, 6.50, 9.50, 11.00, 14.00, 17.00, 20.00])

print("Receipts:")
for x, y in zip(km, price):
    print(f"  {x:4.0f} km   €{y:.2f}")

w = 0.0
b = 0.0
learning_rate = 0.01
steps = 2000
show_at = {0, 1, 5, 10, 50, 100, 200, 500, 1000, 2000}

history_w = [w]
history_b = [b]
history_loss = []

pred = w * km + b
loss = float(np.mean((pred - price) ** 2))
history_loss.append(loss)

print("\nAfter step |        w |        b |     loss")
print("-----------+----------+----------+---------")
print(f"     start | {w:8.4f} | {b:8.4f} | {loss:7.4f}")

for step in range(1, steps + 1):
    pred = w * km + b
    err = pred - price
    grad_w = float(np.mean(2 * err * km))
    grad_b = float(np.mean(2 * err))
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    pred = w * km + b
    loss = float(np.mean((pred - price) ** 2))
    history_w.append(w)
    history_b.append(b)
    history_loss.append(loss)
    if step in show_at:
        print(f"{step:10d} | {w:8.4f} | {b:8.4f} | {loss:7.4f}")

print(f"\nGradient descent:  w = {w:.6f}   b = {b:.6f}")

model = LinearRegression()
model.fit(km.reshape(-1, 1), price)
sk_w = float(model.coef_[0])
sk_b = float(model.intercept_)
print(f"LinearRegression:  w = {sk_w:.6f}   b = {sk_b:.6f}")
print(f"True values:       w = 1.500000   b = 2.000000")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history_loss, color="#d35400")
axes[0].set_xlabel("step")
axes[0].set_ylabel("mean squared error")
axes[0].set_title("Loss")
axes[0].grid(True, color="#dddddd")

axes[1].plot(history_w, color="#3d3c96", label="w (rate)")
axes[1].plot(history_b, color="#1e8449", label="b (base fare)")
axes[1].axhline(1.5, color="#3d3c96", linestyle="--", linewidth=1)
axes[1].axhline(2.0, color="#1e8449", linestyle="--", linewidth=1)
axes[1].set_xlabel("step")
axes[1].set_title("w and b")
axes[1].legend()
axes[1].grid(True, color="#dddddd")
fig.suptitle("Taxi Lab 2 — gradient descent for price = w · km + b")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
xs = np.linspace(0, 13, 50)
ax.scatter(km, price, color="#d35400", zorder=3, label="receipts")
ax.plot(xs, w * xs + b, color="#3d3c96", label="gradient descent")
ax.plot(
    xs,
    sk_w * xs + sk_b,
    color="#1e8449",
    linestyle="--",
    label="LinearRegression",
)
ax.set_xlabel("kilometres")
ax.set_ylabel("price (€)")
ax.set_title("The fitted line, against the receipts")
ax.legend()
ax.grid(True, color="#dddddd")
fig.tight_layout()
plt.show()
