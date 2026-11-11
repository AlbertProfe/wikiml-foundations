"""Taxi Lab 1 — learn the fare per kilometre in pure Python.

Receipts: 2 km for €3, 5 km for €7.50, 10 km for €15.
Model: price = w * km. The true rate is 1.50. Learning rate 0.01.

No libraries for the loop — only matplotlib for the loss curve.

Companion walkthrough:
https://ml.albertprofe.dev/foundations/mlab01-1.html
"""

import matplotlib.pyplot as plt

km = [2, 5, 10]
price = [3.0, 7.50, 15.0]

w = 0.0
learning_rate = 0.01
steps = 20

history_w = [w]
history_loss = []


def mean_squared_error(weight):
    predictions = [weight * x for x in km]
    errors = [p - y for p, y in zip(predictions, price)]
    return sum(e * e for e in errors) / len(km), predictions


loss, predictions = mean_squared_error(w)
history_loss.append(loss)

print("After step |        w | predictions              |     loss")
print("-----------+----------+--------------------------+---------")
print(
    f"     start | {w:8.4f} | "
    + ", ".join(f"€{p:.2f}" for p in predictions)
    + f" | {loss:7.4f}"
)

for step in range(1, steps + 1):
    predictions = [w * x for x in km]
    errors = [p - y for p, y in zip(predictions, price)]
    gradient = sum(2 * e * x for e, x in zip(errors, km)) / len(km)
    w = w - learning_rate * gradient
    loss, predictions = mean_squared_error(w)
    history_w.append(w)
    history_loss.append(loss)
    if step <= 6 or step == steps:
        pred_str = ", ".join(f"€{p:.2f}" for p in predictions)
        print(f"{step:10d} | {w:8.4f} | {pred_str} | {loss:7.4f}")

print(f"\nFinal w = {w:.4f}  (true rate is 1.50)")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(len(history_loss)), history_loss, color="#d35400", marker="o", markersize=4)
ax.set_xlabel("step")
ax.set_ylabel("mean squared error")
ax.set_title("Taxi Lab 1 — loss falls as w approaches €1.50 / km")
ax.set_xlim(0, steps)
ax.grid(True, color="#dddddd")
fig.tight_layout()
plt.show()
