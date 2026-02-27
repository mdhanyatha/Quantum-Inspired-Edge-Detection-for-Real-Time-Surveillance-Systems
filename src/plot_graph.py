import json
import numpy as np
import matplotlib.pyplot as plt

labels = ["Quantum", "Sobel", "Canny"]

# ================= LOAD FPS =================
with open("fps.json") as f:
    fps = json.load(f)

fps_vals = [
    np.mean(fps["1"]) if fps["1"] else 0,
    np.mean(fps["2"]) if fps["2"] else 0,
    np.mean(fps["3"]) if fps["3"] else 0
]

# ================= LOAD PSNR =================
with open("psnr.json") as f:
    ps = json.load(f)

psnr_vals = [
    np.mean(ps["1"]) if ps["1"] else 0,
    np.mean(ps["2"]) if ps["2"] else 0,
    np.mean(ps["3"]) if ps["3"] else 0
]

# ================= FPS GRAPH =================
plt.figure()
plt.bar(labels, fps_vals)
plt.title("FPS Comparison")
plt.ylabel("FPS")
plt.xlabel("Method")
plt.grid(True)
plt.savefig("fps_graph.png")

# ================= PSNR GRAPH =================
plt.figure()
plt.bar(labels, psnr_vals)
plt.title("PSNR Comparison")
plt.ylabel("PSNR (dB)")
plt.xlabel("Method")
plt.grid(True)
plt.savefig("psnr_graph.png")

# show graphs
plt.show()

# ================= PRINT VALUES =================
print("\nAverage FPS:")
for label, val in zip(labels, fps_vals):
    print(f"{label}: {val:.2f}")

print("\nAverage PSNR:")
for label, val in zip(labels, psnr_vals):
    print(f"{label}: {val:.2f}")
