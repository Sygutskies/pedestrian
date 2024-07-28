import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('logs/06_12_2024_20_48_38/phone/training.csv', header=0)

print(df["Accuracy/val"].max(), df["Accuracy/val"].idxmax())
print(df["Accuracy/train"].max(), df["Accuracy/train"].idxmax())
print(df["Loss/val"].min(), df["Loss/val"].idxmin())
print(df["Loss/train"].min(), df["Loss/train"].idxmin())

plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 101)),df["Accuracy/val"])
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy/val', fontsize=20)
plt.legend()
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('accuracy_val.png')