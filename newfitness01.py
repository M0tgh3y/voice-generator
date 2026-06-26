import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import time
import os
import winsound

start_time = time.perf_counter()

# sakht array for ravand jaghir
ravand = []

# sakht poshe braye zakhire ravand ha
folder_name = "ravand"
os.makedirs(folder_name, exist_ok=True)

# -------------------
# Load audio
# -------------------
y, sr = librosa.load("target.wav", sr=22050)
y = y / np.max(np.abs(y))

# -------------------
# Feature extraction
# -------------------
target_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=65)
# print(target_mfcc.shape)
# dar inja darim 20 ta az vizhegi haye voice asli ra kharej mikonim va dr yek shekl matrisi zakhire mikonim

# -------------------
# GA settings
# -------------------
POP_SIZE = 20
GENS = 20000

mfcc_min = target_mfcc.min(axis=1, keepdims=True)
# print("mfcc min ra chap mikone")
# print(mfcc_min.shape)
mfcc_max = target_mfcc.max(axis=1, keepdims=True)
# print("mfcc max ra chap mikone")
# print(mfcc_max.shape)

def create_individual():
    return np.random.uniform(
        mfcc_min,
        mfcc_max,
        target_mfcc.shape
    )

population = [create_individual() for _ in range(POP_SIZE)]

# -------------------
# Fitness
# -------------------
target_flat = target_mfcc.ravel()
target_norm = np.linalg.norm(target_flat)

def fitness(ind):

    ind_flat = ind.ravel()

    sim = np.dot(ind_flat, target_flat) / (
        np.linalg.norm(ind_flat) * target_norm + 1e-8
    )

    return sim

# -------------------
# GA loop
# -------------------
fitness_history = []

for g in range(GENS):

    population = sorted(population, key=fitness, reverse=True)

    best = population[0]

    best_fit = fitness(best)

    fitness_history.append(best_fit)

    if g % 1000 == 0 or g == 19999:
        print(f"Gen {g} | fitness: {best_fit}")
        
    if g % 2000 == 0 or g == 19999:
        ravand.append(best)
        filename = f"ravand/gen_{g}.wav"
        audio = librosa.feature.inverse.mfcc_to_audio(
            best,
            sr=sr
        )

        sf.write(filename, audio, sr)

        winsound.PlaySound(
            filename,
            winsound.SND_FILENAME |
            winsound.SND_ASYNC
        )

    new_pop = population[:5]

    while len(new_pop) < POP_SIZE:

        i1 = np.random.randint(0, 5)
        i2 = np.random.randint(0, 5)

        while i1 == i2:
            i2 = np.random.randint(5, 20)

        p1 = population[i1]
        p2 = population[i2]

        if (i1 > i2):
            child = 0.5 * p2 + 0.5 * p1
        else:
            child = 0.5 * p1 + 0.5 * p2

        # child = (p1 + p2) / 2

        # darsadjahesh = 0.99

        #if np.random.rand() < darsadjahesh:
        sigma = max(0.5 * (1 - g/GENS), 0.01)
        child += np.random.normal(0, sigma, child.shape)

        new_pop.append(child)

    population = new_pop

# -------------------
# Reconstruct (approx)
# -------------------
best = population[0]

reconstructed = librosa.feature.inverse.mfcc_to_audio(best)

sf.write("output.wav", reconstructed, sr)

print("Done → output.wav")

# -------------------
# Time ejraye barname
# -------------------

end_time = time.perf_counter()

print(f"Execution Time: {end_time - start_time:.4f} seconds")

# -------------------
# Table for fitness
# -------------------

plt.figure(figsize=(10,5))

plt.plot(fitness_history)

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Best Fitness Per Generation")

plt.grid(True)

plt.show()