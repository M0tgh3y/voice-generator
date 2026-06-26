# -------------------
# ! READ ME:
# ! oon bakhsh hayi ke ba # ? hastand va rangeshoon abie, 
# ! tavasot hoosh neveshte shodan va man dastkary nakardam.
# ! baghie ye bakhsh ha tavasot man neveshte shodan.
# ! bakhsh haye ba comment sabz tasot man ezafe shodan jahat tozih code ha
# -------------------

import numpy as np # ? chatgpt
import librosa # ? chatgpt
import soundfile as sf # ? chatgpt
import matplotlib.pyplot as plt
import time # ? chatgpt
import os # ? chatgpt
import winsound # ? chatgpt
# az numpy baraye kar kardan ba matrix haye adadi.
# az librosa baraye estekhraj vishegi haye seda va estefade azash be onvan arrey adadi estefade mikonim.
# az soundfile baraye zakhire khorooji soti roye file estefade mishe.
# az matplotlib baraye keshidan nemoodar ha estefade mishe.
# az time baraye andazegiri zaman ejra.
# az os baraye sakht poshe ravand va zakhire toosh estefade mishe.
# az winsound baraye pakhsh seda dar lahze estefade mishe.

start_time = time.perf_counter() # ? chatgpt
# shoroo andaze gereftan

# -------------------
# Sakht array for ravand jaghir
# -------------------
ravand = []
# araye zakhire har 2000 ta

# -------------------
# Sakht poshe braye zakhire ravand ha
# -------------------
folder_name = "ravand" # ? chatgpt
os.makedirs(folder_name, exist_ok=True) # ? chatgpt
# sakht pooshe dar flan adress va check kardan vojood

# -------------------
# Load audio
# -------------------
y, sr = librosa.load("target.wav", sr=22050) # ? chatgpt
y = y / np.max(np.abs(y)) # ? chatgpt
# tabdil seda be adad va normal kardan

# -------------------
# Feature extraction
# -------------------
target_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=70) # ? chatgpt
print(target_mfcc.shape)
# darim ye matrix misazim as vizhegi haye sedamoon, khat dovom abad matrix ra chap mikonad.
# ke baraye voice man (70, 204) hast. yani daram 70 ta vizhegi ro tooye 204 ta fram migiram

# -------------------
# GA settings
# -------------------
POP_SIZE = 20
GENS = 20000
# size jamiat va tedad generation ha.

# -------------------
# Maghadir random entekhsbi daraye mahdoude hastand
# -------------------
mfcc_min = target_mfcc.min(axis=1, keepdims=True) # ? chatgpt
mfcc_max = target_mfcc.max(axis=1, keepdims=True) # ? chatgpt
# az in do khat baraye sakht yek baze baraye coromozom ha estefade mikonim,
# yani miaim va max & min har satr (70 vizhegi) ro peyda mikonim,
# hala dar paiin ke population dare sakhte mishe, az in estefade mikonim ke adad random 
# kheyli part nashan. (idea khodam ke goftam chatgpt codesho bezane.)

# -------------------
# Sakht population avalie
# -------------------
def create_individual(): # ? chatgpt
    return np.random.uniform( # ? chatgpt
        mfcc_min,
        mfcc_max,
        target_mfcc.shape # ? chatgpt
    )

population = [create_individual() for _ in range(POP_SIZE)] # ? chatgpt

# -------------------
# Fitness
# -------------------
target_flat = target_mfcc.ravel() # ? chatgpt
target_norm = np.linalg.norm(target_flat) # ? chatgpt

def fitness(ind): # ? chatgpt

    ind_flat = ind.ravel() # ? chatgpt
    sim = np.dot(ind_flat, target_flat) / ( # ? chatgpt
        np.linalg.norm(ind_flat) * target_norm + 1e-8 # ? chatgpt
    ) # ? chatgpt

    return sim # ? chatgpt

# -------------------
# GA loop
# -------------------
fitness_history = [] # ? chatgpt

for g in range(GENS): # ? chatgpt

    population = sorted(population, key=fitness, reverse=True) # ? chatgpt
    best = population[0] # ? chatgpt
    best_fit = fitness(best) # ? chatgpt
    fitness_history.append(best_fit) # ? chatgpt

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

    while len(new_pop) < POP_SIZE: # ? chatgpt

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

        #child = (p1 + p2) / 2
        #darsadjahesh = 0.99
        #if np.random.rand() < darsadjahesh:
        sigma = max(0.5 * (1 - g/GENS), 0.01)
        child += np.random.normal(0, sigma, child.shape)

        new_pop.append(child) # ? chatgpt

    population = new_pop # ? chatgpt

# -------------------
# Reconstruct (approx)
# -------------------
best = population[0] # ? chatgpt
reconstructed = librosa.feature.inverse.mfcc_to_audio(best) # ? chatgpt
sf.write("output.wav", reconstructed, sr) # ? chatgpt
print("Done → output.wav") # ? chatgpt

# -------------------
# Time ejraye barname
# -------------------
end_time = time.perf_counter() # ? chatgpt
print(f"Execution Time: {end_time - start_time:.4f} seconds") # ? chatgpt

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