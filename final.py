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
# shoroo andaze gereftan zaman

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
# dar in bakhsh ba estefade az min va max har satr miaym va jamiat avalie misazim.
# betori ke dar har satr random bein min & max bashe, ta javab ha kheyli behtar beshan.
# va bad array population ra misazim ke toosh 20 ta az in array haye (70 * 204) hast.

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
# ! tozih in ghesmat be soorat kamel dar file hast. dar inja baraye kholase goftan:
# matrix hara be array yek bodi tabdil mikonim. 
# hala ba estefade az formool zarb dakheli miaim va cos zavie beineshoono bedast miarim.
# harche ke cos be 1 nazdik tar bashe yani fasele do bordar sefre va rooye ham dige hastan.
# fitnesesham mishe hamoon cos zavie beineshoon.

# -------------------
# GA loop
# -------------------
fitness_history = []
# baraye inke akhare kar rooye nemoodar neshoon bedim

for g in range(GENS): # ? chatgpt

    population = sorted(population, key=fitness, reverse=True) # ? chatgpt
    best = population[0] # ? chatgpt
    best_fit = fitness(best) # ? chatgpt
    fitness_history.append(best_fit) # ? chatgpt
    # pop ra moratab mikonim ta behtarin pop biad aval array

    if g % 1000 == 0 or g == 19999:
        print(f"Gen {g} | fitness: {best_fit}")
        # har 1000 bar fitness behtarin ta oon lahze ro chap mikone
        
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
    # har 2000 bar ye file soti misaze va oono pakhsh mikone
        
    new_pop = population[:5]
    # 5 taye aval ro bedoone taghir mizare toye pop badi

    while len(new_pop) < POP_SIZE: # ? chatgpt
        # baraye 15 taye badi miad va bache haro hesab mikone

        i1 = np.random.randint(0, 5)
        i2 = np.random.randint(0, 5)
        # as 5 taye aval random 2 ta parents bar midare

        while i1 == i2:
            i2 = np.random.randint(5, 20)
        # inja baraye inke yekam bekham jahesh bedam, age pop ha mese ham boodan biad , p2 ro az 15
        # taye badi bardare. (kheyli komak konande nabood.)

        p1 = population[i1]
        p2 = population[i2]

        if (i1 > i2):
            child = 0.5 * p2 + 0.5 * p1
        else:
            child = 0.5 * p1 + 0.5 * p2

        #child = (p1 + p2) / 2

        # oon code bala darvaghe hamoon paiinie, vali mikhastam tasir bishtar kardan az valed behtaro bebinam
        # (inam kheyli kar saz nabood. choon dashtim ba adad kar mikardim na masalan ba shekl matrix.)
        # (manzoor az shekl matrix, block block kardane. (test kardam khoob nashod.))

        #darsadjahesh = 0.99
        #if np.random.rand() < darsadjahesh:
        sigma = max(0.5 * (1 - g/GENS), 0.01)
        child += np.random.normal(0, sigma, child.shape)
        # nokte i ke fahmidam in bood ke age hame bache haro hamishe jahesh bedim zoodtar be javab behtar 
        # miresim. montaha jajhat inke dar nasl haye nazdik be javab, jahesh kam bashe sigma ro tarif kardim.
        # dar avayel sigma bozorge va jahesh ham bozorge, dar avakher jahesh kame va sigma ham koochike.

        new_pop.append(child) # ? chatgpt
        # ezafe kardan bache be pop jadidemoon

    population = new_pop # ? chatgpt
    # jagozari pop jadid bejaye ghabli

# -------------------
# Sakht voice behtarin
# -------------------
best = population[0] # ? chatgpt
voiceAkhar = librosa.feature.inverse.mfcc_to_audio(best) # ? chatgpt
sf.write("output.wav", voiceAkhar, sr) # ? chatgpt
print("Done → output.wav") # ? chatgpt
# inja darim bad az 20000 bar behtarin bache i ke peyda kardim ro file soti sho misazim.

# -------------------
# Time ejraye barname
# -------------------
end_time = time.perf_counter() # ? chatgpt
print(f"Execution Time: {end_time - start_time:.4f} seconds") # ? chatgpt
# inja mishe payan barname va zaman ro baramoon chap mikone.

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
# rasm nemoodar bad az payan barname.