import numpy as np
import models_parallel, random, torch
import matplotlib.pyplot as plt
from multiprocessing import Process
from tqdm import tqdm
torch.use_deterministic_algorithms(True)

# Função que executa o treinamento para um valor específico de seed
def run_parallel(seed, inputs, y_true2,number0flayer):
    #print(f"Running with seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    quantump = models_parallel.Train(D=30, epochs=10000, step_size_=500, lr=0.01, number_of_layers=number0flayer)
    quantump.prepare_data(x_=inputs, y_=y_true2)
    quantump.train(
        save=True,
        checkpoint_path=f"examples/regression/dataposterfloripa/quantum/{number0flayer}layer/model_sin{seed}.pth")

def run_parallel2(seed, inputs, y_true2,number0flayer):
    #print(f"Running with seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    quantump = models_parallel.Train(D=30, epochs=10000, step_size_=500, lr=0.01, number_of_layers=number0flayer)
    quantump.prepare_data(x_=inputs, y_=y_true2)
    quantump.train(
        save=True,
        checkpoint_path=f"examples/regression/dataposterfloripa/quantum/{number0flayer}layer/model_Heaviside{seed}.pth")

if __name__ == "__main__":
    
    inputs = np.linspace(-1, 1, 20)
    y_true2 = np.heaviside(inputs,0)
    y_true = np.sin(np.pi * inputs)
    seeds_font = [list(range(i, i+4)) for i in range(12, 50, 4)]
    for N_layer in range(2,6):
        print(f"N_layer:{N_layer}")
        for seeds in tqdm(seeds_font):
            print(f"seeds:{seeds}")
            processes = []
            for seed in seeds:
            
                # Cria um processo para cada seed
                p = Process(target=run_parallel, args=(seed, inputs, y_true,N_layer))#sino
                p2 = Process(target=run_parallel2, args=(seed, inputs, y_true2,N_layer))#Heavi
                processes.append(p)
                processes.append(p2)
                p.start()
                p2.start()
                
            # Aguarda todos os processos terminarem
            for p in processes:
                p.join() 
    