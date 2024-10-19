import matplotlib.pyplot as plt
import numpy as np
import os
import math
def plot3(avg_forb_norm,avg_accuracy, avg_spec_norm,avg_time, std_forb_norm, std_accuracy, std_spec_norm,std_time, AS, dataset, SS):
    num_algorithms = len(avg_forb_norm)
    num_levels = len(avg_forb_norm[0])

    # Create subplots for each metric
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

    # List of data to plot and their respective standard deviations
    data_to_plot = [avg_forb_norm, avg_accuracy, avg_spec_norm, avg_time]
    std_to_plot = [std_forb_norm, std_accuracy, std_spec_norm, std_time]
    #titles = ['Frobenius Norm', 'Accuracy', 'Spectral Norm', 'Time']
    titles = ['Frobenius Norm', 'Accuracy', 'Spectral Norm', 'Isomorphic']
    # Loop through each metric and plot its data
    for ax, data, std, title in zip(axs.flat, data_to_plot, std_to_plot, titles):
        # Set the positions of the bars on the x-axis
        bar_positions = range(num_levels)

        # Width of each bar
        bar_width = 0.1

        # Plot each algorithm's data with error bars
        for i in range(num_algorithms):
            # Offset the position of bars for each algorithm
            offset = bar_width * (i - (num_algorithms - 1) / 2)
            ax.bar([pos + offset for pos in bar_positions], data[i], bar_width,
                yerr=std[i], label=AS[i], capsize=5)

        # Add labels and title
        ax.set_xlabel('Subgraph Size ')
        ax.set_ylabel(f' {title}',fontsize=12)
        ax.set_title(f'Average {title} with standard deviation')

        # Add x-axis ticks
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([f'{SS[j]} %' for j in range(num_levels)])
        
        # Add legend
        ax.legend()
    plt.suptitle(f'{dataset} metrics')
# Adjust layout to prevent overlap
    plt.tight_layout()
    if not os.path.exists('./pltsAuLM'): os.makedirs('./pltsAuLM')
    plt.savefig(f'./pltsAuLM/a{dataset}.png', bbox_inches='tight')
    # Show the plot
    plt.show()
    # Adjust layout to prevent overlap
    
    # Show the plot

def plot2(avg_forb_norm,avg_accuracy, avg_spec_norm,avg_time, std_forb_norm, std_accuracy, std_spec_norm,std_time, AS, dataset, SS):
    num_algorithms = len(avg_forb_norm)
    num_levels = len(avg_forb_norm[0])

    # Create subplots for each metric
    #fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # List of data to plot and their respective standard deviations
    #data_to_plot = [avg_forb_norm, avg_accuracy, avg_spec_norm, avg_time]
    #std_to_plot = [std_forb_norm, std_accuracy, std_spec_norm, std_time]
    #titles = ['Frobenius Norm', 'Accuracy', 'Spectral Norm', 'Time']
    data_to_plot = [ avg_accuracy, avg_spec_norm]
    std_to_plot = [ std_accuracy, std_spec_norm]
    titles = ['Accuracy', 'Spectral Norm']
    # Loop through each metric and plot its data
    for ax, data, std, title in zip(axs.flat, data_to_plot, std_to_plot, titles):
        # Set the positions of the bars on the x-axis
        bar_positions = range(num_levels)

        # Width of each bar
        bar_width = 0.1

        # Plot each algorithm's data with error bars
        for i in range(num_algorithms):
            # Offset the position of bars for each algorithm
            offset = bar_width * (i - (num_algorithms - 1) / 2)
            ax.bar([pos + offset for pos in bar_positions], data[i], bar_width,
                yerr=std[i], label=AS[i], capsize=5)

        # Add labels and title
        ax.set_xlabel('Subgraph Size ')
        ax.set_ylabel(f' {title}',fontsize=12)
        ax.set_title(f'Average {title} with standard deviation')

        # Add x-axis ticks
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([f'{SS[j]} %' for j in range(num_levels)])
        
        # Add legend
        ax.legend()
    print(dataset)
    plt.suptitle(f'{dataset} metrics')
# Adjust layout to prevent overlap
    plt.tight_layout()
    print("plot")
    if not os.path.exists('./plotsSSL1'): os.makedirs('./plotsSSL1')
    print(os.path.exists('./plotsSSL1'))
    plt.savefig(f'./plotsSSL1/{dataset}.png', bbox_inches='tight')
    # Show the plot
    plt.show()
    # Adjust layout to prevent overlap
    
    # Show the plot

# Read the file and store the values in lists
forb_norm = []
accuracy = []
spec_norm = []
iso_=[]
time = []
isocounter=0
SS=[10,20,30,40,50,60,70,80,90]
#SS=[50]
#SS=[80,90]
#SS=[30]
nL=["_Noise5","_Noise10","_Noise15","_Noise20","_Noise25"]
nL1=["5","10","15","20","25"]
#DS=["arenas"]
DS1=["arenas","netscience","multimanga","highschool","voles"]
DS1=["highschool"]

DS=["_94/highschool"]
AS=["Alpine","Alpine_Dummy","Cone","Grampa","Regal","SGWL"]
AS=["Grad","mcmc"]
avg_forb_norm = [[] for _ in range(len(AS))]
avg_accuracy = [[] for _ in range(len(AS))]
avg_spec_norm = [[] for _ in range(len(AS))]
avg_time = avg_time = [[] for _ in range(len(AS))]
avg_iso = [[] for _ in range(len(AS))]

std_forb_norm = [[] for _ in range(len(AS))]
std_accuracy = [[] for _ in range(len(AS))]
std_spec_norm = [[] for _ in range(len(AS))]
std_time = [[] for _ in range(len(AS))]
std_iso = [[] for _ in range(len(AS))]
folderall_ = 'data3_/res'
u=0
for j in range(len(DS)):    
    for i in nL1:#SS before
        u=0
        for t in AS:
            with open(f'{folderall_}/_438/highschool{i}/50/NoiseTest_results{t}.txt', 'r') as file:
                tolerance = 1e-10
                next(file)
                for line in file:
                    # Split the line into values
                    values = line.split()
                    # Convert values to appropriate data types and store them
                    forb_norm.append(float(values[6]))
                    accuracy.append(float(values[7]))
                    spec_norm.append(float(values[8]))
                    time.append(float(values[9]))
                    if (float(values[6]) <tolerance):
                        iso_.append(1)
                        isocounter=isocounter+1
                    else:
                        iso_.append(0)
            # Calculate averages
            avg_forb_norm[u].append(np.mean(forb_norm))
            avg_accuracy[u].append(np.mean(accuracy))
            avg_spec_norm[u].append(np.mean(spec_norm)) 
            avg_time[u].append(np.mean(time)) 
            #print(sum(iso_))
            #avg_iso[u].append(np.mean(iso_)*100)
            avg_iso[u].append(isocounter)
            # Calculate standard deviations
            std_forb_norm[u].append(np.std(forb_norm))
            #print(np.mean(forb_norm))
            #print(np.std(np.array(forb_norm)))
            std_accuracy[u].append(np.std(accuracy)) 
            std_spec_norm[u].append(np.std(spec_norm)) 
            std_time[u].append(np.std(time)) 
            std_iso[u].append(np.std(iso_))
            u+=1
            forb_norm.clear()
            accuracy.clear()
            spec_norm.clear()
            time.clear()
            iso_.clear()
            isocounter=0
    #plot(avg_forb_norm,std_forb_norm,j,SS,AS)
    #plot3(avg_forb_norm,avg_accuracy,avg_spec_norm,
    #      avg_iso,std_forb_norm,std_accuracy,std_spec_norm,std_iso,AS,DS1[j],SS)
    print("Dataset: ",DS1[j] )
    #for aa in range(u):
        #print("Algorithm: ",AS[aa])
    for t in range(len(nL1)): #ss
           #print(SS[t],avg_forb_norm[aa][t],std_forb_norm[aa][t],avg_accuracy[aa][t],std_accuracy[aa][t],avg_spec_norm[aa][t],std_spec_norm[aa][t],avg_time[aa][t],std_time[aa][t])
        #print(SS[t],avg_forb_norm[0][t],avg_forb_norm[1][t],avg_forb_norm[2][t],avg_forb_norm[3][t],avg_forb_norm[4][t],avg_forb_norm[5][t])
        print(SS[t],avg_forb_norm[0][t],avg_forb_norm[1][t])     
        print()
    for t in range(len(nL1)): #ss
        #print(SS[t],avg_accuracy[0][t],avg_accuracy[1][t],avg_accuracy[2][t],avg_accuracy[3][t],avg_accuracy[4][t],avg_accuracy[5][t])
        #print()
        print(f"{SS[t]:.4f}", 
        f"{avg_accuracy[0][t]:.4f}", 
        f"{avg_accuracy[1][t]:.4f}") 
      #f"{avg_accuracy[2][t]:.4f}", 
      #f"{avg_accuracy[3][t]:.4f}", 
      #f"{avg_accuracy[4][t]:.4f}", 
      #f"{avg_accuracy[5][t]:.4f}")        
            
    for t in range(len(AS)):
        avg_forb_norm[t].clear()
        avg_accuracy[t].clear()
        avg_spec_norm[t].clear()
        avg_time[t].clear()
        avg_iso[t].clear()
        std_forb_norm[t].clear()
        std_accuracy[t].clear()
        std_spec_norm[t].clear()
        std_time[t].clear()
        std_iso[t].clear()

