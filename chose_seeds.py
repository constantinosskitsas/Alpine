import random

def sample_ground_truth(input_file, output_file, sample_ratio=0.05, seed=None):
    # Optional: set a random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Read all pairs
    with open(input_file, "r", encoding="utf-8") as f:
        pairs = f.readlines()

    # Determine how many to sample
    sample_size = max(1, int(len(pairs) * sample_ratio))

    # Randomly choose pairs
    sampled_pairs = random.sample(pairs, sample_size)

    # Write sampled pairs to output
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(sampled_pairs)

    print(f"Selected {sample_size} pairs out of {len(pairs)} and saved to {output_file}")
Probs=[0.05,0.1,0.15,0.2]
Dataset=["douban","allmv_tmdb","fb_tw","ppi"]
Dataset=["phone"]

#sample_ground_truth(f'Data/data/douban/douban_ground_True.txt',f'./Data/data/douban/douban_ground_True_0.2_0.txt',sample_ratio=0.2)
#exit()
for i in Probs:
    for j in Dataset:
        for iter in range(5):
            print(i,j,iter)
            sample_ground_truth(f'Data/data/{j}/{j}_ground_True.txt',f'./Data/data/{j}/{j}_ground_True_{i}_{iter}.txt',sample_ratio=i)