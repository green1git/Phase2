def plot_cooks_distances(cooks_distances):
   
    n = len(cooks_distances)  
    threshold = 4 / n  # influential threshold point
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(n), cooks_distances, markerfmt=",", use_line_collection=True)
    plt.axhline(y=threshold, linestyle='--', color='r', label=f'Threshold = (4/n){threshold:.3f}')
    plt.xlabel('Initial Condition')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot for Average Error by Initial Conditions")
    plt.legend()
    plt.show()
