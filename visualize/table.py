import matplotlib.pyplot as plt
import pandas as pd


def makeTable(head,data,name="result_table",):
    data['Cost (×SGD)'] = head
    """  data = {
        'Cost (×SGD)': head,
        'CIFAR-10\nSpeed': [1.00, 1.16, 1.42, 5.76],
        'CIFAR-10\nMemory': [1.00, 1.01, 1.01, 2.12],
        'ImageNet\nSpeed': [1.00, 1.01, 1.23, 11.78],
        'ImageNet\nMemory': [1.00, 1.03, 1.05, 2.51],
        'WMT-14\nSpeed': [1.00, 1.13, 1.19, 8.46],
        'WMT-14\nMemory': [1.00, 1.04, 1.06, 2.47]
    }"""
  
    df = pd.DataFrame(data)

    # Create the table plot
    fig, ax = plt.subplots(figsize=(12, 4))  
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header cells
            cell.set_height(0.15)  #
        else:
             cell.set_height(0.1)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        
    plt.title('Performance Comparison of Different Optimizers', fontsize=16)

    plt.savefig(f'./{name}.png')       