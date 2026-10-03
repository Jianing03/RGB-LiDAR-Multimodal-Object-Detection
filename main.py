# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    ckpt = torch.load("./results/early_saved_model_best_f1.pth", map_location="cpu")
    for k in ckpt["model"].keys():
        if "detect_layers" in k:
            print(k)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
