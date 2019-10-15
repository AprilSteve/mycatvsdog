import os

path = 'E:\Jupyter\catanddog\PetImages\Dog'


for file in os.listdir(path):
    old_file = os.path.join(path, file)
    a = "\dog." + file
    print(a)
    new_file = path+a
    os.rename(old_file, new_file)
