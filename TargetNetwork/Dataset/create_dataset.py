import os

def less(name_a, name_b):
  i = 0
  file = open(name_a, "r")
  for line in file:
    i += 1
  file.close()
  count = i
  i = 0
  file = open(name_b, "r")
  for line in file:
    i += 1
  file.close()
  if count > i:
    count = i
  return count

size = less("anomaly.txt", "normal.txt")
print("count:", size)

train_dir = "./dataset/train"
test_dir = "./dataset/test"
if not os.path.exists(train_dir + "/anomaly"): os.makedirs(train_dir + "/anomaly")
if not os.path.exists(test_dir + "/anomaly"): os.makedirs(test_dir + "/anomaly")
if not os.path.exists(train_dir + "/normal"): os.makedirs(train_dir + "/normal")
if not os.path.exists(test_dir + "/normal"): os.makedirs(test_dir + "/normal")

train_size = size // 2
test_size = size // 2
print("train_size: ", train_size)
print("test_size: ", test_size)

def create_file(path, i, line):
  file_name = str(i) + ".txt"
  f = open(path + "/" + file_name, "w+")
  f.write(line)
  f.close()

file = open("anomaly.txt", "r")
i = 0
for line in file:
  if i < train_size:
    create_file(train_dir + "/anomaly", i, line)
    i += 1
  elif i >= train_size and i < train_size + test_size:
    create_file(test_dir + "/anomaly", i - train_size, line)
    i += 1
  else:
    break
    file.close()

file = open("normal.txt", "r")
i = 0
for line in file:
  if i < train_size:
    create_file(train_dir + "/normal", i, line)
    i += 1
  elif i >= train_size and i < train_size + test_size:
    create_file(test_dir + "/normal", i - train_size, line)
    i += 1
  else:
    break
    file.close()
