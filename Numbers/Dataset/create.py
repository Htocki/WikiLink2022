import os, sys

def check_arguments_count():
  if len(sys.argv) != 4:
    print("Ошибка: задано неверное количество аргументов.")
    exit(1)

def get_input_directory():
  return sys.argv[1]

def get_output_directory():
  return sys.argv[2]

def get_count():
  return int(sys.argv[3])

def is_mts(line):
  return \
    (line[:9] == "+375-29-2") or \
    (line[:9] == "+375-29-5") or \
    (line[:9] == "+375-29-7") or \
    (line[:9] == "+375-29-8") or \
    (line[:7] == "+375-33")

def is_a1(line):
  return \
    (line[:9] == "+375-29-1") or \
    (line[:9] == "+375-29-3") or \
    (line[:9] == "+375-29-6") or \
    (line[:9] == "+375-29-9") or \
    (line[:7] == "+375-44")

def is_live(line):
  return line[:7] == "+375-25"

def is_beltelecom(line):
  return line[:7] == "+375-17"

check_arguments_count()
i = get_input_directory()
o = get_output_directory()

f_all = open(str(i) + "/" + "all.txt", "r")

mts_counter = 0
a1_counter = 0
live_counter = 0
beltelecom_counter = 0
others_counter = 0

train_dir = "/dataset/train/"
test_dir = "/dataset/test/"

if not os.path.exists(str(o) + train_dir + "mts"):
  os.makedirs(str(o) + train_dir + "mts")
if not os.path.exists(str(o) + train_dir + "a1"):
  os.makedirs(str(o) + train_dir + "a1")
if not os.path.exists(str(o) + train_dir + "live"):
  os.makedirs(str(o) + train_dir + "live")
if not os.path.exists(str(o) + train_dir + "beltelecom"):
  os.makedirs(str(o) + train_dir + "beltelecom")
if not os.path.exists(str(o) + train_dir + "others"):
  os.makedirs(str(o) + train_dir + "others")

if not os.path.exists(str(o) + test_dir + "mts"):
  os.makedirs(str(o) + test_dir + "mts")
if not os.path.exists(str(o) + test_dir + "a1"):
  os.makedirs(str(o) + test_dir + "a1")
if not os.path.exists(str(o) + test_dir + "live"):
  os.makedirs(str(o) + test_dir + "live")
if not os.path.exists(str(o) + test_dir + "beltelecom"):
  os.makedirs(str(o) + test_dir + "beltelecom")
if not os.path.exists(str(o) + test_dir + "others"):
  os.makedirs(str(o) + test_dir + "others")

count = get_count()

for line in f_all:
  if is_mts(line):
    if mts_counter < count:
      path = str(o) + train_dir + "mts/"
      name = str(mts_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    elif count <= mts_counter < count * 2:
      path = str(o) + test_dir + "mts/"
      name = str(mts_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    mts_counter += 1
  elif is_a1(line):
    if a1_counter < count:
      path = str(o) + train_dir + "a1/"
      name = str(a1_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    elif count <= a1_counter < count * 2:
      path = str(o) + test_dir + "a1/"
      name = str(a1_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    a1_counter += 1
  elif is_live(line):
    if live_counter < count:
      path = str(o) + train_dir + "live/"
      name = str(live_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    elif count <= live_counter < count * 2:
      path = str(o) + test_dir + "live/"
      name = str(live_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    live_counter += 1
  elif is_beltelecom(line):
    if beltelecom_counter < count:
      path = str(o) + train_dir + "beltelecom/"
      name = str(beltelecom_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    elif count <= beltelecom_counter < count * 2:
      path = str(o) + test_dir + "beltelecom/"
      name = str(beltelecom_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    beltelecom_counter += 1
  else:
    if others_counter < count:
      path = str(o) + train_dir + "others/"
      name = str(others_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    elif count <= others_counter < count * 2:
      path = str(o) + test_dir + "others/"
      name = str(others_counter) + ".txt"
      f = open(path + name, "w+")
      f.write(line)
      f.close()
    others_counter += 1

f_all.close()
