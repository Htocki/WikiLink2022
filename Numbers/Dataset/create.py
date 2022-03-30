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

mts_train_counter = 0
a1_train_counter = 0
live_train_counter = 0
beltelecom_train_counter = 0
others_train_counter = 0

mts_test_counter = 0
a1_test_counter = 0
live_test_counter = 0
beltelecom_test_counter = 0
others_test_counter = 0

train_dir = str(o) + "/dataset/train/"
test_dir = str(o) + "/dataset/test/"

if not os.path.exists(train_dir + "mts"): os.makedirs(train_dir + "mts")
if not os.path.exists(train_dir + "a1"): os.makedirs(train_dir + "a1")
if not os.path.exists(train_dir + "live"): os.makedirs(train_dir + "live")
if not os.path.exists(train_dir + "beltelecom"): os.makedirs(train_dir + "beltelecom")
if not os.path.exists(train_dir + "others"): os.makedirs(train_dir + "others")

if not os.path.exists(test_dir + "mts"): os.makedirs(test_dir + "mts")
if not os.path.exists(test_dir + "a1"): os.makedirs(test_dir + "a1")
if not os.path.exists(test_dir + "live"): os.makedirs(test_dir + "live")
if not os.path.exists(test_dir + "beltelecom"): os.makedirs(test_dir + "beltelecom")
if not os.path.exists(test_dir + "others"): os.makedirs(test_dir + "others")

count = get_count()

def create_file(path, counter, line):
  file_name = str(counter) + ".txt"
  f = open(path + "/" + file_name, "w+")
  f.write(line)
  f.close()

for line in f_all:
  if is_mts(line):
    if mts_train_counter < count:
      create_file(train_dir + "mts", mts_train_counter, line)
      mts_train_counter += 1
    elif mts_train_counter == count and mts_test_counter < count:
      create_file(test_dir + "mts", mts_test_counter, line)
      mts_test_counter += 1
  elif is_a1(line):
    if a1_train_counter < count:
      create_file(train_dir + "a1", a1_train_counter, line)
      a1_train_counter += 1
    elif a1_train_counter == count and a1_test_counter < count:
      create_file(test_dir + "a1", a1_test_counter, line)
      a1_test_counter += 1
  elif is_live(line):
    if live_train_counter < count:
      create_file(train_dir + "live", live_train_counter, line)
      live_train_counter += 1
    elif live_train_counter == count and live_test_counter < count:
      create_file(test_dir + "live", live_test_counter, line)
      live_test_counter += 1
  elif is_beltelecom(line):
    if beltelecom_train_counter < count:
      create_file(train_dir + "beltelecom", beltelecom_train_counter, line)
      beltelecom_train_counter += 1
    elif beltelecom_train_counter == count and beltelecom_test_counter < count:
      create_file(test_dir + "beltelecom", beltelecom_test_counter, line)
      beltelecom_test_counter += 1
  else:
    if others_train_counter < count:
      create_file(train_dir + "others", others_train_counter, line)
      others_train_counter += 1
    elif others_train_counter == count and others_test_counter < count:
      create_file(test_dir + "others", others_test_counter, line)
      others_test_counter += 1
  
  if  mts_train_counter == count and \
      a1_train_counter == count and \
      live_train_counter == count and \
      beltelecom_train_counter == count and \
      others_train_counter == count and \
      mts_test_counter == count and \
      a1_test_counter == count and \
      live_test_counter == count and \
      beltelecom_test_counter == count and \
      others_test_counter == count:
    break

f_all.close()
