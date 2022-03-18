import sys

def check_arguments_count():
  if len(sys.argv) != 3:
    print("Ошибка: задано неверное количество аргументов.")
    exit(1)

def get_input_directory():
  return sys.argv[1]

def get_output_directory():
  return sys.argv[2]

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
f_mts = open(str(o) + "/" + "mts.txt", "w")
f_a1 = open(str(o) + "/" + "a1.txt", "w")
f_live = open(str(o) + "/" + "live.txt", "w")
f_beltelecom = open(str(o) + "/" + "beltelecom.txt", "w")
f_others = open(str(o) + "/" + "others.txt", "w")

for line in f_all:
  if is_mts(line): f_mts.write(line)
  elif is_a1(line): f_a1.write(line)
  elif is_live(line): f_live.write(line)
  elif is_beltelecom(line): f_beltelecom.write(line)
  else: f_others.write(line)

f_all.close()
f_mts.close()
f_a1.close()
f_live.close()
f_beltelecom.close()
f_others.close()
