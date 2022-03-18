def is_mts(line):
  return  (line[:9] == "+375-29-2") or (line[:9] == "+375-29-5") or (line[:9] == "+375-29-7") or (line[:9] == "+375-29-8") or (line[:7] == "+375-33")

f_all = open("all.txt", "r")
f_mts = open("mts.txt", "w")

for line in f_all:
  if is_mts(line):
    f_mts.write(line)

f_all.close()
f_mts.close()
