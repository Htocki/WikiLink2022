with open("all.txt", "w") as f:
  beg = 1000000000;
  end = 2000000000;
  for n in range(beg, end):
    n = str(n)
    n = "+375-" + n[1:3] + "-" + n[3:6] + "-" + n[6:8] + "-" + n[8:]
    f.write(n + "\n")
