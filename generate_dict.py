


start1 = ord('0')
end1 = ord('9') + 1

start2 = ord('A')
end2 = ord('Z') +1 

start3 = ord('a')
end3 = ord('z') + 1

f = open("charlabels.txt", "w")
for x in range(start1,end1):
    f.write(chr(x) +'\n')
for x in range(start2,end2):
    f.write(chr(x)+'\n')
for x in range(start3,end3):
    f.write(chr(x)+'\n')
f.close()