import os
import sys
import random

src= sys.argv[1]
des=sys.argv[2]
num = eval(sys.argv[3])


os.system("rm -r "+des)
os.system("mkdir "+des)

count=0 

files=os.listdir(src)
picked=[]
while count<num:
    index=random.randint(0,len(files)-1)
    if index not in picked: 
        picked.append(index)
        count+=1

for i in picked:
    os.system("cp "+src+files[i]+" "+des)





