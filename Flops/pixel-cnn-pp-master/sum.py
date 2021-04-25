
numbers = []
l=0
with open("ops.txt", 'rt') as handle:
    for ln in handle:
        l=l+1
        numbers.append(int(ln))
print(l)    
print('Flops = ',sum(numbers)) 
#27806798720