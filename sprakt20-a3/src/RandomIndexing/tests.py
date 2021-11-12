lws = 2
rws = 2
w2i ={0:'today', 10:'I', 12:'succeed', 3:'to', 40:'do'}

result = [w2i[k] for k in sorted(w2i.keys(), reverse=False)]
print(result)