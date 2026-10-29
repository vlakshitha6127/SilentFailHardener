with open('src/silentfail/tools.py', 'r') as f:
    lines = f.readlines()
for i in range(128, 155):
    print(f'{i+1}: {repr(lines[i])}')
