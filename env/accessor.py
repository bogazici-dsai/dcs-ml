# 1. Open and write
filename = r"C:\Users\fisne\PycharmProjects\HIRL4UCAV\bc_actor\straight_line\Agent181_100_-4_Harfang_GYM\data\mydocument.txt"
with open(filename, "w") as file:
    file.write("Hello, this is the first line.\n")
    file.write("And this is the second line.\n")

# 2. Re-open and read
with open(filename, "r") as file:
    content = file.read()
    print("File content:")
    print(content)


import pickle

with open(r'C:\Users\fisne\PycharmProjects\HIRL4UCAV\bc_actor\straight_line\Agent181_100_-4_Harfang_GYM\data.pkl', 'rb') as f:
    expert_buffer = pickle.load(f)
    print(expert_buffer)

print("done")