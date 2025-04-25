import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_split", default="test", choices=["train", "test", "val"])
args = parser.parse_args()
data_split = args.data_split

print("data_split: ", data_split)

output = "SentenceSplitting/output_default.txt"
if data_split == "test":
    new_output = "./output_data/test/uslt_ss_supreme_test.txt"
else:
    new_output = f"./output_data/{data_split}/supreme_{data_split}_uslt_ss.txt"

f = open(output,"r")
fr = f.read()
outputs = fr.split("\n")
f.close()
refined_lines = []
counter = 0
flag = False
refined_lines = []

for line in outputs:
    if len(line) > 0:
        if line[0] == "#":
            if flag == True:
                refined_lines.append(sentence_lines)
            flag = True
            sentence_lines = []
            print("YES")
            counter += 1
            big_flag = True
        
    flag_zero = 0
    
    for i in range(len(line)):
        if line[i-3] == line[i-1] == "\t" and flag_zero == 0:
            print("yes")
            flag_zero = 1
            sentence_lines.append(line[i:])
            break
refined_lines.append(sentence_lines)

g = open(new_output,"w")
for sent_lines in refined_lines:
    for line in sent_lines:
        g.write(line)
        if line[-1] != ".":
            g.write(".")
    if sent_lines != refined_lines[-1]:
        g.write("\n")

print(len(refined_lines))