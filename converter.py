def convert_annotated_to_raw(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", encoding="utf-8") as outfile:

        line = infile.readline()
        while line:
            tokens = []
            # 读取直到空行或文件结束
            while line.strip() != "":
                parts = line.strip().split()
                if len(parts) >= 1:
                    tokens.append(parts[0])
                line = infile.readline()
            # 将标记连接成一个字符串并写入
            if tokens:
                outfile.write("".join(tokens) + "\n")
            # 跳过条目之间的空行
            while line.strip() == "":
                line = infile.readline()


def convert_test_conll_to_raw(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", encoding="utf-8") as outfile:
        line = infile.readline()
        while line:
            outfile.write(line.split("\u0001")[1])
            line = infile.readline()


if __name__ == "__main__":
    # convert_annotated_to_raw("data/dev.conll",
    #  "data/dev_raw.txt")
    convert_test_conll_to_raw("data/final_test.txt",
                              "data/test_raw.txt")
