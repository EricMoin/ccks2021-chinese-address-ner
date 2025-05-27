def convert_annotated_to_raw(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
            open(output_path, "w", encoding="utf-8") as outfile:

        line = infile.readline()
        while line:
            tokens = []
            # Read until a blank line or end of file
            while line.strip() != "":
                parts = line.strip().split()
                if len(parts) >= 1:
                    tokens.append(parts[0])
                line = infile.readline()
            # Join tokens into one string and write
            if tokens:
                outfile.write("".join(tokens) + "\n")
            # Skip empty lines between entries
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
