import sys
def main(input1, input2, output):
    first = True
    with open (input1, "rb") as in1:
        with open (input2, "rb") as in2:
            with open (output, "wb") as ou:
                for line1, line2 in zip(in1, in2):
                    if (first):
                        first = False
                        s = line1.split()
                        ou.write(s[0])
                        ou.write(" ".encode("utf8"))
                        ou.write(str(2 * int(s[1])).encode("utf8"))
                        ou.write("\n".encode("utf8"))
                    else:
                        
                        #try:
                            #line1.decode('utf-8')
                        ou.write(' '.join(map(bytes.decode, line1.split() + line2.split()[1:])).encode('utf8'))
                        #except UnicodeError:
                            #continue
                        
                        ou.write("\n".encode("utf8"))                        
                        
if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2], sys.argv[3])
