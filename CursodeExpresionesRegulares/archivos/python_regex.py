#/usr/bin/python

import re

pattern = re.compile(r'^([\d]{4,4})\-\d\d\-\d\d,(.+),(.+),(\d+),(\d+),.*$')

with open("results.csv", "r", encoding="utf-8") as f:
    for line in f:
        res = re.match(pattern, line)
        if res:
            total = int(res.group(4)) + int(res.group(5))
            if total < 1:
                # print("goles: {0:d}, {1:s}: {2:s} [{4:d} - {5:d}] {3:s}\n".format(total, res.group(1), res.group(2), res.group(3), int(res.group(4)), int(res.group(5))))
                print(f'goles: {total}, {res.group(1)}, {res.group(2)}, {res.group(3)}, [{int(res.group(4))}, {int(res.group(5))}]')
                #print(f"{res.group(4)}\n")
        
f.close()

