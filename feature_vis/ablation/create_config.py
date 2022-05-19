import itertools
l1 = [0, 0.2, 2]
l2 = [0, 0.2, 2]
pad = [0,1,2,5, 10]
tv = [0, -0.000000025, -0.000001]
s = [l1, l2, pad, tv]
choices = list(itertools.product(*s))
print(len(choices))
i = 0
for n in choices:
	if n[0] + n[1] == 0:
		continue
	with open("config_"+str(i)+".cfg", "a+") as f:
		f.write(f"--lasso_1={n[0]}\n--lasso_2={n[1]}\n--padding={n[2]}\n--total_variance={format(n[3], '.8f')}\n--file_name='{i}'\n--file_path='abl_{i}'\n--reproduce\n--steps=2000\n--save_every=500")
		i += 1


