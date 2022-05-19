import itertools
l1 = [0, 0.2]
l2 = [0, 2]
pad = [0, 2]
tv = [0, -0.00000025]
s = [l1,l2,pad,tv]
choices = list(itertools.product(*s))
for i, n in enumerate(choices[1:]):
	with open("config_"+str(i)+".cfg", "a+") as f:
		f.write(f"--lasso_1={n[0]}\n--lasso_2={n[1]}\n--padding={n[2]}\n--total_variance={n[3]}\n--file_name='{n[0]}_{n[1]}_{n[2]}_{i}'")


